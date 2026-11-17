#!/usr/bin/env python3
"""Tiny OpenAI-compatible fake model server for CPU tests (owner: worker C). Stdlib only.

CLI (stable):
    python tests/fake_openai_server.py --port N --script file.json [--host 127.0.0.1]
        [--ports-file PATH --ports-key NAME] [--token-file PATH | --token-env VAR]
    --port 0 picks a free port. The bound port is printed on stdout as
    ``FAKE_OPENAI_PORT=<port>`` (one line, flushed) once the server is listening.

In-process use (preferred inside pytest):
    from tests.fake_openai_server import FakeServer
    with FakeServer({"queue": [{"content": "hi"}]}) as fs:
        fs.url            # "http://127.0.0.1:<port>/v1"
        fs.requests       # list of received request bodies (dicts)

Endpoints: GET /v1/models, POST /v1/chat/completions (stream and non-stream),
GET /health, POST /v1/load_lora_adapter, POST /v1/unload_lora_adapter (mimic vLLM),
GET /_fake/requests (received bodies), POST /_fake/script (replace the script),
POST /_fake/reset (clear request log).

Script JSON (every key optional):
{
  "models":  ["improver", "ladder-27b"],      # listed by /v1/models; unknown models still answered
  "strict_models": false,                      # true -> 404 for a model not in "models"
  "latency_s": 0.0,                            # added delay per response
  "queue":  [RESP, ...],                       # consumed in order, all models
  "queues": {"improver": [RESP, ...]},         # per-model queues, tried before "queue"
  "rules":  [{"match": MATCH, "respond": RESP, "times": 3}],   # first matching rule wins
  "default": RESP                              # fallback; built-in default echoes the last user message
}
RESP:
{
  "content": "text" | null,
  "reasoning_content": "...",                  # emitted like vLLM's reasoning parser does
  "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}],   # arguments: dict or raw string
  "finish_reason": "stop" | "tool_calls" | "length",   # default inferred
  "echo": true,                                # content = last user message text
  "status": 503, "error": "busy",              # return an HTTP error instead (simulate a busy vLLM)
  "delay_s": 0.5,
  "times": 2                                   # queue entries only: serve this entry N times
}
MATCH (all given keys must hold):
  model, last_role, has_tools (bool), tool_choice, n_tool_msgs (int), min_tool_msgs (int),
  last_user_contains, last_user_regex, last_contains, any_contains
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["FakeServer", "FakeState", "serve", "main"]


# ---------------------------------------------------------------- helpers
def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _ntok(s: str) -> int:
    return max(1, len(s.split())) if s else 0


def _merge_ports(path: str, key: str, port: int) -> None:
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = b""
        while chunk := os.read(fd, 65536):
            raw += chunk
        try:
            data = json.loads(raw.decode() or "{}")
        except ValueError:
            data = {}
        data[key] = port
        os.lseek(fd, 0, 0)
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(data, indent=1, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


class FakeState:
    def __init__(self, script: dict | None = None, token: str | None = None):
        self.lock = threading.Lock()
        self.token = token
        self.requests: list[dict] = []
        self.set_script(script or {})

    def set_script(self, script: dict) -> None:
        with self.lock:
            self.script = copy.deepcopy(script)
            self.models = list(self.script.get("models") or ["improver", "ladder-27b"])
            self.queue = list(self.script.get("queue") or [])
            self.queues = {k: list(v) for k, v in (self.script.get("queues") or {}).items()}
            self.rules = [dict(r) for r in (self.script.get("rules") or [])]

    # pick the response spec for one request
    def pick(self, body: dict) -> dict:
        model = body.get("model", "")
        with self.lock:
            for q in (self.queues.get(model), self.queue):
                if q:
                    spec = q[0]
                    n = int(spec.get("times", 1))
                    if n <= 1:
                        q.pop(0)
                    else:
                        spec = dict(spec)
                        q[0]["times"] = n - 1
                    return spec
            for r in self.rules:
                if "times" in r and r["times"] <= 0:
                    continue
                if _match(r.get("match") or {}, body):
                    if "times" in r:
                        r["times"] -= 1
                    return r.get("respond") or {}
            return self.script.get("default") or {"echo": True}


def _match(m: dict, body: dict) -> bool:
    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else {}
    users = [x for x in msgs if x.get("role") == "user"]
    last_user = _text_of(users[-1].get("content")) if users else ""
    n_tool = sum(1 for x in msgs if x.get("role") == "tool")
    for k, v in m.items():
        if k == "model":
            ok = body.get("model") == v
        elif k == "last_role":
            ok = last.get("role") == v
        elif k == "has_tools":
            ok = bool(body.get("tools")) == bool(v)
        elif k == "tool_choice":
            ok = body.get("tool_choice") == v
        elif k == "n_tool_msgs":
            ok = n_tool == int(v)
        elif k == "min_tool_msgs":
            ok = n_tool >= int(v)
        elif k == "last_user_contains":
            ok = v in last_user
        elif k == "last_user_regex":
            ok = re.search(v, last_user) is not None
        elif k == "last_contains":
            ok = v in _text_of(last.get("content"))
        elif k == "any_contains":
            ok = any(v in _text_of(x.get("content")) for x in msgs)
        else:
            ok = False  # unknown key never matches (fail loudly in tests)
        if not ok:
            return False
    return True


def _build_message(spec: dict, body: dict) -> tuple[dict, str]:
    msgs = body.get("messages") or []
    content = spec.get("content")
    if spec.get("echo") or ("content" not in spec and not spec.get("tool_calls")):
        users = [x for x in msgs if x.get("role") == "user"]
        content = "echo: " + (_text_of(users[-1].get("content")) if users else "")
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if spec.get("reasoning_content"):
        msg["reasoning_content"] = spec["reasoning_content"]
    tcs = []
    for i, tc in enumerate(spec.get("tool_calls") or []):
        args = tc.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        tcs.append({"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                    "function": {"name": tc.get("name", ""), "arguments": args}})
    if tcs:
        msg["tool_calls"] = tcs
    finish = spec.get("finish_reason") or ("tool_calls" if tcs else "stop")
    return msg, finish


def _usage(body: dict, msg: dict) -> dict:
    p = sum(_ntok(_text_of(m.get("content"))) for m in body.get("messages") or [])
    p += _ntok(json.dumps(body.get("tools"))) if body.get("tools") else 0
    c = _ntok(_text_of(msg.get("content"))) + _ntok(msg.get("reasoning_content") or "")
    c += sum(_ntok(t["function"]["arguments"]) + 1 for t in msg.get("tool_calls") or [])
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _chunks(s: str, n: int = 8) -> list[str]:
    return [s[i:i + n] for i in range(0, len(s), n)] or [""]


# ---------------------------------------------------------------- HTTP
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeOpenAI/1.0"
    state: FakeState  # set on subclass

    def log_message(self, fmt, *args):  # quiet
        if os.environ.get("FAKE_OPENAI_VERBOSE"):
            sys.stderr.write("fake: " + fmt % args + "\n")

    def _json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code: int, msg: str, typ: str = "error") -> None:
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _authed(self) -> bool:
        tok = self.state.token
        if not tok:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {tok}":
            return True
        self._err(401, "invalid api key", "auth_error")
        return False

    def _body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            self._err(400, "invalid json")
            return None

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/health", "/v1/health"):
            return self._json(200, {"ok": True})
        if not self._authed():
            return
        if p == "/v1/models":
            with self.state.lock:
                models = list(self.state.models)
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "fake"} for m in models]})
        if p == "/_fake/requests":
            with self.state.lock:
                return self._json(200, list(self.state.requests))
        self._err(404, f"no route {p}")

    def do_POST(self):
        p = self.path.split("?")[0]
        if not self._authed():
            return
        body = self._body()
        if body is None:
            return
        st = self.state
        if p == "/_fake/script":
            st.set_script(body)
            return self._json(200, {"ok": True})
        if p == "/_fake/reset":
            with st.lock:
                st.requests.clear()
            return self._json(200, {"ok": True})
        if p == "/v1/load_lora_adapter":
            name = body.get("lora_name")
            if not name:
                return self._err(400, "lora_name required")
            with st.lock:
                if name not in st.models:
                    st.models.append(name)
            return self._json(200, {"ok": True})
        if p == "/v1/unload_lora_adapter":
            with st.lock:
                if body.get("lora_name") in st.models:
                    st.models.remove(body["lora_name"])
            return self._json(200, {"ok": True})
        if p != "/v1/chat/completions":
            return self._err(404, f"no route {p}")
        with st.lock:
            st.requests.append(body)
            strict, models = st.script.get("strict_models"), list(st.models)
        if strict and body.get("model") not in models:
            return self._err(404, f"model {body.get('model')!r} not found", "NotFoundError")
        if not isinstance(body.get("messages"), list):
            return self._err(400, "messages must be a list")
        spec = st.pick(body)
        delay = float(spec.get("delay_s", 0)) + float(st.script.get("latency_s", 0))
        if delay:
            time.sleep(delay)
        if spec.get("status"):
            return self._err(int(spec["status"]), spec.get("error", "fake error"))
        msg, finish = _build_message(spec, body)
        usage = _usage(body, msg)
        cid, created, model = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time()), body.get("model", "")
        if not body.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                    "model": model, "usage": usage,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish,
                                                 "logprobs": None}]})
        self._stream(cid, created, model, msg, finish, usage,
                     bool((body.get("stream_options") or {}).get("include_usage")))

    def _stream(self, cid, created, model, msg, finish, usage, include_usage):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(delta: dict | None, fin: str | None = None, usage_obj: dict | None = None):
            ch = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                  "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": fin,
                                                        "logprobs": None}]}
            if usage_obj is not None:
                ch["usage"] = usage_obj
            self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.flush()

        send({"role": "assistant", "content": ""})
        if msg.get("reasoning_content"):
            for piece in _chunks(msg["reasoning_content"]):
                send({"reasoning_content": piece})
        if msg.get("content"):
            for piece in _chunks(msg["content"]):
                send({"content": piece})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            send({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                  "function": {"name": tc["function"]["name"], "arguments": ""}}]})
            for piece in _chunks(tc["function"]["arguments"], 16):
                send({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
        send({}, finish)
        if include_usage:
            send(None, usage_obj=usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(host: str, port: int, state: FakeState) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"state": state})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


class FakeServer:
    """Run the fake in a background thread. ``with FakeServer(script) as fs: fs.url``."""

    def __init__(self, script: dict | None = None, *, host: str = "127.0.0.1", port: int = 0,
                 token: str | None = None):
        self.state = FakeState(script, token)
        self._srv = serve(host, port, self.state)
        self.host, self.port = self._srv.server_address[:2]
        self._th: threading.Thread | None = None

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self.base + "/v1"

    @property
    def requests(self) -> list[dict]:
        with self.state.lock:
            return list(self.state.requests)

    def set_script(self, script: dict) -> None:
        self.state.set_script(script)

    def start(self) -> "FakeServer":
        self._th = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._th.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def __enter__(self) -> "FakeServer":
        return self.start()

    def __exit__(self, *a) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--script", help="script JSON file (see module docstring)")
    ap.add_argument("--token-file", help="require 'Authorization: Bearer <contents>'")
    ap.add_argument("--token-env", help="require 'Authorization: Bearer $<this env var>'")
    ap.add_argument("--ports-file", help="merge {ports-key: port} into this JSON file")
    ap.add_argument("--ports-key", default="fake_openai")
    a = ap.parse_args(argv)
    script = {}
    if a.script:
        with open(a.script) as f:
            script = json.load(f)
    token = None
    if a.token_file:
        with open(a.token_file) as f:
            token = f.read().strip()
    elif a.token_env:
        token = os.environ.get(a.token_env) or None
    state = FakeState(script, token)
    srv = serve(a.host, a.port, state)
    port = srv.server_address[1]
    if a.ports_file:
        _merge_ports(a.ports_file, a.ports_key, port)
    print(f"FAKE_OPENAI_PORT={port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
