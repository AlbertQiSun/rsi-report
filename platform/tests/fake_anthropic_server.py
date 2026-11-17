#!/usr/bin/env python3
"""Tiny Anthropic-Messages-compatible fake model server for CPU tests (mimics vLLM's /v1/messages). Stdlib only.

In-process:
    from tests.fake_anthropic_server import FakeAnthropic
    with FakeAnthropic({"queue": [{"text": "hi"}]}, token="main") as fa:
        fa.base          # "http://127.0.0.1:<port>"  (clients append /v1/messages)
        fa.requests      # received bodies;  fa.headers: received headers (lower-case keys)

CLI:  python tests/fake_anthropic_server.py --port 0 [--script file.json]   -> prints FAKE_ANTHROPIC_PORT=<port>

Endpoints: POST /v1/messages (stream and non-stream, SSE events exactly as the Anthropic API names them),
POST /v1/messages/count_tokens, GET /v1/models, GET /health.

Script: {"queue": [RESP, ...], "default": RESP}. RESP (all optional):
  {"text": "...", "thinking": "...", "tool_uses": [{"name": "Bash", "input": {...}, "id": "toolu_1"}],
   "raw_input": "{bad json",            # emit this partial_json for the (first) tool_use instead of json(input)
   "stop_reason": "end_turn" | "tool_use" | "max_tokens",   # default inferred
   "status": 503, "error": "busy"}      # HTTP error instead
Default response echoes the last user text.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _last_user_text(body: dict) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            return " ".join(b.get("text", "") for b in c or [] if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _ntok(body: dict) -> int:
    return max(1, len(json.dumps(body.get("messages") or [])) // 4)


def build_message(spec: dict, body: dict) -> dict:
    content: list[dict] = []
    if spec.get("thinking"):
        content.append({"type": "thinking", "thinking": spec["thinking"], "signature": ""})
    text = spec.get("text")
    if text is None and not spec.get("tool_uses"):
        text = "echo: " + _last_user_text(body)
    if text:
        content.append({"type": "text", "text": text})
    for i, tu in enumerate(spec.get("tool_uses") or []):
        content.append({"type": "tool_use", "id": tu.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": tu["name"], "input": tu.get("input") or {}})
    stop = spec.get("stop_reason") or ("tool_use" if spec.get("tool_uses") else "end_turn")
    out_tok = max(1, len(json.dumps(content)) // 4)
    return {"id": f"msg_{uuid.uuid4().hex[:16]}", "type": "message", "role": "assistant",
            "model": body.get("model"), "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": _ntok(body), "output_tokens": out_tok}}


def sse_events(msg: dict, raw_input: str | None = None) -> list[tuple[str, dict]]:
    start = dict(msg, content=[], stop_reason=None)
    start["usage"] = {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 1}
    evs: list[tuple[str, dict]] = [("message_start", {"type": "message_start", "message": start})]
    first_tool = True
    for i, b in enumerate(msg["content"]):
        if b["type"] == "text":
            evs.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "text", "text": ""}}))
            t = b["text"]
            for j in range(0, len(t), 7):
                evs.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "text_delta", "text": t[j:j + 7]}}))
        elif b["type"] == "thinking":
            evs.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "thinking", "thinking": ""}}))
            evs.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "thinking_delta", "thinking": b["thinking"]}}))
        else:
            evs.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}))
            raw = raw_input if (raw_input is not None and first_tool) else json.dumps(b["input"])
            first_tool = False
            for j in range(0, len(raw), 5):
                evs.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "input_json_delta", "partial_json": raw[j:j + 5]}}))
        evs.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    evs.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": msg["stop_reason"],
                                                                     "stop_sequence": None},
                                  "usage": {"output_tokens": msg["usage"]["output_tokens"]}}))
    evs.append(("message_stop", {"type": "message_stop"}))
    return evs


class State:
    def __init__(self, script: dict | None, token: str | None):
        self.lock = threading.Lock()
        self.token = token
        self.script = copy.deepcopy(script or {})
        self.queue = list(self.script.get("queue") or [])
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.paths: list[str] = []

    def pick(self) -> dict:
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return self.script.get("default") or {}


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            if not st.token:
                return True
            a = self.headers.get("Authorization", "")
            if a == f"Bearer {st.token}" or self.headers.get("x-api-key") == st.token:
                return True
            self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "bad token"}})
            return False

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._json(200, {"ok": True})
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [{"id": "improver", "object": "model"}]})
            self._json(404, {"error": "no route"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.split("?")[0]
            with st.lock:
                st.requests.append(body)
                st.headers.append({k.lower(): v for k, v in self.headers.items()})
                st.paths.append(self.path)
            if not self._authed():
                return
            if path == "/v1/messages/count_tokens":
                return self._json(200, {"input_tokens": _ntok(body)})
            if path != "/v1/messages":
                return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})
            spec = st.pick()
            if spec.get("status"):
                return self._json(int(spec["status"]), {"type": "error", "error": {
                    "type": "overloaded_error", "message": spec.get("error", "error")}})
            msg = build_message(spec, body)
            if not body.get("stream"):
                return self._json(200, msg)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for name, ev in sse_events(msg, spec.get("raw_input")):
                self.wfile.write(f"event: {name}\ndata: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

    return H


class _Srv(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class FakeAnthropic:
    def __init__(self, script: dict | None = None, *, token: str | None = None, host: str = "127.0.0.1", port: int = 0):
        self.state = State(script, token)
        self._srv = _Srv((host, port), make_handler(self.state))
        self.host, self.port = self._srv.server_address[:2]

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def requests(self) -> list[dict]:
        with self.state.lock:
            return list(self.state.requests)

    @property
    def headers(self) -> list[dict]:
        with self.state.lock:
            return list(self.state.headers)

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--script")
    a = ap.parse_args(argv)
    script = json.load(open(a.script)) if a.script else {}
    fa = FakeAnthropic(script, host=a.host, port=a.port)
    print(f"FAKE_ANTHROPIC_PORT={fa.port}", flush=True)
    try:
        fa._srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
