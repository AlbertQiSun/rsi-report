"""Logging gateway: an OpenAI-compatible proxy in front of vLLM (owner: worker C). Stdlib only.

Every ``POST /v1/chat/completions`` (stream or not) is forwarded to the vLLM instance serving
the requested model, and on completion the gateway
  * emits one ``llm_call`` event (schema v1, no content) into ``<runs_root>/<run>/events.jsonl``;
  * appends the full request/response pair to
    ``<runs_root>/<run>/trajectories/gateway/<agent>.jsonl`` (the LoRA SFT source; see
    ``rsi/runs.py`` for the run-dir layout; ``/`` in agent ids becomes ``__``).

Tagging (all optional; untagged calls go to run ``--default-run``, default ``_untagged``):
  headers  X-RSI-Run, X-RSI-Agent, X-RSI-Gen, X-RSI-Trial, X-RSI-Parent
  or URL prefix  /rsi/t/<run>/<agent, %2F-encoded>/<gen>/<trial or ->/v1/...  (for harnesses
  that can only set a base URL). Headers win over the prefix.

Model routing: ``improver`` -> upstream of base ``improver``; ``improver@<adapter>`` -> the same
upstream, model name passed through unchanged (adapters are loaded into vLLM under the full
``improver@<adapter>`` name, see /rsi/load_adapter); ``ladder-27b`` -> its own upstream.
Upstreams come from ``--upstream base=url`` or from ports.json keys ``vllm_<base>`` (re-read on
change, so a restarted vLLM on a new port is picked up without restarting the gateway).

Busy handling: connection errors and HTTP 429/502/503/504 before any byte reached the client are
retried with exponential backoff and jitter until ``--max-retries`` or ``--retry-deadline-s``;
then the client gets 503 (the orchestrator treats that as "pause", not a failed trial).

Admin (require the main token, i.e. the vLLM API key):
  GET  /rsi/adapters                         -> {base: [model ids]}
  POST /rsi/load_adapter   {"base":"improver","name":"g1","path":"/abs/adapter"}
  POST /rsi/unload_adapter {"base":"improver","name":"g1"}
Other: GET /health, GET /v1/models (union over upstreams), POST /v1/completions (proxied+logged).
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import ports as _ports

try:
    from rsi import events as _events
except Exception:  # pragma: no cover - only if rsi.events is broken
    _events = None

RETRY_STATUSES = {429, 502, 503, 504}
_SAFE_RUN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._\-]{0,199}$")


def safe_agent_file(agent: str) -> str:
    s = agent.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._\-]", "_", s).strip(".") or "agent"
    return s[:200]


def trajectory_path(runs_root: str, run: str, agent: str) -> str:
    return os.path.join(runs_root, run, "trajectories", "gateway", safe_agent_file(agent) + ".jsonl")


def append_locked(path: str, obj: Any) -> None:
    """Append one JSON line under flock, looping on partial writes (big lines on cluster FS)."""
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ------------------------------------------------------------------ config / upstreams
@dataclass
class GatewayConfig:
    runs_root: str = "runs"
    ports_path: str | None = None
    upstreams: dict[str, str] = field(default_factory=dict)   # base -> "http://host:port"
    upstream_token: str | None = None      # vLLM API key (also the admin token)
    inbound_token: str | None = None       # what clients must present; None = no inbound auth
    default_run: str = "_untagged"
    max_retries: int = 8
    retry_deadline_s: float = 120.0
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 15.0
    timeout_s: float = 1800.0
    max_inflight: int = 64
    record: bool = True
    quiet: bool = False


class Upstreams:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._from_ports: dict[str, str] = {}

    def _refresh(self) -> None:
        p = self.cfg.ports_path
        if not p:
            return
        try:
            m = os.stat(p).st_mtime
        except FileNotFoundError:
            return
        if m == self._mtime:
            return
        d = _ports.read_ports(p)
        self._from_ports = {k[len("vllm_"):]: f"http://127.0.0.1:{v}" for k, v in d.items()
                            if k.startswith("vllm_") and isinstance(v, int)}
        self._mtime = m

    def table(self) -> dict[str, str]:
        with self._lock:
            self._refresh()
            t = dict(self._from_ports)
        t.update(self.cfg.upstreams)  # explicit --upstream wins
        return t

    def resolve(self, model: str) -> tuple[str | None, str | None]:
        base = (model or "").split("@", 1)[0]
        return base, self.table().get(base)


# ------------------------------------------------------------------ assembly of responses
class StreamAssembler:
    """Rebuild the final assistant message from SSE chunks (chat or text completions)."""

    def __init__(self):
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.text: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason = None
        self.usage = None
        self.id = None
        self.model = None
        self.is_chat = None

    def feed(self, ch: dict) -> None:
        self.id = self.id or ch.get("id")
        self.model = self.model or ch.get("model")
        if ch.get("usage"):
            self.usage = ch["usage"]
        for c in ch.get("choices") or []:
            if c.get("index", 0) != 0:
                continue
            if c.get("finish_reason"):
                self.finish_reason = c["finish_reason"]
            if "delta" in c:
                self.is_chat = True
                d = c["delta"] or {}
                if d.get("content"):
                    self.content.append(d["content"])
                r = d.get("reasoning_content") or d.get("reasoning")
                if r:
                    self.reasoning.append(r)
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    cur = self.tool_calls.setdefault(i, {"id": None, "type": "function",
                                                         "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name"):
                        cur["function"]["name"] += f["name"]
                    if f.get("arguments"):
                        cur["function"]["arguments"] += f["arguments"]
            elif "text" in c:
                self.is_chat = False
                self.text.append(c.get("text") or "")

    def result(self) -> dict:
        if self.is_chat is False:
            return {"text": "".join(self.text), "finish_reason": self.finish_reason, "id": self.id}
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return {"message": msg, "finish_reason": self.finish_reason, "id": self.id}


def _final_from_body(obj: dict) -> dict:
    ch = (obj.get("choices") or [{}])[0]
    if "message" in ch:
        return {"message": ch.get("message"), "finish_reason": ch.get("finish_reason"), "id": obj.get("id")}
    return {"text": ch.get("text"), "finish_reason": ch.get("finish_reason"), "id": obj.get("id")}


# ------------------------------------------------------------------ the gateway
class Gateway:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self.upstreams = Upstreams(cfg)
        self._sems: dict[str, threading.BoundedSemaphore] = {}
        self._sem_lock = threading.Lock()
        self.stats = {"requests": 0, "errors": 0, "retries": 0}
        self._stats_lock = threading.Lock()

    def sem(self, base: str) -> threading.BoundedSemaphore:
        with self._sem_lock:
            if base not in self._sems:
                self._sems[base] = threading.BoundedSemaphore(self.cfg.max_inflight)
            return self._sems[base]

    def bump(self, k: str, n: int = 1) -> None:
        with self._stats_lock:
            self.stats[k] = self.stats.get(k, 0) + n

    def log(self, msg: str) -> None:
        if not self.cfg.quiet:
            sys.stderr.write(time.strftime("%H:%M:%S ") + "gateway " + msg + "\n")
            sys.stderr.flush()

    # -- upstream HTTP with retry
    def open_upstream(self, url: str, method: str, path: str, body: bytes | None):
        """Return (conn, resp, attempts). resp is None if every attempt failed (error in conn)."""
        u = urllib.parse.urlsplit(url)
        deadline = time.time() + self.cfg.retry_deadline_s
        attempt, last_err = 0, ""
        while True:
            attempt += 1
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.cfg.timeout_s)
            hdrs = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            if self.cfg.upstream_token:
                hdrs["Authorization"] = f"Bearer {self.cfg.upstream_token}"
            try:
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                if resp.status not in RETRY_STATUSES:
                    return conn, resp, attempt
                last_err = f"http {resp.status}"
                resp.read()
            except (ConnectionError, OSError, http.client.HTTPException) as e:
                last_err = f"{type(e).__name__}: {e}"
            conn.close()
            if attempt > self.cfg.max_retries or time.time() >= deadline:
                return last_err, None, attempt
            self.bump("retries")
            sleep = min(self.cfg.backoff_cap_s, self.cfg.backoff_base_s * 2 ** (attempt - 1))
            sleep = min(sleep * (0.5 + random.random()), max(0.0, deadline - time.time()))
            time.sleep(sleep)

    def simple_upstream(self, url: str, method: str, path: str, obj: Any = None) -> tuple[int, Any]:
        body = json.dumps(obj).encode() if obj is not None else None
        conn, resp, _ = self.open_upstream(url, method, path, body)
        if resp is None:
            return 503, {"error": {"message": f"upstream unavailable: {conn}", "type": "upstream_unavailable"}}
        try:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"null")
            except ValueError:
                return resp.status, {"raw": raw.decode(errors="replace")[:2000]}
        finally:
            conn.close()

    def simple_upstream_quick(self, url: str, path: str) -> tuple[int, Any]:
        """GET without retries (used for listings)."""
        u = urllib.parse.urlsplit(url)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        hdrs = {"Authorization": f"Bearer {self.cfg.upstream_token}"} if self.cfg.upstream_token else {}
        try:
            conn.request("GET", path, headers=hdrs)
            r = conn.getresponse()
            raw = r.read()
            return r.status, json.loads(raw or b"null")
        except (OSError, ValueError, http.client.HTTPException):
            return 0, None
        finally:
            conn.close()

    # -- recording
    def record(self, tags: dict, *, model: str, base: str | None, request: dict, final: dict | None,
               usage: dict | None, status: str, http_status: int, t0: float, ttft: float | None,
               attempts: int, stream: bool, error: str | None = None) -> None:
        latency = round(time.time() - t0, 3)
        usage = usage or {}
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        msg = (final or {}).get("message") or {}
        n_tc = len(msg.get("tool_calls") or [])
        run, agent = tags["run"], tags["agent"]
        if self.cfg.record:
            rec = {"t": round(t0, 3), "run": run, "agent": agent, "parent": tags["parent"],
                   "gen": tags["gen"], "trial": tags["trial"], "model": model, "upstream": base,
                   "stream": stream, "status": status, "http_status": http_status,
                   "latency_s": latency, "ttft_s": None if ttft is None else round(ttft, 3),
                   "attempts": attempts, "usage": usage, "request": request, "response": final,
                   "error": error}
            try:
                append_locked(trajectory_path(self.cfg.runs_root, run, agent), rec)
            except OSError as e:
                self.log(f"trajectory write failed: {e}")
        if _events is not None:
            try:
                path = os.path.join(self.cfg.runs_root, run, "events.jsonl")
                _events.get_log(path, run).emit(
                    "llm_call", agent, parent=tags["parent"], gen=tags["gen"], trial=tags["trial"],
                    model=model, prompt_tokens=pt, completion_tokens=ct, latency_s=latency,
                    ttft_s=None if ttft is None else round(ttft, 3), status=status, stream=stream,
                    n_tool_calls=n_tc, finish_reason=(final or {}).get("finish_reason"))
            except Exception as e:  # never fail a model call because of logging
                self.log(f"emit failed: {type(e).__name__}: {e}")
        if status != "ok":
            self.bump("errors")
        self.log(f"{run} {agent} gen={tags['gen']} trial={tags['trial']} {model} {status} "
                 f"{http_status} {latency}s pt={pt} ct={ct} tc={n_tc} tries={attempts}")


def _int_or_none(v: Any) -> int | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RSIGateway/1.0"
    gw: Gateway  # set on subclass

    def log_message(self, fmt, *args):
        pass

    # -- small helpers
    def _json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code: int, msg: str, typ: str = "gateway_error") -> None:
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _bearer(self) -> str:
        a = self.headers.get("Authorization", "")
        return a[7:].strip() if a.startswith("Bearer ") else ""

    def _auth(self, admin: bool = False) -> bool:
        cfg = self.gw.cfg
        need = cfg.upstream_token if admin else cfg.inbound_token
        if admin and not need:
            need = cfg.inbound_token
        if not need or self._bearer() == need:
            return True
        self._err(401, "invalid or missing token", "auth_error")
        return False

    def _read_body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            obj = json.loads(raw or b"{}")
            if not isinstance(obj, dict):
                raise ValueError
            return obj
        except ValueError:
            self._err(400, "body must be a JSON object")
            return None

    def _split_path(self) -> tuple[str, dict]:
        """Strip an optional /rsi/t/<run>/<agent>/<gen>/<trial> prefix; return (path, tags)."""
        p = self.path.split("?")[0]
        tags: dict[str, Any] = {}
        if p.startswith("/rsi/t/"):
            parts = p.split("/")
            # ['', 'rsi', 't', run, agent, gen, trial, 'v1', ...]
            if len(parts) >= 8:
                tags = {"run": urllib.parse.unquote(parts[3]), "agent": urllib.parse.unquote(parts[4]),
                        "gen": parts[5], "trial": parts[6]}
                p = "/" + "/".join(parts[7:])
        return p, tags

    def _tags(self, from_path: dict) -> dict | None:
        h = self.headers
        run = h.get("X-RSI-Run") or from_path.get("run") or self.gw.cfg.default_run
        if not _SAFE_RUN.match(run) or ".." in run:
            self._err(400, "bad X-RSI-Run")
            return None
        agent = h.get("X-RSI-Agent") or from_path.get("agent") or "unknown"
        gen = _int_or_none(h.get("X-RSI-Gen") or from_path.get("gen")) or 0
        trial = _int_or_none(h.get("X-RSI-Trial") or from_path.get("trial"))
        return {"run": run, "agent": agent[:200], "gen": gen, "trial": trial,
                "parent": h.get("X-RSI-Parent") or None}

    # -- routes
    def do_GET(self):
        p, _ = self._split_path()
        if p in ("/health", "/v1/health"):
            return self._json(200, {"ok": True, "upstreams": self.gw.upstreams.table(),
                                    "stats": dict(self.gw.stats)})
        if p == "/v1/models":
            if not self._auth():
                return
            ids, seen = [], set()
            for base, url in self.gw.upstreams.table().items():
                code, obj = self.gw.simple_upstream_quick(url, "/v1/models")
                names = [m.get("id") for m in (obj or {}).get("data", [])] if code == 200 else []
                for m in [base] + [n for n in names if n]:
                    if m not in seen:
                        seen.add(m)
                        ids.append(m)
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "rsi-gateway"} for m in ids]})
        if p == "/rsi/adapters":
            if not self._auth(admin=True):
                return
            out = {}
            for base, url in self.gw.upstreams.table().items():
                code, obj = self.gw.simple_upstream_quick(url, "/v1/models")
                out[base] = [m.get("id") for m in (obj or {}).get("data", [])] if code == 200 else None
            return self._json(200, out)
        self._err(404, f"no route GET {p}")

    def do_POST(self):
        p, ptags = self._split_path()
        if p in ("/rsi/load_adapter", "/rsi/unload_adapter"):
            return self._admin_adapter(p)
        if p not in ("/v1/chat/completions", "/v1/completions"):
            return self._err(404, f"no route POST {p}")
        if not self._auth():
            return
        body = self._read_body()
        if body is None:
            return
        tags = self._tags(ptags)
        if tags is None:
            return
        self._proxy(p, body, tags)

    def _admin_adapter(self, p: str) -> None:
        if not self._auth(admin=True):
            return
        body = self._read_body()
        if body is None:
            return
        base, name = body.get("base", "improver"), body.get("name")
        if not name or "@" in name or "/" in name:
            return self._err(400, "name required (no '@' or '/')")
        url = self.gw.upstreams.table().get(base)
        if not url:
            return self._err(404, f"no upstream for base {base!r}")
        full = f"{base}@{name}"
        if p == "/rsi/load_adapter":
            path = body.get("path")
            if not path or not os.path.isabs(path):
                return self._err(400, "absolute adapter path required")
            req = {"lora_name": full, "lora_path": path}
            if body.get("load_inplace"):
                req["load_inplace"] = True
            code, obj = self.gw.simple_upstream(url, "POST", "/v1/load_lora_adapter", req)
        else:
            code, obj = self.gw.simple_upstream(url, "POST", "/v1/unload_lora_adapter", {"lora_name": full})
        self.gw.log(f"admin {p} {full} -> {code}")
        self._json(code if code else 502, {"model": full, "upstream_status": code, "upstream": obj})

    def _proxy(self, path: str, body: dict, tags: dict) -> None:
        gw, t0 = self.gw, time.time()
        gw.bump("requests")
        model = str(body.get("model") or "")
        base, url = gw.upstreams.resolve(model)
        stream = bool(body.get("stream"))
        common = dict(model=model, base=base, request=body, t0=t0, stream=stream)
        if not url:
            gw.record(tags, final=None, usage=None, status="no_route", http_status=404, ttft=None,
                      attempts=0, error=f"no upstream for model {model!r}", **common)
            return self._err(404, f"model {model!r} not served (known bases: "
                                  f"{sorted(gw.upstreams.table())})", "NotFoundError")
        fwd = dict(body)
        client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        if stream:
            fwd["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        sem = gw.sem(base)
        sem.acquire()
        try:
            conn, resp, attempts = gw.open_upstream(url, "POST", path, json.dumps(fwd).encode())
            if resp is None:
                gw.record(tags, final=None, usage=None, status="upstream_unavailable", http_status=503,
                          ttft=None, attempts=attempts, error=str(conn), **common)
                return self._err(503, f"upstream busy/unavailable after {attempts} attempts: {conn}",
                                 "upstream_unavailable")
            try:
                if resp.status != 200 or not stream:
                    raw = resp.read()
                    ttft = time.time() - t0
                    try:
                        obj = json.loads(raw or b"null")
                    except ValueError:
                        obj = None
                    ok = resp.status == 200 and isinstance(obj, dict)
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.getheader("Content-Type") or "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    try:
                        self.wfile.write(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    gw.record(tags, final=_final_from_body(obj) if ok else obj, usage=(obj or {}).get("usage")
                              if isinstance(obj, dict) else None, status="ok" if ok else f"http_{resp.status}",
                              http_status=resp.status, ttft=ttft if ok else None, attempts=attempts,
                              error=None if ok else raw.decode(errors="replace")[:2000], **common)
                    return
                self._relay_stream(resp, tags, common, attempts, client_wants_usage)
            finally:
                conn.close()
        finally:
            sem.release()

    def _relay_stream(self, resp, tags, common, attempts, client_wants_usage) -> None:
        gw, t0 = self.gw, common["t0"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        asm, ttft, status, err = StreamAssembler(), None, "ok", None
        client_ok = True
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                forward = True
                s = line.strip()
                if s.startswith(b"data:"):
                    payload = s[5:].strip()
                    if payload and payload != b"[DONE]":
                        try:
                            ch = json.loads(payload)
                        except ValueError:
                            ch = None
                        if isinstance(ch, dict):
                            if ch.get("error"):
                                status, err = "stream_error", json.dumps(ch["error"])[:2000]
                            asm.feed(ch)
                            if ttft is None and ch.get("choices"):
                                ttft = time.time() - t0
                            if not client_wants_usage and not ch.get("choices") and ch.get("usage"):
                                forward = False  # we added include_usage; hide the extra chunk
                if forward and client_ok:
                    try:
                        self.wfile.write(line if line.endswith(b"\n") else line + b"\n")
                        if not s:
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        client_ok = False
                        status = "client_disconnected"
                        break  # closing upstream aborts the generation in vLLM
            if client_ok:
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as e:
            status, err = "upstream_stream_error", f"{type(e).__name__}: {e}"
        gw.record(tags, final=asm.result(), usage=asm.usage, status=status, http_status=200, ttft=ttft,
                  attempts=attempts, error=err, **common)


def make_server(cfg: GatewayConfig, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, Gateway]:
    gw = Gateway(cfg)
    handler = type("GatewayHandler", (Handler,), {"gw": gw})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    srv.request_queue_size = 256
    return srv, gw


class GatewayThread:
    """In-process gateway for tests: ``with GatewayThread(cfg) as g: g.url``."""

    def __init__(self, cfg: GatewayConfig, host: str = "127.0.0.1", port: int = 0):
        self.srv, self.gw = make_server(cfg, host, port)
        self.host, self.port = self.srv.server_address[:2]
        self._th = None

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self.base + "/v1"

    def __enter__(self):
        self._th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rsi.gateway", description="RSI logging gateway")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = random free high port")
    ap.add_argument("--ports-file", default=None, help="default $R/run/ports.json")
    ap.add_argument("--ports-key", default="gateway")
    ap.add_argument("--token-file", default=None, help="vLLM API key / admin token; default $R/run/token")
    ap.add_argument("--inbound-token-file", default=None,
                    help="token clients must present (default: same as --token-file)")
    ap.add_argument("--no-inbound-auth", action="store_true", help="accept unauthenticated clients")
    ap.add_argument("--upstream", action="append", default=[], metavar="BASE=URL",
                    help="static upstream, e.g. improver=http://127.0.0.1:23456 (else ports.json vllm_<base>)")
    ap.add_argument("--runs-root", default=None, help="default $RSI_RUNS_ROOT or $R/runs")
    ap.add_argument("--default-run", default="_untagged")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--retry-deadline-s", type=float, default=120.0)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--no-record", action="store_true", help="do not write trajectories")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    ports_path = a.ports_file or _ports.default_ports_path()
    token_file = a.token_file or _ports.default_token_path()
    upstream_token = _ports.read_token(token_file)
    inbound = None if a.no_inbound_auth else _ports.read_token(a.inbound_token_file or token_file)
    if not a.no_inbound_auth and inbound is None:
        print(f"gateway: no token at {a.inbound_token_file or token_file}; use --no-inbound-auth to allow",
              file=sys.stderr)
        return 2
    ups = {}
    for kv in a.upstream:
        k, v = kv.split("=", 1)
        ups[k] = v.rstrip("/").removesuffix("/v1")
    runs_root = a.runs_root or os.environ.get("RSI_RUNS_ROOT") or os.path.join(_ports.default_rsi_root(), "runs")
    cfg = GatewayConfig(runs_root=os.path.abspath(runs_root), ports_path=ports_path, upstreams=ups,
                        upstream_token=upstream_token, inbound_token=inbound, default_run=a.default_run,
                        max_retries=a.max_retries, retry_deadline_s=a.retry_deadline_s, timeout_s=a.timeout_s,
                        max_inflight=a.max_inflight, record=not a.no_record, quiet=a.quiet)
    port = a.port or _ports.free_port(a.host)
    srv, gw = make_server(cfg, a.host, port)
    port = srv.server_address[1]
    _ports.merge_ports({a.ports_key: port}, ports_path)
    print(f"GATEWAY_PORT={port}", flush=True)
    gw.log(f"listening on http://{a.host}:{port}/v1 runs_root={cfg.runs_root} "
           f"upstreams={gw.upstreams.table()} inbound_auth={'on' if inbound else 'off'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
