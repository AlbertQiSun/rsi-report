"""evald HTTP API on 127.0.0.1 with bearer-token auth (stdlib only).

  GET  /health                         no auth: {"ok": true}
  POST /v1/submit                      {run, task, artifact_dir, seeds, lane?, gen?, trial?, events_path?, emit_submit?}
  GET  /v1/result/<job_id>?wait=1&timeout=30
  GET  /v1/lanes
  POST /v1/lanes/<n>/pause | /v1/lanes/<n>/resume
  GET  /v1/jobs?run=<run>
The token comes from a mode-600 file; it is never logged or put on a command line.
"""

from __future__ import annotations

import fcntl
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rsi.evald.service import EvalService

MAX_BODY = 1 << 20


def read_token(path: str) -> str:
    st = os.stat(path)
    if st.st_mode & 0o077:
        raise PermissionError(f"token file {path} must be mode 600")
    with open(path) as f:
        tok = f.read().strip()
    if len(tok) < 16:
        raise ValueError("token too short")
    return tok


def make_handler(service: EvalService, token: str):
    expected = ("Bearer " + token).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "rsi-evald/1"

        def log_message(self, fmt, *args):  # no request logging (headers carry the token)
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            got = (self.headers.get("Authorization") or "").encode()
            if hmac.compare_digest(got, expected):
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            data = self.rfile.read(n) if n else b"{}"
            obj = json.loads(data or b"{}")
            if not isinstance(obj, dict):
                raise ValueError("body must be a JSON object")
            return obj

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authed():
                return
            q = parse_qs(u.query)
            try:
                parts = [p for p in u.path.split("/") if p]
                if parts[:2] == ["v1", "result"] and len(parts) == 3:
                    wait = q.get("wait", ["1"])[0] not in ("0", "false")
                    timeout = min(float(q.get("timeout", ["30"])[0]), 120.0)
                    return self._send(200, service.result(parts[2], wait=wait, timeout_s=timeout))
                if parts == ["v1", "lanes"]:
                    return self._send(200, service.lanes_info())
                if parts == ["v1", "jobs"]:
                    return self._send(200, service.list_jobs(q.get("run", [None])[0]))
                return self._send(404, {"error": "not found"})
            except KeyError:
                return self._send(404, {"error": "unknown job"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": f"{type(e).__name__}: {e}"[:300]})

        def do_POST(self):
            u = urlparse(self.path)
            if not self._authed():
                return
            try:
                parts = [p for p in u.path.split("/") if p]
                if parts == ["v1", "submit"]:
                    b = self._body()
                    jid = service.submit(b["run"], b["task"], b["artifact_dir"], seeds=b.get("seeds") or [0],
                                         lane=b.get("lane"), gen=int(b.get("gen") or 0), trial=b.get("trial"),
                                         events_path=b.get("events_path"),
                                         emit_submit=bool(b.get("emit_submit", True)))
                    return self._send(200, {"job_id": jid})
                if len(parts) == 4 and parts[:2] == ["v1", "lanes"] and parts[3] in ("pause", "resume"):
                    lane = int(parts[2])
                    (service.pause_lane if parts[3] == "pause" else service.resume_lane)(lane)
                    return self._send(200, {"ok": True, "lanes": service.lanes_info()})
                return self._send(404, {"error": "not found"})
            except KeyError as e:
                return self._send(400, {"error": f"missing or unknown: {e}"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": f"{type(e).__name__}: {e}"[:300]})

    return Handler


class EvaldServer:
    def __init__(self, service: EvalService, token: str, host: str = "127.0.0.1", port: int = 0):
        self.service = service
        self.httpd = ThreadingHTTPServer((host, port), make_handler(service, token))
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]
        self._th: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "EvaldServer":
        self._th = threading.Thread(target=self.httpd.serve_forever, name="evald-http", daemon=True)
        self._th.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def write_port(ports_file: str, key: str, port: int, host: str = "127.0.0.1") -> None:
    """Merge {key: port, key_url: url} into the shared ports.json under a lock."""
    os.makedirs(os.path.dirname(os.path.abspath(ports_file)), exist_ok=True)
    with open(ports_file + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(ports_file) as f:
                cur = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cur = {}
        cur[key] = port
        cur[key + "_url"] = f"http://{host}:{port}"
        tmp = ports_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, indent=1, sort_keys=True)
        os.replace(tmp, ports_file)
        fcntl.flock(lk, fcntl.LOCK_UN)
