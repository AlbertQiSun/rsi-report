"""Minimal stdlib client for the gateway (used by toolcall_test, U_s lessons writer, policies).

    from rsi.gateway import GatewayClient, gateway_url
    c = GatewayClient(gateway_url(), token_path="$R/run/token")
    r = c.chat("improver", [{"role": "user", "content": "hi"}], run="r1", agent="thread-0", gen=0, trial=3)
    r["choices"][0]["message"]
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from typing import Any

from . import ports as _ports


def gateway_url(ports_path: str | None = None, key: str = "gateway") -> str:
    """``http://127.0.0.1:<port>/v1`` from ports.json (raises KeyError if not running)."""
    p = _ports.read_ports(ports_path).get(key)
    if not p:
        raise KeyError(f"{key!r} not in {ports_path or _ports.default_ports_path()}")
    return f"http://127.0.0.1:{p}/v1"


def rsi_headers(run: str | None = None, agent: str | None = None, gen: int | None = None,
                trial: int | None = None, parent: str | None = None) -> dict[str, str]:
    h = {}
    for k, v in (("Run", run), ("Agent", agent), ("Gen", gen), ("Trial", trial), ("Parent", parent)):
        if v is not None:
            h[f"X-RSI-{k}"] = str(v)
    return h


class GatewayError(RuntimeError):
    def __init__(self, status: int, body: Any):
        super().__init__(f"gateway HTTP {status}: {str(body)[:500]}")
        self.status, self.body = status, body


class GatewayClient:
    def __init__(self, base_url: str, token_path: str | None = None, token: str | None = None,
                 timeout_s: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.token = token if token is not None else _ports.read_token(token_path)
        self.timeout_s = timeout_s
        u = urllib.parse.urlsplit(self.base_url)
        self._host, self._port, self._prefix = u.hostname, u.port or 80, u.path[: -len("/v1")]

    def _req(self, method: str, path: str, obj: Any = None, headers: dict | None = None):
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout_s)
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        h.update(headers or {})
        conn.request(method, self._prefix + path, body=None if obj is None else json.dumps(obj).encode(), headers=h)
        return conn, conn.getresponse()

    def models(self) -> list[str]:
        conn, r = self._req("GET", "/v1/models")
        try:
            body = json.loads(r.read() or b"{}")
        finally:
            conn.close()
        if r.status != 200:
            raise GatewayError(r.status, body)
        return [m["id"] for m in body.get("data", [])]

    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None,
             run: str | None = None, agent: str | None = None, gen: int | None = None,
             trial: int | None = None, parent: str | None = None, stream: bool = False,
             **params: Any) -> dict:
        """One chat completion. With ``stream=True`` the SSE stream is consumed and reassembled,
        and the result has the non-stream shape plus ``_ttft_s``."""
        body: dict[str, Any] = {"model": model, "messages": messages, **params}
        if tools:
            body["tools"] = tools
        if stream:
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
        t0 = time.time()
        conn, r = self._req("POST", "/v1/chat/completions", body, rsi_headers(run, agent, gen, trial, parent))
        try:
            if r.status != 200:
                raw = r.read()
                try:
                    err = json.loads(raw)
                except ValueError:
                    err = raw.decode(errors="replace")
                raise GatewayError(r.status, err)
            if not stream:
                out = json.loads(r.read())
                out["_latency_s"] = round(time.time() - t0, 3)
                return out
            from .server import StreamAssembler
            asm, ttft = StreamAssembler(), None
            while True:
                line = r.readline()
                if not line:
                    break
                s = line.strip()
                if not s.startswith(b"data:"):
                    continue
                p = s[5:].strip()
                if p == b"[DONE]":
                    break
                ch = json.loads(p)
                if ttft is None and ch.get("choices"):
                    ttft = time.time() - t0
                asm.feed(ch)
            res = asm.result()
            return {"id": res.get("id"), "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": res.get("message"), "finish_reason": res.get("finish_reason")}],
                    "usage": asm.usage, "_ttft_s": ttft, "_latency_s": round(time.time() - t0, 3)}
        finally:
            conn.close()

    def load_adapter(self, name: str, path: str, base: str = "improver", inplace: bool = False) -> dict:
        conn, r = self._req("POST", "/rsi/load_adapter",
                            {"base": base, "name": name, "path": path, "load_inplace": inplace})
        try:
            body = json.loads(r.read() or b"{}")
        finally:
            conn.close()
        if r.status != 200:
            raise GatewayError(r.status, body)
        return body

    def unload_adapter(self, name: str, base: str = "improver") -> dict:
        conn, r = self._req("POST", "/rsi/unload_adapter", {"base": base, "name": name})
        try:
            body = json.loads(r.read() or b"{}")
        finally:
            conn.close()
        if r.status != 200:
            raise GatewayError(r.status, body)
        return body
