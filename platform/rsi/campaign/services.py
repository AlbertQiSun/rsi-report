"""Backends the campaign talks to: evald lanes, the vLLM LoRA admin API, health probes.

Real backends speak HTTP to 127.0.0.1 ports from ``$R/run/ports.json`` with the
token from ``$R/run/token``. The token is read from the file at call time and only
ever goes into an ``Authorization`` header; it is never logged or put in argv.
Dry-run backends live in ``rsi.campaign.fakes``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any


def read_json_file(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_token(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


class Ports:
    def __init__(self, path: str):
        self.path = path

    def get(self, name: str) -> str:
        p = read_json_file(self.path).get(name)
        if isinstance(p, dict):  # {"host":..., "port":...}
            p = p.get("port")
        return str(p) if p is not None else f"<no-port:{name}>"


def http(method: str, url: str, token_path: str | None, body: Any = None, timeout: float = 10.0) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    tok = read_token(token_path) if token_path else None
    if tok:
        headers["Authorization"] = "Bearer " + tok
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw or b"null")
            except ValueError:
                return r.status, raw.decode(errors="replace")[:500]
    except urllib.error.HTTPError as e:
        return e.code, str(e)[:300]
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, str(e)[:300]


# ---------------------------------------------------------------------------
# health probes
# ---------------------------------------------------------------------------

class HttpProbe:
    def __init__(self, token_path: str):
        self.token_path = token_path

    def check(self, name: str, url: str) -> tuple[bool, str]:
        if "<no-port:" in url:
            return False, "port unknown in ports.json"
        # Liveness, not correctness: any HTTP answer below 500 (even 401/404) means the process is up.
        code, _ = http("GET", url, self.token_path, timeout=8)
        return (0 < code < 500), f"http {code}"


def gpu_memory() -> list[dict] | None:
    """[{gpu, used_mb, total_mb}] via nvidia-smi, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.strip().splitlines():
        try:
            i, used, total = [x.strip() for x in line.split(",")]
            rows.append({"gpu": int(i), "used_mb": float(used), "total_mb": float(total)})
        except ValueError:
            continue
    return rows


def disk_free_gb(path: str) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


# ---------------------------------------------------------------------------
# evald lanes (worker B's EvalClient)
# ---------------------------------------------------------------------------

def make_eval_client(base_url: str, token_path: str):
    from rsi.evald.client import EvalClient  # worker B

    return EvalClient(base_url, token_path)


# ---------------------------------------------------------------------------
# vLLM runtime LoRA loading (vLLM --enable-lora with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
# ---------------------------------------------------------------------------

class VllmLoraLoader:
    def __init__(self, load_url: str, models_url: str, token_path: str, unload_url: str | None = None):
        self.load_url, self.models_url, self.unload_url = load_url, models_url, unload_url
        self.token_path = token_path

    def load(self, name: str, path: str) -> tuple[bool, str]:
        code, body = http("POST", self.load_url, self.token_path, {"lora_name": name, "lora_path": path}, timeout=300)
        if 200 <= code < 300:
            return True, "loaded"
        if isinstance(body, str) and "already" in body.lower():
            return True, "already loaded"
        return False, f"http {code}: {str(body)[:200]}"

    def verify(self, name: str) -> bool:
        code, body = http("GET", self.models_url, self.token_path, timeout=20)
        if not (200 <= code < 300) or not isinstance(body, dict):
            return False
        return any(m.get("id") == name for m in body.get("data", []) if isinstance(m, dict))
