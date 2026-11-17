"""EvalClient (INTERFACES.md) — stdlib HTTP client for evald."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class EvalError(RuntimeError):
    pass


class EvalClient:
    def __init__(self, base_url: str, token_path: str):
        self.base_url = base_url.rstrip("/")
        with open(token_path) as f:
            self._auth = "Bearer " + f.read().strip()

    @classmethod
    def from_run_dir(cls, run_dir: str) -> "EvalClient":
        """``run_dir`` = $R/run (holds ports.json with key evald_url, and token)."""
        with open(os.path.join(run_dir, "ports.json")) as f:
            ports = json.load(f)
        url = ports.get("evald_url") or f"http://127.0.0.1:{ports['evald']}"
        return cls(url, os.path.join(run_dir, "token"))

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Authorization": self._auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error")
            except Exception:
                msg = str(e)
            raise EvalError(f"evald {method} {path}: HTTP {e.code}: {msg}") from None

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                return bool(json.loads(r.read()).get("ok"))
        except Exception:
            return False

    def submit(self, run: str, task: str, artifact_dir: str, *, seeds: list[int], lane: int | None = None,
               gen: int = 0, trial: int | None = None, events_path: str | None = None,
               emit_submit: bool = True) -> str:
        body = {"run": run, "task": task, "artifact_dir": os.path.abspath(artifact_dir), "seeds": list(seeds),
                "lane": lane, "gen": gen, "trial": trial, "events_path": events_path, "emit_submit": emit_submit}
        return self._req("POST", "/v1/submit", body)["job_id"]

    def result(self, job_id: str, *, wait: bool = True, timeout_s: float = 3600) -> dict:
        """Result dict (keys = eval_result event fields + job_id, per_seed, sealed, error, log_tail, ...).
        ``sealed`` is for the orchestrator's private ledger only: never show it to an improver
        (Task.visible_feedback strips it). Raises TimeoutError if not done in ``timeout_s``."""
        deadline = time.time() + timeout_s
        while True:
            left = deadline - time.time()
            chunk = max(0.0, min(60.0, left)) if wait else 0.0
            r = self._req("GET", f"/v1/result/{job_id}?wait={1 if wait else 0}&timeout={chunk:.1f}",
                          timeout=chunk + 30)
            if r.get("done") or not wait:
                return r
            if time.time() >= deadline:
                raise TimeoutError(f"evald job {job_id} not done after {timeout_s}s (state={r.get('state')})")

    def lanes(self) -> list[dict]:
        return self._req("GET", "/v1/lanes")

    def jobs(self, run: str | None = None) -> list[dict]:
        return self._req("GET", "/v1/jobs" + (f"?run={run}" if run else ""))

    def pause_lane(self, lane: int, *, wait: bool = True, timeout_s: float = 3600) -> None:
        """Stop the lane from starting new jobs. With wait=True (default) also block until
        the job running on it (if any) has finished, so the GPU is free."""
        self._req("POST", f"/v1/lanes/{int(lane)}/pause", {})
        if not wait:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not any(l["lane"] == int(lane) and l["busy"] for l in self.lanes()):
                return
            time.sleep(2.0)
        raise TimeoutError(f"lane {lane} still busy after {timeout_s}s")

    def resume_lane(self, lane: int) -> None:
        self._req("POST", f"/v1/lanes/{int(lane)}/resume", {})
