"""Local end-to-end smoke run (Mac CPU, no GPU, no internet).

    python -m rsi.smoke [--out samples/events_smoke.jsonl] [--trials 6] [--threads 2] [--task muar/train-modadd-0]

Wires the real components together with a fake model:
    fake OpenAI server (tests/fake_openai_server.py, scripted tool calls)
      <- logging gateway (rsi.gateway, emits llm_call + writes trajectories/gateway/)
      <- mini harness (rsi.harness.mini, sandbox "none" on the Mac) with the lab spool
      <- karpathy_greedy policy, 2 threads mapped to 2 eval lanes
      -> local CPU evaluator (µAR task, real training, subprocess per eval) emitting eval_result/audit_flag
and validates the resulting events.jsonl against the schema. The cluster path differs only in
the harness (OpenCode in bwrap), the evaluator (evald on GPU lanes) and the model (vLLM).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from rsi import events, orchestrator


class LocalEvalClient:
    """EvalClient-compatible local CPU evaluator for µAR (one subprocess per eval, lanes = worker slots)."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.pool = ThreadPoolExecutor(max_workers=int(cfg.get("workers", 2)))
        self.futs: dict[str, object] = {}
        self.lock = threading.Lock()

    def _eval(self, run, task, artifact_dir, seeds, lane, gen, trial) -> dict:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "rsi.tasks.muar.evaluate", "--task", task, "--artifact",
                            artifact_dir, "--seeds", ",".join(map(str, seeds))],
                           capture_output=True, text=True, timeout=600,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        try:
            res = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            res = {"metric": "val_loss", "value": None, "seeds": seeds, "status": "crash",
                   "audit": {"level": "L0", "flags": ["evaluator_crash"]}, "diag": {"error": r.stderr[-300:]}}
        h = pathlib.Path(artifact_dir).name
        for flag in res.get("audit", {}).get("flags", []):
            events.emit(run, "audit_flag", "evald", gen=gen, trial=trial, artifact_hash=h, check=flag,
                        detail=str((res.get("diag") or {}).get("error") or flag)[:200])
        ev = dict(artifact_hash=h, metric=res.get("metric", "val_loss"), value=res.get("value"), seeds=list(seeds),
                  wall_s=round(time.time() - t0, 3), status=res.get("status", "crash"),
                  audit=res.get("audit") or {"level": "L0", "flags": []})
        events.emit(run, "eval_result", "evald", gen=gen, trial=trial, lane=lane, **ev)
        return {**ev, "diag": res.get("diag")}

    def submit(self, run, task, artifact_dir, *, seeds, lane=None, gen=0, trial=None, **_kw) -> str:
        jid = uuid.uuid4().hex[:10]
        with self.lock:
            self.futs[jid] = self.pool.submit(self._eval, run, task, artifact_dir, list(seeds), lane, gen, trial)
        return jid

    def result(self, job_id, *, wait=True, timeout_s=3600) -> dict:
        return self.futs[job_id].result(timeout=timeout_s)  # type: ignore[union-attr]

    def lanes(self):
        return [{"lane": i + 1, "gpu": None, "busy": False, "paused": False} for i in range(2)]

    def pause_lane(self, lane):
        pass

    def resume_lane(self, lane):
        pass


def make_local_evald(cfg: dict) -> LocalEvalClient:
    return LocalEvalClient(cfg)


# The fake improver: look at results, read config.json, weaken the regulariser, submit.
_BUMP = ("python3 -c \"import json;c=json.load(open('config.json'));"
         "c['weight_decay']=round(c['weight_decay']*0.4,6);c['warmup_frac']=round(c['warmup_frac']*0.6,4);"
         "json.dump(c,open('config.json','w'),indent=1,sort_keys=True)\"")

FAKE_SCRIPT = {
    "models": ["improver"],
    "latency_s": 0.05,
    "rules": [
        {"match": {"n_tool_msgs": 0}, "respond": {"content": "Let me look at past results first.",
                                                  "tool_calls": [{"name": "lab_get_results", "arguments": {"n": 5}}]}},
        {"match": {"n_tool_msgs": 1}, "respond": {"content": None, "tool_calls": [
            {"name": "read_file", "arguments": {"path": "config.json"}}]}},
        {"match": {"n_tool_msgs": 2}, "respond": {"content": "Weight decay looks far too strong for this budget; weakening it.",
                                                  "tool_calls": [{"name": "bash", "arguments": {"command": _BUMP}}]}},
        {"match": {"n_tool_msgs": 3}, "respond": {"content": None, "tool_calls": [
            {"name": "lab_submit_eval", "arguments": {"hypothesis": "weight_decay x0.4, warmup x0.6: start config is over-regularised"}}]}},
    ],
    "default": {"content": "Reduced weight decay and warmup, submitted for evaluation."},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/events_smoke.jsonl")
    ap.add_argument("--workdir", default=None, help="where runs/ go (default: a temp dir)")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--task", default="muar/train-modadd-0")
    ap.add_argument("--run", default="smoke-muar-karpathy")
    a = ap.parse_args(argv)

    from rsi.gateway.server import GatewayConfig, GatewayThread
    from tests.fake_openai_server import FakeServer

    work = pathlib.Path(a.workdir or tempfile.mkdtemp(prefix="rsi-smoke-"))
    runs_root = work / "runs"
    os.environ["RSI_MUAR_START_DIR"] = str(work / "muar-start")
    with FakeServer(json.loads(json.dumps(FAKE_SCRIPT))) as fake:
        gcfg = GatewayConfig(runs_root=str(runs_root), upstreams={"improver": fake.url[: -len("/v1")]},
                             quiet=True)
        with GatewayThread(gcfg) as gw:
            cfg = {
                "run": a.run,
                "runs_root": str(runs_root),
                "task": {"name": a.task},
                "policy": {"name": "karpathy_greedy", "threads": a.threads, "seeds": [0, 1]},
                "model": "improver",
                "improver": {"model": "fake/improver (tests/fake_openai_server.py)"},
                "budget": {"trials": a.trials, "episode_s": 120},
                "lanes": [1, 2],
                "evald": {"factory": "rsi.smoke:make_local_evald", "workers": 2},
                "harness": {"kind": "mini", "sandbox": "none", "gateway_url": gw.url, "max_turns": 8},
            }
            summary = orchestrator.run(cfg)
    ev_path = runs_root / a.run / "events.jsonl"
    evs = events.validate_file(str(ev_path))
    counts: dict[str, int] = {}
    for e in evs:
        counts[e["ev"]] = counts.get(e["ev"], 0) + 1
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ev_path, out)
    print(json.dumps({"summary": summary, "events": counts, "copied_to": str(out), "run_dir": str(runs_root / a.run)},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
