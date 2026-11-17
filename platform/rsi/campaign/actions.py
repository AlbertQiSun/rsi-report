"""Thin CLI wrappers the campaign runs as subprocesses around other workers' functions.

    python -m rsi.campaign.actions lessons  --spec spec.json   # rsi.update.write_lessons (worker D)
    python -m rsi.campaign.actions sft      --spec spec.json   # rsi.update.build_sft_data (worker D)
    python -m rsi.campaign.actions endpoint --spec spec.json   # rsi.evald.client.EvalClient (worker B)

Each prints one JSON object on its last stdout line (the runner stores it as ``out``).
Running them out of process keeps a crash in a component from taking down the campaign.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def do_lessons(spec: dict) -> dict:
    from rsi.update import write_lessons

    h = write_lessons(spec["run_dirs"], model=spec["model"], gateway_url=spec["gateway_url"],
                      out_dir=spec["out_dir"], generic_placebo=bool(spec.get("generic_placebo")))
    n = 0
    p = os.path.join(spec["out_dir"], "lessons.md")
    if os.path.exists(p):
        n = sum(1 for line in open(p) if line.lstrip().startswith(("-", "*")))
    return {"scaffold_hash": h, "n_lessons": n, "out_dir": spec["out_dir"]}


def do_sft(spec: dict) -> dict:
    from rsi.update import build_sft_data

    d = build_sft_data(spec["run_dirs"], accepted_only=bool(spec.get("accepted_only", True)),
                       placebo=bool(spec.get("placebo")), out=spec["out"]) or {}
    d = dict(d)
    d.setdefault("out", spec["out"])
    return d


def do_endpoint(spec: dict) -> dict:
    from rsi.campaign.analysis import top_artifacts
    from rsi.evald.client import EvalClient

    client = EvalClient(spec["evald_url"], spec["token_path"])
    cands = top_artifacts(spec["run_dirs"], int(spec["top_k"]))
    if spec.get("include_start") and os.path.isdir(spec["include_start"]):
        cands.append({"artifact_hash": "start", "path": spec["include_start"], "value": None})
    jobs = []
    for c in cands:  # one job per (artifact, seed) so both lanes share the work
        for sd in spec["seeds"]:
            jid = client.submit(spec["run"], spec.get("task", "nanochat"), c["path"], seeds=[int(sd)], gen=0)
            jobs.append((c["artifact_hash"], sd, jid))
    results: dict[str, list] = {}
    for h, sd, jid in jobs:
        r = client.result(jid, wait=True, timeout_s=3600)
        results.setdefault(h, []).append({"seed": sd, "value": r.get("value"), "status": r.get("status")})
    summary = {}
    for h, rs in results.items():
        vals = [r["value"] for r in rs if r["status"] == "ok" and isinstance(r["value"], (int, float))]
        summary[h] = {"n_ok": len(vals), "mean": (sum(vals) / len(vals)) if vals else None}
    return {"n_artifacts": len(cands), "seeds": spec["seeds"], "summary": summary, "t": time.time()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.campaign.actions")
    ap.add_argument("what", choices=["lessons", "sft", "endpoint"])
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    out = {"lessons": do_lessons, "sft": do_sft, "endpoint": do_endpoint}[a.what](spec)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
