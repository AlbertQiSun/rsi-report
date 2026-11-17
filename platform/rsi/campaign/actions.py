"""Thin CLI wrappers the campaign runs as subprocesses around other workers' functions.

    python -m rsi.campaign.actions lessons  --spec spec.json   # rsi.update.write_lessons (worker D)
    python -m rsi.campaign.actions sft      --spec spec.json   # rsi.update.build_sft_data (worker D)
    python -m rsi.campaign.actions endpoint --spec spec.json   # rsi.evald.client.EvalClient (worker B)
    python -m rsi.campaign.actions decision --spec spec.json   # rsi.update.analysis.analyse (worker D)

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

    target = None
    if spec.get("match_words_of"):  # placebo memo of the same length as the real lessons
        man = os.path.normpath(spec["match_words_of"]) + ".manifest.json"
        if os.path.exists(man):
            target = json.load(open(man)).get("words")
    h = write_lessons(spec["run_dirs"], model=spec["model"], gateway_url=spec["gateway_url"],
                      out_dir=spec["out_dir"], generic_placebo=bool(spec.get("generic_placebo")),
                      target_words=target, run=spec.get("run"), gen=int(spec.get("gen", 1)),
                      token_path=spec.get("token_path"))
    n = 0
    p = os.path.join(spec["out_dir"], "lessons.md")
    if os.path.exists(p):
        n = sum(1 for line in open(p) if line.lstrip().startswith(("-", "*")))
    return {"scaffold_hash": h, "n_lessons": n, "out_dir": spec["out_dir"]}


def do_sft(spec: dict) -> dict:
    from rsi.update import build_sft_data

    m = build_sft_data(spec["run_dirs"], accepted_only=bool(spec.get("accepted_only", True)),
                       placebo=bool(spec.get("placebo")), out=spec["out"]) or {}
    n_total = int(m.get("n_train", 0)) + int(m.get("n_valid", 0))
    n_acc = int((m.get("by_label") or {}).get("accepted", 0))
    leak = m.get("leak_check") or {}
    # G3 (deploy/second_order.md): by_label.accepted >= 40 for the real set; the placebo set is size-matched
    return {"out": spec["out"], "n_examples": n_total if spec.get("placebo") else n_acc, "n_total": n_total,
            "n_accepted": n_acc, "heldout_overlap": 0 if leak.get("ok", True) else int(leak.get("rescan_hits") or 1),
            "placebo_shortfall": (m.get("report") or {}).get("placebo_shortfall"), "mode": m.get("mode")}


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


def do_selfdesign(spec: dict) -> dict:
    from rsi.update.selfdesign import design

    return design(spec)


def do_sft_recipe(spec: dict) -> dict:
    from rsi.update.selfdesign import build_recipe_sft

    recipe = json.load(open(spec["recipe_path"]))
    match_n = None
    if spec.get("match_manifest"):
        mm = json.load(open(spec["match_manifest"]))
        match_n = int(mm.get("n_train", 0)) + int(mm.get("n_valid", 0))
    m = build_recipe_sft(spec["run_dirs"], recipe, spec["out"], random_placebo=bool(spec.get("random_placebo")),
                         match_n=match_n, salt=spec.get("salt", "sdr"))
    n = int(m.get("n_train", 0)) + int(m.get("n_valid", 0))
    return {"out": spec["out"], "n_examples": n, "n_total": n, "by_label": m.get("by_label"), "mode": m.get("mode"),
            "heldout_overlap": 0, "placebo_shortfall": (m.get("report") or {}).get("placebo_shortfall"),
            "recipe_path": spec["recipe_path"]}


def do_decision(spec: dict) -> dict:
    """Run worker D's pre-registered analysis and reduce it to what the campaign needs."""
    from rsi.update.analysis import analyse

    raw = json.load(open(spec["sref_path"]))
    s_ref = {t: (v.get("s_ref") if isinstance(v, dict) else v) for t, v in raw.items()}
    arms = {}
    for label, (path, key) in (spec.get("arms_files") or {}).items():
        try:
            arms[label] = json.load(open(path))["arms"][key]
        except (OSError, KeyError, ValueError):
            pass
    # s0 comes from each run's baseline trial; for curves loaded from a file, use sref.json's s0
    s0 = {t: v["s0"] for t, v in raw.items() if isinstance(v, dict) and v.get("s0") is not None}
    a = {"s_ref": s_ref, "s0": s0, "runs": spec["runs"], "arms": arms, "n_trials": spec.get("n_trials"),
         "n_boot": spec.get("n_boot", 10000), "floor": spec.get("floor", "per_task"), "compare": []}
    have = set(spec["runs"]) | set(arms)
    if not spec["compare"] or not {spec["compare"][0][0], spec["compare"][0][1]} <= have:
        return {"pass": False, "harmful": False, "nodata": True, "reason": "missing arm data"}
    for row in spec["compare"]:
        row = [r for r in row if r in have] if row[0] in have and row[1] in have else []
        if row:
            a["compare"].append(row)
    if spec.get("aa") and all(x in have for x in spec["aa"]):
        a["aa"] = spec["aa"]
    if not a["compare"]:
        return {"pass": False, "harmful": False, "nodata": True, "reason": "missing arm data"}
    res = analyse(a)
    summ = res["arms"]

    def main_of(c: dict) -> dict:
        d = c["delta2"]
        return {"delta": d.get("delta"), "ci": d.get("ci"), "p_signflip": d.get("p_signflip", d.get("p")),
                "n_tasks": d.get("n_tasks"), "mean_a": (summ.get(c["new"]) or {}).get("mean"),
                "mean_b": (summ.get(c["old"]) or {}).get("mean")}

    c0 = res["comparisons"][0]
    dec = c0["decision"]
    m = main_of(c0)
    ci = m.get("ci") or [None, None]
    names = spec.get("names") or []
    also = [{"name": names[i] if i < len(names) else f"{c['new']} vs {c['old']}", **main_of(c)}
            for i, c in enumerate(res["comparisons"][1:])]
    return {"pass": bool(dec.get("observed")), "harmful": ci[1] is not None and ci[1] < 0, "main": m,
            "vs_placebo": c0.get("vs_placebo"), "aa": res.get("aa"), "mde": dec.get("mde"),
            "criteria": {k: dec.get(k) for k in ("ci_lower_gt_0", "gt_2x_aa", "beats_placebo", "aa_floor")},
            "reason": dec.get("text"), "also": also}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.campaign.actions")
    ap.add_argument("what", choices=["lessons", "sft", "endpoint", "decision", "selfdesign", "sft_recipe"])
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    out = {"lessons": do_lessons, "sft": do_sft, "endpoint": do_endpoint, "decision": do_decision,
           "selfdesign": do_selfdesign, "sft_recipe": do_sft_recipe}[a.what](spec)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
