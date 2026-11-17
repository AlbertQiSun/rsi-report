"""Read run logs and compute what the campaign needs to decide: trial counts, best
values, the frontier artifact, and the pre-registered Δ₂ rule (platform_design.md §1.3–1.5).

Δ₂ for arm A against base B on held-out tasks (paired by task, common seeds):
    Δ = mean_j [ mean_s nAUC(A; j, s) − mean_s nAUC(B; j, s) ]
95% CI: stratified bootstrap (resample tasks; within each task resample seeds of each arm).
p: exact (≤ 16 tasks) or Monte Carlo sign-flip permutation over the per-task differences.
Promotion rule (all three): CI low > 0;  Δ > 2 × sd(A/A);  CI low of (A − placebo) > 0.
sd(A/A) is the bootstrap sd of the A/A difference (A0b − A0a): the noise floor of the same
statistic. This reading of "A/A 差值的标准差" is an implementation choice (see NOTES_F.md).
"""

from __future__ import annotations

import itertools
import json
import os
import random
from statistics import mean, pstdev
from typing import Any, Iterable

_cache: dict[str, tuple[int, float, dict]] = {}


def run_stats(events_path: str, lower_is_better: bool = True) -> dict:
    """Summary of one run's events.jsonl (cached by size and mtime)."""
    try:
        st = os.stat(events_path)
    except OSError:
        return {"exists": False, "n_trials": 0, "n_ok": 0, "best": None, "run_end": None, "mtime": None,
                "n_run_start": 0}
    key = os.path.abspath(events_path)
    hit = _cache.get(key)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime:
        return hit[2]
    trials: dict[int, dict] = {}
    best = None
    best_hash = None
    run_end = None
    n_run_start = 0
    walls = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ev = e.get("ev")
            if ev == "run_start":
                n_run_start += 1
            elif ev == "eval_result" and e.get("trial") is not None:
                trials[e["trial"]] = e
                if isinstance(e.get("wall_s"), (int, float)):
                    walls.append(e["wall_s"])
                v = e.get("value")
                if e.get("status") == "ok" and isinstance(v, (int, float)):
                    if best is None or (v < best if lower_is_better else v > best):
                        best, best_hash = v, e.get("artifact_hash")
            elif ev == "run_end":
                run_end = e
    n_ok = sum(1 for e in trials.values() if e.get("status") == "ok")
    out = {"exists": True, "n_trials": len(trials), "n_ok": n_ok,
           "ok_rate": (n_ok / len(trials)) if trials else None,
           "mean_wall_s": mean(walls) if walls else None,
           "best": best, "best_hash": best_hash, "run_end": run_end,
           "nAUC": (run_end or {}).get("nAUC"), "mtime": st.st_mtime, "n_run_start": n_run_start}
    _cache[key] = (st.st_size, st.st_mtime, out)
    return out


def frontier(run_dirs: Iterable[str], lower_is_better: bool = True) -> dict | None:
    """Best ok artifact across runs: {run_dir, artifact_hash, value, path}."""
    best = None
    for rd in run_dirs:
        s = run_stats(os.path.join(rd, "events.jsonl"), lower_is_better)
        if s["best"] is None:
            continue
        if best is None or (s["best"] < best["value"] if lower_is_better else s["best"] > best["value"]):
            best = {"run_dir": rd, "artifact_hash": s["best_hash"], "value": s["best"],
                    "path": os.path.join(rd, "artifacts", str(s["best_hash"]))}
    return best


def top_artifacts(run_dirs: Iterable[str], k: int, lower_is_better: bool = True) -> list[dict]:
    """Top-k distinct ok artifacts across runs by dev value."""
    cands: dict[str, dict] = {}
    for rd in run_dirs:
        p = os.path.join(rd, "events.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ev") != "eval_result" or e.get("status") != "ok" or e.get("trial") is None:
                    continue
                v, h = e.get("value"), e.get("artifact_hash")
                if not isinstance(v, (int, float)) or not h:
                    continue
                cur = cands.get(h)
                if cur is None or (v < cur["value"] if lower_is_better else v > cur["value"]):
                    cands[h] = {"artifact_hash": h, "value": v, "run_dir": rd,
                                "path": os.path.join(rd, "artifacts", h)}
    ranked = sorted(cands.values(), key=lambda c: c["value"], reverse=not lower_is_better)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Δ₂
# ---------------------------------------------------------------------------

def arm_table(subruns: dict[str, dict], runs_root: str) -> dict[str, dict[int, float]]:
    """{task: {seed: nAUC}} for one µAR arm from its sub-run records."""
    out: dict[str, dict[int, float]] = {}
    for sub in subruns.values():
        s = run_stats(os.path.join(runs_root, sub["run"], "events.jsonl"))
        v = s.get("nAUC")
        if isinstance(v, (int, float)):
            out.setdefault(sub["task"], {})[int(sub["seed"])] = float(v)
    return out


def _delta(a: dict[str, list[float]], b: dict[str, list[float]], tasks: list[str]) -> float:
    return mean(mean(a[t]) - mean(b[t]) for t in tasks)


def bootstrap(a: dict[str, dict[int, float]], b: dict[str, dict[int, float]], *, n_boot: int = 2000,
              seed: int = 0) -> dict:
    tasks = sorted(set(a) & set(b))
    tasks = [t for t in tasks if a[t] and b[t]]
    if not tasks:
        return {"delta": None, "n_tasks": 0}
    A = {t: list(a[t].values()) for t in tasks}
    B = {t: list(b[t].values()) for t in tasks}
    d = _delta(A, B, tasks)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ts = [rng.choice(tasks) for _ in tasks]
        ra = {i: [rng.choice(A[t]) for _ in A[t]] for i, t in enumerate(ts)}
        rb = {i: [rng.choice(B[t]) for _ in B[t]] for i, t in enumerate(ts)}
        boots.append(mean(mean(ra[i]) - mean(rb[i]) for i in range(len(ts))))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    diffs = [mean(A[t]) - mean(B[t]) for t in tasks]
    return {"delta": d, "ci": [lo, hi], "sd": pstdev(boots), "p_signflip": signflip_p(diffs, seed=seed),
            "n_tasks": len(tasks), "n_a": sum(len(v) for v in A.values()), "n_b": sum(len(v) for v in B.values()),
            "mean_a": mean(x for v in A.values() for x in v), "mean_b": mean(x for v in B.values() for x in v)}


def signflip_p(diffs: list[float], seed: int = 0, n_mc: int = 20000) -> float | None:
    if not diffs:
        return None
    obs = abs(mean(diffs))
    n = len(diffs)
    if n <= 16:
        signs = itertools.product((1, -1), repeat=n)
        total = hits = 0
        for sg in signs:
            total += 1
            hits += abs(mean(s * x for s, x in zip(sg, diffs))) >= obs - 1e-12
        return hits / total
    rng = random.Random(seed)
    hits = sum(abs(mean(rng.choice((1, -1)) * x for x in diffs)) >= obs - 1e-12 for _ in range(n_mc))
    return hits / n_mc


def delta2_decision(arm: dict, base: dict, *, aa: tuple[dict, dict] | None = None,
                    placebo: dict | None = None, n_boot: int = 2000) -> dict[str, Any]:
    """Apply the pre-registered rule. Inputs are {task: {seed: nAUC}} tables."""
    main = bootstrap(arm, base, n_boot=n_boot, seed=1)
    out: dict[str, Any] = {"main": main}
    if main.get("delta") is None:
        out.update({"pass": False, "harmful": False, "reason": "no paired nAUC data"})
        return out
    reasons = []
    ok_ci = main["ci"][0] > 0
    reasons.append(f"CI_lo={main['ci'][0]:+.4f}{'>' if ok_ci else '<='}0")
    ok_aa = True
    if aa is not None:
        aa_res = bootstrap(aa[1], aa[0], n_boot=n_boot, seed=2)
        out["aa"] = aa_res
        if aa_res.get("sd") is not None:
            thr = 2 * aa_res["sd"]
            ok_aa = main["delta"] > thr
            reasons.append(f"Δ={main['delta']:+.4f}{'>' if ok_aa else '<='}2·sdAA={thr:.4f}")
            out["mde"] = thr
    ok_pl = True
    if placebo is not None:
        pl = bootstrap(arm, placebo, n_boot=n_boot, seed=3)
        out["vs_placebo"] = pl
        ok_pl = pl.get("delta") is not None and pl["ci"][0] > 0
        if pl.get("delta") is not None:
            reasons.append(f"vs_placebo CI_lo={pl['ci'][0]:+.4f}{'>' if ok_pl else '<='}0")
    out["pass"] = bool(ok_ci and ok_aa and ok_pl)
    out["harmful"] = bool(main["ci"][1] < 0)
    out["reason"] = "; ".join(reasons)
    return out


def fmt_delta(res: dict | None) -> str:
    if not res or not res.get("main") or res["main"].get("delta") is None:
        return "n/a"
    m = res["main"]
    return f"{m['delta']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}]{' PASS' if res.get('pass') else ''}"
