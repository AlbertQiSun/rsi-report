"""First- and second-order statistics and the pre-registered decision rule (design doc §1.2-1.4).

Data layout used throughout:
    curves = {task: {seed_key: [v_1, ..., v_N]}}   # dev score after each trial (None = failed / illegal)
    s0     = {task: start score},  s_ref = {task: reference best}   (lower is better)

g(t)   = (s0 - best_t) / (s0 - s_ref)                 best_t = best-so-far including s0
nAUC   = mean_t g(t), first per seed, then mean over seeds within a task
Δ₂     = mean_j [nAUC(new; j) - nAUC(old; j)], j over held-out tasks, paired by task
ρ      = nAUC(new) / nAUC(old)
A/A    = the same improver twice with disjoint seeds; d_j = per-task difference
         sd_task = sd(d_j) ; se_mean = sd_task / sqrt(n_tasks)
MDE    = (z_{1-α/2} + z_{power}) * se_mean   (α = 0.05 two-sided, power 0.8 -> 2.80 * se_mean)
Decision ("second-order effect observed") = all of
   1. stratified-bootstrap 95% CI lower bound of Δ₂ > 0  (sign-flip p reported alongside)
   2. Δ₂ > 2 * A/A sd   (floor="per_task": sd_task, literal reading; floor="mean": se_mean)
   3. new arm beats its placebo arm: CI lower bound of Δ₂(new vs placebo) > 0
Otherwise: "not detected at this power; minimum detectable effect ≈ MDE".

CLI: python -m rsi.update.analysis spec.json   where spec.json =
  {"s0": {...}, "s_ref": {...}, "arms": {"A0a": curves, "A0b": curves, "A1": curves, "A1p": curves},
   "aa": ["A0a", "A0b"], "compare": [["A1", "A0a", "A1p"], ...], "n_boot": 10000}
or with "runs": {"A0a": [run_dir, ...], ...} instead of "arms" (curves read from ledgers).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from statistics import NormalDist

import numpy as np


# ---------------------------------------------------------------------------- first order
def best_so_far(s0: float, values: list) -> list[float]:
    best, out = s0, []
    for v in values:
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v < best:
            best = v
        out.append(best)
    return out


def g_curve(s0: float, values: list, s_ref: float) -> list[float]:
    den = s0 - s_ref
    return [(s0 - b) / den for b in best_so_far(s0, values)]


def nauc(s0: float | None, values: list, s_ref: float | None, n_trials: int | None = None) -> float | None:
    if s0 is None or s_ref is None or not values or s0 == s_ref:
        return None
    vals = list(values)
    if n_trials is not None:
        vals = (vals + [None] * n_trials)[:n_trials]
    g = g_curve(s0, vals, s_ref)
    return float(sum(g) / len(g))


def task_naucs(curves: dict, s0: dict, s_ref: dict, n_trials: int | None = None) -> dict:
    """{task: {seed: nAUC}} (tasks with undefined normalisation are dropped)."""
    out = {}
    for t, seeds in curves.items():
        row = {k: nauc(s0.get(t), v, s_ref.get(t), n_trials) for k, v in seeds.items()}
        row = {k: v for k, v in row.items() if v is not None}
        if row:
            out[t] = row
    return out


def arm_summary(curves: dict, s0: dict, s_ref: dict, n_trials: int | None = None) -> dict:
    tn = task_naucs(curves, s0, s_ref, n_trials)
    per_task = {t: float(np.mean(list(v.values()))) for t, v in tn.items()}
    return {"per_task": per_task, "mean": float(np.mean(list(per_task.values()))) if per_task else None,
            "n_tasks": len(per_task)}


def decomposition(records: list[dict]) -> dict:
    """v = legal proposals / proposals; p = improving / legal; m = mean relative step of improvements."""
    props = [r for r in records if r.get("status") not in ("baseline", "pending", "no_change")]
    legal = [r for r in props if r.get("eval_status") == "ok"]
    improving = [r for r in legal if r.get("status") in ("keep", "adopted")]
    steps, best = [], None
    for r in sorted(props, key=lambda r: r.get("trial", 0)):
        if r.get("status") in ("keep", "adopted") and isinstance(r.get("dev"), (int, float)):
            if best is not None:
                steps.append(best - r["dev"])
            best = r["dev"]
        elif r.get("status") == "baseline":
            best = r.get("dev")
    return {"n": len(props), "v": len(legal) / len(props) if props else None,
            "p": len(improving) / len(legal) if legal else None,
            "m": float(np.mean(steps)) if steps else None}


# ---------------------------------------------------------------------------- second order
def _paired_tasks(a: dict, b: dict) -> list[str]:
    return sorted(set(a) & set(b))


def bootstrap_delta(new: dict, old: dict, *, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Stratified bootstrap of Δ₂ over tasks and seeds. ``new``/``old`` = {task: {seed: nAUC}}.

    Resample tasks with replacement; within each drawn task resample each arm's seeds with replacement
    (jointly, by seed key, when both arms used the same seed keys = common random numbers).
    """
    tasks = _paired_tasks(new, old)
    if not tasks:
        return {"delta": None, "ci": [None, None], "n_tasks": 0, "per_task": {}}
    per_task = {t: float(np.mean(list(new[t].values())) - np.mean(list(old[t].values()))) for t in tasks}
    delta = float(np.mean(list(per_task.values())))
    rng = np.random.default_rng(seed)
    arrs = []
    for t in tasks:
        kn, ko = sorted(new[t]), sorted(old[t])
        paired = kn == ko
        arrs.append((np.array([new[t][k] for k in kn]), np.array([old[t][k] for k in ko]), paired))
    boots = np.empty(n_boot)
    nt = len(tasks)
    for i in range(n_boot):
        idx = rng.integers(0, nt, size=nt)
        s = 0.0
        for j in idx:
            an, ao, paired = arrs[j]
            if paired:
                k = rng.integers(0, len(an), size=len(an))
                s += an[k].mean() - ao[k].mean()
            else:
                s += an[rng.integers(0, len(an), size=len(an))].mean() - ao[rng.integers(0, len(ao), size=len(ao))].mean()
        boots[i] = s / nt
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"delta": delta, "ci": [float(lo), float(hi)], "n_tasks": nt, "per_task": per_task,
            "boot_sd": float(boots.std(ddof=1))}


def signflip_test(diffs: list[float], *, n_perm: int = 100000, seed: int = 0) -> float | None:
    """Two-sided paired sign-flip permutation p-value for mean(diffs) != 0 (exact when n <= 20)."""
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    n = len(d)
    if n == 0:
        return None
    obs = abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        stats = np.abs((signs * d).mean(axis=1))
        return float((stats >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float(((stats >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def mde(sd_mean: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = NormalDist()
    return float((z.inv_cdf(1 - alpha / 2) + z.inv_cdf(power)) * sd_mean)


def aa_noise(a: dict, b: dict, *, alpha: float = 0.05, power: float = 0.8) -> dict:
    """A/A noise floor from two runs of the same improver with disjoint seeds ({task: {seed: nAUC}})."""
    tasks = _paired_tasks(a, b)
    d = [float(np.mean(list(a[t].values())) - np.mean(list(b[t].values()))) for t in tasks]
    sd_task = statistics.stdev(d) if len(d) >= 2 else None
    se_mean = sd_task / math.sqrt(len(d)) if sd_task is not None else None
    return {"n_tasks": len(d), "diffs": dict(zip(tasks, d)), "mean": float(np.mean(d)) if d else None,
            "sd_task": sd_task, "se_mean": se_mean, "floor_2sd_task": 2 * sd_task if sd_task is not None else None,
            "floor_2se_mean": 2 * se_mean if se_mean is not None else None,
            "mde": mde(se_mean, alpha, power) if se_mean is not None else None}


def compare(new: dict, old: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    b = bootstrap_delta(new, old, n_boot=n_boot, seed=seed)
    b["p_signflip"] = signflip_test(list(b["per_task"].values())) if b["per_task"] else None
    mo = np.mean([np.mean(list(v.values())) for t, v in old.items() if t in b["per_task"]]) if b["per_task"] else None
    mn = np.mean([np.mean(list(v.values())) for t, v in new.items() if t in b["per_task"]]) if b["per_task"] else None
    b["rho"] = float(mn / mo) if mo not in (None, 0) and mn is not None else None
    return b


def decide(delta: dict, aa: dict, placebo: dict | None, *, floor: str = "per_task") -> dict:
    """Pre-registered rule (design doc §1.4). ``delta``/``placebo`` = compare() outputs; ``aa`` = aa_noise()."""
    d, lo = delta.get("delta"), (delta.get("ci") or [None])[0]
    fl = aa.get("floor_2sd_task") if floor == "per_task" else aa.get("floor_2se_mean")
    c1 = lo is not None and lo > 0
    c2 = d is not None and fl is not None and d > fl
    c3 = placebo is not None and (placebo.get("ci") or [None])[0] is not None and placebo["ci"][0] > 0
    ok = bool(c1 and c2 and c3)
    m = aa.get("mde")
    text = ("second-order effect observed under the pre-registered rule" if ok else
            "not detected at this power; minimum detectable effect ≈ "
            + (f"{m:.4f} nAUC" if m is not None else "unknown (A/A missing)"))
    return {"ci_lower_gt_0": c1, "gt_2x_aa": c2, "beats_placebo": c3, "aa_floor": fl, "floor_kind": floor,
            "observed": ok, "mde": m, "text": text}


# ---------------------------------------------------------------------------- reading runs
def curves_from_runs(run_dirs: list[str], n_trials: int | None = None) -> tuple[dict, dict]:
    """Read ledgers -> ({task: {seed_key: values}}, {task: s0}). Seed key from config.json seed fields."""
    from rsi.runs import read_json
    from rsi.update.trajectories import run_task, trials

    curves: dict = {}
    s0: dict = {}
    for rd in run_dirs:
        task = run_task(rd)
        cfg = read_json(os.path.join(rd, "config.json"), {}) or {}
        key = str(cfg.get("seed", cfg.get("rng_seed", os.path.basename(os.path.normpath(rd)))))
        recs = sorted(trials(rd).values(), key=lambda r: r["trial"])
        base = [r for r in recs if r.get("status") == "baseline"]
        if base and base[-1].get("dev") is not None:
            s0.setdefault(task, base[-1]["dev"])
        vals = [r.get("dev") if r.get("eval_status") == "ok" else None for r in recs
                if r.get("status") not in ("baseline", "pending", "no_change")]
        if n_trials is not None:
            vals = (vals + [None] * n_trials)[:n_trials]
        curves.setdefault(task, {})[key] = vals
    return curves, s0


def analyse(spec: dict) -> dict:
    s0, s_ref = dict(spec.get("s0", {})), dict(spec["s_ref"])
    n_trials, n_boot = spec.get("n_trials"), int(spec.get("n_boot", 10000))
    arms = dict(spec.get("arms", {}))
    for arm, dirs in (spec.get("runs") or {}).items():
        c, s = curves_from_runs(dirs, n_trials)
        arms[arm] = c
        for t, v in s.items():
            s0.setdefault(t, v)
    tn = {a: task_naucs(c, s0, s_ref, n_trials) for a, c in arms.items()}
    out: dict = {"arms": {a: arm_summary(c, s0, s_ref, n_trials) for a, c in arms.items()}}
    aa = aa_noise(tn[spec["aa"][0]], tn[spec["aa"][1]]) if spec.get("aa") else {}
    out["aa"] = aa
    out["comparisons"] = []
    for row in spec.get("compare", []):
        new, old = row[0], row[1]
        plc = row[2] if len(row) > 2 else None
        c = compare(tn[new], tn[old], n_boot=n_boot)
        p = compare(tn[new], tn[plc], n_boot=n_boot) if plc else None
        out["comparisons"].append({"new": new, "old": old, "placebo": plc, "delta2": c, "vs_placebo": p,
                                   "decision": decide(c, aa, p, floor=spec.get("floor", "per_task"))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    with open(a.spec) as f:
        res = analyse(json.load(f))
    text = json.dumps(res, indent=1, sort_keys=True, default=float)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
