"""Continue-from-frontier (cf_*) NanoChat analysis: M1 / M1p / S1 vs M0, 3 reps each, 12 trials per run.

usage: python cf_analysis.py RUNS_DIR OUT_JSON
RUNS_DIR holds ovn1118b-cf-<arm>-r<k>/ledger.jsonl (copied from the cluster).

Per run (same trial semantics as rsi.update.analysis.curves_from_runs): s0 = the run's own re-measured
baseline dev (val_bpb, seed 0); trial values = dev of eval-ok trials, None for crashes; baseline/pending/
no_change rows are not trials; first 12 trials.
  gain     = s0 - min(s0, best dev)            (val_bpb drop; >0 = better)
  auc_gain = mean_t (s0 - best_so_far_t)       (unnormalised area under the improvement curve; NanoChat has no
                                                s_ref, so the nAUC of the µAR decisions cannot be formed)
Arm comparison (unpaired; reps were lane-counterbalanced, not matched): difference of arm means, bootstrap 95% CI
(resampling runs within each arm), exact two-sided permutation p over all C(6,3) = 20 relabelings (smallest
possible p = 0.10), and the minimum detectable effect at alpha 0.05 / power 0.8 from the pooled SD.
"""
import itertools
import json
import os
import random
import re
import statistics as stt
import sys

N_TRIALS = 12
ARMS = ["m0", "m1", "m1p", "s1"]


def run_metrics(d):
    recs = {}
    for line in open(os.path.join(d, "ledger.jsonl")):
        r = json.loads(line)
        if r.get("trial") is not None:
            recs.setdefault(int(r["trial"]), {}).update(r)
    base = [r for r in recs.values() if r.get("status") == "baseline" and r.get("dev") is not None]
    s0 = base[-1]["dev"]
    trials = [recs[t] for t in sorted(recs) if recs[t].get("status") not in ("baseline", "pending", "no_change")]
    vals = [(r.get("dev") if r.get("eval_status") == "ok" else None) for r in trials][:N_TRIALS]
    n_done = len(vals)
    best, curve = s0, []
    for v in vals + [None] * (N_TRIALS - n_done):
        if v is not None:
            best = min(best, v)
        curve.append(s0 - best)
    return {"s0": s0, "best": best, "gain": s0 - best, "auc_gain": sum(curve) / N_TRIALS, "n_trials": n_done,
            "n_crash": sum(1 for v in vals if v is None),
            # finished = no trial still pending (episodes with no code change are not trials, as in the µAR
            # analysis, so a finished run can have < 12 trials; the missing ones count as "no improvement")
            "complete": not any(r.get("status") == "pending" for r in recs.values())}


def compare(a, b, n_boot=20000, seed=0):
    rng = random.Random(seed)
    d = stt.mean(a) - stt.mean(b)
    boots = sorted(stt.mean(rng.choices(a, k=len(a))) - stt.mean(rng.choices(b, k=len(b))) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    pool = a + b
    ext = 0
    combos = list(itertools.combinations(range(len(pool)), len(a)))
    for idx in combos:
        aa = [pool[i] for i in idx]
        bb = [pool[i] for i in range(len(pool)) if i not in idx]
        ext += abs(stt.mean(aa) - stt.mean(bb)) >= abs(d) - 1e-12
    sd = stt.pstdev(a) if len(a) < 2 else (((len(a) - 1) * stt.variance(a) + (len(b) - 1) * stt.variance(b))
                                          / (len(a) + len(b) - 2)) ** 0.5
    t_crit = {4: 2.776, 3: 3.182, 2: 4.303}.get(len(a) + len(b) - 2, 2.776)   # two-sided 0.05
    t_pow = {4: 0.941, 3: 0.978, 2: 1.061}.get(len(a) + len(b) - 2, 0.941)    # power 0.8
    mde = (t_crit + t_pow) * sd * (1 / len(a) + 1 / len(b)) ** 0.5
    return {"delta": d, "ci": [lo, hi], "p_perm": ext / len(combos), "n_perm": len(combos), "pooled_sd": sd,
            "mde": mde, "n_a": len(a), "n_b": len(b)}


def main():
    root, out = sys.argv[1], sys.argv[2]
    runs = {}
    for d in sorted(os.listdir(root)):
        m = re.fullmatch(r"ovn1118b-cf-(m0|m1|m1p|s1)-r(\d)", d)
        if m and os.path.exists(os.path.join(root, d, "ledger.jsonl")):
            runs[d] = {"arm": m.group(1), "rep": int(m.group(2)), **run_metrics(os.path.join(root, d))}
    res = {"runs": runs, "arms": {}, "comparisons": {}, "notes": {
        "metric": "val_bpb (lower is better); gain = run baseline - best; auc_gain = mean over 12 trials of "
                  "(baseline - best so far)",
        "nauc": "not computable: NanoChat has no s_ref",
        "excluded_incomplete": [k for k, v in runs.items() if not v["complete"]]}}
    for arm in ARMS:
        rs = [v for v in runs.values() if v["arm"] == arm and v["complete"]]
        res["arms"][arm] = {"n": len(rs), "gain_mean": stt.mean(r["gain"] for r in rs) if rs else None,
                            "auc_gain_mean": stt.mean(r["auc_gain"] for r in rs) if rs else None,
                            "gains": [r["gain"] for r in sorted(rs, key=lambda r: r["rep"])]}
    for arm in ("m1", "m1p", "s1"):
        for metric in ("gain", "auc_gain"):
            a = [v[metric] for v in runs.values() if v["arm"] == arm and v["complete"]]
            b = [v[metric] for v in runs.values() if v["arm"] == "m0" and v["complete"]]
            if len(a) >= 2 and len(b) >= 2:
                res["comparisons"][f"{arm}_vs_m0_{metric}"] = compare(a, b)
    json.dump(res, open(out, "w"), indent=1)
    for k, v in runs.items():
        print(f"{k:22s} s0 {v['s0']:.4f} best {v['best']:.4f} gain {v['gain']:.4f} auc {v['auc_gain']:.4f} "
              f"trials {v['n_trials']} crash {v['n_crash']}{'' if v['complete'] else '  INCOMPLETE'}")
    for arm, a in res["arms"].items():
        print(arm, a)
    for k, c in res["comparisons"].items():
        print(f"{k:22s} delta {c['delta']:+.4f} CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] p_perm {c['p_perm']:.2f} "
              f"(n_perm {c['n_perm']}) pooled_sd {c['pooled_sd']:.4f} MDE {c['mde']:.4f}")


if __name__ == "__main__":
    main()
