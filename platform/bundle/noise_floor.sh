#!/bin/bash
# S0 noise floor (platform_build_plan.md §3 / §6.4, gate G1a): the vanilla start for
# N seeds on two lanes, then summary statistics of dev BPB, sealed BPB and wall times.
#
# usage: bundle/noise_floor.sh [N=20] [LANES=1,2]
# env:   MODE=direct (default here: in-process, with a generous MARGIN so no seed is
#        killed while we measure startup time; use MODE=evald to go through a running
#        evald with its configured limit), MARGIN=600, R, PY
# Output: $R/out/noise_floor_<ts>.csv (+ .summary.json). Prints numbers only; reading
# them against the G1a thresholds is left to the user.
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
N=${1:-20}
LANES=${2:-1,2}
export MODE=${MODE:-direct}
export MARGIN=${MARGIN:-600}
TS=$(date +%Y%m%dT%H%M%S)
CSV=$R/out/noise_floor_$TS.csv
mkdir -p "$R/out"
R="$R" PY="$PY" bash "$(dirname "$0")/run_vanilla.sh" "0-$((N-1))" "$LANES" "$CSV"

"$PY" - "$CSV" <<'EOF' | tee "${CSV%.csv}.summary.json"
import csv, json, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def col(k, only_ok=True):
    out = []
    for r in rows:
        if only_ok and r["status"] != "ok":
            continue
        try:
            out.append(float(r[k]))
        except (TypeError, ValueError):
            pass
    return out
def stats(v):
    if not v:
        return None
    return {"n": len(v), "mean": round(st.mean(v), 6), "sd": round(st.stdev(v), 6) if len(v) > 1 else None,
            "min": round(min(v), 6), "max": round(max(v), 6)}
dev, sealed = col("dev_bpb"), col("sealed_bpb")
gap = [float(r["dev_bpb"]) - float(r["sealed_bpb"]) for r in rows
       if r["status"] == "ok" and r["dev_bpb"] and r["sealed_bpb"]]
by_lane = {}
for r in rows:
    if r["status"] == "ok" and r["dev_bpb"]:
        by_lane.setdefault(r["lane"], []).append(float(r["dev_bpb"]))
print(json.dumps({
    "csv": sys.argv[1], "n_seeds": len(rows),
    "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
    "dev_bpb": stats(dev), "sealed_bpb": stats(sealed), "dev_minus_sealed": stats(gap),
    "dev_bpb_by_lane": {k: stats(v) for k, v in sorted(by_lane.items())},
    "train_wall_s": stats(col("train_wall_s", False)), "startup_s": stats(col("startup_s")),
    "score_wall_s": stats(col("score_wall_s")), "job_wall_s": stats(col("wall_s", False)),
    "attn_backend": sorted({r["attn_backend"] for r in rows}),
}, indent=1))
EOF
