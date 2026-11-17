#!/bin/bash
# Vanilla NanoChat start, N seeds across lanes, one CSV line per seed:
#   seed,lane,status,dev_bpb,sealed_bpb,wall_s,train_wall_s,score_wall_s,
#   reported_training_s,reported_total_s,startup_s,attn_backend,job_id
#
# usage: bundle/run_vanilla.sh [SEEDS=0-19] [LANES=1,2] [CSV=$R/out/vanilla_<ts>.csv]
# env:   MODE=auto|evald|direct (auto: use a running evald, else run in-process)
#        MARGIN=<s>  startup margin for direct mode (evald mode uses evald's own setting)
#        R, PY (serving-env python; evald host code is stdlib only)
# Run on the compute node (inside `srun --jobid=... --overlap`).
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
PLATFORM=$(cd "$(dirname "$0")/.." && pwd)
SEEDS=${1:-0-19}
LANES=${2:-1,2}
mkdir -p "$R/out"
CSV=${3:-$R/out/vanilla_$(date +%Y%m%dT%H%M%S).csv}
MODE=${MODE:-auto}

# tokenizer (Karpathy prepare.py on the train shards; no-op if present)
R="$R" bash "$PLATFORM/rsi/tasks/nanochat/prepare_data.sh"

EXTRA=()
[ -n "${MARGIN:-}" ] && EXTRA+=(--margin-s "$MARGIN")
cd "$PLATFORM"
R="$R" "$PY" -m rsi.evald vanilla --R "$R" --lanes "$LANES" --seeds "$SEEDS" --mode "$MODE" \
  --csv "$CSV" ${EXTRA[@]+"${EXTRA[@]}"}
echo "CSV: $CSV" >&2
