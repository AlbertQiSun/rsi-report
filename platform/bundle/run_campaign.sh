#!/bin/bash
# Run the overnight campaign on the hold node, with bounded auto-restart of the scheduler itself.
#
#   usage (inside srun on node-a, normally from tmux; see deploy/campaign.md):
#     bash bundle/run_campaign.sh                 # start, or continue after a crash / STOP
#     bash bundle/run_campaign.sh --check         # print the plan and exit
#     bash bundle/run_campaign.sh --dry-run       # rehearsal with fakes (own temp root, CPU only)
#
# The scheduler always runs with --resume: a fresh start when there is no state yet, otherwise
# it continues from runs/campaign-<id>/state.json. If the scheduler process dies, this loop
# restarts it (at most MAX_RESTARTS times, 60 s apart). Its children (orchestrator runs, LoRA
# training) live in their own sessions inside this srun step and are re-adopted on restart.
# Exit codes of the scheduler: 0 done or STOP, 2 refused, 3 another instance running.
set -u
export R=${R:-/scratch/user/rsi}
export HF_HOME=${HF_HOME:-$R/hf}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
CODE=${RSI_CODE:-$(cd "$(dirname "$0")/.." && pwd)}      # platform/ checkout on the cluster
CONFIG=${CONFIG:-$CODE/campaign/overnight.yaml}
MAX_RESTARTS=${MAX_RESTARTS:-5}
mkdir -p "$R/out"
LOG=$R/out/campaign-supervisor.log
cd "$CODE" || { echo "no code dir $CODE"; exit 2; }
export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
export RSI_PY=$PY

for a in "$@"; do
  if [ "$a" = "--check" ] || [ "$a" = "--dry-run" ]; then
    exec "$PY" -m rsi.campaign --config "$CONFIG" "$@"
  fi
done

n=0
while true; do
  echo "[$(date '+%F %T')] campaign start (attempt $((n + 1))) config=$CONFIG" | tee -a "$LOG"
  "$PY" -m rsi.campaign --config "$CONFIG" --resume "$@"
  rc=$?
  echo "[$(date '+%F %T')] campaign exited rc=$rc" | tee -a "$LOG"
  case $rc in
    0|2|3) exit $rc ;;
  esac
  n=$((n + 1))
  if [ $n -ge "$MAX_RESTARTS" ]; then
    echo "[$(date '+%F %T')] giving up after $n restarts; fix and rerun this script" | tee -a "$LOG"
    exit $rc
  fi
  sleep 60
done
