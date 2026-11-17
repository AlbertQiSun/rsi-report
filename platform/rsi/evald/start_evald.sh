#!/bin/bash
# Start evald on the compute node (run inside `srun --jobid=... --overlap`, see deploy/evald.md).
# usage: rsi/evald/start_evald.sh [LANES=1,2] [extra args for `python -m rsi.evald serve`]
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
PLATFORM=$(cd "$(dirname "$0")/../.." && pwd)
LANES=${1:-1,2}
shift || true
mkdir -p "$R/run" "$R/out"
chmod 700 "$R/run"
if [ ! -s "$R/run/token" ]; then
  (umask 077; "$PY" -c "import secrets; print(secrets.token_hex(32))" > "$R/run/token")
fi
chmod 600 "$R/run/token"
cd "$PLATFORM"
exec env R="$R" RSI_RUNS_ROOT="$R/runs" "$PY" -m rsi.evald serve --R "$R" --lanes "$LANES" "$@"
