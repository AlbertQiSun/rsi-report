#!/bin/bash
# runs the fingerprint census against a local rec server under the Mac GPU lock
cd "$(dirname "$0")/../.."
bash bench/gpu_lock.sh acquire flint-census 600 || exit 1
trap '~/.local/bin/rec stop >/dev/null 2>&1; rm -f ${TMPDIR:-/tmp}/rec.tok; bash bench/gpu_lock.sh release flint-census' EXIT
~/.local/bin/rec serve --context 32000 > ${TMPDIR:-/tmp}/rec-census.log 2>&1 &
for i in $(seq 90); do grep -q "is ready" ${TMPDIR:-/tmp}/rec-census.log 2>/dev/null && break; sleep 2; done
umask 077; ~/.local/bin/rec auth token > ${TMPDIR:-/tmp}/rec.tok
python3 -I tools/flint/census.py http://127.0.0.1:11500 ${TMPDIR:-/tmp}/rec.tok research/flint_census.jsonl
