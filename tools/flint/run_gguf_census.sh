#!/bin/bash
# Behavioural census for a GGUF model via llama-server (Mac GPU, under the bench GPU lock).
# usage: run_gguf_census.sh <gguf> <tag>     -> research/<tag>_census.jsonl, research/<tag>_probes2.jsonl
set -u
cd "$(dirname "$0")/../.."
G=$1; TAG=$2; PORT=11710
bash bench/gpu_lock.sh acquire "census-$TAG" 3600 || exit 1
llama-server -m "$G" --port $PORT --host 127.0.0.1 -ngl 999 -c 16384 --jinja --reasoning-format deepseek > "research/${TAG}_llama.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null; bash bench/gpu_lock.sh release "census-$TAG"' EXIT
for i in $(seq 180); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
umask 077; echo none > ${TMPDIR:-/tmp}/gguf.tok
python3 -I tools/flint/census.py http://127.0.0.1:$PORT ${TMPDIR:-/tmp}/gguf.tok "research/${TAG}_census.jsonl" > "research/${TAG}_census.log" 2>&1
python3 -I tools/flint/probes2.py http://127.0.0.1:$PORT ${TMPDIR:-/tmp}/gguf.tok "research/${TAG}_probes2.jsonl" > "research/${TAG}_probes2.log" 2>&1
python3 -I tools/flint/probes.py http://127.0.0.1:$PORT ${TMPDIR:-/tmp}/gguf.tok "research/${TAG}_probes.jsonl" > "research/${TAG}_probes.log" 2>&1
echo "CENSUS_DONE $TAG $(date +%T)"
