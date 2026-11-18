#!/usr/bin/env bash
# Naive-N0.5-Flash-FP8: index -> env check -> smoke + full census (one model load) -> summary.
# Run inside a GPU allocation (4x H20), e.g.:
#   srun --jobid=<JOBID> --overlap --ntasks=1 -N1 --gres=gpu:4 bash /scratch/user/naive/run_naive.sh
# Resumable: rerunning skips the index if it exists and skips prompts already in the jsonl.
set -euo pipefail

WORK=/scratch/user/naive
MODEL=$WORK/Naive-N0.5-Flash-FP8
PY=/scratch/user/venvs/naive/bin/python
PROMPTS=$WORK/prompts.json
OUT=$WORK/naive_census_all.jsonl
LOG=$WORK/run_naive.log

cd "$WORK"
exec > >(tee -a "$LOG") 2>&1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$WORK/.hf_home              # own cache (remote-code modules land here); never another project's
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SECONDS=0
stage() { echo; echo "===== [$(date '+%F %T')] STAGE $* ====="; }
pass()  { echo "PASS: $*"; }
fail()  { echo "FAIL: $*"; echo "elapsed ${SECONDS}s; full log: $LOG"; exit 1; }

echo "########## run_naive.sh start $(date '+%F %T') host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} ##########"

# ---- 0. files present ----
stage "0 files"
for f in "$PY" "$WORK/make_index.py" "$WORK/naive_census.py" "$PROMPTS"; do
  [ -e "$f" ] || fail "missing $f"
done
[ -d "$MODEL" ] || fail "missing model dir $MODEL"
nshard=$(ls "$MODEL" | grep -cE '^model-[0-9]{5}-of-00049\.safetensors$' || true)
[ "$nshard" = 49 ] || fail "expected 49 shards in $MODEL, found $nshard"
pass "49 shards, python, make_index.py, naive_census.py, prompts.json present"

# ---- 1. index ----
stage "1 index"
if [ -f "$MODEL/model.safetensors.index.json" ]; then
  pass "index already exists (not rebuilt)"
else
  "$PY" "$WORK/make_index.py" "$MODEL" || fail "make_index.py failed (truncated shard or duplicate tensor? see message above)"
  pass "index written"
fi
"$PY" - "$MODEL/model.safetensors.index.json" <<'EOF' || fail "index unreadable"
import json, sys
j = json.load(open(sys.argv[1]))
print(f"index: {len(j['weight_map'])} tensors, {len(set(j['weight_map'].values()))} shards, "
      f"total_size {j['metadata']['total_size']/2**30:.1f} GiB (budget 4 x 88 GiB = 352 GiB)")
EOF

# ---- 2. environment ----
stage "2 environment"
nvidia-smi -L || fail "nvidia-smi failed (not inside a GPU allocation?)"
ngpu=$(nvidia-smi -L | grep -c '^GPU' || true)
[ "$ngpu" -ge 4 ] || echo "WARNING: only $ngpu GPU(s) visible; expected 4 (check --gres=gpu:4)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h "$WORK" || true
"$PY" -c "import sys, torch, transformers; print('python', sys.version.split()[0], '| torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| cuda devices', torch.cuda.device_count(), '| cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)" \
  || fail "python/torch import failed"
pass "environment printed"

# ---- 3. smoke + full census in ONE process (model loads once, ~12 min) ----
# naive_census.py (default --only all) first checks loading info, then the 17*23 -> 391 smoke test, and exits non-zero
# with "ABORT: ..." before writing any census line if either fails; the batching check only falls back to batch 1.
stage "3 smoke + census (batch 16) -> $OUT"
t_cen=$SECONDS
if "$PY" "$WORK/naive_census.py" --model "$MODEL" --prompts "$PROMPTS" --out "$OUT" --batch 16; then
  pass "smoke + census finished ($((SECONDS - t_cen))s)"
else
  fail "naive_census.py exited non-zero: look for ABORT (loading info / no 391) or a Traceback above; partial results (if any) kept in $OUT, rerun this script to resume"
fi

# ---- 4. summary ----
stage "4 summary"
"$PY" - "$OUT" "$PROMPTS" <<'EOF'
import json, sys, collections
rows = []
for l in open(sys.argv[1]):
    try: rows.append(json.loads(l))
    except Exception: pass
n_lines = sum(1 for _ in open(sys.argv[1]))
n_prompts = len(json.load(open(sys.argv[2])))
print(f"lines in jsonl: {n_lines} (parsed {len(rows)}; prompts.json has {n_prompts})")
print(f"non-empty think : {sum(1 for r in rows if r.get('reasoning'))}")
print(f"non-empty answer: {sum(1 for r in rows if r.get('content'))}")
print("finish:", dict(collections.Counter(r.get('finish') for r in rows)))
print("by set:", dict(collections.Counter(r.get('set') for r in rows)))
print("STATUS:", "COMPLETE" if len(rows) >= n_prompts else "INCOMPLETE (rerun to resume)")
EOF
echo "total elapsed: ${SECONDS}s ($((SECONDS / 60)) min); log: $LOG"
echo "########## run_naive.sh end $(date '+%F %T') ##########"
