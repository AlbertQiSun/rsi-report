#!/bin/bash
# U_w: PEFT LoRA SFT of Qwen/Qwen3.5-9B (BF16) on ONE A100 lane. Owner: worker D.
#
#   bundle/lora_train.sh <sft_dir> <adapter_out> <gpu>
#
#   <sft_dir>     output of `python -m rsi.update.sft` (train.jsonl, valid.jsonl, manifest.json)
#   <adapter_out> new directory; receives adapter_model.safetensors + adapter_config.json (PEFT format,
#                 loadable by `vllm serve ... --enable-lora --max-lora-rank 16` or POST /v1/load_lora_adapter),
#                 train_log.jsonl, rsi_adapter.json (provenance). Last stdout line = absolute adapter path.
#   <gpu>         physical GPU index (nvidia-smi numbering). The scheduler must pause that eval lane first;
#                 this script refuses to start if the GPU has compute processes (RSI_ALLOW_BUSY_GPU=1 overrides).
#
# Python env: the serving env `rsi` (peft 0.21.3, transformers 5.22.2, torch 2.14). Reason: it is the env
# whose transformers already loads `qwen3_5` for vLLM, so the module names and adapter_config written here are
# exactly what the same vLLM install loads; the NanoChat train env ($R/envs/rsi-train) is not guaranteed to
# have peft or a qwen3_5-aware transformers. Override with RSI_TRAIN_PY.
#
# Defaults (~30-60 min on one A100 80GB for a few hundred examples of 2-6k tokens): rank 16, alpha 32,
# dropout 0.05, lr 1e-4 cosine with 5% warmup, 2 epochs, 8 examples per optimizer step, max_len 6144,
# gradient checkpointing, BF16 weights, loss on the final assistant turn only, best-by-valid checkpoint,
# hard stop at 55 min (LORA_TIME_LIMIT_MIN). Extra flags for rsi.update.lora_train go in LORA_ARGS.
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 <sft_dir> <adapter_out> <gpu>" >&2
  exit 2
fi
SFT_DIR=$1
OUT=$2
GPU=$3

R=${R:-/scratch/user/rsi}
PY=${RSI_TRAIN_PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -n "${RSI_SRC:-}" ]; then SRC=$RSI_SRC
elif [ -f "$HERE/../rsi/__init__.py" ]; then SRC=$(cd "$HERE/.." && pwd)
elif [ -f "$R/platform/rsi/__init__.py" ]; then SRC=$R/platform
else SRC=$R/bundle; fi

[ -f "$SFT_DIR/train.jsonl" ] || { echo "no $SFT_DIR/train.jsonl" >&2; exit 3; }
if [ -e "$OUT/adapter_model.safetensors" ]; then
  echo "refusing to overwrite existing adapter in $OUT" >&2; exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | grep -c . || true)
  if [ "${busy:-0}" != "0" ] && [ "${RSI_ALLOW_BUSY_GPU:-0}" != "1" ]; then
    echo "GPU $GPU has $busy compute process(es); pause its lane first" >&2; exit 4
  fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$SRC${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-$R/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}          # weights were downloaded by UV1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

mkdir -p "$OUT"
echo "[lora_train.sh] sft=$SFT_DIR out=$OUT gpu=$GPU py=$PY src=$SRC" >&2
# shellcheck disable=SC2086
"$PY" -m rsi.update.lora_train --sft-dir "$SFT_DIR" --out "$OUT" \
  --model "${RSI_BASE_MODEL:-Qwen/Qwen3.5-9B}" \
  --time-limit-min "${LORA_TIME_LIMIT_MIN:-55}" ${LORA_ARGS:-} 2>&1 | tee -a "$OUT/train.log"
status=${PIPESTATUS[0]}
[ "$status" -eq 0 ] || exit "$status"
[ -f "$OUT/adapter_model.safetensors" ] || { echo "no adapter written" >&2; exit 5; }
cd "$OUT" && pwd
