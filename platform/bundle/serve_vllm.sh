#!/usr/bin/env bash
# serve_vllm.sh <9b|27b> <gpu> [--lora] [--dry-run] [-- extra vllm args...]   (owner: worker C)
#
# Starts `vllm serve` (env rsi, vLLM 0.34.0) on 127.0.0.1:<random free high port>, pinned to one GPU,
# and merges {"vllm_<served-name>": port} into $R/run/ports.json (flock, never clobbers other keys).
# Sources for every flag: deploy/serving.md ("Sources").
#
#   9b  -> Qwen/Qwen3.5-9B   served as "improver"    (default GPU0)
#   27b -> Qwen/Qwen3.8-27B  served as "ladder-27b"  (ladder phase, GPU2)
#   --lora    enable LoRA + runtime adapter loading (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True);
#             adapters are loaded later through the gateway as "<served-name>@<adapter>".
#   --dry-run print the resolved command and exit (no GPU needed).
#
# Auth: the API key is read from $R/run/token and passed ONLY via the VLLM_API_KEY env var
# (never argv, never logged). Requests need `Authorization: Bearer <token>` (except /health).
#
# Env overrides (all optional):
#   R (=/scratch/user/rsi)  RSI_SERVE_ENV (=/scratch/user/miniconda3/envs/rsi)
#   HF_HOME (=$R/hf)  HF_HUB_OFFLINE (=1)  RSI_TOKEN_FILE (=$R/run/token)  RSI_PORTS (=$R/run/ports.json)
#   PORT (=random)  MAX_MODEL_LEN (9b 131072 / 27b 65536)  GPU_MEM_UTIL (9b 0.85 / 27b 0.92)
#   TOOL_PARSER (=qwen3_coder; fallbacks qwen3_xml (same parser class in 0.34) or hermes)
#   REASONING_PARSER (=qwen3)  RSI_THINKING (=default | off | on)  PREFIX_CACHING (=auto | 1 | 0)
#   MAX_LORAS (=4)  MAX_LORA_RANK (=64)  MAX_CPU_LORAS (=8)
#   LORA_MODULES ("name=/abs/path,name2=/abs/path2": preloaded as "<served-name>@name")
#   MAX_NUM_SEQS (vLLM default if unset)  LOG (=$R/out/vllm-<served-name>.log)
set -euo pipefail
set +x

usage() { sed -n '2,12p' "$0" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
SIZE=$1; GPU=$2; shift 2
LORA=0; DRY=0; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora) LORA=1 ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "gpu must be an integer index, got '$GPU'" >&2; exit 2; }

R=${R:-/scratch/user/rsi}
ENV_DIR=${RSI_SERVE_ENV:-/scratch/user/miniconda3/envs/rsi}
PY=${RSI_SERVE_PY:-$ENV_DIR/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLATFORM=$(cd "$HERE/.." && pwd)
export PYTHONPATH="$PLATFORM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
TOKEN_FILE=${RSI_TOKEN_FILE:-$R/run/token}
PORTS=${RSI_PORTS:-$R/run/ports.json}

case "$SIZE" in
  9b)  HF_ID=Qwen/Qwen3.5-9B;  NAME=improver;   MML=${MAX_MODEL_LEN:-131072}; GMU=${GPU_MEM_UTIL:-0.85} ;;
  27b) HF_ID=Qwen/Qwen3.8-27B; NAME=ladder-27b; MML=${MAX_MODEL_LEN:-65536};  GMU=${GPU_MEM_UTIL:-0.92} ;;
  *) usage ;;
esac
MODEL_PATH=${MODEL_PATH:-$HF_ID}   # an explicit snapshot dir also works

if [[ $DRY -eq 0 ]]; then
  [[ -s "$TOKEN_FILE" ]] || { echo "missing token file $TOKEN_FILE (create: umask 077; openssl rand -hex 24 > \$R/run/token)" >&2; exit 3; }
  perm=$(stat -c %a "$TOKEN_FILE" 2>/dev/null || stat -f %Lp "$TOKEN_FILE")
  [[ "$perm" == 600 || "$perm" == 400 ]] || echo "WARNING: $TOKEN_FILE mode $perm (want 600)" >&2
  VLLM_API_KEY=$(<"$TOKEN_FILE"); export VLLM_API_KEY
fi

PORT=${PORT:-$("$PY" -m rsi.gateway.ports free)}

ARGS=(serve "$MODEL_PATH"
  --host 127.0.0.1 --port "$PORT"
  --served-model-name "$NAME"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$MML"
  --gpu-memory-utilization "$GMU"
  --language-model-only
  --reasoning-parser "${REASONING_PARSER:-qwen3}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}")

case "${RSI_THINKING:-default}" in
  off) ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}') ;;
  on)  ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}') ;;
  default) ;;
  *) echo "RSI_THINKING must be default|on|off" >&2; exit 2 ;;
esac
case "${PREFIX_CACHING:-auto}" in
  1) ARGS+=(--enable-prefix-caching) ;;
  0) ARGS+=(--disable-prefix-caching) ;;
esac
[[ -n "${MAX_NUM_SEQS:-}" ]] && ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")

if [[ $LORA -eq 1 ]]; then
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
  ARGS+=(--enable-lora --max-loras "${MAX_LORAS:-4}" --max-lora-rank "${MAX_LORA_RANK:-64}"
         --max-cpu-loras "${MAX_CPU_LORAS:-8}")
  if [[ -n "${LORA_MODULES:-}" ]]; then
    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
    LM=()
    for m in "${_mods[@]}"; do
      n=${m%%=*}; p=${m#*=}
      LM+=("{\"name\": \"$NAME@$n\", \"path\": \"$p\", \"base_model_name\": \"$NAME\"}")
    done
    ARGS+=(--lora-modules "${LM[@]}")
  fi
fi
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

if [[ -x "$ENV_DIR/bin/vllm" ]]; then VLLM=("$ENV_DIR/bin/vllm"); else VLLM=("$PY" -m vllm.entrypoints.cli.main); fi

if [[ $DRY -eq 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
       "VLLM_API_KEY=<from $TOKEN_FILE>${VLLM_ALLOW_RUNTIME_LORA_UPDATING:+ VLLM_ALLOW_RUNTIME_LORA_UPDATING=True}"
  printf '%q ' "${VLLM[@]}" "${ARGS[@]}"; echo
  echo "ports.json key: vllm_$NAME=$PORT ($PORTS)"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPU
LOG=${LOG:-$R/out/vllm-$NAME.log}
mkdir -p "$(dirname "$LOG")" "$(dirname "$PORTS")" "$R/run"
"$PY" -m rsi.gateway.ports merge --path "$PORTS" "vllm_$NAME=$PORT"
"$PY" - "$R/run/vllm_$NAME.json" <<EOF
import json, sys, time
json.dump({"name": "$NAME", "hf_id": "$HF_ID", "port": $PORT, "gpu": $GPU, "lora": bool($LORA),
           "max_model_len": $MML, "gpu_memory_utilization": $GMU, "tool_parser": "${TOOL_PARSER:-qwen3_coder}",
           "reasoning_parser": "${REASONING_PARSER:-qwen3}", "thinking": "${RSI_THINKING:-default}",
           "started": time.time(), "log": "$LOG"}, open(sys.argv[1], "w"), indent=1)
EOF
echo "[serve_vllm] $NAME ($HF_ID) on GPU$GPU -> http://127.0.0.1:$PORT/v1  log=$LOG  lora=$LORA" | tee -a "$LOG"
echo "[serve_vllm] $(printf '%q ' "${VLLM[@]}" "${ARGS[@]}")" >> "$LOG"
exec "${VLLM[@]}" "${ARGS[@]}" > >(tee -a "$LOG") 2>&1
