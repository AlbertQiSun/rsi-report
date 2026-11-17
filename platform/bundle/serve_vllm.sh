#!/usr/bin/env bash
# serve_vllm.sh <9b|27b> <gpus> [--lora] [--expose] [--dry-run] [-- extra vllm args...]   (owner: worker C)
#
#   9b  -> Qwen/Qwen3.5-9B   served as "improver"
#   27b -> Qwen/Qwen3.8-27B  served as "ladder-27b"
#   <gpus>    "0" | "0,1" (sets CUDA_VISIBLE_DEVICES) | "slurm" (keep what srun --gres set).
#             --tensor-parallel-size = number of GPUs (override: TP=...).
#   --lora    enable LoRA + runtime adapter loading (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True); adapters are
#             loaded later through the gateway as "<served-name>@<adapter>".
#   --expose  for another node to reach this model (27B on node-c): vLLM still binds 127.0.0.1, and an
#             authenticated edge gateway (python -m rsi.gateway --no-record) binds this node's cluster IP.
#             vLLM is never bound off-loopback: its --api-key guards only /v1,/v2,/inference,/cohere,
#             while /invocations, /tokenize, /score, /pooling, /metrics stay unauthenticated.
#   --dry-run print the resolved commands and exit (no GPU, no token needed).
#
# ports.json ($R/run/ports.json, merged under flock, never clobbered):
#   local mode : "vllm_<name>": <port>,  "<name>": "http://127.0.0.1:<port>"
#   --expose   : "<name>": "http://<cluster-ip>:<edge port>"   (and "vllm_<name>" is removed)
# Auth: the token in $R/run/token goes to vLLM only via env VLLM_API_KEY, and to the edge gateway via
# --token-file. It never appears on argv or in logs.
#
# Env overrides: R, RSI_SERVE_ENV, HF_HOME(=$R/hf), HF_HUB_OFFLINE(=1), RSI_TOKEN_FILE, RSI_PORTS, PORT,
#   EDGE_PORT, BIND_IP, TP, MAX_MODEL_LEN, GPU_MEM_UTIL, TOOL_PARSER(=qwen3_coder), REASONING_PARSER(=qwen3),
#   RSI_THINKING(=default|on|off), PREFIX_CACHING(=auto|1|0), MAX_LORAS(=4), MAX_LORA_RANK(=64),
#   MAX_CPU_LORAS(=8), LORA_MODULES("name=/abs/path,..."), MAX_NUM_SEQS, MODEL_PATH, LOG.
# Sources for every flag: deploy/serving.md ("Sources").
set -euo pipefail
set +x

usage() { sed -n '2,15p' "$0" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
SIZE=$1; GPUS=$2; shift 2
LORA=0; DRY=0; EXPOSE=0; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora) LORA=1 ;;
    --expose) EXPOSE=1 ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

if [[ "$GPUS" == slurm ]]; then
  VIS=${CUDA_VISIBLE_DEVICES:-}
  [[ -n "$VIS" || $DRY -eq 1 ]] || { echo "gpus=slurm but CUDA_VISIBLE_DEVICES is unset (run inside srun --gres=gpu:a100:N)" >&2; exit 2; }
  VIS=${VIS:-0}
elif [[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  VIS=$GPUS
else
  echo "gpus must be like 0 or 0,1 or 'slurm', got '$GPUS'" >&2; exit 2
fi
NGPU=$(awk -F, '{print NF}' <<< "$VIS")
TP=${TP:-$NGPU}

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
  9b)  HF_ID=Qwen/Qwen3.5-9B;  NAME=improver
       MML=${MAX_MODEL_LEN:-131072}; GMU=${GPU_MEM_UTIL:-0.85} ;;
  27b) HF_ID=Qwen/Qwen3.8-27B; NAME=${SERVED_NAME:-ladder-27b}   # SERVED_NAME: a second 27B base, e.g. ladder-27b-evo
       if [[ $TP -ge 2 ]]; then MML=${MAX_MODEL_LEN:-131072}; GMU=${GPU_MEM_UTIL:-0.90}
       else MML=${MAX_MODEL_LEN:-65536}; GMU=${GPU_MEM_UTIL:-0.92}; fi ;;
  *) usage ;;
esac
MODEL_PATH=${MODEL_PATH:-$HF_ID}

if [[ $DRY -eq 0 ]]; then
  [[ -s "$TOKEN_FILE" ]] || { echo "missing token file $TOKEN_FILE (create once: umask 077; openssl rand -hex 24 > \$R/run/token)" >&2; exit 3; }
  perm=$(stat -c %a "$TOKEN_FILE" 2>/dev/null || stat -f %Lp "$TOKEN_FILE")
  [[ "$perm" == 600 || "$perm" == 400 ]] || { echo "$TOKEN_FILE has mode $perm; run chmod 600 (other users can read it)" >&2; exit 3; }
  VLLM_API_KEY=$(<"$TOKEN_FILE"); export VLLM_API_KEY
fi

PORT=${PORT:-$("$PY" -m rsi.gateway.ports free)}
ARGS=(serve "$MODEL_PATH"
  --host 127.0.0.1 --port "$PORT"
  --served-model-name "$NAME"
  --tensor-parallel-size "$TP"
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
      LM+=("{\"name\": \"$NAME@${m%%=*}\", \"path\": \"${m#*=}\", \"base_model_name\": \"$NAME\"}")
    done
    ARGS+=(--lora-modules "${LM[@]}")
  fi
fi
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")
if [[ -x "$ENV_DIR/bin/vllm" ]]; then VLLM=("$ENV_DIR/bin/vllm"); else VLLM=("$PY" -m vllm.entrypoints.cli.main); fi

if [[ $EXPOSE -eq 1 ]]; then
  IP=$("$PY" -m rsi.gateway.ports ip)
  [[ "$IP" != 127.* ]] || { echo "cluster IP resolved to $IP; set BIND_IP" >&2; exit 2; }
  EDGE_PORT=${EDGE_PORT:-$("$PY" -m rsi.gateway.ports free --host "$IP" 2>/dev/null || true)}
  if [[ -z "$EDGE_PORT" ]]; then
    [[ $DRY -eq 1 ]] || { echo "cannot bind a port on $IP (wrong BIND_IP?)" >&2; exit 2; }
    EDGE_PORT="<edge-port>"
  fi
  EDGE=("$PY" -m rsi.gateway --host "$IP" --port "$EDGE_PORT" --upstream "$NAME=http://127.0.0.1:$PORT"
        --token-file "$TOKEN_FILE" --ports-file "$PORTS" --ports-key "" --no-record
        --retry-deadline-s 60)
  URL="http://$IP:$EDGE_PORT"
else
  URL="http://127.0.0.1:$PORT"
fi

if [[ $DRY -eq 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES=$VIS HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
       "VLLM_API_KEY=<from $TOKEN_FILE>${VLLM_ALLOW_RUNTIME_LORA_UPDATING:+ VLLM_ALLOW_RUNTIME_LORA_UPDATING=True}"
  printf '%q ' "${VLLM[@]}" "${ARGS[@]}"; echo
  [[ $EXPOSE -eq 1 ]] && echo "edge: $(printf '%q ' "${EDGE[@]}")"
  if [[ $EXPOSE -eq 1 ]]; then echo "ports.json: $NAME=$URL (remove vllm_$NAME)  [$PORTS]"
  else echo "ports.json: vllm_$NAME=$PORT $NAME=$URL  [$PORTS]"; fi
  exit 0
fi

export CUDA_VISIBLE_DEVICES=$VIS
LOG=${LOG:-$R/out/vllm-$NAME.log}
mkdir -p "$(dirname "$LOG")" "$(dirname "$PORTS")" "$R/run"
if [[ $EXPOSE -eq 1 ]]; then
  "$PY" -m rsi.gateway.ports remove --path "$PORTS" "vllm_$NAME"
  "$PY" -m rsi.gateway.ports merge --path "$PORTS" "$NAME=$URL"
else
  "$PY" -m rsi.gateway.ports merge --path "$PORTS" "vllm_$NAME=$PORT" "$NAME=$URL"
fi
"$PY" - "$R/run/vllm_$NAME.json" <<EOF
import json, socket, sys, time
json.dump({"name": "$NAME", "hf_id": "$HF_ID", "url": "$URL", "vllm_port": $PORT, "node": socket.gethostname(),
           "cuda_visible_devices": "$VIS", "tp": $TP, "lora": bool($LORA), "expose": bool($EXPOSE),
           "max_model_len": $MML, "gpu_memory_utilization": $GMU, "tool_parser": "${TOOL_PARSER:-qwen3_coder}",
           "reasoning_parser": "${REASONING_PARSER:-qwen3}", "thinking": "${RSI_THINKING:-default}",
           "slurm_job": "${SLURM_JOB_ID:-}", "started": time.time(), "log": "$LOG"},
          open(sys.argv[1], "w"), indent=1)
EOF
{
  echo "[serve_vllm] $(date "+%F %T") $NAME ($HF_ID) node=$(hostname) GPUs=$VIS tp=$TP -> $URL lora=$LORA expose=$EXPOSE"
  nvidia-smi -L 2>/dev/null || true
  echo "[serve_vllm] $(printf '%q ' "${VLLM[@]}" "${ARGS[@]}")"
} | tee -a "$LOG"

if [[ $EXPOSE -eq 0 ]]; then
  exec "${VLLM[@]}" "${ARGS[@]}" > >(tee -a "$LOG") 2>&1
fi

# --expose: run vLLM (loopback) and the edge gateway (cluster IP) together; if either dies, stop both.
EDGE_LOG=$R/out/edge-$NAME.log
"${VLLM[@]}" "${ARGS[@]}" >> "$LOG" 2>&1 &
VPID=$!
unset VLLM_API_KEY   # the edge gateway reads the token file itself
"${EDGE[@]}" >> "$EDGE_LOG" 2>&1 &
EPID=$!
cleanup() { kill "$VPID" "$EPID" 2>/dev/null || true; sleep 2; kill -9 "$VPID" "$EPID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
echo "[serve_vllm] vllm pid=$VPID (log $LOG)  edge pid=$EPID (log $EDGE_LOG)  url=$URL" | tee -a "$LOG"
while kill -0 "$VPID" 2>/dev/null && kill -0 "$EPID" 2>/dev/null; do sleep 2; done
echo "[serve_vllm] $(date) vllm alive=$(kill -0 "$VPID" 2>/dev/null && echo y || echo n)" \
     "edge alive=$(kill -0 "$EPID" 2>/dev/null && echo y || echo n); stopping both" | tee -a "$LOG"
exit 1
