#!/bin/bash
# Train the NanoChat tokenizer with Karpathy's prepare.py on the 10 TRAIN shards only,
# and install it at $R/data/tokenizer/{tokenizer.pkl,token_bytes.pt}. CPU, ~2 min.
#
# prepare.py reads/writes ~/.cache/autoresearch/{data,tokenizer}. We point HOME at a
# staging dir whose data/ holds symlinks to the train shards plus the dev shard (prepare.py
# expects shard_06542 to exist and skips downloading present files; it excludes that shard
# from tokenizer training). Sealed shards are never linked.
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-$R/envs/rsi-train/bin/python}
PREP=${PREP:-$R/tools/autoresearch/prepare.py}
OUT=$R/data/tokenizer
if [ -s "$OUT/tokenizer.pkl" ] && [ -s "$OUT/token_bytes.pt" ]; then
  echo "tokenizer already present in $OUT"; exit 0
fi
STAGE=$(mktemp -d /tmp/rsi-prep-XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/.cache/autoresearch/data"
n=0
for f in "$R"/data/train/shard_*.parquet; do ln -s "$f" "$STAGE/.cache/autoresearch/data/"; n=$((n+1)); done
ln -s "$R/data/devval/shard_06542.parquet" "$STAGE/.cache/autoresearch/data/"
echo "linked $n train shards + dev shard; training tokenizer with $PREP"
# --num-shards must match the number of train shards so nothing is downloaded
HOME="$STAGE" "$PY" "$PREP" --num-shards "$n"
mkdir -p "$OUT"
cp "$STAGE/.cache/autoresearch/tokenizer/tokenizer.pkl" "$STAGE/.cache/autoresearch/tokenizer/token_bytes.pt" "$OUT/"
chmod 644 "$OUT"/*
"$PY" - "$OUT" <<'EOF'
import os, pickle, sys, torch
d = sys.argv[1]
enc = pickle.load(open(os.path.join(d, "tokenizer.pkl"), "rb"))
tb = torch.load(os.path.join(d, "token_bytes.pt"))
print("tokenizer ok: n_vocab", enc.n_vocab, "token_bytes", tuple(tb.shape), tb.dtype,
      "bos", enc.encode_single_token("<|reserved_0|>"))
EOF
