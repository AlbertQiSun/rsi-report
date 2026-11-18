"""Pre-training forensics for Flint from weights + tokenizer (MLX; reads only embedding and lm_head).

1. Vocab composition by script/category (what corpus the BPE was trained on).
2. Under-trained-token detection (Land & Bartolo 2024 style): tokens whose unembedding rows look like the
   never-trained reserved tokens were (almost) never seen as targets in pre-training.
usage: <rec runtime python> -I pretrain_forensics.py <flint-int4 dir> <out.json>
"""
import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import mlx.core as mx
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
tok = json.load(open(f"{D}/tokenizer.json"))
vocab = tok["model"]["vocab"]
id2 = {i: s for s, i in vocab.items()}
added = {a["id"]: a["content"] for a in tok["added_tokens"]}

# byte-level decode table (GPT-2 style)
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}


def text(s):
    try:
        return bytes(dec[c] for c in s).decode("utf-8")
    except Exception:
        return None  # partial UTF-8 byte sequence


def cat(t):
    if t is None:
        return "partial-utf8-bytes"
    core = t.strip()
    if t and not core:
        return "whitespace/indent"
    if re.fullmatch(r"[0-9]+", core):
        return "digits"
    scripts = Counter()
    for ch in core:
        if ch.isascii():
            scripts["ascii-letter" if ch.isalpha() else "ascii-punct/code"] += 1
            continue
        name = unicodedata.name(ch, "")
        for key, lab in (("CJK", "CJK"), ("HIRAGANA", "japanese-kana"), ("KATAKANA", "japanese-kana"),
                         ("HANGUL", "korean"), ("CYRILLIC", "cyrillic"), ("ARABIC", "arabic"), ("GREEK", "greek"),
                         ("DEVANAGARI", "devanagari"), ("THAI", "thai"), ("HEBREW", "hebrew"), ("LATIN", "latin-ext")):
            if key in name:
                scripts[lab] += 1
                break
        else:
            scripts["other-unicode"] += 1
    if not scripts:
        return "other"
    top = scripts.most_common(1)[0][0]
    if top in ("ascii-letter", "ascii-punct/code") and scripts.get("ascii-punct/code") and scripts.get("ascii-letter"):
        return "ascii-mixed(code-like)"
    return top


cats = {i: cat(text(s)) for i, s in id2.items()}
comp = Counter(cats.values())

# ---- weights: dequantize lm_head (4-bit) and embedding (8-bit)
W = {}
for f in sorted(glob.glob(f"{D}/model-*.safetensors")):
    w = mx.load(f)
    for k in list(w):
        if k.startswith("lm_head.") or k.startswith("model.embed_tokens."):
            W[k] = w[k]
def u32(a):  # checkpoint stores packed bytes as U8; MLX wants the same bytes viewed as uint32
    return mx.array(np.array(a).view(np.uint32))
U = mx.dequantize(u32(W["lm_head.weight"]), W["lm_head.scales"], W["lm_head.biases"], group_size=64, bits=4).astype(mx.float32)
Em = mx.dequantize(u32(W["model.embed_tokens.weight"]), W["model.embed_tokens.scales"], W["model.embed_tokens.biases"],
                   group_size=64, bits=8).astype(mx.float32)
U = np.array(U); Em = np.array(Em)
V = U.shape[0]
reserved = [i for i, c in added.items() if c.startswith("<|reserved_")]
ref = U[reserved].mean(0)
gm = U[:150000].mean(0)
# indicator: cosine(row - global mean, reserved mean - global mean); first principal direction removed
X = U - gm
_, _, vt = np.linalg.svd(X[np.random.default_rng(0).choice(150000, 20000, replace=False)], full_matrices=False)
pc1 = vt[0]
Xr = X - np.outer(X @ pc1, pc1)
r = (ref - gm); r = r - (r @ pc1) * pc1
ind = (Xr @ r) / (np.linalg.norm(Xr, axis=1) * np.linalg.norm(r) + 1e-9)
res_ind = ind[reserved]
thr = float(np.percentile(res_ind, 5))  # tokens at least as "reserved-like" as the bottom 5% of reserved tokens
under = [i for i in range(150000) if ind[i] >= thr]
under_cat = Counter(cats[i] for i in under)
emb_norm = np.linalg.norm(Em, axis=1)
by_cat_norm = defaultdict(list)
for i in range(150000):
    by_cat_norm[cats[i]].append(float(emb_norm[i]))
out = {
    "vocab_composition": dict(comp.most_common()),
    "reserved_indicator": {"n": len(reserved), "mean": float(res_ind.mean()), "min": float(res_ind.min()), "thr_p5": thr},
    "normal_indicator": {"mean": float(ind[:150000].mean()), "p99": float(np.percentile(ind[:150000], 99))},
    "n_undertrained": len(under),
    "undertrained_by_category": dict(under_cat.most_common()),
    "undertrained_rate_by_category": {c: round(under_cat.get(c, 0) / comp[c], 4) for c in comp},
    "undertrained_examples": [repr(text(id2[i]) or id2[i]) for i in sorted(under, key=lambda i: -ind[i])[:60]],
    "embedding_norm_by_category": {c: round(float(np.median(v)), 3) for c, v in by_cat_norm.items()},
    "special_token_indicator": {added[i]: round(float(ind[i]), 3) for i in sorted(added) if not added[i].startswith("<|reserved_")},
}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("vocab_composition", "reserved_indicator", "normal_indicator", "n_undertrained",
                                      "undertrained_by_category", "embedding_norm_by_category")}, ensure_ascii=False, indent=1))
print("examples:", out["undertrained_examples"][:40])
print("special:", out["special_token_indicator"])
