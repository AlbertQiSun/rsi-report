"""Dissect Flint's MoE routing and attention sinks on real text (MLX, read-only use of rec's model code).

usage: <rec runtime python> -I router_stats.py <flint-int4 dir> <rec engine src dir> <out.json>
Records, per MoE layer and per text domain: expert usage histogram, load-balance entropy, coverage,
top-1 routing weight, router score stats; plus attention-sink offsets per layer/head.
"""
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

D, SRC, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC)
from rec_mlx import model as M  # noqa: E402
from rec_mlx.quantized import QATModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEXTS = {
    "code": (ROOT / "rec/engine/src/rec_mlx/engine.py").read_text()[:12000],
    "zh": (ROOT / "research/reading/01_Recursive_FirstSteps_导读.md").read_text()[:6000],
    "en_prose": (ROOT / "first-steps-toward-automated-ai-research/README.md").read_text()[:12000],
    "math": "\n".join(f"Problem {i}: compute the integral of x^{i} * exp(-{i}x) from 0 to infinity, "
                      f"then simplify. Solution: by repeated integration by parts we obtain {i}!/{i}^{i+1}. "
                      f"Check with the Gamma function: Gamma({i + 1}) = {i}!." for i in range(1, 60)),
}

REC = {}
CUR = {"domain": None}


def patched(self, x):
    scores = mx.sigmoid(self.gate(x).astype(mx.float32))
    choice = scores + self.gate.expert_bias.astype(mx.float32)
    idx = mx.stop_gradient(mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[..., : self.top_k])
    w = mx.take_along_axis(scores, idx, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    mx.eval(idx, w, scores)
    REC.setdefault(CUR["domain"], {}).setdefault(id(self), []).append(
        (np.array(idx).reshape(-1, self.top_k), np.array(w).reshape(-1, self.top_k),
         np.array(scores).reshape(-1, scores.shape[-1])))
    routed = (self.switch_mlp(x, idx) * w[..., None].astype(x.dtype)).sum(axis=-2)
    return routed + self.shared_expert(x)


M.SparseMoeBlock.__call__ = patched

t0 = time.time()
qm = QATModel(D)
native = qm.native
tok = AutoTokenizer.from_pretrained(str(D))
moe_layers = [(i, l.mlp) for i, l in enumerate(native.layers) if isinstance(l.mlp, M.SparseMoeBlock)]
layer_of = {id(m): i for i, m in moe_layers}
print(f"loaded in {time.time() - t0:.1f}s; MoE layers: {len(moe_layers)}", flush=True)

ntok = {}
for dom, text in TEXTS.items():
    ids = tok(text, add_special_tokens=False)["input_ids"][:2048]
    ntok[dom] = len(ids)
    CUR["domain"] = dom
    logits = native(mx.array([ids]))
    mx.eval(logits)
    print(f"{dom}: {len(ids)} tokens", flush=True)

E = native.args.num_experts if hasattr(native, "args") else 256
res = {"n_tokens": ntok, "layers": {}, "sinks": {}}
for dom, per in REC.items():
    for mid, chunks in per.items():
        L = layer_of[mid]
        idx = np.concatenate([c[0] for c in chunks])
        w = np.concatenate([c[1] for c in chunks])
        sc = np.concatenate([c[2] for c in chunks])
        cnt = np.bincount(idx.ravel(), minlength=E).astype(float)
        p = cnt / cnt.sum()
        ent = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(E)
        top_sorted = np.sort(p)[::-1]
        d = res["layers"].setdefault(str(L), {})
        d[dom] = {"norm_entropy": round(float(ent), 4), "coverage": int((cnt > 0).sum()),
                  "top10_share": round(float(top_sorted[:10].sum()), 4),
                  "top1_weight_mean": round(float(np.sort(w, axis=1)[:, -1].mean()), 4),
                  "score_mean": round(float(sc.mean()), 4), "score_top6_mean": round(float(np.sort(sc, axis=1)[:, -6:].mean()), 4),
                  "hist": cnt.astype(int).tolist()}
for i, l in enumerate(native.layers):
    s = np.array(l.self_attn.softmax_offset.astype(mx.float32))
    res["sinks"][str(i)] = {"global": not l.self_attn.use_rope, "mean": round(float(s.mean()), 3),
                            "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}
# cross-domain overlap of each layer's 20 most used experts
for L, d in res["layers"].items():
    tops = {dom: set(np.argsort(v["hist"])[::-1][:20].tolist()) for dom, v in d.items()}
    doms = sorted(tops)
    d["jaccard_top20"] = {f"{a}|{b}": round(len(tops[a] & tops[b]) / len(tops[a] | tops[b]), 3)
                          for i, a in enumerate(doms) for b in doms[i + 1:]}
Path(OUT).write_text(json.dumps(res))
print("wrote", OUT, f"total {time.time() - t0:.1f}s")
