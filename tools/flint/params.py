"""Exact parameter counts (total and active per token) from the int4 checkpoint shapes."""
import json, re, struct, sys, glob, os
D = sys.argv[1]; cfg = json.load(open(os.path.join(D, "config.json")))
bits = {"embed_tokens": 8}
P = {}
for f in sorted(glob.glob(os.path.join(D, "model-*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == "__metadata__" or k.endswith((".scales", ".biases")): continue
        s = v["shape"]
        if v["dtype"] == "U8":           # packed: last dim is bytes
            b = 8 if "embed_tokens" in k else 4
            cnt = 1
            for x in s[:-1]: cnt *= x
            cnt *= s[-1] * 8 // b
        else:
            cnt = 1
            for x in s: cnt *= x
        P[k] = cnt
tot = sum(P.values())
E, K = cfg["num_experts"], cfg["num_experts_per_tok"]
routed = sum(v for k, v in P.items() if "switch_mlp" in k)
embed = sum(v for k, v in P.items() if "embed_tokens" in k)
head = sum(v for k, v in P.items() if k.startswith("lm_head"))
active = tot - routed + routed * K / E - embed + cfg["hidden_size"]   # one embedding row is read per token
attn = sum(v for k, v in P.items() if "self_attn" in k)
shared = sum(v for k, v in P.items() if "shared_expert" in k)
dense = sum(v for k, v in P.items() if re.search(r"layers\.(0|1)\.mlp\.(gate|up|down)_proj", k))
router = sum(v for k, v in P.items() if k.endswith("mlp.gate.weight"))
print(f"total params            {tot/1e9:7.3f} B   (card: 39.8B)")
print(f"  routed experts        {routed/1e9:7.3f} B   ({100*routed/tot:.1f}% of total) — {E} experts x 47 layers")
print(f"  embedding (8-bit)     {embed/1e9:7.3f} B ;  lm_head (4-bit) {head/1e9:.3f} B ; attention {attn/1e9:.3f} B")
print(f"  shared expert {shared/1e9:.3f} B ; dense MLP (L0,L1) {dense/1e9:.3f} B ; router {router/1e9:.4f} B")
print(f"active params / token   {active/1e9:7.3f} B   (card: ~2.5B incl. head; body ~2.2B)  -> body {(active-head)/1e9:.3f} B")
print(f"active share            {100*active/tot:.2f}% of total")
pa = {"attention": attn, "routed (6/256)": routed*K/E, "shared expert": shared, "lm_head": head, "dense MLP L0-1": dense, "router": router}
for k, v in pa.items(): print(f"  active {k:16} {v/1e9:6.3f} B  {100*v/active:5.1f}%")
