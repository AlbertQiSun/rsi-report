"""Flint int4 weight inventory from safetensors headers only (no tensor data read)."""
import json, re, struct, sys, glob, os, collections
D = sys.argv[1]
cfg = json.load(open(os.path.join(D, "config.json")))
rows = []
for f in sorted(glob.glob(os.path.join(D, "model-*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        rows.append((k, v["dtype"], v["shape"], b - a))
print("tensors", len(rows), "bytes", sum(r[3] for r in rows))
dt = collections.Counter(r[1] for r in rows); print("dtypes", dict(dt))
# group by component
def comp(k):
    k2 = re.sub(r"layers\.\d+\.", "layers.N.", k)
    k2 = re.sub(r"experts\.\d+\.", "experts.E.", k2)
    return k2
g = collections.defaultdict(lambda: [0, 0, set(), set()])
for k, d, s, b in rows:
    c = comp(k); g[c][0] += 1; g[c][1] += b; g[c][2].add(d); g[c][3].add(tuple(s))
for c in sorted(g, key=lambda c: -g[c][1]):
    n, b, d, s = g[c]
    print(f"{b/1e9:8.3f} GB  n={n:<4} {sorted(d)} {sorted(s)[:2]}  {c}")
# show one layer fully
print("--- layer 5 (full attn) and layer 6 tensors ---")
for k, d, s, b in rows:
    if re.search(r"layers\.(5|6)\.", k) and "experts." not in k:
        print(f"  {k:70} {d:6} {s}")
print("--- non-layer tensors ---")
for k, d, s, b in rows:
    if "layers." not in k:
        print(f"  {k:70} {d:6} {s}")
print("mlx_export:", cfg.get("mlx_export"), "quantization:", cfg.get("quantization"))
