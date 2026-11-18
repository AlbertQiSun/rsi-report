"""Build model.safetensors.index.json from shard headers (reads only the JSON header of each shard).

usage: python make_index.py MODEL_DIR
Refuses to overwrite an existing index; checks no tensor name appears in two shards.
"""
import json, os, re, struct, sys

DT = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
      "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "U32": 4, "U64": 8}

d = sys.argv[1]
out = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(out):
    sys.exit(f"exists: {out}")
shards = sorted(f for f in os.listdir(d) if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", f))
n_expect = int(re.search(r"of-(\d{5})", shards[0]).group(1))
assert len(shards) == n_expect, (len(shards), n_expect)
wm, total = {}, 0
for s in shards:
    p = os.path.join(d, s)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    data_end = 0
    for k, v in h.items():
        if k == "__metadata__":
            continue
        assert k not in wm, f"duplicate tensor {k} in {s} and {wm[k]}"
        wm[k] = s
        nb = DT[v["dtype"]]
        for x in v["shape"]:
            nb *= x
        assert nb == v["data_offsets"][1] - v["data_offsets"][0], (k, v)
        total += nb
        data_end = max(data_end, v["data_offsets"][1])
    size = os.path.getsize(p)
    assert size == 8 + n + data_end, f"{s}: file size {size} != header-implied {8 + n + data_end} (truncated?)"
json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, open(out, "w"), indent=2)
print(f"wrote {out}: {len(wm)} tensors, {len(shards)} shards, total_size {total/2**30:.1f} GiB; all shard sizes match headers")
