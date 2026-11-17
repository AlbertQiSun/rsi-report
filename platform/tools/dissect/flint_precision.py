#!/usr/bin/env python3
"""Flint quantization forensics: BF16 reference vs the released MLX int4/int8 checkpoint.

usage: flint_precision.py --bf16 nonexp.safetensors,exp.safetensors --int4-dir <flint-int4 dir> --out out.json
For every BF16 tensor that has a quantized counterpart: dequantize (MLX affine, group 64; 4-bit, embedding 8-bit),
compute relative error ||Wq - W||/||W||, and compare with a plain min-max affine re-quantization of the BF16
weights at the same bits/group (to see how much the released "scale_rule: mse" recipe buys).
"""
import argparse
import glob
import json
import re

import torch
from safetensors import safe_open

torch.set_num_threads(8)
G = 64


def unpack(packed_u8, bits):
    """MLX packs little-endian into uint32; viewed as bytes, 4-bit: low nibble = even element."""
    p = packed_u8.to(torch.int32)
    if bits == 8:
        return p
    lo, hi = p & 0xF, (p >> 4) & 0xF
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def deq(w_u8, scales, biases, bits):
    q = unpack(w_u8, bits).float()
    shp = q.shape
    q = q.reshape(*shp[:-1], shp[-1] // G, G)
    return (q * scales.float()[..., None] + biases.float()[..., None]).reshape(shp)


def minmax_requant(w, bits):
    shp = w.shape
    x = w.float().reshape(*shp[:-1], shp[-1] // G, G)
    lo, hi = x.min(-1, keepdim=True).values, x.max(-1, keepdim=True).values
    n = (1 << bits) - 1
    sc = (hi - lo).clamp_min(1e-12) / n
    q = torch.round((x - lo) / sc).clamp(0, n)
    return (q * sc + lo).reshape(shp)


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True); ap.add_argument("--int4-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = {}
    for p in a.bf16.split(","):
        f = safe_open(p, framework="pt")
        for k in f.keys():
            ref[k] = f
    qf = {}
    for p in sorted(glob.glob(f"{a.int4_dir}/model-*.safetensors")):
        f = safe_open(p, framework="pt")
        for k in f.keys():
            qf[k] = f
    rows = []
    for k in sorted(ref):
        if not k.endswith(".weight") or ref[k].get_slice(k).get_shape().__len__() != 2:
            continue
        m = re.match(r"(.*\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", k)
        if m:
            base = f"{m.group(1)}.switch_mlp.{m.group(3)}_proj"; idx = int(m.group(2))
        else:
            base = k[: -len(".weight")]; idx = None
        if base + ".weight" not in qf or base + ".scales" not in qf:
            continue
        bits = 8 if "embed_tokens" in k else 4
        W = ref[k].get_tensor(k).float()
        wq = qf[base + ".weight"].get_tensor(base + ".weight")
        sc = qf[base + ".scales"].get_tensor(base + ".scales")
        bi = qf[base + ".biases"].get_tensor(base + ".biases")
        if idx is not None:
            wq, sc, bi = wq[idx], sc[idx], bi[idx]
        Wq = deq(wq, sc, bi, bits)
        if Wq.shape != W.shape:
            rows.append({"name": k, "note": f"shape {list(Wq.shape)} vs {list(W.shape)}"}); continue
        e_rel = rel(Wq, W)
        e_mm = rel(minmax_requant(W, bits), W)
        rows.append({"name": k, "kind": re.sub(r"\.\d+\.", ".N.", k), "bits": bits, "rel_err": e_rel, "minmax_rel_err": e_mm,
                     "gain_vs_minmax": e_mm / max(e_rel, 1e-12), "w_absmax": W.abs().max().item(),
                     "w_kurtosis": (((W - W.mean()) / W.std()) ** 4).mean().item()})
        if len(rows) % 50 == 0:
            print(len(rows), k, f"{e_rel:.4f} vs minmax {e_mm:.4f}", flush=True)
    agg = {}
    for r in rows:
        if "rel_err" in r:
            agg.setdefault(r["kind"], []).append(r)
    summary = {k: {"n": len(v), "mean_rel_err": sum(x["rel_err"] for x in v) / len(v),
                   "mean_minmax_rel_err": sum(x["minmax_rel_err"] for x in v) / len(v),
                   "max_rel_err": max(x["rel_err"] for x in v)} for k, v in sorted(agg.items())}
    json.dump({"summary_by_kind": summary, "rows": rows}, open(a.out, "w"))
    for k, v in summary.items():
        print(f"{v['mean_rel_err']:.4f} (minmax {v['mean_minmax_rel_err']:.4f}, max {v['max_rel_err']:.4f}) n={v['n']:4} {k}")


if __name__ == "__main__":
    main()
