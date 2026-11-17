#!/usr/bin/env python3
"""Per-tensor weight differences between two checkpoints (forensics of continued training).

usage:
  weight_diff.py --a A1.safetensors[,A2...] --b B1.safetensors[,B2...] --mode {same,naive_mimo} --out out.json
Handles BF16/F32 and block-FP8 (weight_scale_inv, 128x128 blocks). In naive_mimo mode, Naive's split
q/k/v_proj are compared against row slices of MiMo's fused (FP8) qkv_proj.
Reports, per tensor: rel_diff = ||A-B||_F / ||B||_F, cosine, and aggregates by module kind and layer.
"""
import argparse
import json
import re
from collections import defaultdict

import torch
from safetensors import safe_open

torch.set_num_threads(8)
BLK = 128


class Ckpt:
    def __init__(self, paths):
        self.files = [safe_open(p, framework="pt") for p in paths]
        self.where = {}
        for f in self.files:
            for k in f.keys():
                self.where[k] = f

    def has(self, k):
        return k in self.where

    def raw(self, k):
        return self.where[k].get_tensor(k)

    def get(self, k):
        t = self.raw(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            s = self.raw(k + "_scale_inv").float()
            r, c = t.shape
            s = s.repeat_interleave(BLK, 0)[:r].repeat_interleave(BLK, 1)[:, :c]
            return t.float() * s
        return t.float()

    def names(self):
        return [k for k in self.where if not k.endswith("_scale_inv")]


def kind(k):
    k2 = re.sub(r"\.\d+\.", ".N.", k)
    k2 = re.sub(r"^model\.(language_model\.)?", "", k2)
    return k2


def layer(k):
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--mode", default="same", choices=["same", "naive_mimo"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = Ckpt(a.a.split(",")), Ckpt(a.b.split(","))
    rows = []
    for k in sorted(A.names()):
        if a.mode == "naive_mimo" and re.search(r"self_attn\.(q|k|v)_proj\.weight$", k):
            fused = re.sub(r"(q|k|v)_proj\.weight$", "qkv_proj.weight", k)
            if not B.has(fused):
                continue
            L = layer(k)
            q, kk, v = (A.raw(re.sub(r"(q|k|v)_proj", x, k)).shape[0] for x in ("q_proj", "k_proj", "v_proj"))
            W = B.get(fused)
            sl = {"q_proj": W[:q], "k_proj": W[q:q + kk], "v_proj": W[q + kk:q + kk + v]}[re.search(r"(q|k|v)_proj", k).group(0)]
            tb = sl
        elif B.has(k):
            tb = B.get(k)
        else:
            continue
        ta = A.get(k)
        if ta.shape != tb.shape:
            rows.append({"name": k, "kind": kind(k), "layer": layer(k), "shape_a": list(ta.shape), "shape_b": list(tb.shape), "note": "shape mismatch"})
            continue
        d = (ta - tb)
        nb = tb.norm().item()
        rel = d.norm().item() / (nb + 1e-12)
        cos = torch.nn.functional.cosine_similarity(ta.flatten(), tb.flatten(), dim=0).item()
        row = {"name": k, "kind": kind(k), "layer": layer(k), "numel": ta.numel(), "rel_diff": rel, "cos": cos,
               "frac_exact_equal": (d == 0).float().mean().item()}
        if ta.dim() == 3:  # stacked experts [E, out, in] (Qwen3.x fused experts)
            per = (d.flatten(1).norm(dim=1) / (tb.flatten(1).norm(dim=1) + 1e-12)).tolist()
            row["per_expert_rel"] = per
        rows.append(row)
    agg = defaultdict(list)
    for r in rows:
        if "rel_diff" in r:
            agg[r["kind"]].append(r["rel_diff"])
    summary = {k: {"n": len(v), "mean_rel": sum(v) / len(v), "max_rel": max(v), "min_rel": min(v)} for k, v in sorted(agg.items())}
    only_a = sorted(set(A.names()) - set(B.names()))
    json.dump({"a": a.a, "b": a.b, "mode": a.mode, "summary_by_kind": summary, "rows": rows,
               "only_in_a": only_a[:200], "n_only_in_a": len(only_a)}, open(a.out, "w"))
    for k, v in summary.items():
        print(f"{v['mean_rel']:9.5f} (min {v['min_rel']:.5f} max {v['max_rel']:.5f}) n={v['n']:4}  {k}")
    print("only in A:", len(only_a), only_a[:8])


if __name__ == "__main__":
    main()
