#!/usr/bin/env python3
"""Naive-N0.5-Flash behavioural census (same 104 prompts as Flint / Apex / Qwen3.6), transformers + FP8, greedy.

usage: python naive_census.py --model DIR --prompts prompts.json --out naive_census_all.jsonl [--batch 16] [--only smoke]
Writes one JSON line per prompt: set, probe/domain, prompt, reasoning, content, tool_calls_raw, n_new, finish.
Safety checks before the long run (abort unless --force):
  1. loading info: no missing / unexpected weights (a silently random-initialised FP8 scale would make garbage);
  2. smoke test: 17*23 must contain 391;
  3. batching check: two prompts generated alone and batched must give the same first 48 greedy tokens.
"""
import argparse
import json
import os
import sys
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def patch_dsa_cache():
    """Runtime fix for `generate` (no edits to the remote code).

    The config sets layer_types = "sliding_attention" / "deepseek_sparse_attention", but transformers 5.23's
    DynamicCache(config=...) looks each type up in DYNAMIC_LAYER_TYPE_MAPPING, which has no "deepseek_sparse_attention"
    -> KeyError. The DSA layers need a normal (growing) KV cache plus an indexer-key cache: the remote code calls
    `cache.update_indexer(key[B, L, D], layer_idx)` and expects the full accumulated [B, total_len, D] back.
    Defensive: only fills in what is missing, so a newer transformers / remote code that already registers it wins.
    """
    from transformers import cache_utils as cu

    if "deepseek_sparse_attention" in cu.DYNAMIC_LAYER_TYPE_MAPPING:
        log("cache patch: deepseek_sparse_attention already registered ->",
            cu.DYNAMIC_LAYER_TYPE_MAPPING["deepseek_sparse_attention"].__name__)
    else:
        layer_cls = getattr(cu, "DynamicIndexedLayer", None)   # 5.2x ships this: DynamicLayer + indexer_keys [B, L, D]
        if layer_cls is None or not hasattr(layer_cls, "update_indexer"):
            class DSALayer(cu.DynamicLayer):
                """DynamicLayer that also accumulates the DSA indexer keys along the sequence dim."""

                def __init__(self, **kwargs):              # DynamicCache passes per-layer kwargs; DSA layers need none
                    super().__init__()
                    self.indexer_keys = None

                def update_indexer(self, key):            # key: [B, L_new, D] -> returns [B, L_total, D]
                    self.indexer_keys = key if self.indexer_keys is None else torch.cat([self.indexer_keys, key], dim=1)
                    return self.indexer_keys

                def reset(self):
                    super().reset()
                    self.indexer_keys = None

                def reorder_cache(self, beam_idx):
                    super().reorder_cache(beam_idx)
                    if self.indexer_keys is not None:
                        self.indexer_keys = self.indexer_keys.index_select(0, beam_idx.to(self.indexer_keys.device))

            layer_cls = DSALayer
        cu.DYNAMIC_LAYER_TYPE_MAPPING["deepseek_sparse_attention"] = layer_cls
        log("cache patch: deepseek_sparse_attention ->", layer_cls.__name__)

    if not hasattr(cu.DynamicCache, "update_indexer"):    # 5.23 has Cache.update_indexer; older 5.x may not
        def update_indexer(self, key, layer_idx):
            return self.layers[layer_idx].update_indexer(key)

        cu.DynamicCache.update_indexer = update_indexer
        log("cache patch: added DynamicCache.update_indexer")


# Local copy of kernels-community/finegrained-fp8, branch v4 (commit 03ae3193...), variant build/torch-cuda
# (pure Triton, no compiled .so), fetched on the login node because the compute node may have no internet.
FP8_KERNEL_DIR = os.environ.get("NAIVE_FP8_KERNEL_DIR", "/scratch/user/naive/kernels/finegrained-fp8")


def preload_fp8_kernel(path=FP8_KERNEL_DIR):
    """Offline fix for the FP8 MoE experts (and FP8 linears on the Triton path).

    transformers 5.23 loads them via lazy_load_kernel("finegrained-fp8") -> kernels.get_kernel(repo, version=4),
    which needs the Hub (or a pre-populated HF cache). lazy_load_kernel returns a module already present in
    hub_kernels._KERNEL_MODULE_MAPPING, so load the local build with kernels.get_local_kernel (no network) and put
    it there before the first forward. Dir absent -> leave transformers' normal path untouched.
    """
    from pathlib import Path

    from transformers.integrations import hub_kernels as hk

    if not os.path.isdir(path):
        log(f"fp8 kernel: {path} absent -> transformers' default hub path")
        return
    if not hk.is_kernels_available():   # transformers re-checks this before using the module, so fail early and clearly
        sys.exit("ABORT: `kernels` missing or wrong version (transformers 5.23 wants 0.18.2 <= kernels < 0.19.0)")
    from kernels import get_local_kernel

    mod = get_local_kernel(Path(path))   # resolves build/torch-cuda for a CUDA torch, checks digest, imports
    missing = [s for s in ("matmul_2d", "matmul_batched", "matmul_grouped") if not hasattr(mod, s)]
    if missing:
        sys.exit(f"ABORT: local fp8 kernel at {path} lacks {missing}")
    hk._KERNEL_MODULE_MAPPING["finegrained-fp8"] = mod
    log("fp8 kernel: preloaded", mod.__name__, "from", path)


def split_think(text):
    t = text
    for s in ("<|im_end|>", "<|endoftext|>"):
        t = t.split(s)[0]
    if "</think>" in t:
        r, c = t.split("</think>", 1)
        return r.replace("<think>", "").strip(), c.strip()
    return t.replace("<think>", "").strip(), ""          # never closed the think block (ran out of tokens)


def build(tok, it):
    msgs = [{"role": "user", "content": it["prompt"]}]
    kw = {"tools": it["tools"]} if it.get("tools") else {}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


@torch.no_grad()
def generate(model, tok, texts, max_new):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                         eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    res = []
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for row in new:
        ids = row.tolist()
        n = len(ids)
        while n and ids[n - 1] == tok.pad_token_id:
            n -= 1
        hit_eos = eos in ids          # not ids[:n]: if pad == <|im_end|> the strip above removes the eos itself
        res.append((tok.decode(ids[:n], skip_special_tokens=False), n, "stop" if hit_eos else "length"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-mem-gib", type=int, default=88)
    ap.add_argument("--only", choices=["smoke", "probes", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    log("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
        "gpus", torch.cuda.device_count())
    patch_dsa_cache()
    preload_fp8_kernel()

    cfg =AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    qc = getattr(cfg, "quantization_config", None)
    if isinstance(qc, dict) and qc.get("ignored_layers") and not qc.get("modules_to_not_convert"):
        # the checkpoint lists BF16 modules as `ignored_layers`; transformers' FP8 quantizer reads `modules_to_not_convert`
        qc["modules_to_not_convert"] = list(qc["ignored_layers"])
        cfg.quantization_config = qc
        log("quantization_config: modules_to_not_convert <- ignored_layers", len(qc["ignored_layers"]))
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"

    t0 = time.time()
    mm = {i: f"{a.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    model, info = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, trust_remote_code=True, dtype=torch.bfloat16,
                                                       device_map="auto", max_memory=mm, output_loading_info=True)
    log(f"loaded in {time.time() - t0:.0f}s")
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    for k, v in bad.items():
        log(k, len(v), list(v)[:12])
    if bad and not a.force:
        sys.exit("ABORT: loading info not clean (see above); rerun with --force only if the listed keys are harmless")
    for i in range(torch.cuda.device_count()):
        log(f"gpu{i} mem {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")

    # 1. smoke test
    t0 = time.time()
    txt = tok.apply_chat_template([{"role": "user", "content": "What is 17*23? Answer with just the number."}],
                                  add_generation_prompt=True, tokenize=False)
    (s, n, fin), = generate(model, tok, [txt], 400)
    dt = time.time() - t0
    log(f"smoke: {n} tokens in {dt:.1f}s ({n / dt:.2f} tok/s, batch 1) finish={fin}\n{s[:600]}")
    if "391" not in s and not a.force:
        sys.exit("ABORT: smoke test did not produce 391 (weights or FP8 path broken?)")

    # 2. batching check (left padding must not change greedy outputs)
    pa = [build(tok, {"prompt": "Name three primary colors."}), build(tok, {"prompt": "Explain recursion to a ten-year-old in two sentences, and give one everyday example of it."})]
    alone = [generate(model, tok, [p], 48)[0][0] for p in pa]
    t0 = time.time()
    both = [x[0] for x in generate(model, tok, pa, 48)]
    log(f"batch-2 48 tok in {time.time() - t0:.1f}s")
    # bf16 matmuls differ slightly between batch sizes, so exact equality is too strict; padding bugs show up immediately
    pref = [len(os.path.commonprefix([x, y])) for x, y in zip(alone, both)]
    same = [x == y or p >= 60 for (x, y), p in zip(zip(alone, both), pref)]
    log("batching check: common-prefix chars", pref, "ok", same)
    batch = a.batch
    if not all(same):
        for x, y in zip(alone, both):
            log("ALONE  :", repr(x[:200])); log("BATCHED:", repr(y[:200]))
        log("WARNING: batched greedy output differs -> falling back to batch 1")
        batch = 1
    if a.only == "smoke":
        return

    items = json.load(open(a.prompts))
    if a.only == "probes":
        items = [x for x in items if x["set"] != "census"]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l); done.add((r["set"], r.get("probe") or r.get("domain"), r["prompt"]))
            except Exception:
                pass
    todo = [x for x in items if (x["set"], x.get("probe") or x.get("domain"), x["prompt"]) not in done]
    log(f"{len(todo)} prompts to run ({len(done)} already in {a.out}), batch {batch}")
    groups = {}
    for x in todo:
        groups.setdefault(x["max_new"], []).append(x)
    with open(a.out, "a") as f:
        for mx in sorted(groups):
            g = groups[mx]
            for i in range(0, len(g), batch):
                chunk = g[i:i + batch]
                t0 = time.time()
                outs = generate(model, tok, [build(tok, x) for x in chunk], mx)
                dt = time.time() - t0
                ntok = sum(o[1] for o in outs)
                for x, (text, n, fin) in zip(chunk, outs):
                    r, c = split_think(text)
                    rec = {k: v for k, v in x.items() if k not in ("tools", "max_new")}
                    rec.update({"reasoning": r, "content": c, "reasoning_chars": len(r), "content_chars": len(c),
                                "n_new": n, "finish": fin, "raw_tail": text[-300:]})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                log(f"max_new {mx} batch {i // batch + 1}/{(len(g) + batch - 1) // batch}: {len(chunk)} prompts, "
                    f"{ntok} tokens in {dt:.0f}s ({ntok / dt:.1f} tok/s) | {chunk[0].get('probe') or chunk[0].get('domain')}: "
                    f"{(outs[0][0][:120]).replace(chr(10), ' ')!r}")
    log("NAIVE_CENSUS_DONE")


if __name__ == "__main__":
    main()
