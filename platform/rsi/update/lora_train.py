"""U_w: PEFT LoRA SFT of Qwen/Qwen3.5-9B (BF16) on one A100, adapter loadable by vLLM --enable-lora.

Called by bundle/lora_train.sh <sft_dir> <adapter_out> <gpu>. Input: train.jsonl / valid.jsonl from
rsi.update.sft (messages format, optional "tools"). Loss is on the final assistant message only.

Module names (verified 2026-11-17 against huggingface.co/Qwen/Qwen3.5-9B config.json and
model.safetensors.index.json, and vLLM main vllm/model_executor/models/qwen3_5.py):
  architecture Qwen3_5ForConditionalGeneration, model_type qwen3_5, 32 text layers
  (3 x linear_attention [Gated DeltaNet] + 1 x full_attention, repeating; full attention at 3,7,...,31)
  full attention : model.language_model.layers.N.self_attn.{q,k,v,o}_proj        (8 layers)
  linear attn    : model.language_model.layers.N.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}
  MLP (all 32)   : model.language_model.layers.N.mlp.{gate,up,down}_proj
  vision tower   : model.visual.blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2}  (no name clash)
  MTP head       : mtp.layers.0.self_attn.*_proj, mtp.layers.0.mlp.*_proj  (dropped by vLLM: "mtp." -> None)
Default targets = q,k,v,o (full-attention layers) + gate,up,down (every layer), language model only.
vLLM packs q/k/v -> qkv_proj and gate/up -> gate_up_proj and supports LoRA on those. The GDN projections
(packed in vLLM as in_proj_qkvz / in_proj_ba) are left out by default: LoRA support for those packed
shards is newer and unverified here; add them with --targets if wanted after a vLLM load test.
Saved tensor keys follow the HF checkpoint naming (base_model.model.model.language_model.layers...),
which vLLM's Qwen3-VL hf_to_vllm_mapper maps onto its own modules. If the text-only class was loaded
(keys base_model.model.model.layers...), keys are renamed to the checkpoint naming before saving.
vLLM needs --max-lora-rank >= --rank (default 16).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
EXCLUDE_PARTS = ("visual", "vision", "mtp", "lm_head", "embed")


# ---------------------------------------------------------------------------- data (CPU-testable)
def normalise_messages(messages: list) -> list:
    """OpenAI -> HF chat-template conventions: tool-call arguments as dicts, no None contents."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"_raw": args}
                tcs.append(tc if "function" in tc else {"type": "function", "function": fn})
            m["tool_calls"] = tcs
        out.append(m)
    return out


def encode_example(tok, ex: dict, max_len: int) -> dict | None:
    """Token ids + labels (-100 everywhere except the final assistant message). None if too long / empty."""
    msgs = normalise_messages(ex["messages"])
    tools = ex.get("tools") or None
    kw = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False, **kw)
    full = tok.apply_chat_template(msgs, tokenize=False, **kw)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    lcp = 0
    for a, b in zip(p_ids, f_ids):
        if a != b:
            break
        lcp += 1
    # trailing newline after the end-of-turn token is not part of the answer
    end = len(f_ids)
    while end > lcp and tok.decode(f_ids[end - 1:end]).strip() == "":
        end -= 1
    if end <= lcp or len(f_ids) > max_len:
        return None
    labels = [-100] * lcp + f_ids[lcp:end] + [-100] * (len(f_ids) - end)
    return {"input_ids": f_ids, "labels": labels, "n_target": end - lcp}


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------------------- model helpers
def select_targets(model, short: list[str]) -> list[str]:
    import torch.nn as nn

    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in short:
            continue
        if any(p in name.split(".") or p in name for p in EXCLUDE_PARTS):
            continue
        names.append(name)
    if not names:
        raise SystemExit(f"no target modules matched {short}")
    return names


INPUT_DEVICE = "cuda"   # set to the embedding layer's device when the model is split over GPUs (--device-map)


def load_model(model_id: str, torch, device_map: str | None = None):
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(model_id)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    candidates = [getattr(transformers, arch, None), getattr(transformers, "AutoModelForImageTextToText", None),
                  transformers.AutoModelForCausalLM]
    last = None
    for cls in [c for c in candidates if c is not None]:
        for dkw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
            if device_map:  # e.g. "auto": layers split over all visible GPUs (27B on 2 x A100)
                dkw = {**dkw, "device_map": device_map}
            try:
                m = cls.from_pretrained(model_id, attn_implementation="sdpa", **dkw)
                print(f"[lora] loaded {model_id} as {type(m).__name__}", flush=True)
                return m
            except TypeError as e:
                last = e
            except (ValueError, KeyError) as e:
                last = e
                break
    raise SystemExit(f"could not load {model_id}: {last}")


def _inner_and_head(model):
    """(callable returning hidden states, lm_head) of the HF model under the PEFT wrapper, or (None, None)."""
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(hf, "model", None)
    head = getattr(hf, "lm_head", None)
    if inner is None or head is None:
        return None, None
    return inner, head


def example_loss(model, batch, torch):
    ids = torch.tensor([batch["input_ids"]], device=INPUT_DEVICE)
    labels = torch.tensor([batch["labels"]], device=INPUT_DEVICE)
    inner, head = _inner_and_head(model)
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if inner is not None:
        try:
            out = inner(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            h = out[0] if isinstance(out, tuple) else out.last_hidden_state
            hs = h[:, :-1][mask.to(h.device)]
            logits = head(hs).float()
            return torch.nn.functional.cross_entropy(logits, shift_labels[mask].to(logits.device))
        except (TypeError, AttributeError):
            pass
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    return out.loss


def fix_saved_adapter(out_dir: str, short_targets: list[str], base_model: str) -> dict:
    """Make the saved adapter look like a standard HF-naming PEFT adapter for vLLM."""
    from safetensors.torch import load_file, save_file

    p = os.path.join(out_dir, "adapter_model.safetensors")
    sd = load_file(p)
    renamed = 0
    new = {}
    for k, v in sd.items():
        k2 = k
        if k.startswith("base_model.model.model.layers."):
            k2 = "base_model.model.model.language_model.layers." + k[len("base_model.model.model.layers."):]
            renamed += 1
        new[k2] = v
    if renamed:
        save_file(new, p, metadata={"format": "pt"})
    cp = os.path.join(out_dir, "adapter_config.json")
    with open(cp) as f:
        cfg = json.load(f)
    cfg["target_modules"] = sorted(short_targets)
    cfg["base_model_name_or_path"] = base_model
    with open(cp, "w") as f:
        json.dump(cfg, f, indent=2)
    mods = sorted({k.split(".lora_")[0].split(".layers.")[-1].split(".", 1)[-1] for k in new})
    return {"n_tensors": len(new), "renamed": renamed, "module_kinds": mods,
            "example_key": sorted(new)[0] if new else None}


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("RSI_BASE_MODEL", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--time-limit-min", type=float, default=55.0, help="stop, evaluate and save at this limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="tokenise and report only (no GPU)")
    ap.add_argument("--device-map", default=os.environ.get("RSI_LORA_DEVICE_MAP") or None,
                    help="e.g. 'auto' to split the base model over all visible GPUs (default: one GPU)")
    a = ap.parse_args(argv)
    t_start = time.time()

    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    data = {}
    for split in ("train", "valid"):
        raw = load_jsonl(os.path.join(a.sft_dir, f"{split}.jsonl"))
        enc = [encode_example(tok, ex, a.max_len) for ex in raw]
        data[split] = [e for e in enc if e is not None]
        print(f"[lora] {split}: {len(raw)} examples, {len(raw) - len(data[split])} dropped (too long / empty), "
              f"{sum(len(e['input_ids']) for e in data[split])} tokens, "
              f"{sum(e['n_target'] for e in data[split])} target tokens", flush=True)
    if not data["train"]:
        raise SystemExit("no training examples")
    if a.dry_run:
        return 0

    import torch
    from peft import LoraConfig, get_peft_model

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    global INPUT_DEVICE
    model = load_model(a.model, torch, device_map=a.device_map)
    if a.device_map:
        INPUT_DEVICE = model.get_input_embeddings().weight.device
        print(f"[lora] device_map={a.device_map}: {sorted({str(p.device) for p in model.parameters()})}", flush=True)
    else:
        model = model.to("cuda")
    model.config.use_cache = False
    short = [s.strip() for s in a.targets.split(",") if s.strip()]
    targets = select_targets(model, short)
    print(f"[lora] {len(targets)} target modules, e.g. {targets[0]} ... {targets[-1]}", flush=True)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.lora_dropout, target_modules=targets,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    n_ex = len(data["train"])
    total_steps = max(1, math.ceil(a.epochs * n_ex / a.grad_accum))
    warm = max(1, int(a.warmup * total_steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))

    def valid_loss():
        if not data["valid"]:
            return None
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for e in data["valid"]:
                tot += float(example_loss(model, e, torch)) * e["n_target"]
                n += e["n_target"]
        model.train()
        return tot / max(1, n)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    model.train()
    step, seen, best_v, stopped = 0, 0, None, "epochs"
    v0 = valid_loss()
    log.write(json.dumps({"step": 0, "valid_loss": v0}) + "\n")
    print(f"[lora] steps={total_steps} valid_loss@0={v0}", flush=True)
    order = []
    while step < total_steps:
        if not order:
            order = list(range(n_ex))
            random.shuffle(order)
        acc = 0.0
        for _ in range(a.grad_accum):
            if not order:
                order = list(range(n_ex))
                random.shuffle(order)
            e = data["train"][order.pop()]
            loss = example_loss(model, e, torch) / a.grad_accum
            loss.backward()
            acc += float(loss)
            seen += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": acc, "lr": lr_at(step - 1), "elapsed_min": (time.time() - t_start) / 60}
        epoch_edge = (seen // n_ex) != ((seen - a.grad_accum) // n_ex)
        timeout = (time.time() - t_start) / 60 > a.time_limit_min
        if epoch_edge or step == total_steps or timeout:
            v = valid_loss()
            rec["valid_loss"] = v
            if best_v is None or v is None or v < best_v:   # keep the best-by-valid checkpoint (train tasks only)
                best_v = v
                model.save_pretrained(a.out)
                rec["saved"] = True
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or "valid_loss" in rec:
            print(f"[lora] {json.dumps(rec)}", flush=True)
        if timeout:
            stopped = "time_limit"
            break
    log.close()
    info = fix_saved_adapter(a.out, short, a.model)
    man = {}
    try:
        with open(os.path.join(a.sft_dir, "manifest.json")) as f:
            man = json.load(f)
    except OSError:
        pass
    meta = {"base_model": a.model, "rank": a.rank, "alpha": a.alpha, "targets": short, "lr": a.lr,
            "epochs": a.epochs, "grad_accum": a.grad_accum, "max_len": a.max_len, "steps": step,
            "stopped": stopped, "n_examples": n_ex, "n_valid": len(data["valid"]), "valid_loss_0": v0,
            "valid_loss_best": best_v, "minutes": round((time.time() - t_start) / 60, 2),
            "sft_sha256": man.get("sha256"), "sft_mode": man.get("mode"), "adapter": info}
    with open(os.path.join(a.out, "rsi_adapter.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)
    print(os.path.abspath(a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
