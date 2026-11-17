#!/usr/bin/env python3
"""Per-token embedding / lm_head drift between a fine-tuned model A and its parent B.

usage: token_drift.py --a A.nonexp.safetensors --b B.nonexp.safetensors --tokenizer tokenizer.json --out out.json
Tokens whose rows moved most were (most) present in A's continued-training data, so the drift by token
category is a coarse fingerprint of the data mix; special tokens show which chat/tool formats were trained.
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict

import torch
from safetensors import safe_open

torch.set_num_threads(8)

bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
DEC = {chr(c): b for b, c in zip(bs, cs)}


def text(s):
    try:
        return bytes(DEC[c] for c in s).decode("utf-8")
    except Exception:
        return None


CODE_KW = re.compile(r"^\s*(def|class|return|import|self|lambda|async|await|const|function|var|let|public|private|static|void|"
                     r"include|struct|nullptr|println|printf|torch|numpy|np|tensor|cuda|kernel|grad|optim|loss|batch|"
                     r"dtype|config|json|http|api|args|kwargs|init|forward|module)\b")


def cat(t):
    if t is None:
        return "partial-bytes"
    core = t.strip()
    if t and not core:
        return "whitespace"
    if re.fullmatch(r"[0-9]+", core):
        return "digits"
    if CODE_KW.match(core) or re.search(r"[_{}();=<>\[\]]|::|->|\.\w", core) and core.isascii():
        return "code-ish"
    if core.isascii():
        return "ascii-word" if core.isalpha() else "ascii-punct"
    for ch in core:
        name = unicodedata.name(ch, "")
        for key, lab in (("CJK", "CJK"), ("HIRAGANA", "kana"), ("KATAKANA", "kana"), ("HANGUL", "korean"),
                         ("CYRILLIC", "cyrillic"), ("ARABIC", "arabic"), ("DEVANAGARI", "devanagari"), ("THAI", "thai"),
                         ("GREEK", "greek"), ("HEBREW", "hebrew"), ("LATIN", "latin-ext"), ("MATHEMATICAL", "math-symbol")):
            if key in name:
                return lab
    return "other"


def rows(path, key_re):
    f = safe_open(path, framework="pt")
    for k in f.keys():
        if re.search(key_re, k):
            return k, f.get_tensor(k).float()
    raise KeyError(key_re)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--tokenizer"); ap.add_argument("--out")
    a = ap.parse_args()
    tok = json.load(open(a.tokenizer))
    vocab = tok["model"]["vocab"]
    id2 = {i: s for s, i in vocab.items()}
    for t in tok.get("added_tokens", []):
        id2[t["id"]] = t["content"]
    special = {t["id"] for t in tok.get("added_tokens", [])}
    out = {"a": a.a, "b": a.b}
    for name, key in (("embedding", r"embed_tokens\.weight$"), ("lm_head", r"^lm_head\.weight$")):
        ka, A = rows(a.a, key)
        kb, B = rows(a.b, key)
        V = min(A.shape[0], B.shape[0])
        d = (A[:V] - B[:V]).norm(dim=1) / (B[:V].norm(dim=1) + 1e-8)
        med = d.median().item()
        by = defaultdict(list)
        for i in range(V):
            if i in special:
                continue
            by[cat(text(id2.get(i, "")) if i in id2 else None)].append(d[i].item())
        cat_stats = {c: {"n": len(v), "mean": sum(v) / len(v), "frac_above_2x_median": sum(x > 2 * med for x in v) / len(v)}
                     for c, v in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}
        top = torch.topk(d, 80).indices.tolist()
        bot = torch.topk(-d, 40).indices.tolist()
        out[name] = {"shape_a": list(A.shape), "shape_b": list(B.shape), "median": med, "mean": d.mean().item(),
                     "by_category": cat_stats,
                     "top_drift": [(id2.get(i), text(id2.get(i, "")) if i not in special else id2.get(i), round(d[i].item(), 4)) for i in top],
                     "least_drift": [(id2.get(i), round(d[i].item(), 5)) for i in bot],
                     "special": {id2[i]: round(d[i].item(), 4) for i in sorted(special) if i < V and i in id2}}
        print(f"== {name}: median {med:.4f}  mean {d.mean().item():.4f}")
        for c, s in list(cat_stats.items()):
            print(f"   {c:14} n={s['n']:6}  mean {s['mean']:.4f}  >2x median {s['frac_above_2x_median']:.1%}")
        print("   top drift:", [t[1] for t in out[name]["top_drift"][:40]])
        sp = sorted(out[name]["special"].items(), key=lambda kv: -kv[1])
        print("   special (most drift):", sp[:15])
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
