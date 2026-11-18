#!/usr/bin/env python3
"""Compare reasoning-style census across models (Flint, Apex, Qwen3.6, Naive) with an extended opener taxonomy.

Naive (2026-11-18): read from research/naive/naive_census_all.jsonl (set == "census"; transformers FP8 greedy on H20).
Extra per-domain fields (added for all models; the original keys n / classes / selfref / median_reasoning_chars are unchanged):
  median_content_chars, empty_content (answer empty, truncated records excluded when the file has `finish`),
  n_truncated (finish == "length"; None when the file has no `finish` field), cjk_share_reasoning (CJK chars / non-space
  chars over all reasoning text), top_openers (first two whitespace tokens of the reasoning, 5 most common).
"""
import json, re, sys, collections
CLASSES = [
    ("heres_thinking_process", r"^\s*Here'?s a (thinking|thought) process"),
    ("gemini_numbered", r"^\s*(1\.\s+\*\*|\*\*Analy[sz]|\*\*Understand|\*\*Identify|Let's break down)"),
    ("telegraphic", r"^\s*(We need|We have|Need to|We should|We must|We are asked|我们需要|我们要|需要)"),
    ("user_asks", r"^\s*(The user (asks|wants|is asking|requests|said|has)|User asks|用户)"),
    ("okay_so", r"^\s*(Okay|OK|Alright|Hmm|Well)[,.! ]"),
    ("simple_task", r"^\s*(Simple|Straightforward|Easy|Quick) (task|question|request)"),
    ("first_person_plan", r"^\s*(I need to|I'll|I will|I should|Let me|I'm going to)"),
    ("empty", r"^\s*$"),
]
CLASSES = [(n, re.compile(p, re.S)) for n, p in CLASSES]
SELF = re.compile(r"\b(as a (Gemini|Google|Claude|GPT|large language model|Qwen)|I am (Gemini|Claude|ChatGPT|a Google|Qwen)|Google AI|Gemini model|Anthropic|OpenAI|Alibaba|Qwen|通义|阿里)\b", re.I)
def cls(r):
    for n, p in CLASSES:
        if p.match(r or ""):
            return n
    return "other"
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
def load(m):
    if m == "naive":
        return [r for r in map(json.loads, open("research/naive/naive_census_all.jsonl")) if r.get("set") == "census"]
    return [json.loads(l) for l in open(f"research/{m}_census.jsonl")]
def trunc(r):
    return r.get("finish") == "length"
models = sys.argv[1:] or ["flint", "apex", "qwen36", "naive"]
rows = {m: load(m) for m in models}
domains = sorted({r["domain"] for m in rows for r in rows[m]})
out = {}
for m in models:
    out[m] = {}
    for d in domains + ["ALL"]:
        rr = [r for r in rows[m] if d == "ALL" or r["domain"] == d]
        c = collections.Counter(cls(r.get("reasoning")) for r in rr)
        selfref = collections.Counter(x.group(0) for r in rr for x in SELF.finditer((r.get("reasoning") or "") + "\n" + (r.get("content") or "")))
        rl = sorted(r.get("reasoning_chars") or 0 for r in rr)
        has_fin = any("finish" in r for r in rr)
        cl = sorted(r.get("content_chars") if r.get("content_chars") is not None else len(r.get("content") or "") for r in rr)
        txt = "".join(r.get("reasoning") or "" for r in rr)
        nonspace = len(re.sub(r"\s", "", txt))
        out[m][d] = {"n": len(rr), "classes": dict(c.most_common()), "selfref": dict(selfref.most_common()),
                     "median_reasoning_chars": rl[len(rl) // 2] if rl else 0,
                     "median_content_chars": cl[len(cl) // 2] if cl else 0,
                     "empty_content": sum(1 for r in rr if not (r.get("content") or "").strip() and not trunc(r)),
                     "n_truncated": sum(1 for r in rr if trunc(r)) if has_fin else None,
                     "cjk_share_reasoning": round(len(CJK.findall(txt)) / nonspace, 3) if nonspace else 0,
                     "top_openers": dict(collections.Counter(" ".join((r.get("reasoning") or "").split()[:2]) or "<empty>"
                                                             for r in rr).most_common(5))}
json.dump(out, open("research/census_compare.json", "w"), ensure_ascii=False, indent=1)
for d in domains + ["ALL"]:
    print(f"\n## {d}")
    for m in models:
        o = out[m][d]
        print(f"  {m:7} n={o['n']:3} med_reason={o['median_reasoning_chars']:6} med_ans={o['median_content_chars']:6} "
              f"empty={o['empty_content']} trunc={o['n_truncated']} cjk={o['cjk_share_reasoning']}  {o['classes']}  self={o['selfref']}")
