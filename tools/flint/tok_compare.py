"""Compare Flint's tokenizer with reference tokenizers: pre-tokenizer rules, vocab overlap, merge-order overlap."""
import json, sys, os
FL = sys.argv[1]; REFS = sys.argv[2:]
def load(p):
    t = json.load(open(p)); m = t["model"]
    vocab = m.get("vocab", {}); merges = m.get("merges", [])
    merges = [tuple(x.split(" ")) if isinstance(x, str) else tuple(x) for x in merges]
    pre = t.get("pre_tokenizer") or {}
    pats = []
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "Split": pats.append(n["pattern"].get("Regex") or n["pattern"].get("String"))
            if n.get("type") == "ByteLevel": pats.append(f"ByteLevel(add_prefix_space={n.get('add_prefix_space')},use_regex={n.get('use_regex')})")
            for v in n.values(): walk(v)
        elif isinstance(n, list):
            for v in n: walk(v)
    walk(pre)
    return {"vocab": vocab, "merges": merges, "pats": pats, "n_added": len(t.get("added_tokens", []))}
f = load(FL)
print(f"FLINT vocab={len(f['vocab'])} merges={len(f['merges'])} added={f['n_added']}")
for i, p in enumerate(f["pats"]): print(f"  pre[{i}] {p[:160]}")
fv = set(f["vocab"]); fm = f["merges"]
for r in REFS:
    g = load(r); name = os.path.basename(r).replace(".tokenizer.json", "")
    gv = set(g["vocab"]); inter = len(fv & gv)
    same_pre = [p for p in f["pats"] if p in g["pats"]]
    k = 1000; fk = set(fm[:k]); gk = set(g["merges"][:k])
    k2 = 20000; fk2 = set(fm[:k2]); gk2 = set(g["merges"][:k2])
    # rank agreement of shared merges (first 5000 of flint)
    gr = {m: i for i, m in enumerate(g["merges"])}
    shared_rank = [(i, gr[m]) for i, m in enumerate(fm[:5000]) if m in gr]
    same_pos = sum(1 for a, b in shared_rank if a == b)
    print(f"\n{name}: vocab={len(gv)} merges={len(g['merges'])}")
    for i, p in enumerate(g["pats"]): print(f"  pre[{i}] {p[:160]}")
    print(f"  identical pre-tokenizer rules with Flint: {len(same_pre)} -> {[p[:40] for p in same_pre]}")
    print(f"  vocab overlap: |F∩G|={inter}  = {inter/len(fv):.1%} of Flint, {inter/len(gv):.1%} of ref, Jaccard {inter/len(fv|gv):.3f}")
    print(f"  merges overlap: top1k {len(fk&gk)/k:.1%}   top20k {len(fk2&gk2)/k2:.1%}   identical rank in Flint's first 5000: {same_pos}")
