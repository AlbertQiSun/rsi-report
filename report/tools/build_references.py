"""Extract every citation from research/ literature files into report/references.md.
Text is copied verbatim from the research files (no rewriting), so nothing is invented."""
import re, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "research"
URL = re.compile(r"https?://[^\s|)）>\]，、`*]+")

def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]

def table_rows(path, start_pat, stop_pat=r"^## "):
    lines = path.read_text().splitlines()
    out, on = [], False
    for i, l in enumerate(lines, 1):
        if re.match(start_pat, l): on = True; continue
        if on and re.match(stop_pat, l): break
        if on and l.startswith("|") and not re.match(r"^\|\s*-", l):
            c = cells(l)
            if c and c[0] in ("#", "论文", "id"): continue
            out.append((i, c, l))
    return out

sections = []
# 01 theory: | # | 来源 | 类型 | 定义 | 主张 | URL |
rows = table_rows(R/"lit_rsi/01_theory.md", r"^## 1\. ")
sections.append(("T", "理论与定义", "research/lit_rsi/01_theory.md",
    [(f"T-{int(c[0]):02d}", c[1], c[2], URL.findall(l), n) for n, c, l in rows if c[0].isdigit()]))
# 02 methods: | 论文 | 发表/类型 | ... | URL |
rows = table_rows(R/"lit_rsi/02_methods.md", r"^## 1\. ")
sections.append(("M", "自我改进方法与实证", "research/lit_rsi/02_methods.md",
    [(f"M-{k:02d}", c[0], c[1], URL.findall(l), n) for k, (n, c, l) in enumerate(rows, 1)]))
# 03 measurement: | # | 来源 | 类型 | ... |
rows = table_rows(R/"lit_rsi/03_measurement.md", r"^## 1\. ")
sections.append(("E", "测量、评测与实验室阈值", "research/lit_rsi/03_measurement.md",
    [(f"E-{int(c[0]):02d}", c[1], c[2], URL.findall(l), n) for n, c, l in rows if c[0].isdigit()]))
# signal hardness: | id | 来源 | 类型 | URL | ... |
rows = table_rows(R/"lit_signal_hardness.md", r"^## 2\. ", r"^### ")
sections.append(("L", "信号硬度与评估器", "research/lit_signal_hardness.md",
    [(c[0], c[1], c[2], URL.findall(l), n) for n, c, l in rows if c[0].startswith("L-")]))


# 02 methods §3–§4: citations that live in bullets / the §4 table, not the §1 table
mlines = (R/"lit_rsi/02_methods.md").read_text().splitlines()
extra_m, on = [], False
for n, l in enumerate(mlines, 1):
    if re.match(r"^## [34]\. ", l): on = True
    elif re.match(r"^## ", l): on = False
    if not on: continue
    m = re.match(r"^- \*\*(.+?)\*\*\s*(?:[（(]([^）)]*)[）)])?", l)
    if m and re.search(r"(19|20)\d\d|et al|arXiv", (m.group(1) or "") + (m.group(2) or "")):
        extra_m.append((m.group(1), m.group(2) or "", URL.findall(l), n))
    elif l.startswith("|") and not re.match(r"^\|\s*-", l):
        c = cells(l)
        if c and c[0] not in ("论文", "研究", "来源") and re.search(r"\d{4}", c[0]):
            extra_m.append((c[0], "", URL.findall(l), n))
sections.append(("M2", "方法：局限与反面证据（正文中的引用）", "research/lit_rsi/02_methods.md §3–4",
    [(f"M2-{k:02d}", f"{a}（{b}）" if b else a, "见原文件", u, n) for k, (a, b, u, n) in enumerate(extra_m, 1)]))
seen = set()
md = ["# 参考文献总表（自动抽取）", "",
      "> 由 `report/tools/build_references.py` 从 `research/` 的文献文件**逐字抽取**，没有改写，也没有补充任何信息。重新运行脚本即可更新。",
      "> - 键：T = `lit_rsi/01_theory.md`，M = `02_methods.md`，E = `03_measurement.md`，L = `lit_signal_hardness.md`；RF/FL/NV/AX = 四份材料的证据台账。",
      "> - 「类型」列沿用原文件的标注（PR = 同行评审，Preprint，blog 等）。原文件里注明\"只读了摘要\"或\"未核实\"的，引用前要看原文件的说明。",
      "> - 同一篇文献可能出现在多个表里（例如 DGM 同时在 M、E、L 中），网页生成时再合并去重。", ""]
for key, title, src, items in sections:
    md += [f"## {key}. {title}（来源：`{src}`）", "", "| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |", "|---|---|---|---|---|"]
    for k, cite, typ, urls, n in items:
        seen.update(urls)
        u = "<br>".join(dict.fromkeys(urls)) or "（原文件未给 URL）"
        md.append(f"| {k} | {cite} | {typ} | {u} | {n} |")
    md.append("")

# four material ledgers
md += ["## 四份材料（证据台账）", "",
       "| 键前缀 | 材料 | 台账文件 | 已读的原始 URL |", "|---|---|---|---|"]
for pre, name, f in [("RF", "Recursive · First Steps Toward Automated AI Research", "recursive_first_steps.md"),
                     ("FL", "Recursive · Introducing Flint", "recursive_flint.md"),
                     ("NV", "NaiveAI · Naive-N0.5-Flash: Building Frontier AI with AI", "naive_research.md"),
                     ("AX", "Apex Intelligence · Introducing Apex-35B-V1.0", "apex_35b.md")]:
    t = (R/"sources"/f).read_text()
    head = t.split("## 数字台账")[0]
    urls = list(dict.fromkeys(URL.findall(head)))
    seen.update(URL.findall(t))
    md.append(f"| {pre}-xx | {name} | `research/sources/{f}` | " + "<br>".join(urls) + " |")
md.append("")

# every other URL appearing anywhere in research/*.md, with its context
md += ["## 其他出处（research/ 下其他文件中出现、上面各表未覆盖的 URL）", "",
       "> 逐条列出所在文件和行号，便于回查上下文。内容未经二次核实，引用前先看原文件。", "",
       "| 键 | URL | 首次出现 |", "|---|---|---|"]
extra = {}
for p in sorted(R.rglob("*.md")):
    for i, l in enumerate(p.read_text().splitlines(), 1):
        for u in URL.findall(l):
            if u not in seen and u not in extra:
                extra[u] = f"`{p.relative_to(ROOT)}`:{i}"
ARX = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.I)
covered = " ".join(list(seen) + list(extra))
for p in sorted(R.rglob("*.md")):
    for i, l in enumerate(p.read_text().splitlines(), 1):
        for a in ARX.findall(l):
            u = f"https://arxiv.org/abs/{a}"
            if a not in covered and u not in extra:
                extra[u] = f"`{p.relative_to(ROOT)}`:{i}（原文只给了 arXiv 编号，URL 由编号拼出）"
for k, (u, where) in enumerate(extra.items(), 1):
    md.append(f"| X-{k:02d} | {u} | {where} |")
out = ROOT/"report/references.md"
out.write_text("\n".join(md) + "\n")
print({k: len(i) for k, _, _, i in sections}, "extra", len(extra), "->", out)
