"""Build site/src/data/content.json from report/draft.md (the single source of truth).

- Text is copied from draft.md as written; only markdown inline syntax is converted.
- Citation brackets like [T-24, E-28 §3.6] become numbered footnotes; the reference
  list is filled from report/references.md (verbatim) and a small table of internal files.
- Headline numbers on the cards are computed here from the experiment data files,
  never typed by hand.
"""
import html, json, pathlib, re

REPORT = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
DRAFT = (REPORT / "draft.md").read_text()
OUT = REPORT / "site" / "src" / "data" / "content.json"

# ---------- references ----------
REF_ROWS = {}
NOTE_FILES = {"T": "research/lit_rsi/01_theory.md", "M": "research/lit_rsi/02_methods.md", "M2": "research/lit_rsi/02_methods.md",
              "E": "research/lit_rsi/03_measurement.md", "L": "research/lit_signal_hardness.md"}
for line in (REPORT / "references.md").read_text().splitlines():
    m = re.match(r"^\| ((?:T|M2|M|E|L|X)-\d+) \| (.+?) \| (.*?) \| (.*?) \| (.*?) \|$", line)
    if m:
        key, cite, typ, urls, srcline = m.groups()
        pre = key.split("-")[0]
        REF_ROWS[key] = {
            "cite": re.sub(r"\*\*", "", cite),
            "type": typ,
            "urls": [u for u in re.split(r"<br>", urls) if u.startswith("http")],
            "src": NOTE_FILES.get(pre), "line": int(srcline) if srcline.strip().isdigit() else None,
        }

for line in (REPORT / "references.md").read_text().splitlines():   # "其他出处" rows: | X-nn | URL | first seen |
    m = re.match(r"^\| (X-\d+) \| (\S+) \| (.*?) \|$", line)
    if m:
        fm = re.search(r"`([^`]+)`:(\d+)", m.group(3))
        REF_ROWS[m.group(1)] = {"cite": m.group(2), "type": "其他出处，首次出现于 " + m.group(3).replace("`", ""), "urls": [m.group(2)] if m.group(2).startswith("http") else [],
                                "src": fm.group(1) if fm else None, "line": int(fm.group(2)) if fm else None}

for ln, line in enumerate((REPORT / "references_extra.md").read_text().splitlines(), 1):  # verified extras; may override labels
    m = re.match(r"^\| ((?:S|X)-\d+) \| (.+?) \| (.*?) \| (.*?) \| (.*?) \|$", line)
    if m:
        key, cite, typ, urls, _ = m.groups()
        REF_ROWS[key] = {"cite": re.sub(r"\*", "", cite), "type": typ, "urls": [u for u in re.split(r"<br>|\s+", urls) if u.startswith("http")],
                         "src": "report/references_extra.md", "line": ln}

# ---------- the four materials: evidence ledgers (research/sources/*.md) ----------
LEDGERS = {
    "RF": ("recursive_first_steps.md", "Recursive, \"First Steps Toward Automated AI Research\""),
    "FL": ("recursive_flint.md", "Recursive, \"Introducing Flint\""),
    "NV": ("naive_research.md", "NaiveAI, \"Naive-N0.5-Flash: Building Frontier AI with AI\""),
    "AX": ("apex_35b.md", "Apex Intelligence, \"Introducing Apex-35B-V1.0\""),
}
LEDGER_LINE = {}
for pre, (fname, title) in LEDGERS.items():
    text = (ROOT / "research/sources" / fname).read_text()
    letters = {}
    for m in re.finditer(r"^\s*- ([A-Z]) = (\S+)", text, re.M):
        letters[m.group(1)] = m.group(2)
    m = re.search(r"`R` = (https?://\S+)", text)
    if m:
        letters["R"] = m.group(1)
    if "B" in letters:
        letters = {k: (letters["B"].rstrip("/") + v[1:] if v.startswith("…") else v) for k, v in letters.items()}
    main_url = letters.get("A") or letters.get("R")
    for ln, line in enumerate(text.splitlines(), 1):
        m = re.match(rf"^\| ({pre}-[A-Z]?\d+) \| (.*)\|\s*$", line)
        if not m:
            continue
        LEDGER_LINE[m.group(1)] = (f"research/sources/{fname}", ln)
        key, cells = m.group(1), [c.strip() for c in m.group(2).split("|")]
        if len(cells) >= 5 and re.match(r"^[A-ZR]", cells[3] or ""):      # value | meaning | quote | URL letter | status
            val, meaning, quote, where, status = cells[:5]
            urls = [letters[x] for x in re.findall(r"\b([A-Z])\b", where) if x in letters] or [main_url]
            cite = f"{title}，{meaning}：{val}" + (f"（原文：{quote}）" if quote and quote != "—" else "")
            REF_ROWS[key] = {"cite": cite, "type": f"证据台账 {key} · {status}", "urls": list(dict.fromkeys(urls)),
                             "work": title, "short": f"{meaning}：{val}" + (f"（原文：{quote}）" if quote and quote != "—" else "")}
        else:                                                              # table rows (benchmarks, kernels, competitor scores)
            cells = [re.sub(r"\*\*", "", c) for c in cells]
            if key.startswith("AX-T") and len(cells) >= 8:                # Apex benchmark table: name the columns
                cols = ["Apex-35B", "Qwen3.6-35B-A3B", "Flint", "GLM-5.3", "GLM-5.3-Flash", "DeepSeek-V4.1-Flash"]
                body = f"{cells[1]}（{cells[0]}）：" + "，".join(f"{c} {v}" for c, v in zip(cols, cells[2:8]))
            else:
                body = " · ".join(c for c in cells if c)
            REF_ROWS[key] = {"cite": f"{title}，{body}", "type": f"证据台账 {key}", "urls": [main_url], "work": title, "short": body}
    sec = text.split("## 原文内部不一致 / 待核", 1)
    if len(sec) == 2:
        base_ln = sec[0].count("\n") + 1
        for m in re.finditer(r"^(\d+)\. (.+)$", sec[1].split("\n## ", 1)[0], re.M):
            LEDGER_LINE[f"{pre}-Q{m.group(1)}"] = (f"research/sources/{fname}", base_ln + sec[1][:m.start()].count("\n"))
            REF_ROWS[f"{pre}-Q{m.group(1)}"] = {"cite": f"{title}，核对记录第 {m.group(1)} 条：" + re.sub(r"\*\*", "", m.group(2)),
                                               "type": "证据台账 · 不一致 / 待核", "urls": [main_url],
                                               "work": title, "short": f"核对记录第 {m.group(1)} 条：" + re.sub(r"\*\*", "", m.group(2))}

INTERNAL = {
    "LOG E": ("RSI-mini 实验总日志，E 节（实验结果）", "RSI-mini experiment log, section E (results)", "research/REPORT_LOG.md"),
    "decisions/decisions_0905.json": ("各项比较的效应、95% 置信区间与置换检验（11-18 09:05 导出）", "Effects, 95% CIs and permutation tests for every comparison (exported 11-18 09:05)", "research/decisions/decisions_0905.json"),
    "lit_signal_hardness.md": ("信号硬度与自动化顺序的文献梳理", "Literature review on signal hardness and the order of automation", "research/lit_signal_hardness.md"),
    "ling_rsi_pipeline.md": ("给 Ling 的完整 RSI 管线设计：MoE 训练难点、组件、节奏与算力粗算（11-18 14:35）", "Full RSI pipeline design for Ling: MoE training constraints, components, cadence and compute estimates (11-18 14:35)", "research/ling_rsi_pipeline.md"),
    "decisions/decisions_1420.json": ("各项比较的效应、95% 置信区间与置换检验（11-18 14:20 导出，含两个补充实验）", "Effects, 95% CIs and permutation tests for every comparison (exported 11-18 14:20, incl. the two follow-up experiments)", "research/decisions/decisions_1420.json"),
    "decisions/decisions_1235.json": ("各项比较的效应、95% 置信区间与置换检验（11-18 12:35 导出，含让模型自己定训练方法的第三轮）", "Effects, 95% CIs and permutation tests for every comparison (exported 11-18 12:35, incl. round 3 of the self-designed recipe)", "research/decisions/decisions_1235.json"),
    "decisions/decisions_1019.json": ("各项比较的效应、95% 置信区间与置换检验（11-18 10:19 导出，含让模型自己定训练方法的两轮）", "Effects, 95% CIs and permutation tests for every comparison (exported 11-18 10:19, incl. both rounds of the self-designed recipe)", "research/decisions/decisions_1019.json"),
    "decisions/decisions_0958.json": ("各项比较的效应、95% 置信区间与置换检验（11-18 09:58 导出，含让模型自己定训练方法的第一轮）", "Effects, 95% CIs and permutation tests for every comparison (exported 11-18 09:58, incl. round 1 of the self-designed recipe)", "research/decisions/decisions_0958.json"),
    "replay/final/nc-g0-final.summary.txt": ("Qwen3.5-9B 在 NanoChat 上的研究运行摘要", "Summary of the Qwen3.5-9B research run on NanoChat", "research/replay/final/nc-g0-final.summary.txt"),
    "replay/final/ovn1118b-nc-g0/events.jsonl": ("Qwen3.5-9B 在 NanoChat 上研究运行的逐事件日志（提案、评测、判定）", "Event log of the Qwen3.5-9B research run on NanoChat (proposals, evaluations, decisions)", "research/replay/final/ovn1118b-nc-g0/events.jsonl"),
    "LOG A": ("RSI 实验与调研总日志，A 节（材料框架与信号硬度文献核查）", "Research log, section A (framing and literature check on signal hardness)", "research/REPORT_LOG.md"),
    "lit_rsi_academic.md": ("学术界如何看 RSI：约 96 个来源的文献综述汇总", "How academia treats RSI: summary of a review of about 96 sources", "research/lit_rsi_academic.md"),
    "platform_design.md": ("RSI-mini 实验平台设计：定义、判定规则与对照", "RSI-mini platform design: definitions, decision rules and controls", "research/platform_design.md"),
    "LOG B": ("RSI 实验与调研总日志，B 节（开源权重与代码的一手核对）", "Research log, section B (first-hand checks of open weights and code)", "research/REPORT_LOG.md"),
    "LOG C": ("RSI 实验与调研总日志，C 节（推理优化实测）", "Research log, section C (inference measurements)", "research/REPORT_LOG.md"),
    "LOG F": ("RSI 实验与调研总日志，F 节（其他模型的逆向）", "Research log, section F (reverse engineering of the other models)", "research/REPORT_LOG.md"),
    "flint_dissection.md": ("Flint 本地拆解：参数清单与 MoE 路由", "Local dissection of Flint: parameter inventory and MoE routing", "research/flint_dissection.md"),
    "artifacts_weights_configs.md": ("开源权重与 config 的逐字段核对", "Field-by-field check of released weights and configs", "research/artifacts_weights_configs.md"),
    "census_compare.json": ("四个模型思考风格普查的汇总数据", "Summary of the reasoning-style census across four models", "research/census_compare.json"),
    "reading/04_Apex35B_导读.md": ("Apex-35B 导读：AI4AI 一节与 Sokoban 图注的解读", "Reading guide to Apex-35B: the AI4AI section and the Sokoban figure caption", "research/reading/04_Apex35B_导读.md"),
    "reading/00_总览.md": ("四份材料的总览与矛盾清单", "Overview of the four releases and list of contradictions", "research/reading/00_总览.md"),
    "model_dissection_apex_naive.md": ("Apex 与 Naive 的权重差分和行为指纹", "Weight diffs and behaviour fingerprints of Apex and Naive", "research/model_dissection_apex_naive.md"),
    "flint_reverse_engineering.md": ("Flint 的预训练与后训练逆向", "Reverse engineering of Flint's pre- and post-training", "research/flint_reverse_engineering.md"),
    "decisions/cf_analysis.json": ("从最好版本继续改：四种改进者各 3 次运行的降幅与比较", "Continue-from-best: drops and comparisons for four improvers, 3 runs each", "research/decisions/cf_analysis.json"),
    "LOG G": ("RSI-mini 实验总日志，G 节（人工干预记录）", "RSI-mini experiment log, section G (record of human interventions)", "research/REPORT_LOG.md"),
    "LOG H": ("RSI-mini 实验总日志，H 节（概念讨论与核实记录）", "RSI-mini experiment log, section H (discussion and checks)", "research/REPORT_LOG.md"),
    "platform_build_plan.md": ("RSI-mini 平台搭建计划与决策记录", "RSI-mini build plan and decision record", "research/platform_build_plan.md"),
    "deploy/second_order.md": ("RSI-mini 部署说明：二阶实验的 LoRA 训练与加载", "RSI-mini deployment notes: LoRA training and loading for the second-order runs", "platform/deploy/second_order.md"),
}

KEY_RE = re.compile(r"^((?:T|M2|M|E|L|X|S)-\d+|(?:RF|FL|NV|AX)-[A-Z]?\d+|LOG [A-H])(?:[·\s](.*))?$")
FILE_RE = re.compile(r"^([\w./-]+\.(?:md|jsonl|json|txt))(?:\s+(.*))?$")


class Refs:
    def __init__(self):
        self.order, self.loc = [], {}

    def num(self, key, loc=None):
        if key not in self.order:
            self.order.append(key)
            self.loc[key] = []
        if loc and loc not in self.loc[key]:
            self.loc[key].append(loc)
        return self.order.index(key) + 1

    def parse_bracket(self, inner):
        """Return list of footnote numbers, or None if this bracket is not a citation."""
        items = [x.replace("`", "").strip() for x in re.split(r"[，,；;]", inner) if x.strip()]
        nums, last = [], None
        for it in items:
            m = KEY_RE.match(it) or FILE_RE.match(it)
            if m:
                last = m.group(1)
                nums.append(self.num(last, (m.group(2) or "").strip() or None))
            elif last and re.match(r"^(§|Fig|图)", it):
                self.num(last, it)
            else:
                return None
        return list(dict.fromkeys(nums))

    def entries(self):
        out = []
        for i, k in enumerate(self.order, 1):
            loc = "，".join(self.loc[k])
            pre = k.split("-")[0]
            if k in REF_ROWS:
                r = REF_ROWS[k]
                src, line = LEDGER_LINE.get(k) or (r.get("src"), r.get("line"))
                out.append({"n": i, "key": k, "zh": r.get("short") or r["cite"], "en": r.get("short") or r["cite"], "work": r.get("work", ""),
                            "note": r["type"], "urls": r["urls"], "loc": loc,
                            "group": "ledger" if pre in LEDGERS else "lit", "src": src, "line": line, "jumps": []})
            elif k in INTERNAL:
                zh, en, path = INTERNAL[k]
                line = locate(path, f"LOG {k[4:]}") if k.startswith("LOG ") else None
                jumps = [{"label": x, "line": locate(path, x)} for x in self.loc[k] if x.startswith("§")]
                out.append({"n": i, "key": k, "zh": zh, "en": en, "note": path, "urls": [], "loc": loc,
                            "group": "data", "src": path, "line": line,
                            "jumps": list({j["label"]: j for j in jumps if j["line"]}.values())})
            else:
                out.append({"n": i, "key": k, "zh": k, "en": k, "note": "未登记的出处，请核对", "urls": [], "loc": loc,
                            "group": "data", "src": None, "line": None, "jumps": []})
        for e in out:
            e["line"] = repo_line(e["src"], e["line"])
            for j in e["jumps"]:
                j["line"] = repo_line(e["src"], j["line"])
        return out


# Line numbers must match the published repository, which carries a few redacted files (e.g. cluster names
# removed from REPORT_LOG.md). When the publish checkout exists, every recorded line is re-anchored to the
# same text in that copy.
PUB = ROOT / "publish" / "rsi-report-public"


def repo_line(path, line):
    if not (path and line and PUB.exists()):
        return line
    loc, pub = ROOT / path, PUB / path
    if not (loc.exists() and pub.exists()) or loc.suffix not in (".md", ".txt", ".json", ".jsonl"):
        return line
    a, b = loc.read_text().splitlines(), pub.read_text().splitlines()
    if line > len(a):
        return line
    text = a[line - 1]
    if line <= len(b) and b[line - 1] == text:
        return line
    hits = [i for i, l in enumerate(b, 1) if l == text]
    if hits:
        return min(hits, key=lambda i: abs(i - line))
    import difflib
    best = difflib.get_close_matches(text, b, n=1, cutoff=0.6)
    return (b.index(best[0]) + 1) if best else line


def locate(path, what):
    """Line number of a heading in a repo file: 'LOG E' -> '## E. ...'; '§1.4' -> first heading containing 1.4."""
    f = ROOT / path
    if not f.exists() or f.suffix != ".md":
        return None
    lines = f.read_text().splitlines()
    if what.startswith("LOG "):
        pat = re.compile(rf"^##\s+{re.escape(what[4:])}[.．、\s]")
    else:
        num = re.match(r"§\s*([\d.]+)", what)
        if not num:
            return None
        n = num.group(1).rstrip(".").split("–")[0]
        pat = re.compile(rf"^#+\s.*?(?<![\d.]){re.escape(n)}(?![\d])")
    for i, l in enumerate(lines, 1):
        if pat.match(l):
            return i
    return None


REFS = Refs()


def inline(text):
    """Markdown inline -> HTML, citation brackets -> footnote sups."""
    out, pos = [], 0
    for m in re.finditer(r"\[([^\[\]]+)\]", text):
        nums = REFS.parse_bracket(m.group(1))
        if nums is None:
            continue
        out.append(fmt(text[pos:m.start()].rstrip()))
        out.append('<sup class="fn">' + ",".join(f'<a href="#ref-{n}">{n}</a>' for n in nums) + "</sup>")
        pos = m.end()
    out.append(fmt(text[pos:]))
    return "".join(out)


def fmt(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\$([^$\n]+?)\$", lambda m: '<i class="math">' + m.group(1).replace("\\approx", "≈").replace("\\times", "×")
               .replace("\\le", "≤").replace("\\ge", "≥").replace("\\,", " ").replace(" ", "\u2009") + "</i>", t)
    t = re.sub(r"〔(推断|估算|自报|inference|estimate|self-reported)〕", r'<span class="tag">\1</span>', t)
    return t


ABSTRACT = (REPORT / "sections" / "00_abstract.md").read_text()


def block(start, end, text=None):
    text = DRAFT if text is None else text
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    body = text[i:j].split("\n", 1)[1]
    return body.strip()


def abs_part(name):
    """Parts of sections/00_abstract.md: cards_zh, cards_en, zh, en."""
    heads = {"cards_zh": ("#### 卡片（中文）", "#### Cards (English)"), "cards_en": ("#### Cards (English)", "#### 正文（中文）"),
             "zh": ("#### 正文（中文）", "#### Text (English)"), "en": ("#### Text (English)", None)}
    a, b = heads[name]
    return block(a, b, ABSTRACT)


def paragraphs(body):
    paras = []
    for p in re.split(r"\n\s*\n", body):
        p = " ".join(x.strip() for x in p.splitlines()).strip()
        if not p:
            continue
        pending = bool(re.match(r"^(〔预留|\[Reserved)", p))
        if pending:
            p = re.sub(r"^〔预留[：:]?\s*|〕$|^\[Reserved[：:]?\s*|\]$", "", p)
        paras.append({"html": inline(p), "pending": pending})
    return paras


def cards(body):
    res = []
    for m in re.finditer(r"^\d\. \*\*(.+?)\*\*[：:]\s*(.+?)$((?:\n {2,4}- .+)*)", body, re.M):
        title, text, subs = m.groups()
        notes = []
        for s in re.findall(r"^ {2,4}- (.+)$", subs or "", re.M):
            pending = bool(re.match(r"^(〔预留〕|\[Reserved\])", s))
            s = re.sub(r"^(⚠️\s*|〔预留〕\s*|\[Reserved\]\s*)", "", s)
            notes.append({"html": inline(s), "pending": pending})
        res.append({"title": fmt(title), "html": inline(text), "notes": notes})
    return res


LIST_RE = re.compile(r"^(-|\d+\.)\s+")


def blocks(body):
    """Paragraphs and pipe tables -> list of {type, html|head|rows, pending}."""
    out = []
    for chunk in re.split(r"\n\s*\n", body.strip()):
        lines = [l.strip() for l in chunk.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if len(lines) == 1 and lines[0].startswith("### "):
            out.append({"type": "h3", "html": inline(lines[0][4:].strip())})
            continue
        fig = re.match(r"^\[\[figure:([\w-]+)｜(.*)\]\]$", " ".join(lines))
        if fig:
            out.append({"type": "figure", "id": fig.group(1), "html": inline(fig.group(2))})
            continue
        if all(l.startswith("|") for l in lines):
            rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines if not re.match(r"^\|\s*-", l)]
            out.append({"type": "table", "head": [inline(c) for c in rows[0]], "rows": [[inline(c) for c in r] for r in rows[1:]]})
        elif any(LIST_RE.match(l) for l in lines):
            lead, items, ordered = [], [], False
            for l in lines:
                m = LIST_RE.match(l)
                if m:
                    ordered = ordered or m.group(1)[0].isdigit()
                    items.append(l[m.end():])
                elif items:
                    items[-1] += " " + l
                else:
                    lead.append(l)
            if lead:
                out.append({"type": "p", "html": inline(" ".join(lead)), "pending": False})
            out.append({"type": "list", "ordered": ordered, "items": [inline(x) for x in items]})
        else:
            p = " ".join(lines)
            pending = bool(re.match(r"^(〔预留|\[Reserved)", p))
            if pending:
                p = re.sub(r"^〔预留[：:]?\s*|〕$|^\[Reserved[：:]?\s*|\]$", "", p)
            out.append({"type": "p", "html": inline(p), "pending": pending})
    return out


def section_text(n):
    """Text blocks of draft section n: '#### 正文（中文…' and '#### Text (English…'."""
    files = sorted((REPORT / "sections").glob(f"{n:02d}_*.md"))
    if files:                                   # a section file, when present, is the source of truth
        sec = files[0].read_text()
    else:
        m = re.search(rf"^## {n}\. .*$", DRAFT, re.M)
        if not m:
            return None
        nxt = re.search(r"^## ", DRAFT[m.end():], re.M)
        sec = DRAFT[m.end(): m.end() + nxt.start()] if nxt else DRAFT[m.end():]
    zh = re.search(r"^#### 正文（中文.*$", sec, re.M)
    en = re.search(r"^#### Text \(English.*$", sec, re.M)
    if not (zh and en):
        return None
    stop = re.search(r"^(> \*\*✍️|\*\*📎|#### (?!Text)|---\s*$)", sec[en.end():], re.M)
    en_body = sec[en.end(): en.end() + stop.start()] if stop else sec[en.end():]
    for k, v in {"推断": "inference", "估算": "estimate", "自报": "self-reported"}.items():
        en_body = en_body.replace(f"〔{k}〕", f"〔{v}〕")      # English page gets English tags
    out = {"zh": blocks(sec[zh.end(): en.start()]), "en": blocks(en_body)}
    figs = REPORT / "site" / "public" / "figs"
    for lang, bl in out.items():
        for blk in bl:
            if blk["type"] == "figure" and (figs / f"{blk['id']}.mp4").exists():
                blk["video"] = f"{blk['id']}.mp4"                                     # one video for both languages
            if blk["type"] == "figure" and (figs / f"{blk['id']}.{lang}.svg").exists():
                blk["svg"] = (figs / f"{blk['id']}.{lang}.svg").read_text()    # inline so hover tooltips work
    return out


def with_svg(bl, lang):
    figs = REPORT / "site" / "public" / "figs"
    for blk in bl:
        if blk["type"] == "figure" and (figs / f"{blk['id']}.{lang}.svg").exists():
            blk["svg"] = (figs / f"{blk['id']}.{lang}.svg").read_text()
    return bl


# ---------- computed headline numbers ----------
# authoritative values: the run's own records (the replay summary's "best" is per-thread, not the run best)
_ev = [json.loads(l) for l in (ROOT / "research/replay/final/ovn1118b-nc-g0/events.jsonl").read_text().splitlines()]
s0 = next(e for e in _ev if e["ev"] == "decision" and e["trial"] == 0)["best_after"]
best = next(e for e in _ev if e["ev"] == "run_end")["best_value"]
DEC_FILE = sorted((ROOT / "research/decisions").glob("decisions_*.json"))[-1]  # newest export
dec = json.loads(DEC_FILE.read_text())
us = dec["dec_us"]["main"]
lessons_pct = (us["mean_a"] - us["mean_b"]) / us["mean_b"] * 100
vs_m0 = [dec["dec_g1"]["main"]] + [a for k in ("dec_g2", "dec_g3") for a in dec[k]["also"] if a["name"].endswith("vs M0")]
assert len(vs_m0) == 3
faster = sum(1 for x in vs_m0 if x["ci"][0] > 0)
FIG = [
    {"value": f"{s0:.4f} → {best:.4f}", "zh": "验证损失（每字节比特数，越低越好）", "en": "validation loss (bits per byte, lower is better)"},
    {"value": f"{lessons_pct:+.0f}%".replace("-", "−"), "zh": f"改进速度（{us['mean_b']:.3f} → {us['mean_a']:.3f}）", "en": f"speed of improvement ({us['mean_b']:.3f} → {us['mean_a']:.3f})"},
    {"value": f"{faster} / {len(vs_m0)}", "zh": "比原始模型改得更快的后继代数", "en": "successor generations faster than the original"},
]

# ---------- assemble ----------
zh_cards = cards(abs_part("cards_zh"))
en_cards = cards(abs_part("cards_en"))
for c in en_cards:
    c["html"] = c["html"][:1].upper() + c["html"][1:]
for c, f in zip(zh_cards, FIG):
    c["figure"] = {"value": f["value"], "label": f["zh"]}
for c, f in zip(en_cards, FIG):
    c["figure"] = {"value": f["value"], "label": f["en"]}

content = {
    "abstract": {
        "zh": with_svg(blocks(abs_part("zh")), "zh"),
        "en": with_svg(blocks(abs_part("en")), "en"),
    },
    "cards": {"zh": zh_cards, "en": en_cards},
    "sections": {str(n): t for n in range(1, 8) if (t := section_text(n))},
    "refs": REFS.entries(),
}

# ---------- full plain-text report (report/report_full.md), always regenerated ----------
def md_cites(raw):
    def sub(m):
        nums = REFS.parse_bracket(m.group(1))
        return m.group(0) if nums is None else "[" + ",".join(str(n) for n in nums) + "]"
    raw = re.sub(r"\[\[figure:([\w-]+)｜(.*?)\]\]", lambda m: f"![{m.group(1)}](site/public/figs/{m.group(1)}.LANG.gif)\n\n*{m.group(2)}*", raw)
    return re.sub(r"(?<!\[)\[([^\[\]]+)\](?!\()", sub, raw)

def raw_section(n, lang):
    files = sorted((REPORT / "sections").glob(f"{n:02d}_*.md"))
    if not files:
        return None
    sec = files[0].read_text()
    zh = re.search(r"^#### 正文（中文.*$", sec, re.M); en = re.search(r"^#### Text \(English.*$", sec, re.M)
    body = sec[zh.end():en.start()] if lang == "zh" else sec[en.end():]
    return body.strip()

SEC_TITLES = {1: ("什么才算 RSI", "What counts as RSI"), 2: ("四份材料", "The four releases"), 3: ("没人测过的一环", "The missing measurement"),
              4: ("实验：换上后继模型", "Experiment: swapping in the successor"), 5: ("与文献对照", "Against the literature"), 6: ("给 Ling 的计划", "A plan for Ling"),
              7: ("写在最后（补充分析）", "Afterword (supplementary analysis)")}
full = ["<!-- 此文件由 report/tools/build_content.py 自动生成，请勿直接修改。改 report/draft.md（摘要）或 report/sections/*.md（各节），再运行构建。 -->", "",
        "# Recursive Self-Improvement in AI: A Survey of Claims, Evidence, and Empirical Tests", "", "孙祺（Qi Sun）", "", "## 中文", "", "### 摘要", "", md_cites(abs_part("zh"))]
for n, (zt, et) in SEC_TITLES.items():
    body = raw_section(n, "zh")
    if body:
        full += ["", f"### {n}. {zt}", "", md_cites(body).replace(".LANG.", ".zh.").replace("\n### ", "\n#### ")]
full += ["", "## English", "", "### Abstract", "", md_cites(abs_part("en"))]
for n, (zt, et) in SEC_TITLES.items():
    body = raw_section(n, "en")
    if body:
        full += ["", f"### {n}. {et}", "", md_cites(body).replace(".LANG.", ".en.").replace("\n### ", "\n#### ")]
full += ["", "## 参考文献 / References", ""]
for r in REFS.entries():
    line = f"{r['n']}. " + (f"{r['work']}，" if r.get("work") else "") + f"{r['zh']}" + (f"（{r['loc']}）" if r["loc"] else "") + f" — {r['note']}"
    if r["urls"]:
        line += " — " + " ".join(r["urls"])
    full.append(line)
(REPORT / "report_full.md").write_text("\n".join(full) + "\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(content, ensure_ascii=False, indent=1))
print(f"wrote {OUT.relative_to(REPORT)}: sections {list(content['sections'])}, {len(content['abstract']['zh'])} zh paras, "
      f"{len(content['abstract']['en'])} en paras, {len(zh_cards)} cards, {len(content['refs'])} refs; figures {[f['value'] for f in FIG]}")
