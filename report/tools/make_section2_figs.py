"""Figures for report §2 (inline SVG, zh + en, hover tooltips via <title>).

Every value comes from a research file (ledger keys / dissection tables / census json);
geometry is computed here. Palette: categorical slots 1–2 of the dataviz reference palette
(#2a78d6 blue, #eb6834 orange), validated on the page surface #fbfaf7.
Output: report/site/public/figs/<id>.<lang>.svg
"""
import html, json, pathlib

REPORT = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
OUT = REPORT / "site" / "public" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, NEUTRAL = "#1c1f23", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d6d3cb"
W = 720


def esc(s):
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, h):
        self.h, self.items = h, []

    def add(self, s):
        self.items.append(s)

    def text(self, x, y, s, size=12, fill=INK2, anchor="start", weight=400, cls=""):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{f" class={chr(34)}{cls}{chr(34)}" if cls else ""}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, tip=None, rx=3):
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}">{t}</rect>')

    def line(self, x1, y1, x2, y2, stroke=GRID, w=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>')

    def circle(self, x, y, r, fill, stroke=None, tip=None, sw=2):
        t = f"<title>{esc(tip)}</title>" if tip else ""
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{s}>{t}</circle>')

    def raw(self, s):
        self.add(s)

    def save(self, name, lang, label):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" role="img" aria-label="{esc(label)}" '
               f'style="font-family:inherit;width:100%;height:auto">' + "".join(self.items) + "</svg>")
        (OUT / f"{name}.{lang}.svg").write_text(svg)


def legend(s, x, y, items):
    for label, color, kind in items:
        if kind == "bar":
            s.rect(x, y - 9, 14, 10, color, rx=2)
        else:
            s.circle(x + 6, y - 4, 5, color)
        s.text(x + 20, y, label, size=12, fill=INK2)
        x += 26 + 7.2 * len(label) + (6 * sum(1 for ch in label if ord(ch) > 0x2e80))


# ------------------------------------------------------------------ F1: what each report discloses
def fig_disclosure(lang):
    zh = lang == "zh"
    cols = ["权重", "代码", "研究系统", "驱动模型", "多次运行与方差", "逐代测量"] if zh else \
           ["Weights", "Code", "Research system", "Driving model", "Repeated runs", "Per-generation data"]
    EN_NOTES = {
        "Recursive · First Steps": ["not a model release", "training code, per-seed results, 10 of 235 kernels [RF-29, RF-31]", "system not released [RF-Q9]",
                                    "model not stated [RF-Q9]", "10 random seeds; Speedrun with significance test [RF-04, RF-33]", "none"],
        "Naive · N0.5-Flash": ["weights and inference code under MIT", "inference code released; training code and research harness not; NaiveRT source not yet available at release [NV-68]",
                               "in-house research harness not released", "architecture and NaiveRT only say \"AI models\"; AutoWM run by N0.5-Flash [NV-Q6]",
                               "comparison with Recursive gives no seeds or variance [NV-Q2]", "none"],
        "Recursive · Flint": ["BF16 weights under Apache 2.0", "rec inference engine released; training code, data and recipe not", "Eureka Machine not released",
                              "model driving the Eureka Machine not stated", "pass@1 averaged over at least 5 attempts, no variance [FL-14, FL-15]", "\"Our bet is…\", no data [FL-Q7]"],
        "Apex · Apex-35B": ["BF16 weights under Apache 2.0", "deployment docs only, no training code", "Apex Research not released [AX-Q10]",
                            "model driving Apex Research not stated [AX-Q10]", "each task run once [AX-43]", "none"],
    }
    rows = [
        ("Recursive · First Steps", ["na", "part", "no", "no", "yes", "no"],
         ["不是模型发布 / not a model release", "训练代码、每个种子的结果、235 个 kernel 中的 10 个 [RF-29, RF-31]",
          "系统本身未公开 [RF-Q9]", "未说明所用模型 [RF-Q9]", "10 个随机种子；Speedrun 附显著性检验 [RF-04, RF-33]", "无"]),
        ("Naive · N0.5-Flash", ["yes", "part", "no", "part", "no", "no"],
         ["权重与推理代码 MIT 开源 [NV 基本信息]", "推理代码公开；训练代码、研究框架未公开；NaiveRT 源码发布时尚未开放 [NV-68]",
          "in-house 研究框架未公开", "架构与 NaiveRT 只写 \"AI models\"；AutoWM 由 N0.5-Flash 执行 [NV-Q6]",
          "与 Recursive 的比较未报告种子数和方差 [NV-Q2]", "无"]),
        ("Recursive · Flint", ["yes", "part", "no", "no", "part", "no"],
         ["BF16 权重 Apache 2.0 [FL 基本信息]", "推理引擎 rec 开源；训练代码、数据、配方未公开", "Eureka Machine 未公开",
          "未说明驱动 Eureka Machine 的模型", "pass@1 取至少 5 次的平均，未报告方差 [FL-14, FL-15]", "\"Our bet is…\"，无数据 [FL-Q7]"]),
        ("Apex · Apex-35B", ["yes", "no", "no", "no", "no", "no"],
         ["BF16 权重 Apache 2.0 [AX 基本信息]", "只有部署文档，无训练代码", "Apex Research 未公开 [AX-Q10]",
          "未说明驱动 Apex Research 的模型 [AX-Q10]", "每个任务只运行一次 [AX-43]", "无"]),
    ]
    lab = {"yes": "公开 / 有" if zh else "released / yes", "part": "部分" if zh else "partial",
           "no": "未公开 / 无" if zh else "not released / none", "na": "不适用" if zh else "n/a"}
    x0, cw, y0, rh = 210, 85, 64, 40
    s = SVG(y0 + rh * len(rows) + 46)
    for j, c in enumerate(cols):
        cx = x0 + cw * j + cw / 2
        s.text(cx, 40, c, size=12, fill=INK, anchor="middle", weight=500)
    if not zh:
        rows = [(n, m, EN_NOTES[n]) for n, m, _ in rows]
    for i, (name, marks, notes) in enumerate(rows):
        cy = y0 + rh * i + rh / 2
        if i:
            s.line(16, y0 + rh * i, W - 16, y0 + rh * i, GRID)
        s.text(16, cy + 4, name, size=13, fill=INK, weight=500)
        for j, m in enumerate(marks):
            cx = x0 + cw * j + cw / 2
            tip = f"{name} · {cols[j]}：{lab[m]}。{notes[j]}" if zh else f"{name} · {cols[j]}: {lab[m]}. {notes[j]}"
            if m == "yes":
                s.circle(cx, cy, 7, BLUE, tip=tip)
            elif m == "part":
                s.circle(cx, cy, 7, "#ffffff", stroke=BLUE, tip=tip)
                s.raw(f'<path d="M{cx:.1f},{cy - 7:.1f} A7,7 0 0,0 {cx:.1f},{cy + 7:.1f} Z" fill="{BLUE}"><title>{esc(tip)}</title></path>')
            elif m == "no":
                s.circle(cx, cy, 6.5, "#ffffff", stroke=AXIS, tip=tip)
            else:
                s.text(cx, cy + 4, "—", size=13, fill=MUTED, anchor="middle")
    ly = y0 + rh * len(rows) + 30
    s.circle(22, ly - 4, 6, BLUE); s.text(34, ly, lab["yes"], fill=INK2)
    s.circle(152, ly - 4, 6, "#ffffff", stroke=BLUE)
    s.raw(f'<path d="M152,{ly - 10} A6,6 0 0,0 152,{ly + 2} Z" fill="{BLUE}"/>'); s.text(164, ly, lab["part"], fill=INK2)
    s.circle(242, ly - 4, 5.5, "#ffffff", stroke=AXIS); s.text(254, ly, lab["no"], fill=INK2)
    s.text(392, ly, "—  " + lab["na"], fill=INK2)
    s.save("fig2-disclosure", lang, "四份报告公开了什么" if zh else "What each report discloses")


# ------------------------------------------------------------------ F2: NaiveRT optimisation attempts
def fig_naivert(lang):
    zh = lang == "zh"
    stages = [("阶段 1 · 全网络工程" if zh else "Stage 1 · whole-network engineering", 43, 28, "NV-57"),
              ("阶段 2 · 真实 checkpoint 执行" if zh else "Stage 2 · real-checkpoint execution", 45, 15, "NV-58"),
              ("阶段 3 · W8A8 kernel 精修" if zh else "Stage 3 · W8A8 kernel refinement", 63, 20, "NV-59")]
    total = ("合计" if zh else "Total", 151, 63, "NV-46, NV-60")
    x0, xw, vmax = 250, 430, 160
    sx = lambda v: x0 + v / vmax * xw
    s = SVG(250)
    legend(s, 16, 22, [("被采纳" if zh else "adopted", BLUE, "bar"), ("未采纳（未通过验证、回滚或探索性尝试）" if zh else "not adopted (failed validation, rolled back, or exploratory)", NEUTRAL, "bar")])
    for v in range(0, 161, 40):
        s.line(sx(v), 40, sx(v), 210, GRID)
        s.text(sx(v), 226, v, size=11, fill=MUTED, anchor="middle")
    y = 52
    for k, (name, n, a, key) in enumerate(stages + [total]):
        if k == 3:
            s.line(16, y - 8, W - 16, y - 8, AXIS)
            y += 6
        tip = f"{name}：{n} 次尝试，{a} 次被采纳 [{key}]" if zh else f"{name}: {n} attempts, {a} adopted [{key}]"
        s.text(16, y + 13, name, size=12, fill=INK, weight=500 if k == 3 else 400)
        s.rect(sx(0), y, sx(a) - sx(0), 18, BLUE, tip=tip)
        s.rect(sx(a) + 2, y, sx(n) - sx(a) - 2, 18, NEUTRAL, tip=tip)
        s.text(sx(n) + 8, y + 13, f"{a} / {n}", size=12, fill=INK2)
        y += 38
    s.text(sx(80), 244, "优化尝试次数" if zh else "optimisation attempts", size=11, fill=MUTED, anchor="middle")
    s.save("fig2-naivert", lang, "NaiveRT 的优化尝试" if zh else "NaiveRT optimisation attempts")


# ------------------------------------------------------------------ F3: where the weights changed
def fig_weights(lang):
    zh = lang == "zh"
    # (a) Apex vs Qwen3.6: reported value or range, % relative change  [model_dissection_apex_naive.md §1]
    apex = [("routed 专家（平均）" if zh else "routed experts (mean)", 13.1, 13.3),
            ("shared expert", 11, 15), ("router", 9.0, 9.0),
            ("full attention q/k/v/o", 6.7, 8.7), ("线性注意力投影" if zh else "linear-attention proj.", 7.4, 9.3),
            ("embedding", 7.5, 7.5), ("lm_head", 10.9, 10.9), ("归一化层" if zh else "norm layers", 1.0, 1.0)]
    # (b) Naive: medians in the 9 DSA layers vs the 39 SWA layers  [§3]
    naive = [("q_proj", 28.2, 9.5), ("k_proj", 25.8, 8.2), ("v_proj", 51.9, 10.6), ("o_proj", 9.0, 10.0)]
    s = SVG(330)
    # panel a
    s.text(16, 20, "(a) Apex-35B 相对 Qwen3.6 的改动" if zh else "(a) Apex-35B vs Qwen3.6", size=13, fill=INK, weight=500)
    ax0, axw, amax = 170, 160, 20
    ax = lambda v: ax0 + v / amax * axw
    for v in (0, 10, 20):
        s.line(ax(v), 34, ax(v), 290, GRID)
        s.text(ax(v), 306, f"{v}%", size=11, fill=MUTED, anchor="middle")
    for i, (name, lo, hi) in enumerate(apex):
        y = 48 + i * 30
        s.text(16, y + 4, name, size=11.5, fill=INK2)
        tip = (f"{name}：{lo}%" if lo == hi else f"{name}：{lo}–{hi}%") + " [model_dissection_apex_naive.md §1]"
        s.line(ax(0), y, ax(hi), y, GRID, 1)
        if lo == hi:
            s.circle(ax(lo), y, 5, BLUE, tip=tip)
        else:
            s.rect(ax(lo), y - 5, ax(hi) - ax(lo), 10, BLUE, tip=tip, rx=5)
        s.text(ax(hi) + 10, y + 4, f"{lo}%" if lo == hi else f"{lo}–{hi}%", size=11, fill=MUTED)
    # panel b
    bx0 = 400
    s.text(bx0, 20, "(b) N0.5-Flash 相对 MiMo-V2.5 的改动（中位数）" if zh else "(b) N0.5-Flash vs MiMo-V2.5 (median)", size=13, fill=INK, weight=500)
    legend(s, bx0, 44, [("9 个 DSA 层" if zh else "9 DSA layers", BLUE, "bar"), ("39 个 SWA 层" if zh else "39 SWA layers", ORANGE, "bar")])
    px0, pxw, pmax = bx0 + 60, 230, 60
    px = lambda v: px0 + v / pmax * pxw
    for v in (0, 20, 40, 60):
        s.line(px(v), 58, px(v), 290, GRID)
        s.text(px(v), 306, f"{v}%", size=11, fill=MUTED, anchor="middle")
    for i, (name, dsa, swa) in enumerate(naive):
        y = 70 + i * 56
        s.text(bx0, y + 18, name, size=12, fill=INK2)
        s.rect(px(0), y, px(dsa) - px(0), 14, BLUE, tip=f"{name} · DSA：{dsa}% [§3]")
        s.text(px(dsa) + 6, y + 11, f"{dsa}%", size=11, fill=INK2)
        s.rect(px(0), y + 16, px(swa) - px(0), 14, ORANGE, tip=f"{name} · SWA：{swa}% [§3]")
        s.text(px(swa) + 6, y + 27, f"{swa}%", size=11, fill=INK2)
    s.text(W - 16, 324, "相对改动 ‖A − B‖ / ‖B‖" if zh else "relative change ‖A − B‖ / ‖B‖", size=11, fill=MUTED, anchor="end")
    s.save("fig2-weights", lang, "权重改动的位置" if zh else "Where the weights changed")


# ------------------------------------------------------------------ F4: reasoning-style census
def fig_census(lang):
    zh = lang == "zh"
    c = json.loads((ROOT / "research/census_compare.json").read_text())
    def top(m):
        cl = c[m]["ALL"]["classes"]
        return sorted(cl.items(), key=lambda kv: -kv[1])
    names = {"qwen36": "Qwen3.6（基座）" if zh else "Qwen3.6 (base)", "apex": "Apex-35B", "naive": "Naive N0.5-Flash", "flint": "Flint"}
    pat = {"heres_thinking_process": "\"Here's a thinking process\"", "user_asks": "\"The user asks/wants…\"",
           "telegraphic": "\"We need to…\"（电报体）" if zh else "\"We need to…\" (telegraphic)",
           "gemini_numbered": "Gemini 式编号" if zh else "Gemini-style numbering"}
    order = ["qwen36", "apex", "naive", "flint"]
    s = SVG(280)
    s.text(16, 20, "(a) 最常见的思考开头（80 条提示中的条数）" if zh else "(a) Most common reasoning opening (of 80 prompts)", size=13, fill=INK, weight=500)
    x0, xw = 190, 170
    sx = lambda v: x0 + v / 80 * xw
    for v in (0, 40, 80):
        s.line(sx(v), 34, sx(v), 226, GRID)
        s.text(sx(v), 242, v, size=11, fill=MUTED, anchor="middle")
    for i, m in enumerate(order):
        y = 44 + i * 46
        (k1, v1), (k2, v2) = top(m)[0], top(m)[1]
        label = pat.get(k1, k1)
        if m == "flint":
            label = ("电报体与 Gemini 式编号并列" if zh else "telegraphic tied with Gemini-style")
        s.text(16, y + 12, names[m], size=12, fill=INK)
        s.text(16, y + 27, label, size=10.5, fill=MUTED)
        tip = f"{names[m]}：{label} {v1}/80 [census_compare.json]"
        s.rect(sx(0), y + 4, sx(v1) - sx(0), 16, BLUE, tip=tip)
        s.text(sx(v1) + 6, y + 16, f"{v1}" if m != "flint" else f"{v1} + {v2}", size=11, fill=INK2)
    # panel b: median reasoning length
    bx0 = 455
    s.text(bx0, 20, "(b) 思考长度中位数（字符）" if zh else "(b) Median reasoning length (chars)", size=13, fill=INK, weight=500)
    px0, pxw, pmax = bx0 + 95, 115, 3500
    px = lambda v: px0 + v / pmax * pxw
    for v in (0, 1000, 2000, 3000):
        s.line(px(v), 34, px(v), 226, GRID)
        s.text(px(v), 242, f"{v // 1000}k" if v else "0", size=11, fill=MUTED, anchor="middle")
    for i, m in enumerate(order):
        y = 44 + i * 46
        v = c[m]["ALL"]["median_reasoning_chars"]
        s.text(bx0, y + 16, names[m].split("（")[0].split(" (")[0].replace("Naive N0.5-Flash", "Naive"), size=12, fill=INK)
        lower = m == "qwen36"
        tip = f"{names[m]}：{'≥' if lower else ''}{v} [census_compare.json；model_dissection_apex_naive.md §5]"
        s.rect(px(0), y + 4, px(v) - px(0), 16, ORANGE, tip=tip)
        s.text(px(v) + 6, y + 16, f"{'≥' if lower else ''}{v:,}", size=11, fill=INK2)
    s.text(W - 16, 270, ("Qwen3.6 有 22 条在 1200 token 内没写完，中位数是下限" if zh else
                         "Qwen3.6: 22 answers hit the 1200-token cap, so its median is a lower bound"), size=11, fill=MUTED, anchor="end")
    s.save("fig2-census", lang, "思考方式普查" if zh else "Reasoning-style census")


# ------------------------------------------------------------------ F5: same metric, different measurer
def fig_measure(lang):
    zh = lang == "zh"
    rows = [("Flint · Terminal-Bench 2.1", 54.9, 59.6, "FL-24", "AX-T13", "Apex 测得" if zh else "measured by Apex"),
            ("Flint · SWE-bench Pro v2", 89.7, 85.2, "FL-25", "AX-T14", "Apex 测得" if zh else "measured by Apex"),
            ("Flint · SWE-bench Verified", 74.2, 74.8, "FL-26", "AX-T15", "Apex 测得" if zh else "measured by Apex"),
            ("Recursive · SOL-ExecBench（×100）" if zh else "Recursive · SOL-ExecBench (×100)", 75.4, 67.56, "RF-26", "NV-23",
             "Naive 引用" if zh else "as quoted by Naive")]
    x0, xw = 290, 380
    sx = lambda v: x0 + v / 100 * xw
    s = SVG(250)
    legend(s, 16, 22, [("报告方自报" if zh else "self-reported", BLUE, "dot"), ("另一方测得或引用" if zh else "measured or quoted by another party", ORANGE, "dot")])
    for v in range(0, 101, 25):
        s.line(sx(v), 40, sx(v), 205, GRID)
        s.text(sx(v), 222, v, size=11, fill=MUTED, anchor="middle")
    for i, (name, a, b, ka, kb, who) in enumerate(rows):
        y = 60 + i * 40
        s.text(16, y + 4, name, size=12, fill=INK)
        s.line(sx(min(a, b)), y, sx(max(a, b)), y, AXIS, 3)
        s.circle(sx(a), y, 6, BLUE, stroke="#fbfaf7", tip=f"{name} · 自报 {a} [{ka}]" if zh else f"{name} · self-reported {a} [{ka}]")
        s.circle(sx(b), y, 6, ORANGE, stroke="#fbfaf7", tip=f"{name} · {who} {b} [{kb}]")
        d = b - a
        s.text(sx(max(a, b)) + 12, y + 4, f"{d:+.1f}", size=11, fill=INK2)
    s.text(sx(50), 244, "分数（百分制）" if zh else "score (0–100)", size=11, fill=MUTED, anchor="middle")
    s.save("fig2-measure", lang, "同一指标，不同测量方" if zh else "Same metric, different measurer")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        for f in (fig_disclosure, fig_naivert, fig_weights, fig_census, fig_measure):
            f(lang)
    print(sorted(p.name for p in OUT.glob("fig2-*.svg")))
