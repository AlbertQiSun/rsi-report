"""Figures for report §5 (inline SVG, zh + en, hover tooltips via <title>).

fig5-map    : the five results of §4 next to the closest literature and whether they agree (judgement = inference)
fig5-memory : relative change after adding reflection / experience / memory, across studies
fig5-iter   : round-by-round results of fixed-loop self-training (literature) vs this report's generations

Literature numbers are the values the papers report, as checked against the originals in
research/lit_rsi/02_methods.md (table / section noted in each tooltip). This report's numbers are read from
research/decisions/decisions_1235.json. Relative changes and geometry are computed here.
Output: report/site/public/figs/<id>.{zh,en}.svg
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import W, BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL, esc, legend  # noqa: E402
from make_section4_flow import Fig, L, tw, PAPER, BLUE_T, ORANGE_T  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEC = json.loads((ROOT / "research/decisions/decisions_1235.json").read_text())
M0 = DEC["dec_g1"]["main"]["mean_b"]
G = [M0, DEC["dec_g1"]["main"]["mean_a"], DEC["dec_g2"]["main"]["mean_a"], DEC["dec_g3"]["main"]["mean_a"]]
US = DEC["dec_us"]["main"]["mean_a"]


# ------------------------------------------------------------------ fig5-map
def fig_map(lang):
    rows = [
        (L("经验笔记让改进变慢", "Lessons note slows improvement", lang),
         L(f"0.588 → {US:.3f}；比通用建议低 0.239", f"0.588 → {US:.3f}; 0.239 below generic tips", lang),
         L("ExpeL、自我纠错、ACE、Xiong 等", "ExpeL, self-correction, ACE, Xiong et al.", lang),
         L("加入反思或不加筛选的记忆后变差", "worse after adding reflection or unfiltered memory", lang),
         L("一致", "consistent", lang), True,
         "ExpeL 39.0→29.0 (Table 3) [M-25]; GPT-4 GSM8K 95.5→89.0 (Table 3) [M2-10]; ACE 57.1 < 63.7 (Fig. 2) [M-27]; Xiong add-all 4/4 agents worse (Table 1) [M-26]"),
        (L("成功与失败记录训练无差别", "Successful = failed records", lang),
         L("第一代 − 安慰剂 = +0.004（p = 0.91）", "gen 1 − placebo = +0.004 (p = 0.91)", lang),
         L("Spurious Rewards、Mind the Gap", "Spurious Rewards, Mind the Gap", lang),
         L("随机奖励也能提升；部分提升来自格式", "random rewards also help; part of the gain is format", lang),
         L("一致 · 新对照", "consistent · new control", lang), True,
         "Spurious Rewards: random +21.4 vs true +29.1 pts, Qwen2.5-Math-7B MATH-500 [M2-07]; Mind the Gap §5: format-following [M2-01]"),
        (L("三代没有加速，第三代变差", "No speed-up; gen 3 worse", lang),
         L("第三代 − 第二代 = −0.087（p = 0.0625）", "gen 3 − gen 2 = −0.087 (p = 0.0625)", lang),
         L("SPIN、ReST-EM、Shumailov、Gerstgrasser", "SPIN, ReST-EM, Shumailov, Gerstgrasser", lang),
         L("几轮内饱和；累积数据可避免坍缩", "saturate in a few rounds; accumulation avoids collapse", lang),
         L("比文献更早出现", "earlier than in lit.", lang), True,
         "SPIN 58.14→63.16, shrinking gains (Fig. 2) [M-20]; ReST-EM APPS regresses after round 1 [M-17]; Shumailov tails vanish [M2-04]; Gerstgrasser accumulate avoids collapse [M2-05]"),
        (L("模型自定配方没有显著提升", "Model-set recipes: no gain", lang),
         L("0.605 / 0.595，原始模型 0.588", "0.605 / 0.595 vs original 0.588", lang),
         "SEAL",
         L("自我编辑 20% → 72.5%，靠外层强化学习训练", "self-edits 20% → 72.5%, with outer-loop RL training", lang),
         L("一致 · 缺外层训练", "consistent · no outer RL", lang), True,
         "SEAL ARC subset: 72.5% vs 20% (untrained self-edit) vs 0% (§4) [M-13]"),
        (L("改进过的模型设计下一次更新，结果更差", "Improved model, worse next update", lang),
         L(f"第三轮 {DEC['dec_sd3']['main']['mean_a']:.3f}，显著低于第二轮", f"round 3 {DEC['dec_sd3']['main']['mean_a']:.3f}, significantly below round 2", lang),
         "STOP、HGM" if lang == "zh" else "STOP, HGM",
         L("弱模型改写自己会退化；当前得分与产出后代的能力弱相关", "weak models degrade; score ≠ ability to improve", lang),
         L("一致", "consistent", lang), True,
         "STOP: GPT-3.5 and Mixtral degrade (Fig. 4) [M-01]; HGM weighted Pearson 0.285–0.444 (Table 1) [M-04]"),
    ]
    rh, top = 66, 54
    s = Fig(top + len(rows) * rh + 14)
    s.arrowheads(AXIS)
    cx = [16, 284, 592]
    s.text(cx[0], 30, L("第 4 节的结果", "Result in §4", lang), size=11, fill=MUTED)
    s.text(cx[1], 30, L("最接近的文献", "Closest literature", lang), size=11, fill=MUTED)
    s.text(cx[2] + 52, 30, L("对照", "Agreement", lang), size=11, fill=MUTED, anchor="middle")
    for i, (res, num, lit, litnote, verdict, full, tip) in enumerate(rows):
        y = top + i * rh
        s.box(cx[0], y, 250, rh - 12, fill=BLUE_T, stroke="none", rx=10, cls="pop", style=f"--d:{.1 + .12 * i:.2f}s")
        s.text(cx[0] + 12, y + 21, res, size=11.5, fill=INK, weight=600)
        s.text(cx[0] + 12, y + 40, num, size=10.5, fill=INK2)
        s.path(f"M{cx[0] + 254},{y + (rh - 12) / 2} L{cx[1] - 6},{y + (rh - 12) / 2}", stroke=AXIS, w=1.4, cls="flow")
        s.box(cx[1], y, 292, rh - 12, fill=PAPER, stroke=GRID, rx=10, tip=tip, cls="pop", style=f"--d:{.2 + .12 * i:.2f}s")
        s.text(cx[1] + 12, y + 21, lit, size=11.5, fill=INK, weight=600)
        s.text(cx[1] + 12, y + 40, litnote, size=10, fill=INK2)
        s.path(f"M{cx[1] + 296},{y + (rh - 12) / 2} L{cx[2] - 6},{y + (rh - 12) / 2}", stroke=AXIS, w=1.4, cls="flow")
        pw = 112
        s.box(cx[2], y + 12, pw, 30, fill=BLUE if full else PAPER, stroke=BLUE, rx=15, cls="pop", style=f"--d:{.35 + .12 * i:.2f}s")
        s.text(cx[2] + pw / 2, y + 31, verdict, size=10.5 if len(verdict) > 12 else 11, fill="#ffffff" if full else BLUE, anchor="middle", weight=600)
    s.save("fig5-map", lang, L("第 4 节的结果与文献的对应关系", "Results of §4 against the literature", lang))


# ------------------------------------------------------------------ fig5-memory
def fig_memory(lang):
    rel = lambda a, b: (b - a) / a * 100
    rows = [  # (label, condition, [relative changes %], tip)
        (L("本报告：经验笔记", "This report: lessons note", lang), L("自由文本 · 无筛选 · 迁移到新任务", "free text · unfiltered · new tasks", lang),
         [rel(M0, US)], f"{M0:.3f} → {US:.3f} [decisions_1235.json: dec_us]"),
        (L("ExpeL：加入自我反思", "ExpeL: with self-reflection", lang), L("HotpotQA 成功率", "HotpotQA success", lang),
         [rel(39.0, 29.0)], "39.0 → 29.0 (Table 3) [M-25]"),
        (L("Xiong：不加筛选地累积记忆", "Xiong: add every memory", lang), L("四个 agent", "four agents", lang),
         [rel(67.53, 55.48), rel(16.75, 13.05), rel(40.11, 32.32), rel(71.50, 59.90)],
         "67.53→55.48, 16.75→13.05, 40.11→32.32, 71.50→59.90 (Table 1) [M-26]"),
        (L("ACE：整体重写上下文", "ACE: whole-context rewrite", lang), L("相对不做适应的基线", "vs no-adaptation baseline", lang),
         [rel(63.7, 57.1)], "63.7 → 57.1 (§2.2, Fig. 2) [M-27]"),
        (L("自我纠错（无正确答案）", "Self-correction (no oracle)", lang), "GPT-4 · GSM8K",
         [rel(95.5, 89.0)], "95.5 → 89.0 (Table 3) [M2-10]"),
        (L("Reflexion", "Reflexion", lang), "MBPP Python",
         [rel(0.80, 0.77)], "0.80 → 0.77 (Table 2) [M-23]"),
        (L("Sarukkai：筛选后的成功轨迹", "Sarukkai: filtered successes", lang), L("ALFWorld · 按相似度检索", "ALFWorld · similarity retrieval", lang),
         [rel(73, 89)], "73% → 89% (abstract) [M-28]"),
    ]
    rh, top = 38, 50
    s = Fig(top + len(rows) * rh + 44)
    x0, xw = 300, 400
    lo, hi = -40, 30
    X = lambda v: x0 + (v - lo) / (hi - lo) * xw
    for v in (-40, -30, -20, -10, 0, 10, 20, 30):
        s.line(X(v), top - 8, X(v), top + len(rows) * rh - 6, AXIS if v == 0 else "#f0efea", 1.4 if v == 0 else 1)
        s.text(X(v), top + len(rows) * rh + 12, f"{v:+d}%" if v else "0", size=10, fill=MUTED, anchor="middle")
    s.text(X(-20), top + len(rows) * rh + 32, L("← 变差", "← worse", lang), size=11, fill=ORANGE, anchor="middle")
    s.text(X(15), top + len(rows) * rh + 32, L("变好 →", "better →", lang), size=11, fill=BLUE, anchor="middle")
    s.text(16, 24, L("加入反思、经验或记忆之后的相对变化（各研究指标不同，只比较方向和幅度）",
                     "Relative change after adding reflection, experience or memory (metrics differ; compare direction and size only)", lang), size=11, fill=MUTED)
    for i, (lab, cond, vals, tip) in enumerate(rows):
        y = top + i * rh + 10
        ours = i == 0
        s.text(16, y + 1, lab, size=11.5, fill=INK, weight=600 if ours else 400)
        s.text(16, y + 15, cond, size=9.8, fill=MUTED)
        mean = sum(vals) / len(vals)
        col = ORANGE if mean < 0 else BLUE
        a, b = sorted((X(0), X(mean)))
        s.box(a, y - 8, b - a, 14, fill=col if ours or mean > 0 else ORANGE_T, stroke=col if not ours else "none", rx=4, sw=1,
              cls="grow" if mean > 0 else "", tip=tip)
        if len(vals) > 1:
            for v in vals:
                s.circle(X(v), y - 1, 4, ORANGE, stroke="#ffffff", sw=1.2, tip=tip)
        lab_v = (f"{min(vals):+.1f}% … {max(vals):+.1f}%" if len(vals) > 1 else f"{vals[0]:+.1f}%")
        lx = X(max(vals)) + 8 if mean > 0 else X(min(vals)) - 8
        s.text(lx, y + 3, lab_v, size=10.5, fill=INK2, anchor="start" if mean > 0 else "end")
    s.save("fig5-memory", lang, L("加入经验之后的相对变化", "Relative change after adding experience", lang))


# ------------------------------------------------------------------ fig5-iter
def fig_iter(lang):
    panels = [
        ("SPIN", L("Open LLM 平均分", "Open LLM average", lang), [58.14, 60.80, 62.12, 62.97, 63.16], ["0", "1", "2", "3", "4"],
         "Fig. 2 [M-20]", BLUE, None),
        ("Meta-Rewarding", L("长度控制胜率 %", "LC win rate %", lang), [22.92, 27.85, 32.66, 35.45, 39.44], ["0", "1", "2", "3", "4"],
         "Table 1 [M-19]", BLUE, None),
        ("Self-Rewarding", L("AlpacaEval 2 胜率 %", "AlpacaEval 2 win %", lang), [9.94, 15.38, 20.44], ["M1", "M2", "M3"],
         "Table 1 [M-18]", BLUE, L("同期 GSM8K：60.27 → 57.70", "GSM8K meanwhile: 60.27 → 57.70", lang)),
        (L("本报告", "This report", lang), L("未见过任务上的改进速度分", "score on unseen tasks", lang), G,
         [L("原始", "orig.", lang), "1", "2", "3"], "decisions_1235.json", ORANGE, L("虚线：原始模型", "dashed: original model", lang)),
    ]
    gx, x0 = 16, 22
    pw = (W - x0 - 14 - 3 * gx) / 4
    ph, y0 = 150, 70
    s = Fig(y0 + ph + 80)
    s.text(16, 22, L("固定循环下自我训练的逐轮结果（横轴：轮次 / 代）", "Fixed-loop self-training, round by round (x: round / generation)", lang), size=11, fill=MUTED)
    for i, (name, metric, vals, xl, src, col, note) in enumerate(panels):
        px = x0 + i * (pw + gx)
        s.text(px, y0 - 26, name, size=12, fill=INK, weight=600)
        s.text(px, y0 - 10, metric, size=10, fill=INK2)
        s.box(px, y0, pw, ph, fill=PAPER, stroke=GRID, rx=6)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25 or 0.05
        lo, hi = lo - pad, hi + pad
        X = lambda k: px + 16 + k / (len(vals) - 1) * (pw - 32)
        Y = lambda v: y0 + 12 + (hi - v) / (hi - lo) * (ph - 24)
        if i == 3:
            s.line(px + 4, Y(vals[0]), px + pw - 4, Y(vals[0]), MUTED, 1, "4 3")
        pts = [(X(k), Y(v)) for k, v in enumerate(vals)]
        ln = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1)) + 1
        s.path("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts), stroke=col, w=2, marker=False, cls="draw", style=f"--len:{ln:.0f};--d:{.2 * i:.1f}s")
        for k, (a, b) in enumerate(pts):
            fmt = f"{vals[k]:.3f}" if i == 3 else f"{vals[k]:.2f}"
            s.circle(a, b, 4, col, stroke="#ffffff", sw=1.5, tip=f"{name} · {xl[k]}: {fmt} [{src}]")
            fmt = f"{vals[k]:.3f}" if i == 3 else f"{vals[k]:.1f}"
            if k in (0, len(vals) - 1) or i == 3:
                low = (k == 0 or vals[k] < vals[k - 1]) and (k == len(vals) - 1 or vals[k] < vals[k + 1])
                s.text(a + (8 if k == 0 and i != 3 else 0), b + 16 if low else b - 9, fmt, size=9.5, fill=INK2, anchor="start" if k == 0 and i != 3 else "middle")
            s.text(a, y0 + ph + 14, xl[k], size=9.5, fill=MUTED, anchor="middle")
        if note:
            s.text(px, y0 + ph + 32, note, size=9.8, fill=MUTED)
    s.text(16, y0 + ph + 54, L("文献中的曲线在被优化的指标上上升、增量递减；", "Published curves rise on the optimized metric, with shrinking gains;", lang), size=10.3, fill=MUTED)
    s.text(16, y0 + ph + 70, L("本报告测的是当改进者的速度，三代都没有超过原始模型。", "this report measures speed as an improver, and no generation beats the original.", lang), size=10.3, fill=MUTED)
    s.save("fig5-iter", lang, L("固定循环下自我训练的逐轮结果", "Fixed-loop self-training, round by round", lang))


if __name__ == "__main__":
    for lang in ("zh", "en"):
        fig_map(lang); fig_memory(lang); fig_iter(lang)
    print("wrote fig5-{map,memory,iter}.{zh,en}.svg")
