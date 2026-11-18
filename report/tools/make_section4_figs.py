"""Figures 8 and 9 for report §4 (inline SVG, zh + en, hover tooltips via <title>).

fig4-scores : improvement-speed score (nAUC) on the 8 held-out micro-research tasks for every improver
              produced by a self-update approach, next to the score of its own control arm
              (placebo / "more data" / random recipe). The control's score is derived as
              mean_a - vs_placebo.delta (paired over the same 8 tasks). Dashed line = original model.
fig4-pertask: per-task difference between each improver and its own control (vs_placebo.per_task).

All values are read from research/decisions/decisions_1235.json; nothing is typed by hand.
Output: report/site/public/figs/fig4-scores.{zh,en}.svg, fig4-pertask.{zh,en}.svg
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import SVG, W, BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, legend  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEC_NAME = "decisions_1235.json"
DEC = json.loads((ROOT / "research/decisions" / DEC_NAME).read_text())
SURFACE = "#fbfaf7"


def rows(zh):
    """(group, label, decision key, control label)."""
    return [
        ("经验笔记（权重不变）" if zh else "Lessons note (weights fixed)", [
            ("经验笔记" if zh else "Lessons note", "dec_us", "同长度的通用建议" if zh else "generic tips of equal length")]),
        ("用自己的成功记录微调" if zh else "Fine-tuned on its own successes", [
            ("第一代后继模型" if zh else "Generation 1", "dec_g1", "安慰剂模型" if zh else "placebo model"),
            ("第二代后继模型" if zh else "Generation 2", "dec_g2", "“多给数据”对照模型" if zh else "\"more data\" control"),
            ("第三代后继模型" if zh else "Generation 3", "dec_g3", "“多给数据”对照模型" if zh else "\"more data\" control")]),
        ("让模型自己决定怎么训练自己" if zh else "Model designs its own training", [
            ("第一轮" if zh else "Round 1", "dec_sd1", "随机配方" if zh else "random recipe"),
            ("第二轮" if zh else "Round 2", "dec_sd2", "随机配方" if zh else "random recipe"),
            ("第三轮" if zh else "Round 3", "dec_sd3", "随机配方" if zh else "random recipe")]),
    ]


def scores(key):
    d = DEC[key]
    a = d["main"]["mean_a"]
    return a, a - d["vs_placebo"]["delta"]


ORIGINAL = DEC["dec_g1"]["main"]["mean_b"]


# ------------------------------------------------------------------ Figure 8: scores
def fig_scores(lang):
    zh = lang == "zh"
    groups = rows(zh)
    lo_v, hi_v = 0.30, 0.70
    x0, xw = 250, 370
    sx = lambda v: x0 + (v - lo_v) / (hi_v - lo_v) * xw
    top, rh, gap = 64, 30, 18
    n = sum(len(g[1]) for g in groups)
    H = top + n * rh + len(groups) * gap + 52
    s = SVG(H)
    legend(s, 16, 22, [("该做法得到的改进者" if zh else "improver produced by the approach", BLUE, "dot"),
                       ("它的对照" if zh else "its control", ORANGE, "dot")])
    yb = H - 40
    for v in (0.3, 0.4, 0.5, 0.6, 0.7):
        s.line(sx(v), top - 10, sx(v), yb, GRID)
        s.text(sx(v), yb + 16, f"{v:.1f}", size=11, fill=MUTED, anchor="middle")
    s.line(sx(ORIGINAL), top - 14, sx(ORIGINAL), yb, INK2, 1.5, dash="4 3")
    s.text(sx(ORIGINAL), top - 20, (f"原始模型 {ORIGINAL:.3f}" if zh else f"original model {ORIGINAL:.3f}"), size=11, fill=INK2, anchor="middle")
    s.text(x0 + xw / 2, yb + 32, "改进速度分（0–1，越高越好）" if zh else "improvement-speed score (0–1, higher is better)", size=11, fill=INK2, anchor="middle")
    s.text(W - 12, top - 20, "改进者 / 对照" if zh else "improver / control", size=10.5, fill=MUTED, anchor="end")
    y = top
    for gname, items in groups:
        s.text(16, y + 4, gname, size=12, fill=INK, weight=600)
        y += gap
        for label, key, clabel in items:
            a, c = scores(key)
            s.text(32, y + 4, label, size=12, fill=INK2)
            s.line(sx(min(a, c)), y, sx(max(a, c)), y, AXIS, 2)
            tip_c = (f"{label}的对照（{clabel}）：{c:.3f}，由差值推出 [{DEC_NAME}: {key}.main.mean_a − {key}.vs_placebo.delta]" if zh else
                     f"Control for {label} ({clabel}): {c:.3f}, derived from the difference [{DEC_NAME}: {key}.main.mean_a − {key}.vs_placebo.delta]")
            tip_a = (f"{label}：{a:.3f}；原始模型 {ORIGINAL:.3f} [{DEC_NAME}: {key}.main.mean_a]" if zh else
                     f"{label}: {a:.3f}; original model {ORIGINAL:.3f} [{DEC_NAME}: {key}.main.mean_a]")
            s.circle(sx(c), y, 5.5, ORANGE, stroke="#ffffff", tip=tip_c)
            s.circle(sx(a), y, 6.5, BLUE, stroke="#ffffff", tip=tip_a)
            s.text(W - 12, y + 4, f"{a:.3f} / {c:.3f}", size=11, fill=INK2, anchor="end")
            y += rh
        y += gap - 18
    s.save("fig4-scores", lang, "各改进者的改进速度分" if zh else "Improvement-speed score of each improver")


# ------------------------------------------------------------------ Figure 9: per-task differences vs control
TASKS = ["id-modadd-0", "id-parity-0", "id-teacher-0", "id-cls2d-0", "ood-copyrev-0", "ood-dyck-0", "ood-icl-0", "ood-charlm-0"]
TASK_ZH = ["模加法", "稀疏奇偶", "师生回归", "二维分类", "序列反转", "括号匹配", "上下文回归", "字符级语言模型"]
TASK_EN = ["mod. add.", "parity", "teacher", "2-D class.", "copy/rev.", "Dyck", "in-context", "char LM"]


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def fig_pertask(lang):
    zh = lang == "zh"
    names = TASK_ZH if zh else TASK_EN
    flat = [(label, key, clabel) for _, items in rows(zh) for label, key, clabel in items]
    x0, cw, ch, top = 232, 54, 30, 74
    xm = x0 + 8 * cw + 8
    H = top + len(flat) * ch + 56
    s = SVG(H)
    s.text(x0 + 2 * cw, 20, "同类任务（新参数）" if zh else "same family, new settings", size=11.5, fill=INK, anchor="middle", weight=600)
    s.text(x0 + 6 * cw, 20, "不同类任务" if zh else "different kind", size=11.5, fill=INK, anchor="middle", weight=600)
    s.line(x0 + 4 * cw, 8, x0 + 4 * cw, top + len(flat) * ch, AXIS, 1.5)
    for i, nm in enumerate(names):
        cx = x0 + i * cw + cw / 2
        parts = [nm] if len(nm) <= 5 or not zh else [nm[:3], nm[3:]]
        if not zh and " " in nm and len(nm) > 8:
            parts = nm.split(" ", 1)
        for k, ptxt in enumerate(parts):
            s.text(cx, 42 + k * 14 - (7 if len(parts) > 1 else 0) + 10, ptxt, size=10.5, fill=INK2, anchor="middle")
    s.text(xm + 28, 52, "平均" if zh else "mean", size=10.5, fill=INK2, anchor="middle", weight=600)
    for r, (label, key, clabel) in enumerate(flat):
        y = top + r * ch
        pt = DEC[key]["vs_placebo"]["per_task"]
        s.text(16, y + ch / 2 + 4, f"{label} − {clabel}", size=11, fill=INK2)
        for i, t in enumerate(TASKS):
            v = pt["muar/" + t]
            col = mix(SURFACE, BLUE if v > 0 else ORANGE, min(abs(v) / 0.6, 1.0))
            tip = (f"{label} 相对 {clabel}，{names[i]}：{v:+.3f} [{DEC_NAME}: {key}.vs_placebo.per_task]" if zh else
                   f"{label} vs {clabel}, {names[i]}: {v:+.3f} [{DEC_NAME}: {key}.vs_placebo.per_task]")
            s.rect(x0 + i * cw + 1, y + 1, cw - 2, ch - 2, col, tip=tip, rx=2)
            s.text(x0 + i * cw + cw / 2, y + ch / 2 + 4, "0.00" if abs(v) < 0.005 else f"{v:+.2f}", size=10.5,
                   fill="#ffffff" if abs(v) > 0.4 else INK, anchor="middle")
        d = DEC[key]["vs_placebo"]
        s.text(xm + 28, y + ch / 2 + 4, f"{d['delta']:+.3f}", size=11, fill=INK, anchor="middle", weight=600)
    yb = top + len(flat) * ch + 22
    # colour key
    for k in range(13):
        v = -0.6 + k * 0.1
        s.rect(x0 + k * 24, yb, 24, 10, mix(SURFACE, BLUE if v > 0 else ORANGE, min(abs(v) / 0.6, 1.0)), rx=0)
    s.text(x0 - 6, yb + 9, "−0.6", size=10, fill=MUTED, anchor="end")
    s.text(x0 + 13 * 24 + 6, yb + 9, "+0.6", size=10, fill=MUTED)
    s.text(x0 + 13 * 24 + 40, yb + 9, ("改进者减去对照" if zh else "improver minus control"), size=10.5, fill=INK2)
    s.save("fig4-pertask", lang, "逐任务差值" if zh else "Per-task differences")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        fig_scores(lang)
        fig_pertask(lang)
    for _, items in rows(True):
        for label, key, clabel in items:
            a, c = scores(key)
            print(f"{label:10s} {a:.3f}  control {c:.3f}  vs control {DEC[key]['vs_placebo']['delta']:+.3f}")
    print("original", round(ORIGINAL, 3), "| file", DEC_NAME)
