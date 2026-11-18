"""Overview figure for the abstract: does swapping in a successor improve faster?

Forest plot of each self-update approach vs the original model (Qwen3.5-9B): point = difference
in improvement-speed score (nAUC), whisker = 95% CI, grey band = A/A noise (±2 SD across tasks).
All values are read from the newest research/decisions/decisions_*.json; nothing is typed by hand.
Output: report/site/public/figs/fig0-forest.{zh,en}.svg
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import SVG, W, BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, legend  # noqa: E402

REPORT = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
DEC_FILE = sorted((ROOT / "research/decisions").glob("decisions_*.json"))[-1]
DEC = json.loads(DEC_FILE.read_text())


def vs_m0(key):
    d = DEC[key]
    if d["main"].get("mean_b") and abs(d["main"]["mean_b"] - 0.5883) < 1e-3:   # main comparison is already vs M0
        m = d["main"]
        return m["delta"], m["ci"], m["p_signflip"], key + ".main"
    for a in d.get("also", []):
        if a["name"].endswith("vs M0"):
            return a["delta"], a["ci"], a["p_signflip"], f"{key}.also[{a['name']}]"
    raise KeyError(key)


def render(lang):
    zh = lang == "zh"
    groups = [
        ("用自己的成功记录微调" if zh else "Fine-tuned on its own successes", [
            ("第一代后继模型" if zh else "Generation 1", "dec_g1"),
            ("第二代后继模型" if zh else "Generation 2", "dec_g2"),
            ("第三代后继模型" if zh else "Generation 3", "dec_g3")]),
        ("让模型写经验笔记（权重不变）" if zh else "Model writes lessons (weights fixed)", [
            ("经验笔记" if zh else "Lessons note", "dec_us")]),
        ("让模型自己决定怎么训练自己" if zh else "Model designs its own training", [
            ("第一轮" if zh else "Round 1", "dec_sd1"),
            ("第二轮（看过第一轮的反馈）" if zh else "Round 2 (after feedback)", "dec_sd2")]
            + ([("第三轮（由第一轮微调后的模型来定配方）" if zh else "Round 3 (recipe set by the round-1 model)", "dec_sd3")] if "dec_sd3" in DEC else [])),
    ]
    aa = DEC["dec_g1"]["aa"]["floor_2sd_task"]
    base = DEC["dec_g1"]["main"]["mean_b"]
    lo_v, hi_v = -0.40, 0.20
    x0, xw = 300, 390
    sx = lambda v: x0 + (v - lo_v) / (hi_v - lo_v) * xw
    rows = sum(len(g[1]) for g in groups)
    top = 56
    rh = 30
    gap = 18
    H = top + rows * rh + len(groups) * gap + 70
    s = SVG(H)
    legend(s, 16, 22, [("差值（点）与 95% 置信区间（横线）" if zh else "difference (dot) with 95% CI (line)", BLUE, "dot"),
                       ("置信区间不含 0" if zh else "CI excludes 0", ORANGE, "dot")])
    yb = H - 52
    # A/A noise band and zero line
    s.rect(sx(-aa), top - 8, sx(aa) - sx(-aa), yb - top + 8, "#efede7", rx=0,
           tip=(f"同一个模型跑两遍的随机波动：±{aa:.3f}（2 个标准差）[{DEC_FILE.name}: dec_g1.aa]" if zh else
                f"Noise when the same model runs twice: ±{aa:.3f} (2 SD) [{DEC_FILE.name}: dec_g1.aa]"))
    for v in (-0.4, -0.3, -0.2, -0.1, 0.1, 0.2):
        s.line(sx(v), top - 8, sx(v), yb, GRID)
    s.line(sx(0), top - 12, sx(0), yb, AXIS, 1.5)
    for v in (-0.4, -0.2, 0.0, 0.2):
        s.text(sx(v), yb + 16, f"{v:+.1f}" if v else "0", size=11, fill=MUTED, anchor="middle")
    s.text(sx(0), top - 16, "和原始模型一样快" if zh else "as fast as the original", size=11, fill=MUTED, anchor="middle")
    s.text(sx(-0.22), yb + 34, "← 比原始模型慢" if zh else "← slower than the original", size=11, fill=INK2, anchor="middle")
    s.text(sx(0.12), yb + 34, "比原始模型快 →" if zh else "faster →", size=11, fill=INK2, anchor="middle")
    s.text(sx(aa) + 4, yb - 6, "随机波动范围" if zh else "noise band", size=10.5, fill=MUTED)
    y = top
    for gname, items in groups:
        s.text(16, y + 4, gname, size=12, fill=INK, weight=600)
        y += gap
        for label, key in items:
            d, ci, p, src = vs_m0(key)
            sig = ci[0] > 0 or ci[1] < 0
            col = ORANGE if sig else BLUE
            tip = (f"{label}：相对原始模型 {d:+.3f}，95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]，置换检验 p = {p:.2f} [{DEC_FILE.name}: {src}]" if zh else
                   f"{label}: {d:+.3f} vs original, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], permutation p = {p:.2f} [{DEC_FILE.name}: {src}]")
            s.text(32, y + 4, label, size=12, fill=INK2)
            s.line(sx(max(ci[0], lo_v)), y, sx(min(ci[1], hi_v)), y, col, 2.5)
            s.circle(sx(d), y, 6, col, stroke="#ffffff", tip=tip)
            s.raw(f'<rect x="{sx(ci[0]):.1f}" y="{y - 9:.1f}" width="{sx(ci[1]) - sx(ci[0]):.1f}" height="18" fill="transparent" class="has-tip-hit"><title>{tip}</title></rect>')
            s.text(W - 12, y + 4, f"{d:+.3f}", size=11, fill=INK2, anchor="end")
            y += rh
        y += gap - 18
    s.text(16, H - 8, (f"原始模型的改进速度分为 {base:.3f}；8 个未见过的小型研究任务 × 5 个种子" if zh else
                       f"Original model's improvement-speed score: {base:.3f}; 8 unseen micro-research tasks × 5 seeds"), size=10.5, fill=MUTED)
    s.save("fig0-forest", lang, "换上后继模型之后的改进速度" if zh else "Speed of improvement after swapping in a successor")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        render(lang)
    for k in [k for k in ("dec_g1", "dec_g2", "dec_g3", "dec_us", "dec_sd1", "dec_sd2", "dec_sd3") if k in DEC]:
        print(k, [round(x, 3) if isinstance(x, float) else x for x in (vs_m0(k)[0], *vs_m0(k)[1], vs_m0(k)[2])], vs_m0(k)[3])
    print("A/A ±", round(DEC["dec_g1"]["aa"]["floor_2sd_task"], 3), "| file", DEC_FILE.name)
