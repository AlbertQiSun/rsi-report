"""Figure 21 for report §6: Ling's self-improvement pipeline (conceptual flow diagram, no data).

Output: report/site/public/figs/fig6-pipeline.{zh,en}.svg
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL  # noqa: E402
from make_section4_flow import Fig, L, tw, PAPER, BLUE_T, ORANGE_T  # noqa: E402


def fig_pipeline(lang):
    s = Fig(440)
    s.arrowheads(AXIS, BLUE, ORANGE, MUTED)

    def node(x, y, w, h, title, sub=None, fill=PAPER, stroke=AXIS, tsize=12, sub2=None, tip=None):
        s.box(x, y, w, h, fill=fill, stroke=stroke, rx=10, tip=tip)
        cy = y + h / 2 - (8 if sub else 0) - (7 if sub2 else 0)
        s.text(x + w / 2, cy + 4, title, size=tsize, fill=INK, anchor="middle", weight=600)
        if sub:
            s.text(x + w / 2, cy + 21, sub, size=10.2, fill=INK2, anchor="middle")
        if sub2:
            s.text(x + w / 2, cy + 36, sub2, size=10.2, fill=INK2, anchor="middle")

    # research environment pool -> Ling (generation n)
    node(20, 62, 140, 46, L("研发环境池", "R&D environment pool", lang), L("kernel · 配方 · 设计更新", "kernels · recipes · designs", lang), fill="#f6f5f1", stroke="none")
    node(20, 150, 140, 84, L("Ling 第 n 代", "Ling, generation n", lang), L("研究 agent", "research agent", lang), fill=BLUE_T, stroke=BLUE, tsize=13)
    s.path("M90,110 L90,146", stroke=AXIS, w=1.4, cls="flow")

    # mechanism repository + trajectory store
    node(204, 74, 172, 92, L("机制仓库", "Mechanism repository", lang), L("数据规则 · 训练配方", "data rules · training recipes", lang),
         sub2=L("环境 · 评测 · 训练代码", "envs · evals · training code", lang), fill=PAPER, stroke=INK2)
    node(204, 196, 172, 58, L("轨迹库", "Trajectory store", lang), L("成功 · 失败 · 非法，全部保留", "successes · failures · invalid", lang), fill=PAPER, stroke=AXIS)
    s.path("M162,172 C182,172 182,120 200,120", stroke=BLUE, w=1.5, cls="flow")
    s.text(182, 140, L("提交改动", "commits", lang), size=9.8, fill=BLUE, anchor="middle")
    s.path("M162,212 L200,224", stroke=AXIS, w=1.4, cls="flow")

    # small-model testbed with a paired experiment
    tx, ty, tw_, th = 412, 56, 186, 214
    s.box(tx, ty, tw_, th, fill="#f6f5f1", stroke="none", rx=12)
    s.text(tx + 14, ty + 22, L("小模型试验场", "Small-model testbed", lang), size=12.5, fill=INK, weight=600)
    s.text(tx + 14, ty + 38, L("每个改动都成对实测", "every change tested in pairs", lang), size=10.2, fill=INK2)
    s.box(tx + 14, ty + 52, 74, 34, fill=PAPER, stroke=BLUE, rx=8)
    s.text(tx + 51, ty + 73, L("有这条改动", "with change", lang), size=10.2, fill=BLUE, anchor="middle")
    s.box(tx + 98, ty + 52, 74, 34, fill=PAPER, stroke=AXIS, rx=8)
    s.text(tx + 135, ty + 73, L("没有这条改动", "without", lang), size=10.2, fill=INK2, anchor="middle")
    s.path(f"M{tx + 51},{ty + 88} L{tx + 51},{ty + 108}", stroke=AXIS, w=1.2, cls="flow")
    s.path(f"M{tx + 135},{ty + 88} L{tx + 135},{ty + 108}", stroke=AXIS, w=1.2, cls="flow")
    s.box(tx + 14, ty + 112, 158, 42, fill=PAPER, stroke=AXIS, rx=8)
    s.text(tx + 93, ty + 130, L("后继换上去当改进者", "successors act as improvers", lang), size=10.2, fill=INK2, anchor="middle")
    s.text(tx + 93, ty + 146, L("比较改进速度", "compare improvement speed", lang), size=10.2, fill=INK, anchor="middle", weight=600)
    # tiny paired bars
    s.box(tx + 30, ty + 168, 96, 8, fill=BLUE, stroke="none", rx=3, cls="grow", style="--d:.3s")
    s.box(tx + 30, ty + 182, 64, 8, fill=NEUTRAL, stroke="none", rx=3, cls="grow", style="--d:.5s")
    s.chip(tx + 14, ty + 206, L("+ 失败记录训练的对照", "+ failure-trained control", lang), fill=ORANGE_T, ink=INK2, size=9.8)
    s.path("M378,118 L408,118", stroke=AXIS, w=1.4, cls="flow")
    s.path("M378,226 C394,226 394,200 408,200", stroke=AXIS, w=1.4, cls="flow")

    # sealed evaluator
    node(tx, ty + th + 12, tw_, 34, L("封存评估器打分", "Sealed evaluator scores", lang), fill=PAPER, stroke=AXIS, tsize=11)
    s.path(f"M{tx + tw_ / 2},{ty + th + 10} L{tx + tw_ / 2},{ty + th - 2}", stroke=AXIS, w=1.2, marker=False)

    # update engine -> next generation
    node(624, 84, 86, 62, L("更新引擎", "Update engine", lang), L("胜出的配方", "winning recipe", lang), fill=PAPER, stroke=INK2, tsize=11.5)
    node(624, 196, 86, 70, L("Ling", "Ling", lang), L("第 n+1 代", "gen n+1", lang), fill=BLUE_T, stroke=BLUE, tsize=13)
    s.path(f"M{tx + tw_ + 2},116 L620,116", stroke=AXIS, w=1.4, cls="flow")
    s.path("M667,148 L667,192", stroke=AXIS, w=1.4, cls="flow")

    # loop back: the next generation takes over the same repository
    loop = "M667,268 L667,330 Q667,338 659,338 L98,338 Q90,338 90,330 L90,238"
    s.path(loop, stroke=BLUE, w=1.8, cls="flow")
    s.text(380, 356, L("下一代接手同一个仓库，成为新的改进者", "the next generation takes over the same repository as the new improver", lang),
           size=10.8, fill=BLUE, anchor="middle", weight=600)
    s.add('<circle r="5" fill="#2a78d6" opacity=".9"><animateMotion dur="7s" repeatCount="indefinite" '
          'path="M160,172 C182,172 182,120 204,120 L408,118 L620,116 L667,116 L667,268 L667,338 L90,338 L90,234 Z"/></circle>')

    # reward arc: measured speed trains Ling's update-design skill
    s.add(f'<path d="M505,54 C505,14 230,10 150,148" fill="none" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 5" '
          f'marker-end="url(#ah-{ORANGE[1:]})" class="flow"/>')
    s.text(330, 22, L("奖励：后继当改进者的速度 → 训练 Ling 设计更新的能力", "reward: successor's speed as improver → trains Ling's update design", lang),
           size=10.5, fill=ORANGE, anchor="middle", weight=600)

    # lineage archive
    ay = 398
    s.text(20, ay + 4, L("谱系档案", "Lineage archive", lang), size=11.5, fill=INK, weight=600)
    xs = [130 + i * 50 for i in range(7)]
    for i, x in enumerate(xs):
        if i:
            s.line(xs[i - 1] + 7, ay, x - 7, ay, AXIS, 1.2)
        s.add(f'<circle cx="{x}" cy="{ay}" r="7" fill="{BLUE if i == len(xs) - 1 else BLUE_T}" stroke="{BLUE}" stroke-width="1.2" class="pop" style="--d:{.1 * i:.1f}s"/>')
    s.line(xs[2], ay + 7, xs[2] + 20, ay + 24, AXIS, 1.2)
    s.add(f'<circle cx="{xs[2] + 26}" cy="{ay + 26}" r="5" fill="{BLUE_T}" stroke="{BLUE}" stroke-width="1"/>')
    s.text(xs[-1] + 16, ay + 4, L("按实测的改进速度分叉、分配研究工作", "branch and assign work by measured speed", lang), size=10.2, fill=INK2)
    s.save("fig6-pipeline", lang, L("Ling 的自我改进管线", "Ling's self-improvement pipeline", lang))


if __name__ == "__main__":
    for lang in ("zh", "en"):
        fig_pipeline(lang)
    print("wrote fig6-pipeline.{zh,en}.svg")
