"""Flow diagrams and data figures for report §4 (inline SVG, zh + en, hover tooltips via <title>).

fig4-loop      : one attempt on the platform (improver -> sandbox -> sealed evaluator -> keep/discard -> feedback)
fig4-scoredef  : how the improvement-speed score is computed (schematic; the curve is NOT experiment data)
fig4-lineage   : where the training data of each successor generation and its control comes from
fig4-rounds    : the three self-designed training rounds (designer -> recipe -> LoRA -> score)
fig4-behavior  : call count vs median output length on the unseen tasks (heavily vs lightly trained models)
fig4-cf        : continuing from the best NanoChat version: best-so-far gain per attempt, 12 runs
fig4-audit     : output-cap truncation audit (larger-model control run vs the 9B groups)

Numbers: research/decisions/decisions_1235.json, research/decisions/cf_analysis.json,
research/cf/*/ledger.jsonl (per-attempt curves), and research/REPORT_LOG.md sections E/G (typed below with the
LOG line they come from). Geometry is computed here. Animations are CSS/SMIL inside the SVG and start when the
figure scrolls into view (.in is set by the page); prefers-reduced-motion disables them (global.css).
Output: report/site/public/figs/<id>.{zh,en}.svg
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import SVG, W, BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL, esc, legend  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEC_NAME = "decisions_1235.json"
DEC = json.loads((ROOT / "research/decisions" / DEC_NAME).read_text())
CF = json.loads((ROOT / "research/decisions/cf_analysis.json").read_text())
PAPER = "#ffffff"
BLUE_T, ORANGE_T, BLUE_M = "#e3eefb", "#fde8de", "#8fb8ea"

STYLE = """<style>
.f4 .flow{stroke-dasharray:5 6;}
.in .f4 .flow{animation:f4dash 1.1s linear infinite;}
@keyframes f4dash{to{stroke-dashoffset:-11}}
.f4 .draw{stroke-dasharray:var(--len);stroke-dashoffset:var(--len);}
.in .f4 .draw{animation:f4draw 2.4s cubic-bezier(.3,.7,.2,1) forwards;animation-delay:var(--d,0s);}
@keyframes f4draw{to{stroke-dashoffset:0}}
.f4 .pop{opacity:0;}
.in .f4 .pop{animation:f4pop .5s ease-out forwards;animation-delay:var(--d,0s);}
@keyframes f4pop{from{opacity:0}to{opacity:1}}
.f4 .grow{transform-box:fill-box;transform-origin:left center;transform:scaleX(0);}
.in .f4 .grow{animation:f4grow 1.2s cubic-bezier(.3,.7,.2,1) forwards;animation-delay:var(--d,0s);}
@keyframes f4grow{to{transform:scaleX(1)}}
.f4 .pulse{animation:f4pulse 2.4s ease-in-out infinite;}
@keyframes f4pulse{0%,100%{opacity:.55}50%{opacity:1}}
</style>"""


def L(zh, en, lang):
    return zh if lang == "zh" else en


def tw(s, size):
    """Approximate rendered text width."""
    return sum(size * (1.0 if ord(c) > 0x2e80 else 0.56) for c in s)


class Fig(SVG):
    def __init__(self, h):
        super().__init__(h)
        self.add(STYLE + '<g class="f4">')

    def box(self, x, y, w, h, fill=PAPER, stroke=AXIS, rx=10, tip=None, cls="", sw=1.2, style=""):
        t = f"<title>{esc(tip)}</title>" if tip else ""
        c = f' class="{cls}"' if cls else ""
        st = f' style="{style}"' if style else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{c}{st}>{t}</rect>')

    def path(self, d, stroke=AXIS, w=1.5, fill="none", cls="", style="", marker=True, tip=None):
        c = f' class="{cls}"' if cls else ""
        st = f' style="{style}"' if style else ""
        m = f' marker-end="url(#ah-{stroke[1:]})"' if marker else ""
        t = f"<title>{esc(tip)}</title>" if tip else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{c}{st}{m}>{t}</path>')

    def arrowheads(self, *colors):
        defs = "".join(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
                       f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>' for c in colors)
        self.add(f"<defs>{defs}</defs>")

    def chip(self, x, y, s, fill=BLUE_T, ink=INK2, size=11, anchor="start", tip=None):
        w = tw(s, size) + 14
        x0 = x - w / 2 if anchor == "middle" else x
        self.box(x0, y - size - 4, w, size + 9, fill=fill, stroke="none", rx=(size + 9) / 2, tip=tip)
        self.text(x0 + w / 2, y, s, size=size, fill=ink, anchor="middle")
        return w

    def save(self, name, lang, label):
        self.add("</g>")
        super().save(name, lang, label)


def dec(key, sub=None):
    d = DEC[key]
    if sub is None:
        return d["main"]
    if sub == "placebo":
        return d["vs_placebo"]
    return next(a for a in d["also"] if a["name"] == sub)


def score(key):
    return DEC[key]["main"]["mean_a"]


def ctrl_score(key):
    return DEC[key]["main"]["mean_a"] - DEC[key]["vs_placebo"]["delta"]


M0 = DEC["dec_g1"]["main"]["mean_b"]


# ------------------------------------------------------------------ fig4-loop
def fig_loop(lang):
    s = Fig(392)
    s.arrowheads(AXIS, BLUE, ORANGE, MUTED)
    src = "[platform_build_plan.md; platform_design.md §5.1; LOG E]"
    # nodes
    nodes = {
        "imp": (24, 70, 150, 74),
        "box": (216, 50, 214, 114),
        "eval": (472, 70, 128, 74),
        "keep": (620, 44, 84, 40),
        "drop": (620, 132, 84, 40),
    }
    x, y, w, h = nodes["imp"]
    s.box(x, y, w, h, fill=BLUE_T, stroke=BLUE, tip=L(f"改进者：Qwen3.5-9B 原版 BF16 权重，vLLM 部署在 A100 上 {src}",
                                                    f"Improver: Qwen3.5-9B, original BF16 weights, served by vLLM on an A100 {src}", lang))
    s.text(x + w / 2, y + 28, L("改进者", "Improver", lang), size=13, fill=INK, anchor="middle", weight=600)
    s.text(x + w / 2, y + 47, "Qwen3.5-9B", size=11.5, fill=INK2, anchor="middle")
    s.text(x + w / 2, y + 63, "vLLM · A100", size=10.5, fill=MUTED, anchor="middle")
    x, y, w, h = nodes["box"]
    s.box(x, y, w, h, fill="#f6f5f1", stroke=AXIS, rx=12,
          tip=L("隔离沙箱：12 项隔离检查全部通过 [LOG E]", "Sandbox: all 12 isolation checks passed [LOG E]", lang))
    s.text(x + 14, y + 22, L("隔离沙箱", "Sandbox", lang), size=12.5, fill=INK, weight=600)
    s.chip(x + w - 12 - (tw(L("12 项检查通过", "12 checks pass", lang), 10.5) + 14), y + 23, L("12 项检查通过", "12 checks pass", lang), fill=BLUE_T, size=10.5)
    for i, t in enumerate([L("断网", "no network", lang), L("看不到验证集与封存集", "no dev / sealed data", lang),
                           L("诱饵文件被访问即报警", "decoy files raise alarms", lang)]):
        s.text(x + 26, y + 46 + i * 18, t, size=11, fill=INK2)
        s.circle(x + 17, y + 42 + i * 18, 2.6, BLUE)
    s.box(x + 14, y + 90, w - 28, 16, fill=PAPER, stroke=GRID, rx=5)
    s.text(x + w / 2, y + 102, L("候选方案：训练脚本 / JSON 配置", "candidate: training script / JSON config", lang), size=10.5, fill=INK2, anchor="middle")
    x, y, w, h = nodes["eval"]
    s.box(x, y, w, h, fill=PAPER, stroke=INK2, tip=L("封存的评估器统一打分 [platform_design.md §5.1]", "Sealed evaluator scores every candidate [platform_design.md §5.1]", lang))
    s.text(x + w / 2, y + 30, L("封存评估器", "Sealed evaluator", lang), size=12.5, fill=INK, anchor="middle", weight=600)
    s.text(x + w / 2, y + 50, L("训练 + 打分", "train + score", lang), size=11, fill=MUTED, anchor="middle")
    x, y, w, h = nodes["keep"]
    s.box(x, y, w, h, fill=BLUE_T, stroke=BLUE, rx=20)
    s.text(x + w / 2, y + 25, L("保留", "keep", lang), size=12, fill=BLUE, anchor="middle", weight=600)
    x, y, w, h = nodes["drop"]
    s.box(x, y, w, h, fill=ORANGE_T, stroke=ORANGE, rx=20)
    s.text(x + w / 2, y + 25, L("丢弃", "discard", lang), size=12, fill=ORANGE, anchor="middle", weight=600)
    # arrows
    s.path("M176,107 L212,107", stroke=AXIS, w=1.6, cls="flow")
    s.path("M432,107 L468,107", stroke=AXIS, w=1.6, cls="flow")
    s.path("M602,98 C612,80 610,66 616,64", stroke=BLUE, w=1.6, cls="flow")
    s.path("M602,116 C612,134 610,148 616,150", stroke=ORANGE, w=1.6, cls="flow")
    s.text(662, 36, L("优于当前最好", "beats current best", lang), size=10.5, fill=BLUE, anchor="middle")
    s.text(650, 192, L("否则", "otherwise", lang), size=10.5, fill=ORANGE, anchor="end")
    # feedback path back to improver
    fb = "M662,174 C662,236 420,236 99,236 L99,148"
    s.path(fb, stroke=MUTED, w=1.5, cls="flow")
    s.chip(380, 241, L("反馈只有：验证分数 + 训练曲线", "feedback: dev score + training curve only", lang), fill="#f1efe9", size=11,
           anchor="middle", tip=L("改进者收到的反馈只有验证分数和训练曲线 [platform_design.md §5.1]",
                                  "The improver only receives the dev score and the training curve [platform_design.md §5.1]", lang))
    # travelling token
    s.add('<circle r="5" fill="#2a78d6" opacity=".9"><animateMotion dur="5.5s" repeatCount="indefinite" '
          'path="M99,107 L470,107 L540,107 C600,107 610,70 640,64 C700,60 662,120 662,174 C662,236 420,236 99,236 L99,148 Z"/></circle>')
    # hidden from the improver
    yb = 290
    s.text(24, yb, L("改进者看不到：", "Hidden from the improver:", lang), size=11.5, fill=INK, weight=600)
    xx = 24 + tw(L("改进者看不到：", "Hidden from the improver:", lang), 11.5) + 10
    for t in [L("封存集分数", "sealed scores", lang), L("评估器代码", "evaluator code", lang), L("参考最优值", "reference optimum", lang)]:
        w = s.chip(xx, yb, t, fill="#f1efe9", ink=MUTED, size=11)
        s.line(xx + 6, yb - 4, xx + w - 6, yb - 4, MUTED, 1)
        xx += w + 8
    # two task families
    yt = 330
    s.text(24, yt, L("两类任务", "Two task families", lang), size=11.5, fill=INK, weight=600)
    s.chip(110 if lang == "zh" else 140, yt, L("NanoChat：改训练脚本，保留前换种子复测", "NanoChat: edit the training script, re-test on a new seed before keeping", lang),
           fill=BLUE_T, size=11, tip=L("NanoChat 单次评测噪声：20 个种子均值 1.4007，标准差 0.0268 [LOG E]",
                                       "NanoChat single-eval noise: 20 seeds, mean 1.4007, SD 0.0268 [LOG E]", lang))
    s.chip(110 if lang == "zh" else 140, yt + 30, L("8 个小型研究任务：改约 12 个旋钮的 JSON，训练按步数封顶", "8 micro-research tasks: edit a ~12-knob JSON, training capped by steps", lang),
           fill=BLUE_T, size=11, tip=L("[platform_design.md §5.1；LOG H]", "[platform_design.md §5.1; LOG H]", lang))
    s.save("fig4-loop", lang, L("实验平台上的一次尝试", "One attempt on the platform", lang))


# ------------------------------------------------------------------ fig4-scoredef (schematic)
def fig_scoredef(lang):
    s = Fig(300)
    s.arrowheads(MUTED)
    x0, x1, ytop, ybot = 84, 420, 52, 238
    s0y, refy = 70, 226            # start score (high loss) and reference optimum (low loss); schematic only
    sx = lambda t: x0 + (t - 0.5) / 12 * (x1 - x0)
    # schematic best-so-far step curve (fractions of the gap closed; illustration only)
    frac = [0, 0, .18, .18, .30, .30, .30, .52, .52, .58, .58, .66, .66]
    pts = []
    for t in range(1, 13):
        y = s0y + frac[t] * (refy - s0y)
        pts.append((sx(t - .5), y)); pts.append((sx(t + .5), y))
    area = f"M{pts[0][0]:.1f},{s0y} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts) + f" L{pts[-1][0]:.1f},{s0y} Z"
    s.add(f'<path d="{area}" fill="{BLUE_T}" class="grow" style="--d:.2s"><title>{esc(L("阴影 = 每次尝试时已经缩小的差距；改进速度分 = 阴影面积 ÷ 整个矩形面积", "Shaded = gap already closed at each attempt; score = shaded area ÷ whole rectangle", lang))}</title></path>')
    s.box(x0, s0y, x1 - x0, refy - s0y, fill="none", stroke=GRID, rx=0)
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    ln = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    s.path(line, stroke=BLUE, w=2.2, marker=False, cls="draw", style=f"--len:{ln:.0f}")
    s.line(x0 - 6, s0y, x1 + 8, s0y, INK2, 1.2, "4 3")
    s.line(x0 - 6, refy, x1 + 8, refy, INK2, 1.2, "4 3")
    s.text(x1 + 12, s0y + 4, L("起点分数", "start score", lang), size=11, fill=INK2)
    s.text(x1 + 12, refy + 4, L("参考最优", "reference optimum", lang), size=11, fill=INK2)
    for t in range(1, 13):
        s.text(sx(t), ybot + 20, str(t), size=10, fill=MUTED, anchor="middle")
    s.text((x0 + x1) / 2, ybot + 38, L("尝试次数（每次研究运行 12 次）", "attempt (12 per research run)", lang), size=11, fill=INK2, anchor="middle")
    s.text(x0 - 10, (s0y + refy) / 2, L("验证损失", "dev loss", lang), size=11, fill=INK2, anchor="end")
    s.text(x0 - 10, (s0y + refy) / 2 + 15, L("越低越好", "lower = better", lang), size=10, fill=MUTED, anchor="end")
    # formula card
    cx = 600
    s.box(cx - 104, 64, 208, 150, fill="#f6f5f1", stroke="none", rx=12)
    s.text(cx, 88, L("改进速度分", "Improvement-speed score", lang), size=12.5, fill=INK, anchor="middle", weight=600)
    s.text(cx, 116, L("每次尝试：已缩小的差距比例", "per attempt: fraction of gap closed", lang), size=10.5, fill=INK2, anchor="middle")
    s.text(cx, 136, L("→ 12 次尝试取平均", "→ mean over 12 attempts", lang), size=10.5, fill=INK2, anchor="middle")
    s.text(cx, 156, L("→ 5 个种子取平均", "→ mean over 5 seeds", lang), size=10.5, fill=INK2, anchor="middle")
    s.text(cx, 176, L("→ 8 个任务逐任务配对比较", "→ paired over 8 tasks", lang), size=10.5, fill=INK2, anchor="middle")
    s.text(cx, 200, L("0 到 1，越高越好", "0 to 1, higher = better", lang), size=10.5, fill=BLUE, anchor="middle", weight=600)
    s.text(16, 24, L("示意图：曲线形状为说明用，不是实验数据", "Schematic: the curve is illustrative, not experiment data", lang), size=11, fill=MUTED)
    s.save("fig4-scoredef", lang, L("改进速度分的算法", "How the improvement-speed score is computed", lang))


# ------------------------------------------------------------------ fig4-lineage
def fig_lineage(lang):
    s = Fig(366)
    s.arrowheads(AXIS, BLUE, MUTED)
    P0, P1, P2 = NEUTRAL, BLUE_M, BLUE           # records produced by: original, generation 1, generation 2
    legend(s, 16, 22, [(L("原始模型产生的记录", "records from the original model", lang), P0, "bar"),
                       (L("第一代产生的记录", "records from generation 1", lang), P1, "bar"),
                       (L("第二代产生的记录", "records from generation 2", lang), P2, "bar")])
    rows = [
        (L("第一代后继模型", "Generation 1", lang), [(P0, L("被采纳记录 438 条", "438 accepted records", lang))], "dec_g1", None),
        (L("安慰剂模型", "Placebo model", lang), [("#efede7", L("等量的被否决记录", "equal number of rejected records", lang))], "dec_g1", "placebo"),
        (L("第二代后继模型", "Generation 2", lang), [(P0, L("原始模型", "original", lang)), (P1, L("第一代 · 新种子", "gen 1 · new seeds", lang))], "dec_g2", None),
        (L("“多给数据”对照", "\"More data\" control", lang), [(P0, L("原始模型", "original", lang)), (P0, L("原始模型 · 新种子", "original · new seeds", lang))], "dec_g2", "placebo"),
        (L("第三代后继模型", "Generation 3", lang), [(P0, L("原始模型", "original", lang)), (P1, L("第一代", "gen 1", lang)), (P2, L("第二代 · 新种子", "gen 2 · new seeds", lang))], "dec_g3", None),
        (L("“多给数据”对照", "\"More data\" control", lang), [(P0, L("原始模型", "original", lang)), (P0, L("原始模型", "original", lang)), (P0, L("原始模型 · 新种子", "original · new seeds", lang))], "dec_g3", "placebo"),
    ]
    xl, xd, dw, xm, xs = 16, 150, 300, 506, 660
    y = 58
    s.text(xd, y - 12, L("训练数据（由谁产生）", "training data (who produced it)", lang), size=11, fill=MUTED)
    s.text(xm + 40, y - 12, L("从原始权重做 LoRA", "LoRA from original weights", lang), size=11, fill=MUTED, anchor="middle")
    s.text(xs, y - 12, L("改进速度分", "score", lang), size=11, fill=MUTED, anchor="middle")
    for i, (name, segs, key, kind) in enumerate(rows):
        yy = y + i * 46 + (8 if i >= 2 else 0) + (8 if i >= 4 else 0)
        ctrl = kind is not None
        s.text(xl, yy + 18, name, size=12, fill=INK if not ctrl else INK2, weight=600 if not ctrl else 400)
        seg_w = dw / 3
        for j, (col, lab) in enumerate(segs):
            ww = seg_w - 4 if len(segs) > 1 or i > 1 else dw * 0.5
            xx = xd + j * seg_w
            s.box(xx, yy + 2, ww, 24, fill=col, stroke="none", rx=5, cls="grow", style=f"--d:{.1 + .15 * i + .1 * j:.2f}s")
            s.text(xx + 8, yy + 18, lab, size=10.5, fill=INK if col != P2 else "#ffffff")
        s.path(f"M{xd + dw + 8},{yy + 14} L{xm - 4},{yy + 14}", stroke=AXIS, w=1.4, cls="flow")
        s.box(xm, yy + 1, 80, 26, fill=PAPER, stroke=AXIS, rx=13)
        s.text(xm + 40, yy + 18, "LoRA", size=11, fill=INK2, anchor="middle")
        s.path(f"M{xm + 84},{yy + 14} L{xs - 30},{yy + 14}", stroke=AXIS, w=1.4, cls="flow")
        v = ctrl_score(key) if ctrl else score(key)
        col = ORANGE if ctrl else BLUE
        delta = v - M0
        s.circle(xs, yy + 14, 6, col, stroke="#ffffff",
                 tip=L(f"{name}：{v:.3f}（原始模型 {M0:.3f}）[{DEC_NAME}: {key}{'.vs_placebo，由差值推出' if ctrl else '.main'}]",
                       f"{name}: {v:.3f} (original {M0:.3f}) [{DEC_NAME}: {key}{'.vs_placebo, derived' if ctrl else '.main'}]", lang))
        s.text(xs + 12, yy + 18, f"{v:.3f}", size=11.5, fill=INK2)
    # recursion arrows: gen-1 model produces records for gen 2; gen-2 model produces records for gen 3
    s.path(f"M{xs},{58 + 20} C{xs + 40},{58 + 60} {xd + 1.5 * dw / 3 + 40},{58 + 70} {xd + 1.5 * dw / 3},{58 + 2 * 46 + 8}",
           stroke=BLUE, w=1.3, cls="flow", tip=L("第一代在训练任务上用新种子跑出的记录，加入第二代的训练数据 [LOG H]",
                                                "Records generation 1 produced on the training tasks with new seeds go into generation 2's data [LOG H]", lang))
    s.path(f"M{xs},{58 + 2 * 46 + 8 + 20} C{xs + 40},{58 + 2 * 46 + 60} {xd + 2.5 * dw / 3 + 40},{58 + 3 * 46 + 70} {xd + 2.5 * dw / 3},{58 + 4 * 46 + 16}",
           stroke=BLUE, w=1.3, cls="flow", tip=L("第二代新跑出的记录加入第三代的训练数据 [LOG E]",
                                                "Records generation 2 produced go into generation 3's data [LOG E]", lang))
    s.text(16, 340, L(f"LoRA：rank 16，学习率 1e-4，2 个 epoch；原始模型的改进速度分为 {M0:.3f}。",
                      f"LoRA: rank 16, lr 1e-4, 2 epochs. The original model scores {M0:.3f}.", lang), size=10.5, fill=MUTED)
    s.text(16, 356, L("第二、三代与各自对照的数据量相同，只有新增部分的来源不同。",
                      "Generations 2–3 and their controls use the same amount of data; only the source of the added part differs.", lang), size=10.5, fill=MUTED)
    s.save("fig4-lineage", lang, L("三代后继模型与对照的训练数据来源", "Where each generation's training data comes from", lang))


# ------------------------------------------------------------------ fig4-rounds
def fig_rounds(lang):
    s = Fig(530)
    s.arrowheads(AXIS, BLUE, ORANGE, MUTED)
    cols = [  # (title, designer, designer note, recipe rows, training, key)
        (L("第一轮", "Round 1", lang), L("原始模型", "Original model", lang), L("一次通过", "1 answer", lang),
         [L("被采纳 + hindsight", "accepted + hindsight", lang), L("阈值：不设", "min gain: none", lang), L("每次尝试：全部调用", "calls: all", lang), L("每任务上限 150", "cap 150 / task", lang)],
         694, (32, "3e-4", 4), L("55 分钟到时停止 · 约 2.4 epoch", "stopped at 55 min · ~2.4 epochs", lang), "dec_sd1"),
        (L("第二轮", "Round 2", lang), L("原始模型", "Original model", lang), L("看过反馈表 · 一次通过", "saw feedback · 1 answer", lang),
         [L("只用被采纳", "accepted only", lang), L("阈值：1%", "min gain: 1%", lang), L("每次尝试：最后一次调用", "calls: last only", lang), L("每任务上限 100", "cap 100 / task", lang)],
         137, (16, "2e-5", 2), L("约 10 分钟、31 步 · 验证损失 0.144→0.145", "~10 min, 31 steps · val loss 0.144→0.145", lang), "dec_sd2"),
        (L("第三轮", "Round 3", lang), L("第一轮微调后的模型", "Round-1 model", lang), L("看过同样的反馈表 · 3 次回答", "same feedback · 3 answers", lang),
         [L("被采纳 + hindsight", "accepted + hindsight", lang), L("阈值：1%", "min gain: 1%", lang), L("每次尝试：全部调用", "calls: all", lang), L("每任务上限 150", "cap 150 / task", lang)],
         539, (32, "3e-4", 4), L("55 分钟到时停止 · 第 183 步 · 约 2.7 epoch", "stopped at 55 min · step 183 · ~2.7 epochs", lang), "dec_sd3"),
    ]
    cw, gap, x0 = 212, 18, 30
    ytop = 70
    s.text(16, 24, L("三轮共用原始模型的同一份记录池，都从原始权重开始做 LoRA；唯一变化的是设计者和它看到的信息",
                     "All three rounds share the original model's record pool and start LoRA from the original weights; only the designer and what it sees change", lang), size=11, fill=MUTED)
    # feedback table chip
    fbx = x0 + cw + gap + cw / 2 + cw / 2 + gap / 2
    s.box(fbx - 120, 34, 240, 24, fill="#f1efe9", stroke="none", rx=12,
          tip=L("反馈表：第一轮模型在训练任务（换了种子）上 0.652，原始模型 0.525，第一代 0.483 [LOG E]",
                "Feedback table: round-1 model on training tasks (new seeds) 0.652, original 0.525, generation 1 0.483 [LOG E]", lang))
    s.text(fbx, 50, L("反馈表（训练任务）：0.652 / 0.525 / 0.483", "Feedback (training tasks): 0.652 / 0.525 / 0.483", lang), size=10.5, fill=INK2, anchor="middle")
    s.path(f"M{fbx - 60},{58} L{x0 + cw + gap + cw / 2},{ytop - 2}", stroke=MUTED, w=1.2, cls="flow")
    s.path(f"M{fbx + 60},{58} L{x0 + 2 * (cw + gap) + cw / 2},{ytop - 2}", stroke=MUTED, w=1.2, cls="flow")
    maxcalls = 694
    for i, (title, designer, dnote, recipe, calls, (rank, lr, ep), train, key) in enumerate(cols):
        x = x0 + i * (cw + gap)
        cx = x + cw / 2
        rec = i == 2
        # designer
        s.box(x, ytop, cw, 52, fill=ORANGE_T if rec else BLUE_T, stroke=ORANGE if rec else BLUE, rx=10)
        s.text(cx, ytop + 21, f"{title} · " + L("设计者：", "designer: ", lang) + designer, size=11.5, fill=INK, anchor="middle", weight=600)
        s.text(cx, ytop + 40, dnote, size=10.5, fill=INK2, anchor="middle")
        if rec:   # 3 answers, first two hit the cap
            for k in range(3):
                bx = cx + 8 + k * 14
                s.box(bx, ytop + 60, 10, 9, fill=ORANGE if k < 2 else BLUE, stroke="none", rx=2,
                      tip=L("前两次回答在思考阶段写满 8192 token 上限、正文为空；第三次给出配方 [LOG E]",
                            "The first two answers filled the 8192-token cap while thinking, with empty output; the third gave the recipe [LOG E]", lang) if k < 2 else
                      L("第三次回答给出配方 [LOG E]", "Third answer gave the recipe [LOG E]", lang))
            s.text(cx - 8, ytop + 68, L("前两次被 8192 上限截断", "2 answers cut at 8192", lang), size=9.5, fill=ORANGE, anchor="end")
        s.path(f"M{cx},{ytop + 54} L{cx},{ytop + 74}", stroke=AXIS, w=1.4, cls="flow")
        # recipe card
        ry = ytop + 78
        s.box(x, ry, cw, 156, fill=PAPER, stroke=AXIS, rx=10)
        s.text(x + 12, ry + 20, L("配方", "Recipe", lang), size=11, fill=MUTED)
        for k, r in enumerate(recipe):
            s.text(x + 12, ry + 40 + k * 17, r, size=10.8, fill=INK2)
        by = ry + 116
        s.text(x + 12, by, L("选出的调用数", "calls selected", lang), size=10, fill=MUTED)
        bw = (cw - 60) * calls / maxcalls
        s.box(x + 12, by + 6, bw, 10, fill=ORANGE if rec else BLUE, stroke="none", rx=3, cls="grow", style=f"--d:{.2 + .2 * i:.1f}s",
              tip=L(f"选出 {calls} 条调用（固定配方为 438 条）[LOG E]", f"{calls} calls selected (fixed recipe: 438) [LOG E]", lang))
        s.text(x + 16 + bw, by + 15, str(calls), size=10.5, fill=INK2)
        s.text(x + 12, by + 33, f"LoRA rank {rank} · lr {lr} · {ep} epoch" + ("s" if lang == "en" else ""), size=10.5, fill=INK2)
        s.path(f"M{cx},{ry + 158} L{cx},{ry + 178}", stroke=AXIS, w=1.4, cls="flow")
        # training
        ty = ry + 182
        s.box(x, ty, cw, 34, fill="#f6f5f1", stroke="none", rx=8, tip=L(f"{train} [LOG E；deploy/second_order.md]", f"{train} [LOG E; deploy/second_order.md]", lang))
        s.text(cx, ty + 21, train, size=10.3, fill=INK2, anchor="middle")
        s.path(f"M{cx},{ty + 36} L{cx},{ty + 56}", stroke=AXIS, w=1.4, cls="flow")
        # score
        sy = ty + 60
        v, c = score(key), ctrl_score(key)
        s.box(x, sy, cw, 70, fill=PAPER, stroke=ORANGE if rec else AXIS, rx=10, sw=1.6 if rec else 1.2)
        s.text(x + 12, sy + 20, L("未见过任务上的改进速度分", "score on unseen tasks", lang), size=10, fill=MUTED)
        s.text(x + 12, sy + 50, f"{v:.3f}", size=24, fill=ORANGE if rec else BLUE, weight=600)
        s.text(x + cw - 12, sy + 40, L(f"随机配方 ≈ {c:.3f}", f"random recipe ≈ {c:.3f}", lang), size=10.5, fill=INK2, anchor="end")
        s.text(x + cw - 12, sy + 56, L(f"原始模型 {M0:.3f}", f"original {M0:.3f}", lang), size=10.5, fill=MUTED, anchor="end")
        s.add(f'<rect x="{x:.1f}" y="{sy:.1f}" width="{cw}" height="70" fill="transparent"><title>{esc(L(f"{title}：{v:.3f}；随机配方约 {c:.3f}（由差值推出）[{DEC_NAME}: {key}]", f"{title}: {v:.3f}; random recipe ≈ {c:.3f} (derived) [{DEC_NAME}: {key}]", lang))}</title></rect>')
    # recursion: round-1 model becomes round-3 designer
    x1c = x0 + cw / 2
    x3 = x0 + 2 * (cw + gap)
    ys = ytop + 78 + 182 + 60 + 72
    gxm = x3 - gap / 2
    s.path(f"M{x1c},{ys} L{x1c},{ys + 18} Q{x1c},{ys + 26} {x1c + 8},{ys + 26} L{gxm - 8},{ys + 26} Q{gxm},{ys + 26} {gxm},{ys + 18} "
           f"L{gxm},{ytop + 34} Q{gxm},{ytop + 26} {gxm + 4},{ytop + 26} L{x3 - 1},{ytop + 26}",
           stroke=ORANGE, w=1.8, cls="flow", tip=L("第一轮微调后的模型成为第三轮的设计者 [LOG E]", "The round-1 model becomes the round-3 designer [LOG E]", lang))
    s.text(x0 + cw + gap + cw / 2, ys + 44, L("第一轮微调后的模型 → 第三轮的设计者", "round-1 model → round-3 designer", lang), size=11, fill=ORANGE, anchor="middle", weight=600)
    s.save("fig4-rounds", lang, L("让模型自己决定怎么训练自己：三轮", "The model designs its own training: three rounds", lang))


# ------------------------------------------------------------------ fig4-behavior
def fig_behavior(lang):
    s = Fig(330)
    # REPORT_LOG §E (dec_sd3 behaviour line and the 10:4x truncation audit line):
    # calls on the unseen tasks / median completion tokens
    pts = [(L("原始模型", "Original", lang), 2030, 111, False),
           (L("第二轮（轻度训练）", "Round 2 (light)", lang), 2007, 101, False),
           (L("第一轮", "Round 1", lang), 14689, 51, True),
           (L("第三轮", "Round 3", lang), 11287, 54, True),
           (L("第三轮的随机配方", "Round-3 random recipe", lang), 15087, 47, True)]
    x0, x1, y0, y1 = 90, 690, 270, 50
    lx = lambda v: x0 + (math.log10(v) - math.log10(1500)) / (math.log10(20000) - math.log10(1500)) * (x1 - x0)
    ly = lambda v: y0 - (v - 30) / (130 - 30) * (y0 - y1)
    for v in (2000, 5000, 10000, 20000):
        s.line(lx(v), y1, lx(v), y0, GRID)
        s.text(lx(v), y0 + 18, f"{v:,}", size=10.5, fill=MUTED, anchor="middle")
    for v in (40, 60, 80, 100, 120):
        s.line(x0, ly(v), x1, ly(v), GRID)
        s.text(x0 - 8, ly(v) + 4, str(v), size=10.5, fill=MUTED, anchor="end")
    s.text((x0 + x1) / 2, y0 + 38, L("未见过任务上的调用次数（对数刻度）", "calls on the unseen tasks (log scale)", lang), size=11, fill=INK2, anchor="middle")
    s.text(16, 30, L("每次输出长度中位数（token）", "median output length (tokens)", lang), size=11, fill=INK2)
    # cluster halos
    s.add(f'<ellipse cx="{lx(2018):.1f}" cy="{ly(106):.1f}" rx="44" ry="34" fill="{BLUE_T}" class="pulse"/>')
    s.add(f'<ellipse cx="{lx(13500):.1f}" cy="{ly(50.5):.1f}" rx="112" ry="34" fill="{ORANGE_T}" class="pulse"/>')
    s.text(lx(2018) + 54, ly(122), L("未训练 / 轻度训练", "untrained / light", lang), size=11, fill=BLUE, weight=600)
    s.text(lx(13500), ly(74), L("高强度训练（rank 32 · lr 3e-4）", "heavy training (rank 32 · lr 3e-4)", lang), size=11, fill=ORANGE, anchor="middle", weight=600)
    for k, (name, calls, med, heavy) in enumerate(pts):
        col = ORANGE if heavy else BLUE
        s.add(f'<circle cx="{lx(calls):.1f}" cy="{ly(med):.1f}" r="7" fill="{col}" stroke="#ffffff" stroke-width="2" class="pop" style="--d:{.2 + .2 * k:.1f}s">'
              f'<title>{esc(L(f"{name}：调用 {calls:,} 次，输出中位数 {med} token [LOG E]", f"{name}: {calls:,} calls, median output {med} tokens [LOG E]", lang))}</title></circle>')
        dx, dy, anchor = {0: (12, -8, "start"), 1: (12, 16, "start"), 2: (0, -14, "middle"),
                          3: (-12, 4, "end"), 4: (6, 22, "start")}[k]
        s.text(lx(calls) + dx, ly(med) + dy, name, size=10.8, fill=INK2, anchor=anchor)
    s.save("fig4-behavior", lang, L("调用次数与输出长度", "Call count vs output length", lang))


# ------------------------------------------------------------------ fig4-cf
def cf_curves():
    """Best-so-far gain after each evaluated attempt (keep / discard / crash; episodes with no code change are skipped)."""
    out = {}
    for f in sorted((ROOT / "research/cf").glob("*/ledger.jsonl")):
        run = f.parent.name
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        base = next(r["dev"] for r in recs if r.get("status") == "baseline")
        last = {}
        for r in recs:
            if r.get("status") in ("keep", "discard", "crash") and r.get("trial"):
                last[r["trial"]] = r
        best, curve, marks = base, [0.0], []
        for t in sorted(last):
            r = last[t]
            if r.get("status") == "keep" and r.get("dev") is not None:
                best = min(best, r["dev"])
            curve.append(base - best)
            marks.append(r["status"])
        assert len(curve) - 1 == CF["runs"][run]["n_trials"], run
        assert abs(curve[-1] - CF["runs"][run]["gain"]) < 1e-9, run
        out[run] = (curve, marks)
    return out


def fig_cf(lang):
    curves = cf_curves()
    arms = [("m0", L("原始模型", "Original model", lang), "#6b6a65"), ("m1", L("第一代后继模型", "Generation 1", lang), BLUE),
            ("m1p", L("安慰剂模型", "Placebo model", lang), "#a3a097"), ("s1", L("带经验笔记的原始模型", "Original + lessons note", lang), ORANGE)]
    gx, x0, xr = 14, 46, 14
    pw = (W - x0 - xr - 3 * gx) / 4
    ph, y0 = 170, 84
    ymax = 0.036
    s = Fig(342)
    s.text(16, 22, L("从最好版本（1.2308）出发，每组 3 次运行，每次 12 次尝试", "From the best version (1.2308): 3 runs per improver, 12 attempts each", lang), size=11.5, fill=INK2)
    s.text(16, 40, L("纵轴：本次重测的基线减去到第 t 次为止的最好值，越高越好", "y: run baseline minus best so far after attempt t (higher = better)", lang), size=10.5, fill=MUTED)
    for i, (arm, name, col) in enumerate(arms):
        px = x0 + i * (pw + gx)
        mean = CF["arms"][arm]["gain_mean"]
        s.text(px, y0 - 24, name, size=11.5, fill=INK, weight=600)
        s.text(px, y0 - 8, L(f"平均降幅 {mean:.4f}", f"mean gain {mean:.4f}", lang), size=10.5, fill=col if arm != "m1p" else INK2)
        s.box(px, y0, pw, ph, fill=PAPER, stroke=GRID, rx=6)
        X = lambda t: px + 8 + t / 12 * (pw - 16)
        Yv = lambda v: y0 + ph - 8 - v / ymax * (ph - 16)
        for v in (0.01, 0.02, 0.03):
            s.line(px + 1, Yv(v), px + pw - 1, Yv(v), "#f0efea")
            if i == 0:
                s.text(px - 6, Yv(v) + 4, f"{v:.2f}", size=9.5, fill=MUTED, anchor="end")
        if i == 0:
            s.text(px - 6, Yv(0) + 4, "0", size=9.5, fill=MUTED, anchor="end")
        s.line(px + 1, Yv(mean), px + pw - 1, Yv(mean), col, 1, "3 3")
        runs = sorted(r for r in curves if r.split("-")[2] == arm)
        for k, r in enumerate(runs):
            c, marks = curves[r]
            pts = []
            for t, g in enumerate(c):
                if pts:
                    pts.append((X(t), pts[-1][1]))
                pts.append((X(t), Yv(g)))
            ln = sum(math.dist(pts[q], pts[q + 1]) for q in range(len(pts) - 1)) + 1
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            g = CF["runs"][r]["gain"]
            tip = L(f"{name} 第 {k + 1} 次运行：降幅 {g:.4f}，{CF['runs'][r]['n_trials']} 次尝试，其中 {marks.count('keep')} 次保留、{marks.count('crash')} 次评测崩溃 [cf_analysis.json；cf/{r}/ledger.jsonl]",
                    f"{name}, run {k + 1}: gain {g:.4f}, {CF['runs'][r]['n_trials']} attempts, {marks.count('keep')} kept, {marks.count('crash')} crashed [cf_analysis.json; cf/{r}/ledger.jsonl]", lang)
            s.path(d, stroke=col, w=1.7, marker=False, cls="draw", style=f"--len:{ln:.0f};--d:{.15 * k + .2 * i:.2f}s", tip=tip)
            for t, m in enumerate(marks, start=1):
                if m == "keep":
                    s.add(f'<circle cx="{X(t):.1f}" cy="{Yv(c[t]):.1f}" r="2.6" fill="{col}" stroke="#ffffff" stroke-width="1" class="pop" style="--d:{.6 + .15 * k + .2 * i:.2f}s"/>')
            s.add(f'<circle cx="{X(len(c) - 1):.1f}" cy="{Yv(c[-1]):.1f}" r="4" fill="{col}" stroke="#ffffff" stroke-width="1.5"><title>{esc(tip)}</title></circle>')
        s.text(px + 8, y0 + ph + 16, "0", size=9.5, fill=MUTED, anchor="middle")
        s.text(px + pw - 8, y0 + ph + 16, "12", size=9.5, fill=MUTED, anchor="middle")
        s.text(px + pw / 2, y0 + ph + 16, L("尝试", "attempt", lang), size=9.5, fill=MUTED, anchor="middle")
    s.text(16, y0 + ph + 42, L("小圆点是被保留的尝试，虚线是该改进者 3 次运行的平均降幅。",
                               "Small dots are kept attempts; the dashed line is the improver's mean gain over its 3 runs.", lang), size=10.5, fill=MUTED)
    s.text(16, y0 + ph + 58, L("每组 3 次运行时，置换检验能给出的最小 p 值是 0.10。",
                               "With 3 runs per arm, the smallest possible permutation p is 0.10.", lang), size=10.5, fill=MUTED)
    s.save("fig4-cf", lang, L("从最好版本继续改", "Continuing from the best version", lang))


# ------------------------------------------------------------------ fig4-audit
def fig_audit(lang):
    s = Fig(300)
    # REPORT_LOG §G (11-18 00:46 row): 8 of 48 calls hit the 8192-token cap; §E (10:4x audit line) for the 9B groups
    s.text(22, 14, L("更大模型的对照运行", "Larger-model control run", lang), size=12, fill=INK, weight=600)
    s.text(22, 30, L("返回码 0，日志像“没有找到改进”", "exit code 0, log reads \"no improvement found\"", lang), size=10.5, fill=INK2)
    for k in range(48):
        r, c = divmod(k, 12)
        x, y = 22 + c * 24, 44 + r * 24
        trunc = k in (3, 9, 14, 20, 27, 33, 38, 45)     # positions are illustrative; the count (8 of 48) is from LOG G
        s.add(f'<rect x="{x}" y="{y}" width="18" height="18" rx="4" fill="{ORANGE if trunc else NEUTRAL}" class="pop" style="--d:{k * 0.02:.2f}s">'
              f'<title>{esc(L("被 8192 token 输出上限截断：这一回合改不了代码" if trunc else "正常结束的调用", "Cut at the 8192-token cap: this turn could not edit code" if trunc else "call finished normally", lang))} [LOG G]</title></rect>')
    s.text(22, 160, L("48 次调用中 8 次被截断（方块位置为示意）", "8 of 48 calls truncated (square positions illustrative)", lang), size=10.5, fill=INK2)
    legend(s, 22, 182, [(L("被截断", "truncated", lang), ORANGE, "bar"), (L("正常", "normal", lang), NEUTRAL, "bar")])
    # 9B groups
    gx = 330
    s.text(gx + 30, 14, L("报告使用的 9B 各组", "9B groups used in the report", lang), size=12, fill=INK, weight=600)
    s.text(gx + 30, 30, L("研究调用中被截断的次数", "truncated research calls", lang), size=10.5, fill=INK2)
    groups = [(L("原始模型", "Original", lang), 0, 2030), (L("第一代", "Generation 1", lang), 1, 1988), (L("第二代", "Generation 2", lang), 0, 2126),
              (L("第三代", "Generation 3", lang), 0, 2481), (L("第一轮", "Round 1", lang), 0, 14689), (L("第一轮随机配方", "Round-1 random", lang), 1, 3394),
              (L("第二轮", "Round 2", lang), 0, 2007), (L("第二轮随机配方", "Round-2 random", lang), 0, 2047)]
    for k, (name, n, tot) in enumerate(groups):
        y = 52 + k * 24
        s.text(gx + 30, y + 4, name, size=11, fill=INK2)
        bx, bw = gx + 150, 160
        s.box(bx, y - 6, bw, 12, fill="#f1efe9", stroke="none", rx=4)
        if n:
            s.box(bx, y - 6, max(2.5, bw * n / tot), 12, fill=ORANGE, stroke="none", rx=1)
        s.text(bx + bw + 8, y + 4, f"{n} / {tot:,}", size=10.5, fill=INK2,
               cls="")
        s.add(f'<rect x="{bx}" y="{y - 8}" width="{bw}" height="16" fill="transparent"><title>{esc(L(f"{name}：{tot:,} 次研究调用中 {n} 次被截断 [LOG E]", f"{name}: {n} of {tot:,} research calls truncated [LOG E]", lang))}</title></rect>')
    s.text(gx + 30, 260, L("每组最多 1 次：截断不影响各组改进速度分的比较", "At most 1 per group: truncation does not affect the score comparisons", lang), size=10.5, fill=MUTED)
    s.text(gx + 30, 278, L("截断只出现在开启思考的配方设计调用里", "Truncation only appeared in design calls with thinking on", lang), size=10.5, fill=MUTED)
    s.save("fig4-audit", lang, L("输出上限截断的核查", "Output-cap truncation audit", lang))


# ------------------------------------------------------------------ fig4-replay (animated, real events)
RP_CYCLE, RP_PLAY = 34.0, 30.0           # seconds per loop; seconds the 176.6-minute run is compressed into


def fig_replay(lang):
    ev = [json.loads(l) for l in (ROOT / "research/replay/final/ovn1118b-nc-g0/events.jsonl").read_text().splitlines()]
    t0 = next(e["t"] for e in ev if e["ev"] == "run_start")
    t1 = next(e["t"] for e in ev if e["ev"] == "run_end")
    span = (t1 - t0) / 60
    best_final = next(e["best_value"] for e in ev if e["ev"] == "run_end")
    sub = {e["job_id"]: e for e in ev if e["ev"] == "eval_submit"}
    prop = {e["trial"]: e for e in ev if e["ev"] == "proposal"}
    decs = sorted((e for e in ev if e["ev"] == "decision"), key=lambda e: e["t"])
    res = {e["trial"]: e for e in ev if e["ev"] == "eval_result"}
    mins = lambda t: (t - t0) / 60
    dpct, last = [], -1.0
    for e in decs:
        v = max(min((e["t"] - t0) / 60 / span * RP_PLAY / RP_CYCLE * 100, RP_PLAY / RP_CYCLE * 100), last + 0.25)
        dpct.append(v); last = v
    pct = lambda m: min(m / span * RP_PLAY / RP_CYCLE * 100, RP_PLAY / RP_CYCLE * 100)
    end_pct = RP_PLAY / RP_CYCLE * 100

    s = Fig(466)
    css = [".f4 .rw{opacity:0}@media (prefers-reduced-motion: reduce){.f4 .rlast{opacity:1}}"]

    def window(cls, a, b=None):
        """Element visible from a% to b% of the loop (b=None: until the loop restarts)."""
        b = 99.0 if b is None else b
        b = max(b, a + 0.15)                     # keyframe offsets must stay ordered
        css.append(f".in .f4 .{cls}{{animation:{cls} {RP_CYCLE}s linear infinite}}"
                   f"@keyframes {cls}{{0%,{a:.3f}%{{opacity:0}}{a + .05:.3f}%,{b:.3f}%{{opacity:1}}{b + .05:.3f}%,100%{{opacity:0}}}}")

    x0, xw = 84, W - 84 - 20
    X = lambda m: x0 + m / span * xw
    py0, ph = 92, 196
    vhi, vlo = 1.46, 1.21
    Y = lambda v: py0 + (vhi - min(v, vhi)) / (vhi - vlo) * ph
    # frame
    s.box(x0, py0, xw, ph, fill=PAPER, stroke=GRID, rx=6)
    for v in (1.25, 1.30, 1.35, 1.40, 1.45):
        s.line(x0, Y(v), x0 + xw, Y(v), "#f0efea")
        s.text(x0 - 6, Y(v) + 4, f"{v:.2f}", size=10, fill=MUTED, anchor="end")
    s.line(x0, Y(1.4152), x0 + xw, Y(1.4152), MUTED, 1, "4 3")
    s.text(x0 + 6, Y(1.4152) - 5, L("起点 1.4152", "start 1.4152", lang), size=10, fill=MUTED)
    s.text(16, py0 + 12, "val_bpb", size=11, fill=INK2)
    s.text(16, py0 + 26, L("越低越好", "lower = better", lang), size=10, fill=MUTED)
    for m in (0, 30, 60, 90, 120, 150):
        s.text(X(m), py0 + ph + 16, f"{m}", size=10, fill=MUTED, anchor="middle")
    s.text(X(span), py0 + ph + 16, f"{span:.1f} " + L("分钟", "min", lang), size=10, fill=MUTED, anchor="end")
    # GPU lanes
    lanes = sorted({r["lane"] for r in res.values()})
    ly0 = py0 + ph + 30
    for k, ln in enumerate(lanes):
        y = ly0 + k * 15
        s.box(x0, y, xw, 11, fill="#f6f5f1", stroke="none", rx=3)
        s.text(x0 - 6, y + 9, L(f"GPU {k + 1}", f"GPU {k + 1}", lang), size=9.5, fill=MUTED, anchor="end")
    # per-trial elements
    best_cur = None
    for i, d in enumerate(decs):
        tr = d["trial"]
        r = res.get(tr)
        md = mins(d["t"])
        a = pct(md)
        # GPU span: running (light) then finished (coloured)
        if r:
            ra, rb = mins(r["t"] - (r.get("wall_s") or 0)), mins(r["t"])
            k = lanes.index(r["lane"])
            y = ly0 + k * 15
            col = ORANGE if r["status"] != "ok" else (BLUE if d["verdict"] == "adopt" else NEUTRAL)
            window(f"rr{i}", pct(ra), pct(rb))
            s.add(f'<rect x="{X(ra) + .6:.1f}" y="{y + 1}" width="{max(1.2, X(rb) - X(ra) - 1.2):.1f}" height="9" rx="2" fill="{BLUE_T}" class="rw rr{i}"/>')
            window(f"rf{i}", pct(rb))
            s.add(f'<rect x="{X(ra) + .6:.1f}" y="{y + 1}" width="{max(1.2, X(rb) - X(ra) - 1.2):.1f}" height="9" rx="2" fill="{col}" class="rw rlast rf{i}"/>')
        # result dot
        if r and r["status"] == "ok":
            v = r["value"]
            col = BLUE if d["verdict"] == "adopt" else "#b9b6ad"
            window(f"rd{i}", a)
            tip = (f"#{tr} · {d['verdict']} · val_bpb {v:.4f}" + (" (>1.46)" if v > vhi else ""))
            if v > vhi:
                s.add(f'<path d="M{X(md) - 4:.1f},{py0 + 8} L{X(md):.1f},{py0 + 2} L{X(md) + 4:.1f},{py0 + 8}" fill="none" stroke="#b9b6ad" stroke-width="1.6" class="rw rlast rd{i}"><title>{esc(tip)} [events.jsonl]</title></path>')
            else:
                s.add(f'<circle cx="{X(md):.1f}" cy="{Y(v):.1f}" r="{4.2 if d["verdict"] == "adopt" else 3.2}" fill="{col}" '
                      f'stroke="#ffffff" stroke-width="1.2" class="rw rlast rd{i}"><title>{esc(tip)} [events.jsonl]</title></circle>')
        elif r:
            window(f"rd{i}", a)
            s.add(f'<path d="M{X(md) - 3.5:.1f},{py0 + ph - 10} l7,7 m0,-7 l-7,7" stroke="{ORANGE}" stroke-width="1.6" class="rw rlast rd{i}">'
                  f'<title>#{tr} · {esc(L("评测崩溃", "evaluation crashed", lang))} [events.jsonl]</title></path>')
        # run-best step line (running minimum across threads)
        prev = best_cur
        best_cur = d["best_after"] if best_cur is None else min(best_cur, d["best_after"])
        if prev is not None:
            mprev = mins(decs[i - 1]["t"])
            window(f"rb{i}", a)
            s.add(f'<path d="M{X(mprev):.1f},{Y(prev):.1f} L{X(md):.1f},{Y(prev):.1f} L{X(md):.1f},{Y(best_cur):.1f}" fill="none" '
                  f'stroke="{BLUE}" stroke-width="2" stroke-linejoin="round" class="rw rlast rb{i}"/>')
    window("rbend", end_pct)
    s.add(f'<path d="M{X(mins(decs[-1]["t"])):.1f},{Y(best_cur):.1f} L{X(span):.1f},{Y(best_cur):.1f}" stroke="{BLUE}" stroke-width="2" class="rw rlast rbend"/>')
    # playhead
    css.append(f".f4 .ph{{transform:translateX(0)}}.in .f4 .ph{{animation:ph {RP_CYCLE}s linear infinite}}"
               f"@keyframes ph{{0%{{transform:translateX(0)}}{end_pct:.3f}%,100%{{transform:translateX({xw:.1f}px)}}}}")
    s.add(f'<g class="ph"><line x1="{x0}" y1="{py0 - 4}" x2="{x0}" y2="{ly0 + 46}" stroke="{INK}" stroke-width="1" opacity=".55"/></g>')
    # counters + feed, one group per decision window
    adopt = 0
    hdr_y, feed_y = 20, ly0 + 66
    s.text(16, feed_y - 4, L("最新判定", "latest decisions", lang), size=10.5, fill=MUTED)
    s.box(x0, feed_y - 16, xw, 64, fill="#f6f5f1", stroke="none", rx=8)
    for i, d in enumerate(decs):
        if d["verdict"] == "adopt" and d["trial"] != 0:
            adopt += 1
        nxt = dpct[i + 1] if i + 1 < len(decs) else None
        window(f"rc{i}", dpct[i], nxt)
        best = min(x["best_after"] for x in decs[: i + 1])
        g = [f'<g class="rw{" rlast" if i == len(decs) - 1 else ""} rc{i}">']
        clock = mins(d["t"])
        items = [(L("已运行", "elapsed", lang), f"{int(clock // 60)}:{int(clock % 60):02d}"),
                 (L("已判定的尝试", "attempts decided", lang), f"{i} / 72"),
                 (L("保留", "kept", lang), f"{adopt}"),
                 (L("当前最好", "best", lang), f"{best:.4f}")]
        xx = 16
        for lab, val in items:
            g.append(f'<text x="{xx}" y="{hdr_y}" font-size="10.5" fill="{MUTED}">{esc(lab)}</text>'
                     f'<text x="{xx}" y="{hdr_y + 22}" font-size="18" font-weight="600" fill="{INK}" style="font-variant-numeric:tabular-nums">{esc(val)}</text>')
            xx += 120 if lab != L("当前最好", "best", lang) else 0
        for j, dd in enumerate(decs[max(0, i - 2): i + 1][::-1]):
            trj = dd["trial"]
            p = prop.get(trj)
            hyp = (p["hypothesis"] if p else L("起点基线", "baseline", lang)).replace("\n", " ")
            hyp = hyp if len(hyp) <= 78 else hyp[:76].rstrip() + "…"
            vv = res.get(trj)
            vstr = f"{vv['value']:.4f}" if vv and vv["status"] == "ok" else L("崩溃", "crash", lang)
            verdict = {"adopt": L("保留", "keep", lang), "reject": L("丢弃", "discard", lang)}.get(dd["verdict"], dd["verdict"])
            col = BLUE if dd["verdict"] == "adopt" else (ORANGE if vv and vv["status"] != "ok" else INK2)
            yy = feed_y + j * 18
            op = 1 if j == 0 else .55
            g.append(f'<text x="{x0 + 10}" y="{yy}" font-size="10.5" fill="{col}" opacity="{op}" font-weight="600">#{trj} {esc(verdict)} {vstr}</text>'
                     f'<text x="{x0 + 132}" y="{yy}" font-size="10.5" fill="{INK2}" opacity="{op}">{esc(p["agent"] if p else "orchestrator")} · {esc(hyp)}</text>')
        g.append("</g>")
        s.add("".join(g))
    s.text(W - 20, hdr_y, L(f"真实运行回放 · 约 {span * 60 / RP_PLAY:.0f} 倍速 · 自动循环", f"real-run replay · ~{span * 60 / RP_PLAY:.0f}× speed · loops", lang), size=10.5, fill=MUTED, anchor="end")
    s.text(W - 20, hdr_y + 22, L(f"终点 {best_final:.4f}", f"final {best_final:.4f}", lang), size=12, fill=BLUE, anchor="end", weight=600)
    legend(s, x0, py0 - 16, [(L("评测后保留", "kept", lang), BLUE, "dot"), (L("评测后丢弃", "discarded", lang), "#b9b6ad", "dot"),
                             (L("评测崩溃", "crashed", lang), ORANGE, "bar"), (L("评测中", "evaluating", lang), BLUE_T, "bar")])
    s.items[0] = s.items[0].replace("</style>", "".join(css) + "</style>")
    s.save("fig4-replay", lang, L("NanoChat 研究运行回放", "Replay of the NanoChat research run", lang))


if __name__ == "__main__":
    for lang in ("zh", "en"):
        fig_loop(lang); fig_scoredef(lang); fig_replay(lang); fig_lineage(lang); fig_rounds(lang)
        fig_behavior(lang); fig_cf(lang); fig_audit(lang)
    print("wrote fig4-{loop,scoredef,replay,lineage,rounds,behavior,cf,audit}.{zh,en}.svg")
