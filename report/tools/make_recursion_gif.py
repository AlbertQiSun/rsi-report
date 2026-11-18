"""Figure 1 for report §1 (animated GIF, paper-style, three panels).

(a) non-recursive optimization, replayed from a real run: Qwen3.5-9B improving NanoChat training
    code (research/replay/final/ovn1118b-nc-g0/events.jsonl). Every point, value, hypothesis,
    diff size and eval time shown comes from that log.
(b) recursive self-improvement: each generation runs (a), its trajectory updates the improver,
    the next generation takes over. Concept only; loop durations are illustrative.
(c) the rate of return r (Epoch / Forethought): with research done by the AI itself,
    dR/dt = S and S = R^r, so S(t) is linear for r = 0.5, exponential for r = 1 and blows up in
    finite time for r > 1. Curves and doubling marks are computed here from those equations.

Output: report/site/public/figs/recursion-loop.{zh,en}.gif
"""
import json, math, pathlib
from PIL import Image, ImageDraw, ImageFont

REPORT = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
OUT = REPORT / "site" / "public" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
EVENTS = ROOT / "research/replay/final/ovn1118b-nc-g0/events.jsonl"

S = 2
W, H = 1040, 640
BG = (251, 250, 247); PAPER = (255, 255, 255); LINE = (228, 225, 218); GRID = (238, 236, 230)
TEXT = (28, 31, 35); MUTED = (94, 102, 114); FAINT = (150, 157, 168)
ACC = (61, 99, 221); ACC_SOFT = (234, 239, 253); ACC_MID = (160, 180, 238)
VER = (224, 102, 58); VER_SOFT = (252, 236, 228); VER_MID = (238, 178, 152)
SAGE = (96, 150, 110)

ZH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
EN = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def F(lang, size, bold=False, mono=False):
    if mono:
        return ImageFont.truetype(MONO, int(size * S))
    if lang == "zh":
        return ImageFont.truetype(ZH, int(size * S), index=2 if bold else 0)
    return ImageFont.truetype(EN, int(size * S), index=1 if bold else 0)


# ---------------- real trial log ----------------
ev = [json.loads(l) for l in EVENTS.read_text().splitlines()]
props = {e["trial"]: e for e in ev if e["ev"] == "proposal"}
evals = {e["trial"]: e for e in ev if e["ev"] == "eval_result"}
decs = sorted((e for e in ev if e["ev"] == "decision"), key=lambda x: x["t"])
base = next(e for e in decs if e["trial"] == 0)          # trial 0 = baseline start artifact, not a proposal
s0 = base["best_after"]
TRIALS = []
for e in decs:
    if e["trial"] == 0:
        continue
    r = evals.get(e["trial"], {}); p = props.get(e["trial"], {})
    TRIALS.append({
        "trial": e["trial"], "verdict": e["verdict"], "best": e["best_after"], "thread": p.get("agent", "?"),
        "value": r.get("value") if r.get("status") == "ok" else None,
        "wall": r.get("wall_s"), "hyp": (p.get("hypothesis") or "").strip(),
        "plus": (p.get("diff_stat") or {}).get("plus"), "minus": (p.get("diff_stat") or {}).get("minus"),
    })
NT = len(TRIALS)
# each of the parallel threads keeps its own incumbent; the run's best is the running minimum over all
THREADS = sorted({r["thread"] for r in TRIALS})
run_best = s0
for r in TRIALS:
    run_best = min(run_best, r["best"]); r["run_best"] = run_best

# ---------------- returns model (panel c) ----------------
T_MAX, S_MAX = 4.0, 1000.0
def S_of(r, t):
    if r == 1:
        return math.exp(t)
    if r < 1:
        return ((1 - r) * t + 1) ** (r / (1 - r))
    base = 1 - (r - 1) * t
    return float("inf") if base <= 0 else base ** (-r / (r - 1))
CURVES = [(0.5, (150, 157, 168)), (1.0, ACC), (1.5, VER)]
def doublings(r):
    out, n = [], 1
    # bisection for times where S = 2^n
    while 2 ** n <= S_MAX:
        lo, hi = 0.0, T_MAX
        if S_of(r, hi) < 2 ** n:
            break
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if S_of(r, mid) < 2 ** n else (lo, mid)
        out.append(hi); n += 1
    return out
DOUBLE = {r: doublings(r) for r, _ in CURVES}

TXT = {
    "zh": {
        "fig": "FIG. 1", "title": "两种改进：改进别的东西，还是改进改进者本身",
        "a": "(a) 一个部件优化另一个部件 · non-recursive optimization",
        "a_sub": "改进者固定不变，被改进的是训练代码。下面回放一次真实运行：Qwen3.5-9B 改进 NanoChat 训练脚本，共 {n} 次尝试",
        "nodes": [("改进者", "权重 θ、脚手架 σ 固定"), ("提出补丁", "改动 +{p} / -{m} 行"), ("沙箱训练", "本次 {w} 秒"), ("封存评估器", "val_bpb，越低越好"), ("保留或丢弃", "只留更好的版本")],
        "plot": "val_bpb（点：每次尝试；细线：3 个并行线程各自的最好版本；粗线：全局最好）", "log": "原始日志（按判定顺序）",
        "keep": "保留", "reject": "丢弃", "crash": "崩溃",
        "b": "(b) 改进者改进它自己 · recursive self-improvement",
        "b_sub": "每一代把 (a) 整个跑一遍，用留下的记录更新自己，再由下一代接手（示意，各代用时为假设）",
        "bn": [("第 {k} 代改进者", "权重 θ、脚手架 σ"), ("跑一遍 (a)", "留下全部尝试记录 D"), ("更新自己", "微调权重或改写笔记"), ("第 {k1} 代改进者", "接替成为改进者")],
        "speed": "每代用时",
        "c": "(c) 什么时候会越改越快",
        "c_sub": "Epoch / Forethought：累积研发投入每翻一倍，软件效率翻 r 倍。研发由 AI 自己完成时：",
        "c_eq": "dR/dt = S,   S = R^r",
        "c_x": "时间", "c_y": "软件效率 S（对数刻度）",
        "lab": {0.5: "r = 0.5  减速：每次翻倍越来越慢", 1.0: "r = 1  匀速：每次翻倍用时相同", 1.5: "r = 1.5  加速：有限时间内发散"},
        "dbl": "● 每翻一倍",
        "note": "(a) 为真实日志回放；(b)(c) 为示意",
    },
    "en": {
        "fig": "FIG. 1", "title": "Two kinds of improvement: improving something else, or improving the improver",
        "a": "(a) One part optimizes another · non-recursive optimization",
        "a_sub": "The improver stays fixed; what changes is the training code. Replay of a real run: Qwen3.5-9B improving NanoChat training, {n} attempts",
        "nodes": [("Improver", "weights θ, scaffold σ fixed"), ("Propose patch", "+{p} / -{m} lines"), ("Sandbox training", "this run {w} s"), ("Sealed evaluator", "val_bpb, lower is better"), ("Keep or discard", "keep only if better")],
        "plot": "val_bpb (dots: attempts; thin: best of each of 3 threads; thick: overall best)", "log": "raw log (decision order)",
        "keep": "keep", "reject": "discard", "crash": "crash",
        "b": "(b) The improver improves itself · recursive self-improvement",
        "b_sub": "Each generation runs (a), updates itself from the records, and hands over to the next (concept; loop times are assumed)",
        "bn": [("Improver gen {k}", "weights θ, scaffold σ"), ("Run (a)", "keep every attempt as D"), ("Update itself", "fine-tune or rewrite notes"), ("Improver gen {k1}", "takes over")],
        "speed": "time per gen",
        "c": "(c) When does improvement speed up?",
        "c_sub": "Epoch / Forethought: software efficiency doubles r times per doubling of cumulative research input. With the AI doing the research:",
        "c_eq": "dR/dt = S,   S = R^r",
        "c_x": "time", "c_y": "software efficiency S (log scale)",
        "lab": {0.5: "r = 0.5  slowing: each doubling takes longer", 1.0: "r = 1  steady: equal doubling times", 1.5: "r = 1.5  speeding up: blows up in finite time"},
        "dbl": "● each doubling",
        "note": "(a) replays a real log; (b) and (c) are illustrative",
    },
}


def wrap(d, text, font, width):
    words = list(text) if any("一" <= ch <= "鿿" for ch in text) else text.split(" ")
    lines, cur = [], ""
    sep = "" if words is not None and len(words) and len(words[0]) == 1 and any("一" <= ch <= "鿿" for ch in text) else " "
    for w_ in words:
        trial = (cur + sep + w_) if cur else w_
        if d.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines


def arrow(d, a, b, color, width=1.4):
    d.line([a, b], fill=color, width=int(width * S))
    ang = math.atan2(b[1] - a[1], b[0] - a[0]); L = 7 * S
    d.polygon([b, (b[0] - L * math.cos(ang - .45), b[1] - L * math.sin(ang - .45)),
               (b[0] - L * math.cos(ang + .45), b[1] - L * math.sin(ang + .45))], fill=color)


def node(d, x, y, w, h, t1, t2, f1, f2, active, soft, edge):
    d.rounded_rectangle([x, y, x + w, y + h], radius=9 * S, fill=soft if active else PAPER,
                        outline=edge if active else LINE, width=int((1.6 if active else 1) * S))
    d.text((x + w / 2, y + h / 2 - 7 * S), t1, font=f1, fill=TEXT, anchor="mm")
    d.text((x + w / 2, y + h / 2 + 9 * S), t2, font=f2, fill=MUTED, anchor="mm")


def render(lang):
    t = TXT[lang]
    fT, fP, fs, fn1, fn2 = F(lang, 15, bold=True), F(lang, 12.5, bold=True), F(lang, 10.5), F(lang, 11, bold=True), F(lang, 9.5)
    fm, fmS = F(lang, 9.5, mono=True), F(lang, 8.5, mono=True)
    N = 150
    gens = [42, 32, 25, 20, 16, 15]                     # illustrative loop lengths (frames)
    gstart = [sum(gens[:i]) for i in range(len(gens))]
    frames = []
    X = lambda v: int(v * S)

    for f in range(N):
        im = Image.new("RGB", (W * S, H * S), BG)
        d = ImageDraw.Draw(im)
        d.text((X(28), X(18)), t["fig"], font=fm, fill=FAINT)
        d.text((X(80), X(15)), t["title"], font=fT, fill=TEXT)
        d.line([(X(28), X(44)), (X(1012), X(44))], fill=LINE, width=S)

        # ================= (a) =================
        ax, ay, aw = 28, 58, 632
        d.text((X(ax), X(ay)), t["a"], font=fP, fill=TEXT)
        k_trial = min(NT - 1, int(f / (N - 10) * NT))
        tr = TRIALS[k_trial]
        for i, line in enumerate(wrap(d, t["a_sub"].format(n=NT), fs, X(aw))[:2]):
            d.text((X(ax), X(ay + 20 + i * 15)), line, font=fs, fill=MUTED)
        nw, gap, ny, nh = 112, 18, ay + 58, 44
        phase = (f % 10) / 10
        active = int(phase * 5)
        labels = [(a, b.format(p=tr["plus"] or 0, m=tr["minus"] or 0, w=int(round(tr["wall"])) if tr["wall"] else "—")) for a, b in t["nodes"]]
        for i, (t1, t2) in enumerate(labels):
            x = ax + i * (nw + gap)
            node(d, X(x), X(ny), X(nw), X(nh), t1, t2, fn1, fn2, i == active, ACC_SOFT, ACC)
            if i < 4:
                arrow(d, (X(x + nw + 2), X(ny + nh / 2)), (X(x + nw + gap - 2), X(ny + nh / 2)), FAINT)
        # feedback arrow keep -> improver
        yb = ny + nh + 12
        d.line([(X(ax + 4 * (nw + gap) + nw / 2), X(ny + nh)), (X(ax + 4 * (nw + gap) + nw / 2), X(yb)),
                (X(ax + nw / 2), X(yb))], fill=FAINT, width=int(1.2 * S))
        arrow(d, (X(ax + nw / 2), X(yb)), (X(ax + nw / 2), X(ny + nh + 2)), FAINT)
        # packet on the pipeline
        px = ax + nw / 2 + phase * 4 * (nw + gap)
        d.ellipse([X(px - 4), X(ny - 9), X(px + 4), X(ny - 1)], fill=ACC)

        # mini plot of the real run
        gx, gy, gw, gh = ax + 34, ny + nh + 34, 330, 132
        lo, hi = 1.20, 1.46
        yv = lambda v: gy + gh - (min(max(v, lo), hi) - lo) / (hi - lo) * gh
        xv = lambda i: gx + i / (NT - 1) * gw
        d.text((X(ax), X(gy - 18)), t["plot"], font=fs, fill=MUTED)
        for v in (1.25, 1.30, 1.35, 1.40, 1.45):
            d.line([(X(gx), X(yv(v))), (X(gx + gw), X(yv(v)))], fill=GRID, width=S)
            d.text((X(gx - 6), X(yv(v))), f"{v:.2f}", font=fmS, fill=FAINT, anchor="rm")
        tcol = [(214, 190, 180), (190, 200, 228), (190, 214, 196)]
        for ti, th in enumerate(THREADS):
            tp, prev = [], s0
            for i in range(k_trial + 1):
                if TRIALS[i]["thread"] == th:
                    tp += [(X(xv(i)), X(yv(prev))), (X(xv(i)), X(yv(TRIALS[i]["best"])))]
                    prev = TRIALS[i]["best"]
            tp.append((X(xv(k_trial)), X(yv(prev))))
            if len(tp) > 1:
                d.line(tp, fill=tcol[ti % 3], width=int(1.2 * S))
        pts = []
        for i in range(k_trial + 1):
            r = TRIALS[i]
            if r["value"] is None:
                cx, cy = X(xv(i)), X(gy + gh - 3)
                d.line([(cx - 3 * S, cy - 3 * S), (cx + 3 * S, cy + 3 * S)], fill=FAINT, width=S)
                d.line([(cx - 3 * S, cy + 3 * S), (cx + 3 * S, cy - 3 * S)], fill=FAINT, width=S)
            else:
                c = ACC if r["verdict"] == "adopt" else ACC_MID
                rad = 3.2 if r["verdict"] == "adopt" else 2
                vy = yv(r["value"])
                d.ellipse([X(xv(i) - rad), X(vy - rad), X(xv(i) + rad), X(vy + rad)], fill=c)
            pts += [(X(xv(i)), X(yv(TRIALS[i - 1]["run_best"] if i else s0))), (X(xv(i)), X(yv(r["run_best"])))]
        if len(pts) > 1:
            d.line(pts, fill=VER, width=int(1.8 * S))
        d.text((X(gx + gw), X(gy + gh + 10)), f"{k_trial + 1}/{NT}   best {TRIALS[k_trial]['run_best']:.4f}", font=fmS, fill=MUTED, anchor="ra")

        # ticker of the last attempts (raw log)
        lx, lw = gx + gw + 26, aw - (gx + gw + 26 - ax)
        d.text((X(lx), X(gy - 18)), t["log"], font=fs, fill=MUTED)
        for j, i in enumerate(range(max(0, k_trial - 5), k_trial + 1)):
            r = TRIALS[i]; yy = gy + j * 22
            tag = t["keep"] if r["verdict"] == "adopt" else (t["crash"] if r["value"] is None else t["reject"])
            col = ACC if r["verdict"] == "adopt" else FAINT
            val = f"{r['value']:.4f}" if r["value"] is not None and r["value"] < 9 else "  —  "
            d.text((X(lx), X(yy)), f"#{r['trial']:>2}", font=fm, fill=FAINT)
            d.text((X(lx + 28), X(yy)), val, font=fm, fill=TEXT)
            d.text((X(lx + 76), X(yy)), tag, font=fs, fill=col)
            hyp = r["hyp"]
            while hyp and d.textlength(hyp + "…", font=fmS) > X(lw - 110):
                hyp = hyp[:-1]
            d.text((X(lx + 110), X(yy + 1)), (hyp + "…") if hyp != r["hyp"] else hyp, font=fmS, fill=MUTED)

        # ================= (b) =================
        by = 400
        d.line([(X(ax), X(by - 14)), (X(ax + aw), X(by - 14))], fill=LINE, width=S)
        d.text((X(ax), X(by)), t["b"], font=fP, fill=TEXT)
        for i, line in enumerate(wrap(d, t["b_sub"], fs, X(aw))[:2]):
            d.text((X(ax), X(by + 20 + i * 15)), line, font=fs, fill=MUTED)
        g = max(i for i, s in enumerate(gstart) if f >= s)
        local = min(1.0, (f - gstart[g]) / gens[g]); k = g + 1
        bw_, bgap, bny, bnh = 136, 28, by + 64, 46
        stage = min(3, int(local * 4))
        for i, (t1, t2) in enumerate(t["bn"]):
            x = ax + i * (bw_ + bgap)
            node(d, X(x), X(bny), X(bw_), X(bnh), t1.format(k=k, k1=k + 1), t2, fn1, fn2, i == stage, VER_SOFT, VER)
            if i < 3:
                arrow(d, (X(x + bw_ + 2), X(bny + bnh / 2)), (X(x + bw_ + bgap - 2), X(bny + bnh / 2)), VER_MID)
        yb2 = bny + bnh + 16
        xl, xr = ax + bw_ / 2, ax + 3 * (bw_ + bgap) + bw_ / 2
        d.line([(X(xr), X(bny + bnh)), (X(xr), X(yb2)), (X(xl), X(yb2))], fill=VER_MID, width=int(1.2 * S))
        arrow(d, (X(xl), X(yb2)), (X(xl), X(bny + bnh + 2)), VER_MID)
        bpx = ax + bw_ / 2 + local * 3 * (bw_ + bgap)
        d.ellipse([X(bpx - 4), X(bny - 9), X(bpx + 4), X(bny - 1)], fill=VER)
        # time per generation bars (lengths ∝ assumed loop lengths)
        sx, sy = ax, yb2 + 20
        d.text((X(sx), X(sy)), t["speed"], font=fs, fill=MUTED)
        for i in range(k):
            x0 = sx + 70 + i * 0
            yy = sy + 2 + i * 7
            d.rounded_rectangle([X(x0), X(yy), X(x0 + gens[i] * 4.2), X(yy + 4)], radius=2 * S, fill=VER if i == g else VER_MID)

        # ================= (c) =================
        cx0, cy0, cw = 690, 58, 322
        d.line([(X(cx0 - 15), X(58)), (X(cx0 - 15), X(620))], fill=LINE, width=S)
        d.text((X(cx0), X(cy0)), t["c"], font=fP, fill=TEXT)
        for i, line in enumerate(wrap(d, t["c_sub"], fs, X(cw))[:4]):
            d.text((X(cx0), X(cy0 + 20 + i * 15)), line, font=fs, fill=MUTED)
        d.text((X(cx0), X(cy0 + 86)), t["c_eq"], font=fm, fill=TEXT)
        px0, py0, pw, ph = cx0 + 34, cy0 + 120, cw - 44, 300
        lx_ = lambda tt: px0 + tt / T_MAX * pw
        ly_ = lambda s: py0 + ph - math.log10(min(max(s, 1), S_MAX)) / math.log10(S_MAX) * ph
        for dec in (1, 10, 100, 1000):
            d.line([(X(px0), X(ly_(dec))), (X(px0 + pw), X(ly_(dec)))], fill=GRID, width=S)
            d.text((X(px0 - 6), X(ly_(dec))), f"{dec:g}", font=fmS, fill=FAINT, anchor="rm")
        d.line([(X(px0), X(py0)), (X(px0), X(py0 + ph)), (X(px0 + pw), X(py0 + ph))], fill=MUTED, width=S)
        d.text((X(px0 + pw), X(py0 + ph + 8)), t["c_x"] + " →", font=fs, fill=MUTED, anchor="ra")
        d.text((X(px0), X(py0 - 16)), t["c_y"], font=fs, fill=MUTED)
        tnow = min(T_MAX, f / 120 * T_MAX)
        for r, col in CURVES:
            seg = []
            steps = 240
            for s_ in range(steps + 1):
                tt = s_ / steps * tnow
                v = S_of(r, tt)
                if v >= S_MAX:
                    seg.append((X(lx_(tt)), X(ly_(S_MAX)))); break
                seg.append((X(lx_(tt)), X(ly_(v))))
            if len(seg) > 1:
                d.line(seg, fill=col, width=int(2 * S), joint="curve")
            for td in DOUBLE[r]:
                if td <= tnow:
                    vx, vy = X(lx_(td)), X(ly_(S_of(r, td)))
                    d.ellipse([vx - 2.6 * S, vy - 2.6 * S, vx + 2.6 * S, vy + 2.6 * S], fill=col)
        ly0 = py0 + ph + 34
        for i, (r, col) in enumerate(CURVES[::-1]):
            yy = ly0 + i * 18
            d.line([(X(cx0), X(yy + 6)), (X(cx0 + 18), X(yy + 6))], fill=col, width=int(2 * S))
            d.text((X(cx0 + 26), X(yy)), t["lab"][r], font=fs, fill=TEXT)
        d.text((X(cx0), X(ly0 + 3 * 18 + 4)), t["dbl"], font=fs, fill=MUTED)

        d.text((X(1012), X(628)), t["note"], font=fs, fill=FAINT, anchor="rs")
        frames.append(im.resize((W, H), Image.LANCZOS))

    pal = frames[N - 20].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    durations = [70] * (N - 1) + [2400]                # hold the final frame
    path = OUT / f"recursion-loop.{lang}.gif"
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    return path


if __name__ == "__main__":
    print(f"trials {NT}, s0 {s0:.4f} -> best {TRIALS[-1]['best']:.4f}, kept {sum(r['verdict']=='adopt' for r in TRIALS)}")
    print("doubling times:", {r: [round(x, 2) for x in v[:5]] for r, v in DOUBLE.items()})
    for lang in ("zh", "en"):
        p = render(lang)
        print(p.relative_to(REPORT), f"{p.stat().st_size / 1024:.0f} KB")
