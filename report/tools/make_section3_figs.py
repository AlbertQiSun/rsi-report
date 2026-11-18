"""Figure 7 for report §3: where a real research loop spends its time.

(a) Timeline of the Qwen3.5-9B NanoChat run (176.6 min), from research/replay/final/ovn1118b-nc-g0/events.jsonl:
    top   = best val_bpb so far (running minimum across the 3 parallel threads),
    lanes = the 3 GPU evaluation lanes; each block is one evaluation job (submit → result), coloured by outcome,
    agents= the 3 research threads; each sliver is one model call (latency).
(b) Agent-time split (evaluation vs thinking), from nc-g0-final.summary.txt.
(c) Amdahl's law on that split: loop speed-up if only thinking, or only evaluation, gets faster.
Output: report/site/public/figs/fig3-amdahl.{zh,en}.svg
"""
import json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_section2_figs import SVG, W, BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL, legend  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = [json.loads(l) for l in (ROOT / "research/replay/final/ovn1118b-nc-g0/events.jsonl").read_text().splitlines()]
SUMM = (ROOT / "research/replay/final/nc-g0-final.summary.txt").read_text()
m = re.search(r"thinking ([\d.]+) min \(([\d.]+)%\) · tool use ([\d.]+) min \(([\d.]+)%\) · evaluation ([\d.]+) min \(([\d.]+)%\)", SUMM)
THINK, TOOLS, EVAL = float(m.group(2)) / 100, float(m.group(4)) / 100, float(m.group(6)) / 100

T0 = next(e["t"] for e in EV if e["ev"] == "run_start")
T1 = next(e["t"] for e in EV if e["ev"] == "run_end")
SPAN = (T1 - T0) / 60
SUB = {e["job_id"]: e for e in EV if e["ev"] == "eval_submit"}
VERD = {e["trial"]: e["verdict"] for e in EV if e["ev"] == "decision"}
JOBS = []
for r in (e for e in EV if e["ev"] == "eval_result"):
    s = SUB[r["job_id"]]
    JOBS.append({"lane": r["lane"], "a": (r["t"] - (r.get("wall_s") or 0) - T0) / 60, "b": (r["t"] - T0) / 60, "trial": r["trial"], "status": r["status"],
                 "value": r.get("value") if r["status"] == "ok" else None, "wall": r.get("wall_s"),
                 "verdict": "baseline" if r["trial"] == 0 else VERD.get(r["trial"], "?")})
LLM = [((e["t"] - e["latency_s"] - T0) / 60, (e["t"] - T0) / 60, e["agent"]) for e in EV if e["ev"] == "llm_call"]
DEC = sorted((e for e in EV if e["ev"] == "decision"), key=lambda x: x["t"])
BEST, cur = [], None
for d in DEC:
    v = d["best_after"]
    cur = v if cur is None else min(cur, v)
    BEST.append(((d["t"] - T0) / 60, cur))
LANES = sorted({j["lane"] for j in JOBS})
AGENTS = sorted({a for _, _, a in LLM})
LANE_BUSY_CALC = {ln: sum(j["b"] - j["a"] for j in JOBS if j["lane"] == ln) / SPAN for ln in LANES}
LANE_MEAN = float(re.search(r"mean ([\d.]+)%", SUMM).group(1)) / 100          # authoritative, from the run summary


def loop_speedup(frac_fast, s):
    return 1 / ((1 - frac_fast) + frac_fast / s)


def render(lang):
    zh = lang == "zh"
    H = 560
    s = SVG(H)
    # ---------------- (a) timeline ----------------
    s.text(16, 20, "(a) 一次真实研究运行的时间线：Qwen3.5-9B 改进 NanoChat，176.6 分钟" if zh else
           "(a) Timeline of a real research run: Qwen3.5-9B improving NanoChat, 176.6 minutes", size=13, fill=INK, weight=500)
    legend(s, 16, 42, [("评测后保留" if zh else "evaluated, kept", BLUE, "bar"), ("评测后丢弃" if zh else "evaluated, discarded", NEUTRAL, "bar"),
                       ("评测崩溃" if zh else "evaluation crashed", ORANGE, "bar"), ("模型思考（一次调用）" if zh else "model thinking (one call)", INK, "bar")])
    x0, xw = 112, W - 112 - 18
    X = lambda mnt: x0 + mnt / SPAN * xw
    # best-so-far strip
    by0, bh = 58, 58
    vals = [v for _, v in BEST]
    vhi, vlo = max(vals) + 0.005, min(vals) - 0.005
    Yb = lambda v: by0 + (vhi - v) / (vhi - vlo) * bh
    s.text(16, by0 + 14, "当前最好" if zh else "best so far", size=11.5, fill=INK2)
    s.text(16, by0 + 28, "val_bpb", size=11, fill=MUTED)
    for v in (1.25, 1.30, 1.35, 1.40):
        if vlo <= v <= vhi:
            s.line(x0, Yb(v), x0 + xw, Yb(v), GRID)
            s.text(x0 - 6, Yb(v) + 4, f"{v:.2f}", size=10, fill=MUTED, anchor="end")
    pts, prev = [], None
    for t, v in BEST:
        if prev is not None:
            pts.append(f"{X(t):.1f},{Yb(prev):.1f}")
        pts.append(f"{X(t):.1f},{Yb(v):.1f}")
        prev = v
    pts.append(f"{X(SPAN):.1f},{Yb(prev):.1f}")
    s.raw(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    s.text(X(SPAN), Yb(prev) + 14, f"{prev:.4f}", size=10.5, fill=INK2, anchor="end")
    s.text(X(0) + 4, Yb(BEST[0][1]) - 5, f"{BEST[0][1]:.4f}", size=10.5, fill=INK2)
    # GPU lanes
    ly0, lh, lgap = 136, 20, 6
    for i, ln in enumerate(LANES):
        y = ly0 + i * (lh + lgap)
        s.text(16, y + 14, (f"GPU 评测线 {ln}" if zh else f"GPU lane {ln}"), size=11.5, fill=INK2)
        s.rect(x0, y, xw, lh, "#f3f1ec", rx=3)
        for j in (j for j in JOBS if j["lane"] == ln):
            col = ORANGE if j["status"] != "ok" else (BLUE if j["verdict"] in ("adopt", "baseline") else NEUTRAL)
            if zh:
                desc = "起点基线" if j["verdict"] == "baseline" else ("保留" if j["verdict"] == "adopt" else "丢弃")
                tip = (f"尝试 #{j['trial']} · 评测线 {ln} · 用时 {j['wall']:.0f} 秒 · " +
                       (f"val_bpb {j['value']:.4f} · {desc}" if j["value"] and j["value"] < 9 else ("评测崩溃" if j["status"] != "ok" else desc)))
            else:
                desc = "baseline" if j["verdict"] == "baseline" else ("kept" if j["verdict"] == "adopt" else "discarded")
                tip = (f"attempt #{j['trial']} · lane {ln} · {j['wall']:.0f} s · " +
                       (f"val_bpb {j['value']:.4f} · {desc}" if j["value"] and j["value"] < 9 else ("crashed" if j["status"] != "ok" else desc)))
            s.rect(X(j["a"]) + 0.6, y + 1, max(1.2, X(j["b"]) - X(j["a"]) - 1.2), lh - 2, col, tip=tip + " [events.jsonl]", rx=2)
    # agent threads
    ay0 = ly0 + len(LANES) * (lh + lgap) + 10
    for i, ag in enumerate(AGENTS):
        y = ay0 + i * (lh + lgap)
        s.text(16, y + 14, (f"研究线程 {i + 1}" if zh else f"research thread {i + 1}"), size=11.5, fill=INK2)
        busy = sum(b - a for a, b, g in LLM if g == ag) / SPAN
        s.rect(x0, y, xw, lh, "#f3f1ec", rx=3)
        parts = []
        for a, b, g in LLM:
            if g == ag:
                parts.append(f"M{X(a):.2f},{y + 3}h{max(0.8, X(b) - X(a)):.2f}v{lh - 6}h-{max(0.8, X(b) - X(a)):.2f}Z")
        n = sum(1 for *_, g in LLM if g == ag)
        tip = (f"研究线程 {i + 1}：{n} 次模型调用，合计占全程 {busy * 100:.1f}%" if zh else f"research thread {i + 1}: {n} model calls, {busy * 100:.1f}% of the run")
        s.raw(f'<path d="{"".join(parts)}" fill="{INK}"><title>{tip} [events.jsonl]</title></path>')
    ty = ay0 + len(AGENTS) * (lh + lgap) + 4
    for mnt in range(0, int(SPAN) + 1, 30):
        s.line(X(mnt), by0, X(mnt), ty - 2, GRID)
        s.text(X(mnt), ty + 12, f"{mnt}", size=10.5, fill=MUTED, anchor="middle")
    s.text(x0 + xw, ty + 12, "分钟" if zh else "min", size=10.5, fill=MUTED, anchor="end")
    s.text(16, ty + 12, (f"评测线平均占用 {LANE_MEAN * 100:.1f}%" if zh else f"lanes busy {LANE_MEAN * 100:.1f}% on average"), size=10, fill=MUTED)

    # ---------------- (b) split ----------------
    py = ty + 40
    s.line(16, py - 14, W - 16, py - 14, GRID)
    s.text(16, py + 6, "(b) agent 时间的构成" if zh else "(b) Split of agent time", size=13, fill=INK, weight=500)
    bx, bw = 16, 280
    yb = py + 22
    xx = bx
    for label, f, col in ((("评测（训练 + 打分）" if zh else "evaluation (train + score)"), EVAL, ORANGE), (("模型思考" if zh else "model thinking"), THINK, INK)):
        s.rect(xx, yb, f * bw - 2, 22, col, tip=f"{label}：{f * 100:.1f}% [nc-g0-final.summary.txt]", rx=3)
        xx += f * bw
    s.text(bx, yb + 44, f"{EVAL * 100:.1f}%", size=18, fill=INK, weight=500)
    s.text(bx, yb + 60, "评测（训练 + 打分）" if zh else "evaluation (train + score)", size=11, fill=INK2)
    s.text(bx + 200, yb + 44, f"{THINK * 100:.1f}%", size=18, fill=INK, weight=500)
    s.text(bx + 200, yb + 60, "模型思考" if zh else "model thinking", size=11, fill=INK2)
    s.text(bx, yb + 80, (f"工具调用 {TOOLS * 100:.1f}%" if zh else f"tool calls {TOOLS * 100:.1f}%"), size=10.5, fill=MUTED)

    # ---------------- (c) Amdahl ----------------
    cx0 = 360
    s.text(cx0, py + 6, "(c) 只加速一部分时，整个循环快多少" if zh else "(c) Loop speed-up if only one part gets faster", size=13, fill=INK, weight=500)
    px0, pw, py0, ph = cx0 + 34, 250, py + 22, 96
    smax, ymax = 10, 5
    Xa = lambda v: px0 + (v - 1) / (smax - 1) * pw
    Ya = lambda v: py0 + ph - (v - 1) / (ymax - 1) * ph
    for v in (1, 2, 3, 4, 5):
        s.line(px0, Ya(v), px0 + pw, Ya(v), GRID)
        s.text(px0 - 6, Ya(v) + 4, f"{v}×", size=10, fill=MUTED, anchor="end")
    for v in (1, 2, 4, 6, 8, 10):
        s.text(Xa(v), py0 + ph + 14, f"{v}×", size=10, fill=MUTED, anchor="middle")
    s.text(px0 + pw / 2, py0 + ph + 28, "被加速部分的提速倍数" if zh else "speed-up of the accelerated part", size=10.5, fill=MUTED, anchor="middle")
    for frac, col, name in ((THINK, INK, "模型思考" if zh else "thinking"), (EVAL, ORANGE, "评测" if zh else "evaluation")):
        pts = " ".join(f"{Xa(1 + i * (smax - 1) / 90):.1f},{Ya(min(ymax, loop_speedup(frac, 1 + i * (smax - 1) / 90))):.1f}" for i in range(91))
        s.raw(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for sp in (2, 10):
            v = loop_speedup(frac, sp)
            tip = (f"{name}快 {sp} 倍 → 整个循环快 {v:.2f} 倍" if zh else f"{name} {sp}× faster → loop {v:.2f}× faster")
            s.circle(Xa(sp), Ya(min(ymax, v)), 4.5, col, stroke="#ffffff", tip=tip + " [Amdahl；比例取自 (b)]")
        s.text(Xa(10) + 7, Ya(min(ymax, loop_speedup(frac, 10))) + 4, f"{loop_speedup(frac, 10):.2f}×", size=10.5, fill=INK2)
    s.text(Xa(2) + 6, Ya(loop_speedup(EVAL, 2)) - 7, f"{loop_speedup(EVAL, 2):.2f}×", size=10.5, fill=INK2)
    s.text(Xa(2) + 6, Ya(loop_speedup(THINK, 2)) - 6, f"{loop_speedup(THINK, 2):.2f}×", size=10.5, fill=INK2)
    s.text(Xa(10) + 7, Ya(loop_speedup(EVAL, 10)) + 18, "评测" if zh else "evaluation", size=10, fill=MUTED)
    s.text(Xa(10) + 7, Ya(loop_speedup(THINK, 10)) - 8, "思考" if zh else "thinking", size=10, fill=MUTED)
    s.save("fig3-amdahl", lang, "研究循环的时间分配" if zh else "Time split of the research loop")


if __name__ == "__main__":
    for lang in ("zh", "en"):
        render(lang)
    busy_agents = {a: round(sum(b - x for x, b, g in LLM if g == a) / SPAN, 3) for a in AGENTS}
    print("lane busy (calc from wall_s):", {k: round(v, 3) for k, v in LANE_BUSY_CALC.items()})
    print(f"span {SPAN:.1f} min; jobs {len(JOBS)}; lane mean {LANE_MEAN:.3f}; thread busy {busy_agents}; best {BEST[0][1]:.4f} -> {BEST[-1][1]:.4f}")
    print("speedups think/eval @2x:", round(loop_speedup(THINK, 2), 3), round(loop_speedup(EVAL, 2), 3))
