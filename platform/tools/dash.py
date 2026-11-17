#!/usr/bin/env python3
"""Text dashboard for RSI-mini on the cluster (stdlib only, read-only, cheap).

Run on the login node:  python3 $R/platform/tools/dash.py [--width 118]
It only reads files under $R (ports.json, GPU sidecar files, evald job dirs, runs/*/events.jsonl,
campaign status.json, vLLM/gateway logs). It never talks to Slurm, GPUs or services.
"""
import argparse
import glob
import json
import os
import statistics as st
import time
from datetime import datetime, timedelta, timezone

R = os.environ.get("R", "/scratch/user/rsi")
BJ = timezone(timedelta(hours=8))
NOW = time.time()
LANE_MAP = {1: ("node-a", 1), 2: ("node-a", 2), 3: ("node-b", 0), 4: ("node-b", 1), 5: ("node-b", 2)}
EVAL_EST_S = 480.0  # fallback per-eval wall estimate (300 s train + startup + 2 scoring passes)


def bj(t):
    return datetime.fromtimestamp(t, BJ).strftime("%H:%M")


def dur(s):
    s = int(max(0, s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def bar(frac, w=24):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * w)
    part = " ▏▎▍▌▋▊▉"[int((frac * w - full) * 8)] if full < w else ""
    return "█" * full + part + "·" * (w - full - (1 if part else 0))


def spark(vals, lo=None, hi=None):
    if not vals:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    ticks = "▁▂▃▄▅▆▇█"
    return "".join(ticks[min(7, int((v - lo) / (hi - lo) * 7.999))] if hi > lo else ticks[3] for v in vals)


def read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def read_events(p, tail_bytes=None):
    out = []
    try:
        with open(p, "rb") as f:
            if tail_bytes:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - tail_bytes))
                f.readline()
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except OSError:
        pass
    return out


def section(title, width):
    return f"── {title} " + "─" * max(0, width - len(title) - 4)


def gpu_table():
    g = {}
    for f in glob.glob(os.path.join(R, "run", "gpu_*.txt")):
        node = os.path.basename(f)[4:-4]
        try:
            lines = open(f).read().split("\n")
        except OSError:
            continue
        for ln in lines[1:]:
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 6:
                g[(node, int(p[0]))] = {"util": int(float(p[1])), "mem": int(float(p[2])), "tot": int(float(p[3])),
                                        "temp": p[4], "pw": float(p[5])}
    return g


def lane_jobs():
    """Current evald job per lane: newest job dir per lane whose state is running, or that has a fresh train log."""
    cur = {}
    jobs = sorted(glob.glob(os.path.join(R, "evald", "jobs", "j*")), key=os.path.getmtime, reverse=True)[:200]
    for jd in jobs:
        j = read_json(os.path.join(jd, "job.json"), {})
        lane = j.get("lane")
        if lane is None or lane in cur:
            continue
        logs = glob.glob(os.path.join(jd, "train_s*.log"))
        fresh = [l for l in logs if NOW - os.path.getmtime(l) < 90]
        done = os.path.exists(os.path.join(jd, "result.json")) or j.get("state") in ("done", "failed")
        if fresh and not done:
            t0 = j.get("t_start") or min(os.path.getctime(l) for l in fresh)
            seed = os.path.basename(fresh[0])[7:-4]
            cur[lane] = {"run": j.get("run", "?"), "seed": seed, "trial": j.get("trial"), "t0": t0,
                         "hash": j.get("artifact_hash", "")[:8]}
    return cur


def run_summaries(max_age_h=36):
    runs = []
    for ev in glob.glob(os.path.join(R, "runs", "*", "events.jsonl")):
        if NOW - os.path.getmtime(ev) > max_age_h * 3600:
            continue
        name = os.path.basename(os.path.dirname(ev))
        if name.startswith(("toolcall-", "selftest", "_untagged")):
            continue
        evs = read_events(ev)
        if not evs:
            continue
        rs = next((e for e in evs if e.get("ev") == "run_start"), {})
        results = [e for e in evs if e.get("ev") == "eval_result"]
        ok = [e for e in results if e.get("status") == "ok" and isinstance(e.get("value"), (int, float))]
        lower = rs.get("lower_is_better", True) if rs else True
        decisions = [e for e in evs if e.get("ev") == "decision"]
        adopted = [d for d in decisions if d.get("verdict") == "adopt"]
        proposals = [e for e in evs if e.get("ev") == "proposal"]
        submits = [e for e in evs if e.get("ev") == "eval_submit"]
        budget = (rs.get("budget") or {}).get("trials")
        trials = max([e.get("trial") or 0 for e in evs if e.get("trial") is not None] or [0])
        best_curve, best = [], None
        for e in ok:
            v = e["value"]
            best = v if best is None else (min(best, v) if lower else max(best, v))
            best_curve.append(best)
        t0 = evs[0].get("t", NOW)
        ended = any(e.get("ev") == "run_end" for e in evs)
        runs.append({"name": name, "policy": rs.get("policy", "-"), "task": rs.get("task", "-"),
                     "improver": (rs.get("improver") or {}).get("model", "-") if isinstance(rs.get("improver"), dict)
                     else rs.get("improver", "-"),
                     "budget": budget, "trials": max(trials, len(proposals)), "evals": len(results), "ok": len(ok),
                     "submitted": len(submits), "adopted": len(adopted), "best": best, "first": ok[0]["value"] if ok else None,
                     "curve": best_curve, "vals": [e["value"] for e in ok], "t0": t0, "tlast": evs[-1].get("t", NOW),
                     "ended": ended, "events": evs, "s_ref": rs.get("s_ref")})
    runs.sort(key=lambda r: r["tlast"], reverse=True)
    return runs


def noise_floor_block(runs, width, target=20):
    """All vanilla-* runs of the last few hours are the same artifact (vanilla start): pool their seeds."""
    nf = [r for r in runs if r["name"].startswith("vanilla-") and NOW - r["tlast"] < 4 * 3600]
    if not nf:
        return []
    res = []
    for r in nf:
        res += [e for e in r["events"] if e.get("ev") == "eval_result"]
    ok = [e for e in res if e.get("status") == "ok" and isinstance(e.get("value"), (int, float))]
    vals = [e["value"] for e in sorted(ok, key=lambda e: e.get("t", 0))]
    done = len(vals)
    lines = [section(f"G1a NOISE FLOOR · vanilla start · pooled over {len(nf)} run(s) · target {target} ok seeds", width)]
    walls = [e.get("wall_s") for e in ok if e.get("wall_s")]
    per = st.mean(walls) if walls else EVAL_EST_S
    remaining = max(0, target - done)
    eta = NOW + -(-remaining // 5) * per
    lines.append(f"  progress [{bar(done / target, 30)}] {done}/{target} ok seeds   per-seed wall ≈ {per:.0f}s   "
                 + (f"ETA {bj(eta)} (in {dur(eta - NOW)})" if remaining else "TARGET REACHED"))
    if vals:
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(f"  dev BPB   mean {st.mean(vals):.4f}  sd {sd:.4f}  min {min(vals):.4f}  max {max(vals):.4f}  "
                     f"(n={len(vals)})   per-seed {spark(vals)}")
    stat = {}
    for e in res:
        stat[e.get("status")] = stat.get(e.get("status"), 0) + 1
    lines.append(f"  all results by status {stat}   (crashes = the aborted OOM run)   "
                 f"ref: Recursive vanilla 1.0587 (B200+FA3); ours A100+SDPA")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=118)
    a = ap.parse_args()
    W = a.width
    out = []
    ports = read_json(os.path.join(R, "run", "ports.json"), {})
    gpus = gpu_table()
    cur = lane_jobs()
    runs = run_summaries()

    # ---- lanes
    out.append(section("EVAL LANES (NanoChat: 300 s train + startup + scoring)", W))
    out.append(f"  {'lane':4} {'node':10} {'gpu':3} {'util':>4} {'mem GB':>9} {'running':42} {'elapsed':>16}")
    for lane, (node, gi) in LANE_MAP.items():
        g = gpus.get((node, gi), {})
        c = cur.get(lane)
        what = f"{c['run'][:26]} s{c['seed']}" + (f" t{c['trial']}" if c and c.get("trial") is not None else "") if c else "idle"
        el = f"{dur(NOW - c['t0'])} [{bar((NOW - c['t0']) / EVAL_EST_S, 6)}]" if c else ""
        out.append(f"  {lane:<4} {node:10} {gi:<3} {str(g.get('util', '?')) + '%':>4} "
                   f"{(g.get('mem', 0) / 1024):5.1f}/{(g.get('tot', 0) / 1024):<3.0f} {what[:42]:42} {el:>16}")
    for node, gi, role in (("node-a", 0, "vLLM Qwen3.5-9B  (improver)"), ("node-c", 0, "vLLM Qwen3.8-27B TP0"),
                           ("node-c", 1, "vLLM Qwen3.8-27B TP1")):
        g = gpus.get((node, gi), {})
        out.append(f"  {'—':4} {node:10} {gi:<3} {str(g.get('util', '?')) + '%':>4} "
                   f"{(g.get('mem', 0) / 1024):5.1f}/{(g.get('tot', 0) / 1024):<3.0f} {role}")

    # ---- services
    out.append(section("SERVICES", W))
    svc = []
    for m in ("improver", "ladder-27b"):
        lg = os.path.join(R, "out", f"vllm-{m}.log")
        up = False
        n = 0
        try:
            txt = open(lg, errors="replace").read()
            up = "Application startup complete" in txt
            n = txt.count("POST /v1/chat/completions")
        except OSError:
            pass
        svc.append(f"vLLM {m}: {'UP' if up else 'down'} ({n} req)")
    for k in ("gateway", "gateway_agent", "evald"):
        svc.append(f"{k}: {'port ' + str(ports[k]) if k in ports else 'not started'}")
    out.append("  " + " · ".join(svc))

    # ---- noise floor
    out += noise_floor_block(runs, W)

    # ---- campaign
    camp = sorted(glob.glob(os.path.join(R, "runs", "campaign-*", "status.json")), key=os.path.getmtime)
    if camp:
        s = read_json(camp[-1], {})
        out.append(section(f"CAMPAIGN {os.path.basename(os.path.dirname(camp[-1]))} (updated {dur(NOW - os.path.getmtime(camp[-1]))} ago)", W))
        steps = s.get("steps", {})
        cnt = {}
        for v in steps.values():
            cnt[v.get("status")] = cnt.get(v.get("status"), 0) + 1
        total = max(1, len(steps))
        fin = sum(cnt.get(k, 0) for k in ("done", "ok", "skipped", "failed"))
        out.append(f"  steps [{bar(fin / total, 30)}] {fin}/{total} finished   {cnt}")
        for k, v in steps.items():
            if v.get("status") in ("running", "waiting", "paused"):
                out.append(f"  ▶ {k:34} {v.get('status'):8} {str(v.get('progress') or '')[:60]}")
        if s.get("paused_lanes"):
            out.append(f"  paused lanes: {s['paused_lanes']}")
        for d in (s.get("decisions") or [])[-3:]:
            out.append(f"  decision: {json.dumps(d)[:W - 14]}")
    else:
        out.append(section("CAMPAIGN", W))
        out.append("  not started yet (starts after gate G1a)")

    # ---- runs
    act = [r for r in runs if not r["name"].startswith("vanilla-")]
    if act:
        out.append(section("RUNS (improver loops)", W))
        out.append(f"  {'run':40} {'trials':>11} {'':14} {'evals':>5} {'adopt':>5} {'best':>8} {'Δ vs 1st':>9}  curve")
        for r in act[:8]:
            b = r["budget"] or 0
            frac = r["trials"] / b if b else 0
            rate = (r["tlast"] - r["t0"]) / max(1, r["trials"])
            eta = "" if r["ended"] or not b else f" ETA {bj(r['tlast'] + rate * (b - r['trials']))}"
            delta = (r["best"] - r["first"]) if r["best"] is not None and r["first"] is not None else None
            out.append(f"  {r['name'][:40]:40} {r['trials']:>4}/{b or '?':<5} [{bar(frac, 12)}] {r['evals']:>5} "
                       f"{r['adopted']:>5} {r['best'] if r['best'] is None else round(r['best'], 4)!s:>8} "
                       f"{'' if delta is None else f'{delta:+.4f}':>9}  {spark(r['curve'][-30:])}"
                       f"{' DONE' if r['ended'] else eta}")

    # ---- activity
    feed = []
    for r in runs[:6]:
        for e in r["events"][-40:]:
            evn = e.get("ev")
            if evn == "proposal":
                txt = f"proposal  {e.get('hypothesis', '')[:70]!r}"
            elif evn == "eval_result":
                v = e.get("value")
                txt = f"eval      {e.get('status')} {'' if v is None else round(v, 4)} lane{e.get('lane', '?')} {e.get('wall_s', '')}s"
            elif evn == "decision":
                txt = f"decision  {e.get('verdict')}  best {e.get('best_before')} → {e.get('best_after')}  {str(e.get('reason', ''))[:40]}"
            elif evn == "audit_flag":
                txt = f"AUDIT     {e.get('check')} {str(e.get('detail', ''))[:60]}"
            elif evn in ("lessons_update", "weights_update", "run_start", "run_end", "agent_spawn"):
                txt = f"{evn:9} {json.dumps({k: v for k, v in e.items() if k not in ('t', 'run', 'ev')})[:70]}"
            else:
                continue
            feed.append((e.get("t", 0), r["name"], e.get("agent", ""), txt))
    feed.sort()
    if feed:
        out.append(section("RECENT ACTIVITY", W))
        for t, run, agent, txt in feed[-10:]:
            out.append(f"  {bj(t)} {run[:22]:22} {agent[:14]:14} {txt}"[:W])

    print("\n".join(out))


if __name__ == "__main__":
    main()
