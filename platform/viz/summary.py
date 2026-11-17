#!/usr/bin/env python3
"""Run statistics from an events.jsonl (RSI-mini event schema v1).

Per run: per-thread trial counts and adoptions, adoption rate, eval status mix,
GPU-lane utilization, the thinking / tool-use / evaluation time split, and the
nAUC of the best-so-far curve when it can be computed.

  python3 summary.py samples/synth_events.jsonl
  python3 summary.py run/events.jsonl --s-ref 0.99 --json

Definitions (match research/platform_design.md §1):
  g(t)  = (s0 - best_t) / (s0 - s_ref)  (lower-is-better; sign flipped otherwise)
  nAUC  = (1/N) * sum_{t=1..N} g(t), t = evaluator calls (trials), N = budget.trials
          if logged, else the number of decided trials. If the run decided fewer
          than N trials, best_t is carried forward and the row is marked "padded".
  s0    = baseline eval (trial 0) value, else best_before of the first decision.
  s_ref = --s-ref, else run_start.task.s_ref / run_start.s_ref if the platform logs
          one. Without s_ref, nAUC is not computable; the logged run_end.nAUC (if any)
          and the unnormalized mean gain (1/N) sum (s0 - best_t) are shown instead.
  Trial -> thread attribution: the proposal's own agent if it is not the
  orchestrator, else the non-orchestrator agent that sent a message / made a call
  tagged with that trial (first one wins).
  Time split: thinking = sum llm_call.latency_s, tool use = sum tool_call.latency_s,
  evaluation = sum of lane busy time (eval_submit -> eval_result, or wall_s).
  These are summed agent-seconds, so they can exceed wall-clock with parallel agents.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict, defaultdict


def load(path):
    evs = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(e, dict) and "t" in e and "ev" in e:
                e["_i"] = i
                evs.append(e)
            else:
                bad += 1
    evs.sort(key=lambda e: (e["t"], e["_i"]))
    return evs, bad


def lower_is_better(decisions, metric):
    for d in decisions:
        b, a = d.get("best_before"), d.get("best_after")
        if d.get("verdict") == "adopt" and b is not None and a is not None and a != b:
            return a < b
    return not re.search(r"acc|score|reward|pass|win|f1|auc", metric or "", re.I)


def find_s_ref(run_start, cli):
    if cli is not None:
        return cli, "cli"
    if run_start:
        task = run_start.get("task")
        if isinstance(task, dict) and task.get("s_ref") is not None:
            return float(task["s_ref"]), "run_start.task.s_ref"
        if run_start.get("s_ref") is not None:
            return float(run_start["s_ref"]), "run_start.s_ref"
    return None, None


def summarize_run(evs, s_ref_cli=None):
    rid = evs[0]["run"]
    rs = next((e for e in evs if e["ev"] == "run_start"), None)
    re_ = next((e for e in evs if e["ev"] == "run_end"), None)
    t_start = rs["t"] if rs else evs[0]["t"]
    t_end = re_["t"] if re_ else evs[-1]["t"]
    wall = max(1e-9, t_end - t_start)
    gen = (rs or evs[0]).get("gen")

    roles = Counter(e.get("role", "?") for e in evs if e["ev"] == "agent_spawn")
    decisions = [e for e in evs if e["ev"] == "decision"]
    results = [e for e in evs if e["ev"] == "eval_result"]
    proposals = [e for e in evs if e["ev"] == "proposal"]
    metric = next((e.get("metric") for e in results if e.get("metric")), "metric")
    lower = lower_is_better(decisions, metric)

    # ---- trial -> thread attribution
    owner = {}
    for p in proposals:
        if p.get("agent") not in (None, "orchestrator"):
            owner[p.get("trial")] = p["agent"].split("/")[0]
    for e in evs:
        tr = e.get("trial")
        if tr is None or tr in owner or tr == 0:
            continue
        if e["ev"] in ("message", "tool_call", "llm_call") and e.get("agent") not in ("orchestrator", "evald"):
            owner[tr] = e["agent"].split("/")[0]
    threads = OrderedDict()
    for e in evs:
        if e["ev"] == "agent_spawn" and e.get("role") == "researcher":
            threads.setdefault(e["agent"], Counter())
    for p in proposals:
        threads.setdefault(owner.get(p.get("trial"), "unattributed"), Counter())["trials"] += 1
    for d in decisions:
        th = owner.get(d.get("trial"), "unattributed")
        threads.setdefault(th, Counter())[d.get("verdict", "?")] += 1
    status_of = {r.get("trial"): r.get("status") for r in results}
    for tr, st in status_of.items():
        if tr in owner and st != "ok":
            threads[owner[tr]]["failed_eval"] += 1

    verdicts = Counter(d.get("verdict") for d in decisions)
    decided = sum(v for k, v in verdicts.items() if k != "rollback")
    statuses = Counter(r.get("status", "?") for r in results if r.get("trial") != 0)

    # ---- lanes
    open_jobs, busy = {}, defaultdict(float)
    eval_time = 0.0
    for e in evs:
        if e["ev"] == "eval_submit":
            open_jobs[e.get("artifact_hash")] = (e["t"], str(e.get("lane")))
        elif e["ev"] == "eval_result":
            j = open_jobs.pop(e.get("artifact_hash"), None)
            if j:
                dt = e["t"] - j[0]
                busy[j[1]] += dt
            else:
                dt = float(e.get("wall_s") or 0)
                busy[str(e.get("lane", "?"))] += dt
            eval_time += dt
    for h, (t0, lane) in open_jobs.items():  # still running at end of log
        busy[lane] += t_end - t0
        eval_time += t_end - t0
    lanes = {k: round(v / wall, 3) for k, v in sorted(busy.items())}

    # ---- time split
    think = sum(float(e.get("latency_s") or 0) for e in evs if e["ev"] == "llm_call")
    tool_by = Counter()
    for e in evs:
        if e["ev"] == "tool_call":
            tool = e.get("tool") or "other"
            tool_by["mcp" if tool.startswith("mcp") else tool] += float(e.get("latency_s") or 0)
    tool = sum(tool_by.values())
    tot = think + tool + eval_time or 1e-9
    llm_tokens = sum(int(e.get("prompt_tokens") or 0) + int(e.get("completion_tokens") or 0) for e in evs if e["ev"] == "llm_call")

    # ---- best-so-far and nAUC
    base = next((r for r in results if r.get("trial") == 0 and r.get("value") is not None), None)
    s0 = base["value"] if base else next((d.get("best_before") for d in decisions if d.get("best_before") is not None), None)
    best = s0
    trace = []
    for d in decisions:
        if d.get("best_after") is not None:
            best = d["best_after"]
        if d.get("verdict") != "rollback":
            trace.append(best)
    budget = (rs or {}).get("budget") or {}
    N = int(budget.get("trials") or 0) or len(trace)
    padded = 0 < len(trace) < N
    tr = (trace + [trace[-1]] * (N - len(trace)))[:N] if trace else []
    s_ref, s_ref_src = find_s_ref(rs, s_ref_cli)
    sign = 1 if lower else -1
    nauc = None
    raw_gain = None
    if s0 is not None and tr:
        raw_gain = sum(sign * (s0 - b) for b in tr) / N
        if s_ref is not None and s_ref != s0:
            nauc = sum((s0 - b) / (s0 - s_ref) for b in tr) / N

    return OrderedDict(
        run=rid, gen=gen, wall_s=round(wall, 1), policy=(rs or {}).get("policy"),
        task=(rs or {}).get("task"), metric=metric, lower_is_better=lower,
        agents_spawned=dict(roles),
        threads={k: dict(v) for k, v in threads.items()},
        trials_proposed=len(proposals), trials_decided=decided, verdicts=dict(verdicts),
        adoption_rate=round(verdicts.get("adopt", 0) / decided, 3) if decided else None,
        eval_status=dict(statuses), audit_flags=sum(1 for e in evs if e["ev"] == "audit_flag"),
        lessons_updates=sum(1 for e in evs if e["ev"] == "lessons_update"),
        lane_utilization=lanes,
        lane_utilization_mean=round(sum(lanes.values()) / len(lanes), 3) if lanes else None,
        time_split_s=OrderedDict(thinking=round(think, 1), tool_use=round(tool, 1), evaluation=round(eval_time, 1)),
        time_split_pct=OrderedDict(thinking=round(100 * think / tot, 1), tool_use=round(100 * tool / tot, 1), evaluation=round(100 * eval_time / tot, 1)),
        tool_time_s={k: round(v, 1) for k, v in tool_by.most_common()},
        tool_calls=sum(1 for e in evs if e["ev"] == "tool_call"),
        llm_calls=sum(1 for e in evs if e["ev"] == "llm_call"), llm_tokens=llm_tokens,
        s0=s0, best_final=best, best_gain=None if s0 is None or best is None else round(sign * (s0 - best), 6),
        nAUC_N=N, nAUC_padded=padded, s_ref=s_ref, s_ref_source=s_ref_src,
        nAUC=None if nauc is None else round(nauc, 4),
        mean_gain_unnormalized=None if raw_gain is None else round(raw_gain, 6),
        nAUC_logged=(re_ or {}).get("nAUC"),
    )


def fmt_run(s):
    L = []
    L.append(f"== {s['run']}  (gen {s['gen']}, {s['policy']}, {s['task'] if isinstance(s['task'], str) else json.dumps(s['task'])})")
    L.append(f"   wall {s['wall_s'] / 60:.1f} min · metric {s['metric']} ({'lower' if s['lower_is_better'] else 'higher'} is better) · agents spawned {s['agents_spawned']}")
    ar = "n/a" if s["adoption_rate"] is None else f"{100 * s['adoption_rate']:.1f}%"
    L.append(f"   trials proposed {s['trials_proposed']}, decided {s['trials_decided']} · verdicts {s['verdicts']} · adoption rate {ar}")
    L.append(f"   eval status (excl. baseline) {s['eval_status']} · audit flags {s['audit_flags']} · lessons updates {s['lessons_updates']}")
    L.append("   per thread:      trials  adopt  reject  explore  rollback  failed_eval")
    for th, c in s["threads"].items():
        L.append(f"     {th:<15} {c.get('trials', 0):>5}  {c.get('adopt', 0):>5}  {c.get('reject', 0):>6}  {c.get('explore', 0):>7}  {c.get('rollback', 0):>8}  {c.get('failed_eval', 0):>11}")
    lu = ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["lane_utilization"].items())
    L.append(f"   GPU lane utilization (busy / run wall): {lu or 'n/a'}" + (f"  · mean {100 * s['lane_utilization_mean']:.1f}%" if s['lane_utilization_mean'] is not None else ""))
    ts, tp = s["time_split_s"], s["time_split_pct"]
    L.append(f"   time split (agent-seconds): thinking {ts['thinking'] / 60:.1f} min ({tp['thinking']}%) · tool use {ts['tool_use'] / 60:.1f} min ({tp['tool_use']}%) · "
             f"evaluation {ts['evaluation'] / 60:.1f} min ({tp['evaluation']}%)")
    L.append(f"   tool time by tool (s): {s['tool_time_s']} · {s['tool_calls']} tool calls · {s['llm_calls']} llm calls · {s['llm_tokens']:,} tokens")
    L.append(f"   s0 {s['s0']} -> best {s['best_final']} (gain {s['best_gain']})")
    if s["nAUC"] is not None:
        L.append(f"   nAUC {s['nAUC']} (N={s['nAUC_N']}{', padded' if s['nAUC_padded'] else ''}, s_ref {s['s_ref']} from {s['s_ref_source']})"
                 + (f" · logged run_end.nAUC {s['nAUC_logged']}" if s["nAUC_logged"] is not None else ""))
    else:
        L.append(f"   nAUC: not computable (no s_ref; pass --s-ref). logged run_end.nAUC = {s['nAUC_logged']} · "
                 f"unnormalized mean gain (1/N)Σ(s0-best_t) = {s['mean_gain_unnormalized']} (N={s['nAUC_N']})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+")
    ap.add_argument("--s-ref", type=float, default=None, help="reference best score for nAUC normalization")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = ap.parse_args()
    out = []
    for path in a.events:
        evs, bad = load(path)
        if bad:
            print(f"warning: {path}: {bad} unparseable line(s) skipped", file=sys.stderr)
        by_run = OrderedDict()
        for e in evs:
            by_run.setdefault(e.get("run"), []).append(e)
        for rid, revs in by_run.items():
            s = summarize_run(revs, a.s_ref)
            s["file"] = path
            out.append(s)
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print("\n\n".join(fmt_run(s) for s in out))


if __name__ == "__main__":
    main()
