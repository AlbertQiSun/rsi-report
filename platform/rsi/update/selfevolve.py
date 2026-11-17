"""Self-evolving data (build plan §4.4, Apex): failures -> synthesize -> verify -> SFT.

    classify_failures(run_dirs)                 -> {category: count, "examples": {...}}
    repair_config(obj, fallback)                -> nearest legal µAR config (clipped / filled) or None
    synthesize_repairs(run_dirs, verify)        -> SFT examples tagged "repaired", evaluator-verified only

Categories: illegal_tool_call (tool-call arguments not valid JSON), crash (eval crash / nan / timeout),
illegal_config (L0 audit_fail), no_change (episode made no edit), duplicate (artifact already tried in this
run), no_improvement (legal but rejected), budget_exhausted (episode not ok / harness timeout).
Repairs apply to JSON-proposal trajectories: the assistant's final content contains a ``{"config": ...}``
(or bare config) object. Tool-calling trajectories are classified but not rewritten.
``verify(task, cfg) -> (ok: bool, value: float | None)`` must call the sealed evaluator's dev path; a repair
is kept only if it is legal AND beats the parent state's dev score (never hand-picked).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from rsi.tasks.muar import schema
from rsi.update.trajectories import load_calls, run_task, trials


def classify_failures(run_dirs: list[str]) -> dict:
    counts: Counter = Counter()
    examples: dict = defaultdict(list)
    for rd in run_dirs:
        run_id = os.path.basename(os.path.normpath(rd))
        tr = trials(rd)
        seen = set()
        for t, r in sorted(tr.items()):
            st, es = r.get("status"), r.get("eval_status")
            cat = None
            if st == "no_change":
                cat = "budget_exhausted" if r.get("episode_ok") is False else "no_change"
            elif r.get("artifact_hash") and r["artifact_hash"] in seen and st != "baseline":
                cat = "duplicate"
            elif es == "audit_fail" or st == "audit_fail":
                cat = "illegal_config"
            elif es in ("crash", "nan", "timeout") or st == "crash":
                cat = "crash"
            elif st in ("discard", "rolled_back", "rejected"):
                cat = "no_improvement"
            if r.get("artifact_hash"):
                seen.add(r["artifact_hash"])
            if cat:
                counts[cat] += 1
                if len(examples[cat]) < 5:
                    examples[cat].append({"run": run_id, "trial": t, "feedback": (r.get("feedback") or "")[:200]})
        for c in load_calls(rd):
            for tc in c.response.get("tool_calls") or []:
                args = (tc.get("function") or tc).get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args or "{}")
                    except json.JSONDecodeError:
                        counts["illegal_tool_call"] += 1
                        if len(examples["illegal_tool_call"]) < 5:
                            examples["illegal_tool_call"].append({"run": run_id, "trial": c.trial, "args": args[:200]})
    return {**dict(counts), "examples": dict(examples)}


def repair_config(obj, fallback: dict) -> dict | None:
    """Clip numbers into range, coerce ints, fill missing keys from ``fallback``, drop unknown keys."""
    if not isinstance(obj, dict):
        return None
    cfg = {}
    for k in schema.KEYS:
        v = obj.get(k, fallback.get(k))
        if k in schema.ENUMS:
            v = v if v in schema.ENUMS[k] else fallback.get(k)
        elif k in schema.FLOATS:
            lo, hi, _ = schema.FLOATS[k]
            try:
                v = float(min(hi, max(lo, float(v))))
            except (TypeError, ValueError):
                v = fallback.get(k)
        else:
            lo, hi = schema.INTS[k]
            try:
                v = int(min(hi, max(lo, int(round(float(v))))))
            except (TypeError, ValueError):
                v = fallback.get(k)
        cfg[k] = v
    try:
        return schema.validate_config(cfg)
    except schema.IllegalConfig:
        return None


_OBJ = re.compile(r"\{.*\}", re.S)


def _extract_config(text: str):
    m = _OBJ.search(text or "")
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        return obj, obj["config"]
    return None, obj if isinstance(obj, dict) else None


def synthesize_repairs(run_dirs: list[str], verify, *, max_per_run: int = 20) -> list[dict]:
    from rsi.tasks.muar import start_config

    out = []
    for rd in run_dirs:
        task = run_task(rd)
        if not (task or "").startswith("muar/"):
            continue
        tr = trials(rd)
        fallback = start_config(task)
        best_by_parent = {}
        for r in tr.values():   # dev score of every legal artifact = score of that parent state
            if r.get("artifact_hash") and r.get("eval_status") == "ok" and r.get("dev") is not None:
                best_by_parent[r["artifact_hash"]] = r["dev"]
        n = 0
        for c in load_calls(rd):
            r = tr.get(c.trial) if c.trial is not None else None
            if not r or r.get("eval_status") != "audit_fail" or n >= max_per_run:
                continue
            wrapper, cfg = _extract_config(c.response.get("content") or "")
            fixed = repair_config(cfg, fallback) if cfg is not None else None
            if fixed is None:
                continue
            ok, value = verify(task, fixed)
            parent_dev = best_by_parent.get(r.get("parent_hash"))
            if not ok or value is None or (parent_dev is not None and value >= parent_dev):
                continue
            new = dict(wrapper) if wrapper else None
            content = json.dumps({**new, "config": fixed}) if new else json.dumps(fixed)
            out.append({"messages": list(c.messages) + [{"role": "assistant", "content": content}],
                        "meta": {"run": os.path.basename(os.path.normpath(rd)), "task": task, "trial": c.trial,
                                 "label": "repaired", "verified_value": value}})
            n += 1
    return out
