"""Readers for a run directory: gateway trajectories, ledger, run task / meta.

Gateway record format (worker C, rsi/gateway/server.py Gateway.record). Tolerant; per JSONL line:
  request messages : rec["request"]["messages"] | rec["body"]["messages"] | rec["messages"]
  tools            : rec["request"]["tools"] | rec["tools"]
  response message : rec["response"]["message"] (rsi.gateway) | rec["response"]["choices"][0]["message"]
                     | rec["response_message"] | rec["completion"]
  tags             : run / agent / gen / trial at top level, or in rec["meta"] / rec["tags"],
                     or as X-RSI-* entries in rec["headers"]
  time             : rec["t"] | rec["ts"] | rec["time"]
  failure          : rec["status"] != "ok" (string) or an int >= 400, rec["http_status"] >= 400,
                     or rec["error"] -> skipped (streams cut by the client are not training data)
Files: runs/<id>/trajectories/gateway/**/*.jsonl (and *.jsonl.gz is not supported on purpose).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

from rsi.runs import read_json, read_jsonl

ACCEPT_STATUSES = {"keep", "adopted"}                  # trial made the thread's new best
REJECT_STATUSES = {"discard", "rolled_back", "rejected"}  # legal (eval ok) but not adopted
NON_TRIAL_STATUSES = {"baseline", "pending", "no_change"}


@dataclass
class Call:
    run: str
    agent: str
    gen: int | None
    trial: int | None
    t: float
    model: str | None
    messages: list
    tools: list | None
    response: dict
    source: str = ""
    extra: dict = field(default_factory=dict)


def _tag(rec: dict, key: str) -> Any:
    for container in (rec, rec.get("meta") or {}, rec.get("tags") or {}):
        if isinstance(container, dict) and container.get(key) is not None:
            return container[key]
    hdr = rec.get("headers") or {}
    if isinstance(hdr, dict):
        for k, v in hdr.items():
            if k.lower() == f"x-rsi-{key}":
                return v
    return None


def _int(v) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def parse_record(rec: dict, source: str = "") -> Call | None:
    status, http = rec.get("status"), rec.get("http_status")
    if rec.get("error") or (isinstance(http, int) and http >= 400) or (isinstance(status, int) and status >= 400):
        return None
    if isinstance(status, str) and status != "ok":      # gateway: ok | http_<n> | client_disconnected | ...
        return None
    req = rec.get("request") or rec.get("body") or rec
    if not isinstance(req, dict):
        return None
    messages = req.get("messages") or rec.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    resp = rec.get("response")
    msg = None
    if isinstance(resp, dict) and isinstance(resp.get("choices"), list) and resp["choices"]:
        msg = resp["choices"][0].get("message")
    elif isinstance(resp, dict) and isinstance(resp.get("message"), dict):   # rsi.gateway: {message, finish_reason}
        msg = resp["message"]
    msg = msg or rec.get("response_message") or rec.get("completion")
    if isinstance(msg, str):
        msg = {"role": "assistant", "content": msg}
    if not isinstance(msg, dict):
        return None
    msg = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_content") and v is not None}
    msg["role"] = "assistant"
    if not msg.get("content") and not msg.get("tool_calls"):
        return None
    return Call(run=str(_tag(rec, "run") or ""), agent=str(_tag(rec, "agent") or ""), gen=_int(_tag(rec, "gen")),
                trial=_int(_tag(rec, "trial")), t=float(rec.get("t") or rec.get("ts") or rec.get("time") or 0.0),
                model=req.get("model") or rec.get("model"), messages=messages,
                tools=req.get("tools") or rec.get("tools"), response=msg, source=source)


def load_calls(run_dir: str) -> list[Call]:
    files = sorted(glob.glob(os.path.join(run_dir, "trajectories", "gateway", "**", "*.jsonl"), recursive=True))
    out = []
    for fp in files:
        with open(fp, errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = parse_record(rec, f"{os.path.relpath(fp, run_dir)}:{i + 1}")
                if c is not None:
                    if not c.run:
                        c.run = os.path.basename(os.path.normpath(run_dir))
                    out.append(c)
    out.sort(key=lambda c: (c.t, c.source))
    return out


def run_task(run_dir: str) -> str | None:
    cfg = read_json(os.path.join(run_dir, "config.json"), {}) or {}
    if isinstance(cfg.get("task"), str):
        return cfg["task"]
    if isinstance(cfg.get("task"), dict) and cfg["task"].get("name"):
        return cfg["task"]["name"]
    ev = os.path.join(run_dir, "events.jsonl")
    if os.path.exists(ev):
        with open(ev) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("ev") == "run_start":
                    return e.get("task")
    return None


def trials(run_dir: str) -> dict[int, dict]:
    """Latest merged record per trial (same semantics as Ledger.trials())."""
    latest: dict[int, dict] = {}
    for r in read_jsonl(os.path.join(run_dir, "ledger.jsonl")):
        if r.get("trial") is not None:
            latest.setdefault(int(r["trial"]), {}).update(r)
    return latest


def label(rec: dict) -> str:
    """accepted | rejected (legal, not adopted) | illegal (crash / nan / audit / timeout) | other."""
    st = rec.get("status")
    if st in ACCEPT_STATUSES:
        return "accepted"
    if st in REJECT_STATUSES and rec.get("eval_status", "ok") == "ok":
        return "rejected"
    if st in ("crash", "audit_fail") or rec.get("eval_status") not in (None, "ok"):
        return "illegal"
    return "other"
