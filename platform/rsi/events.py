"""Append-only event log, schema v1 (see platform/EVENTS_SCHEMA.md).

Every component writes through ``emit()``. One JSON object per line in
``runs/<run_id>/events.jsonl``. Several processes on one node append to the
same file (orchestrator, gateway thread, evald), so each line is written with a
single ``os.write`` on an ``O_APPEND`` descriptor under an exclusive ``flock``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from typing import Any

SCHEMA_VERSION = 1

COMMON_FIELDS = ("t", "run", "ev", "agent", "parent", "gen", "trial")

# Required extra fields per event type (schema v1). Extra optional fields are allowed.
EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "run_start": ("policy", "task", "improver", "budget", "config_hash"),
    "agent_spawn": ("role", "model"),
    "agent_end": ("reason",),
    "message": ("to", "summary", "tokens"),
    "llm_call": ("model", "prompt_tokens", "completion_tokens", "latency_s"),
    "tool_call": ("tool", "arg_summary", "ok", "latency_s"),
    "proposal": ("hypothesis", "diff_stat", "artifact_hash"),
    "eval_submit": ("artifact_hash", "lane", "seeds"),
    "eval_result": ("artifact_hash", "metric", "value", "seeds", "wall_s", "status", "audit"),
    "decision": ("artifact_hash", "verdict", "best_before", "best_after", "reason"),
    "audit_flag": ("artifact_hash", "check", "detail"),
    "lessons_update": ("scaffold_hash_before", "scaffold_hash_after", "n_lessons"),
    "weights_update": ("adapter_path", "n_examples", "base_model"),
    "run_end": ("best_value", "n_trials", "nAUC"),
}

ROLES = {"researcher", "subagent", "critic", "lessons-writer"}
EVAL_STATUSES = {"ok", "timeout", "crash", "nan", "audit_fail"}
VERDICTS = {"adopt", "reject", "rollback", "explore"}

LIMITS = {"summary": 200, "arg_summary": 200, "hypothesis": 300}

# Redact things that look like credentials before they reach the log.
_SECRET_RES = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{4,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}"),
]


def redact(text: str) -> str:
    for rx in _SECRET_RES:
        if rx.groups:
            text = rx.sub(lambda m: m.group(1) + "<redacted>", text)
        else:
            text = rx.sub("<redacted>", text)
    return text


def clip(text: Any, n: int) -> str:
    s = redact(" ".join(str(text if text is not None else "").split()))
    return s if len(s) <= n else s[: n - 1] + "…"


class SchemaError(ValueError):
    pass


def validate(event: dict) -> None:
    """Raise SchemaError if ``event`` violates schema v1."""
    for f in COMMON_FIELDS:
        if f not in event:
            raise SchemaError(f"missing common field {f!r} in {event.get('ev')}")
    ev = event["ev"]
    if ev not in EVENT_FIELDS:
        raise SchemaError(f"unknown event type {ev!r}")
    for f in EVENT_FIELDS[ev]:
        if f not in event:
            raise SchemaError(f"{ev}: missing field {f!r}")
    if not isinstance(event["t"], (int, float)):
        raise SchemaError("t must be a number")
    if not isinstance(event["run"], str) or not isinstance(event["agent"], str):
        raise SchemaError("run and agent must be strings")
    if event["parent"] is not None and not isinstance(event["parent"], str):
        raise SchemaError("parent must be str or null")
    if not isinstance(event["gen"], int):
        raise SchemaError("gen must be int")
    if event["trial"] is not None and not isinstance(event["trial"], int):
        raise SchemaError("trial must be int or null")
    for f, n in LIMITS.items():
        if f in event and isinstance(event[f], str) and len(event[f]) > n:
            raise SchemaError(f"{ev}: {f} longer than {n} chars")
    if ev == "agent_spawn" and event["role"] not in ROLES:
        raise SchemaError(f"agent_spawn: bad role {event['role']!r}")
    if ev == "eval_result":
        if event["status"] not in EVAL_STATUSES:
            raise SchemaError(f"eval_result: bad status {event['status']!r}")
        a = event["audit"]
        if not (isinstance(a, dict) and "level" in a and isinstance(a.get("flags"), list)):
            raise SchemaError("eval_result: audit must be {level, flags[]}")
    if ev == "decision" and event["verdict"] not in VERDICTS:
        raise SchemaError(f"decision: bad verdict {event['verdict']!r}")
    if ev == "proposal":
        d = event["diff_stat"]
        if not (isinstance(d, dict) and {"files", "plus", "minus"} <= set(d)):
            raise SchemaError("proposal: diff_stat must be {files, plus, minus}")
    if ev == "tool_call" and not isinstance(event["ok"], bool):
        raise SchemaError("tool_call: ok must be bool")


class EventLog:
    """Writer bound to one run's ``events.jsonl``."""

    def __init__(self, path: str, run: str, strict: bool = True):
        self.path = path
        self.run = run
        self.strict = strict
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def emit(self, ev: str, agent: str, *, parent: str | None = None, gen: int = 0,
             trial: int | None = None, t: float | None = None, **fields: Any) -> dict:
        event = {
            "t": round(time.time() if t is None else t, 3),
            "run": self.run,
            "ev": ev,
            "agent": agent,
            "parent": parent,
            "gen": int(gen),
            "trial": trial,
        }
        for k, v in fields.items():
            if k in LIMITS and isinstance(v, str):
                v = clip(v, LIMITS[k])
            event[k] = v
        if self.strict:
            validate(event)
        line = (json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        return event


_logs: dict[str, EventLog] = {}
_run_paths: dict[str, str] = {}
_logs_lock = threading.Lock()


def runs_root() -> str:
    """Root of all run dirs: $RSI_RUNS_ROOT, else ./runs."""
    return os.environ.get("RSI_RUNS_ROOT") or os.path.abspath("runs")


def register_run(run: str, events_path: str) -> None:
    """Bind a run id to an explicit events.jsonl path (otherwise <runs_root>/<run>/events.jsonl)."""
    with _logs_lock:
        _run_paths[run] = os.path.abspath(events_path)


def events_path(run: str) -> str:
    return _run_paths.get(run) or os.path.join(runs_root(), run, "events.jsonl")


def get_log(path: str, run: str) -> EventLog:
    key = os.path.abspath(path)
    with _logs_lock:
        if key not in _logs:
            _logs[key] = EventLog(path, run)
        return _logs[key]


def emit(run: str, ev: str, agent: str, *, parent: str | None = None, gen: int = 0,
         trial: int | None = None, **fields: Any) -> dict:
    """Write one schema-v1 event for ``run`` (INTERFACES.md signature).

    The file is ``register_run(run, path)`` if called in this process, else
    ``$RSI_RUNS_ROOT/<run>/events.jsonl``. Raises SchemaError on an invalid event.
    """
    return get_log(events_path(run), run).emit(ev, agent, parent=parent, gen=gen, trial=trial, **fields)


def read_events(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate_file(path: str) -> list[dict]:
    """Validate every line; returns the events. Also checks basic replayability."""
    events = read_events(path)
    if not events:
        raise SchemaError("empty event log")
    runs = {e.get("run") for e in events}
    if len(runs) != 1:
        raise SchemaError(f"multiple run ids in one log: {runs}")
    for i, e in enumerate(events):
        try:
            validate(e)
        except SchemaError as err:
            raise SchemaError(f"line {i + 1}: {err}") from None
    if events[0]["ev"] != "run_start":
        raise SchemaError("first event must be run_start")
    return events


if __name__ == "__main__":  # python -m rsi.events <events.jsonl>
    import sys

    evs = validate_file(sys.argv[1])
    counts: dict[str, int] = {}
    for e in evs:
        counts[e["ev"]] = counts.get(e["ev"], 0) + 1
    print(json.dumps({"ok": True, "n": len(evs), "counts": counts}, indent=1))
