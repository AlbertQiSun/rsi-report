"""Run directory layout, content-addressed artifacts, and the append-only ledger.

Layout (one directory per run, ``<runs_root>/<run_id>/``)::

    config.json            frozen config of the run (hash in run_start)
    state.json             resumable orchestrator state (rewritten atomically)
    events.jsonl           event log, schema v1 (replay source for the visualizer)
    ledger.jsonl           one line per trial state change (append-only)
    sealed.jsonl           sealed-split scores (evald only; never shown to agents)
    PAUSE                  if present, threads stop at the next trial boundary
    artifacts/<hash>/      immutable snapshot of the editable files of one proposal
    trajectories/gateway/  full chat requests/responses per agent (LoRA data)
    trajectories/harness/  raw harness output (OpenCode NDJSON, exports)
    threads/<tid>/workspace/  the agent-writable working copy
    threads/<tid>/lab/        lab MCP spool: history.jsonl (ro view), outbox/ (submissions)
    threads/<tid>/home/       HOME of the harness inside the sandbox
    audits/                L3 review queue, audit reports
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_hash(obj: Any, n: int = 12) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())[:n]


def atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = (json.dumps(obj, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file that other threads/processes may be appending to.

    A trailing line without a newline (a concurrent append in progress) is ignored; a malformed
    complete line is still an error.
    """
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    complete, tail = lines[:-1], lines[-1]
    out = []
    for line in complete:
        line = line.strip()
        if line:
            out.append(json.loads(line))
    if tail.strip():
        try:
            out.append(json.loads(tail))
        except json.JSONDecodeError:
            pass  # torn write in progress
    return out


@dataclass
class RunDir:
    root: str
    run_id: str

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.run_id)

    def p(self, *parts: str) -> str:
        return os.path.join(self.path, *parts)

    @property
    def events(self) -> str:
        return self.p("events.jsonl")

    @property
    def ledger_path(self) -> str:
        return self.p("ledger.jsonl")

    @property
    def state_path(self) -> str:
        return self.p("state.json")

    @property
    def config_path(self) -> str:
        return self.p("config.json")

    @property
    def sealed_path(self) -> str:
        return self.p("sealed.jsonl")

    @property
    def pause_flag(self) -> str:
        return self.p("PAUSE")

    def thread_dir(self, tid: str, *parts: str) -> str:
        return self.p("threads", tid, *parts)

    def create(self) -> "RunDir":
        for d in ("artifacts", "trajectories/gateway", "trajectories/harness", "threads", "audits"):
            os.makedirs(self.p(d), exist_ok=True)
        return self

    def paused(self) -> bool:
        return os.path.exists(self.pause_flag)


# ---------------------------------------------------------------------------
# Artifacts: content-addressed snapshots of the editable files
# ---------------------------------------------------------------------------

def artifact_hash(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return h.hexdigest()[:12]


def snapshot_files(workspace: str, names: Iterable[str]) -> dict[str, bytes]:
    out = {}
    for n in names:
        fp = os.path.join(workspace, n)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                out[n] = f.read()
    return out


class ArtifactStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, files: dict[str, bytes]) -> str:
        h = artifact_hash(files)
        d = os.path.join(self.root, h)
        if not os.path.isdir(d):
            tmp = tempfile.mkdtemp(dir=self.root, prefix=".tmp-")
            for name, data in files.items():
                fp = os.path.join(tmp, name)
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "wb") as f:
                    f.write(data)
            try:
                os.rename(tmp, d)
            except OSError:  # raced with another writer
                shutil.rmtree(tmp, ignore_errors=True)
        return h

    def dir(self, h: str) -> str:
        return os.path.join(self.root, h)

    def get(self, h: str) -> dict[str, bytes]:
        d = self.dir(h)
        out = {}
        for base, _, files in os.walk(d):
            for fn in files:
                fp = os.path.join(base, fn)
                with open(fp, "rb") as f:
                    out[os.path.relpath(fp, d)] = f.read()
        return out

    def checkout(self, h: str, workspace: str) -> None:
        """Make the editable files in ``workspace`` equal to artifact ``h``."""
        for name, data in self.get(h).items():
            fp = os.path.join(workspace, name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as f:
                f.write(data)


def diff_stat(old: dict[str, bytes], new: dict[str, bytes]) -> dict[str, int]:
    files = plus = minus = 0
    for name in sorted(set(old) | set(new)):
        a = old.get(name, b"").decode(errors="replace").splitlines()
        b = new.get(name, b"").decode(errors="replace").splitlines()
        if a == b:
            continue
        files += 1
        for line in difflib.unified_diff(a, b, lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++"):
                plus += 1
            elif line.startswith("-") and not line.startswith("---"):
                minus += 1
    return {"files": files, "plus": plus, "minus": minus}


def unified_diff(old: dict[str, bytes], new: dict[str, bytes], max_lines: int = 400) -> str:
    out: list[str] = []
    for name in sorted(set(old) | set(new)):
        a = old.get(name, b"").decode(errors="replace").splitlines()
        b = new.get(name, b"").decode(errors="replace").splitlines()
        out.extend(difflib.unified_diff(a, b, f"a/{name}", f"b/{name}", lineterm="", n=2))
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... ({len(out) - max_lines} more diff lines)"]
    return "\n".join(out)
