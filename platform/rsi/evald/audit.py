"""Audit checks for evald (platform_build_plan.md §2.5).

L0 (every submission, before anything runs; static):
  file count / size / type limits, no symlinks, required contract files, no shadowing
  of the guard, forbidden imports and calls (network, subprocess, ctypes, frame and
  interpreter introspection, exec/eval), string literals naming forbidden locations.
L1 (every evaluation; runtime): see service.py (guard channel, honeypots, wall-clock,
  reported-vs-measured time, checkpoint size) and scorer.py (logits-only contract,
  causality / batch / order probes, parameter hash, canary, non-finite values).

A flag is {"check", "detail", "severity"}; severity "hard" makes the evaluation
``audit_fail``; "soft" is reported but does not change the status.
"""

from __future__ import annotations

import ast
import os
import re

AUDIT_RULES_VERSION = "2026-11-17.1"

ALLOWED_EXT = {".py", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".cfg"}
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")

HARD_IMPORTS = {
    "ctypes": "native memory access", "cffi": "native memory access", "_ctypes": "native memory access",
    "socket": "network", "ssl": "network", "urllib": "network", "http": "network", "requests": "network",
    "httpx": "network", "aiohttp": "network", "ftplib": "network", "smtplib": "network", "telnetlib": "network",
    "paramiko": "network", "websocket": "network", "websockets": "network", "huggingface_hub": "network/hub",
    "subprocess": "process spawning", "pty": "process spawning", "builtins": "interpreter tampering",
    "sitecustomize": "guard", "rsi_guard": "guard", "faulthandler": "interpreter tampering",
}
SOFT_IMPORTS = {"importlib": "dynamic import", "inspect": "introspection", "pickle": "deserialization",
                "marshal": "deserialization", "dill": "deserialization", "mmap": "raw memory",
                "shutil": "file operations", "glob": "file discovery"}

HARD_ATTRS = {
    ("os", "system"), ("os", "popen"), ("os", "fork"), ("os", "forkpty"), ("os", "posix_spawn"),
    ("os", "posix_spawnp"), ("os", "execv"), ("os", "execve"), ("os", "execvp"), ("os", "execvpe"),
    ("os", "execl"), ("os", "execle"), ("os", "execlp"), ("os", "execlpe"), ("os", "spawnv"), ("os", "spawnve"),
    ("os", "spawnl"), ("os", "spawnle"), ("sys", "_getframe"), ("sys", "settrace"), ("sys", "setprofile"),
    ("sys", "addaudithook"), ("sys", "modules"), ("sys", "_current_frames"), ("gc", "get_objects"),
    ("gc", "get_referrers"), ("gc", "get_referents"), ("inspect", "currentframe"), ("inspect", "stack"),
    ("inspect", "getouterframes"), ("threading", "settrace"), ("threading", "setprofile"),
}
SOFT_ATTRS = {("os", "kill"), ("os", "killpg"), ("os", "chdir"), ("os", "walk"), ("os", "scandir"),
              ("os", "putenv"), ("os", "unsetenv")}
HARD_NAMES = {"exec", "eval", "compile", "__import__", "__builtins__", "breakpoint", "globals"}
FRAME_ATTRS = {"f_back", "f_locals", "f_globals", "tb_frame", "gi_frame", "cr_frame", "__subclasses__",
               "__globals__", "__code__", "__closure__"}

HARD_STRINGS = ["devval", "sealed", ".ssh", "otherproj", "rsi_guard", "sitecustomize", "RSI_AUDIT_PIPE",
                "RSI_GUARD", "/opt/rsi/guard", "/opt/rsi/scorer", "scorer.py", "evald", "/proc/self/mem",
                "LD_PRELOAD", "shard_06540", "shard_06541", "shard_06542"]
SOFT_STRINGS = ["/scratch", "/home/", "/root", "/etc/passwd", "huggingface.co", "http://", "https://", "/proc/"]

ALLOWED_ABS_PREFIXES = ("/data/train", "/data/tokenizer", "/work", "/tmp", "/opt/rsi/lib", "/dev/null",
                        "/dev/shm")

REQUIRED_FILES = ("train.py", "model.py")
FORBIDDEN_FILES = {"sitecustomize.py": "hard", "usercustomize.py": "hard", "rsi_guard.py": "hard",
                   "scorer.py": "hard", "rsilib.py": "soft"}


def _flag(check, detail, severity="hard", file=None):
    d = f"{file}: {detail}" if file else detail
    return {"check": check, "detail": str(d)[:300], "severity": severity}


def collect_artifact(src_dir: str, *, max_files: int, max_file_bytes: int, max_total_bytes: int):
    """Read an artifact dir without following symlinks. Returns (files, flags)."""
    files: dict[str, bytes] = {}
    flags: list[dict] = []
    total = 0
    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir) or os.path.islink(src_dir):
        return {}, [_flag("artifact_missing", f"not a directory: {src_dir}")]
    for base, dirs, names in os.walk(src_dir, followlinks=False):
        rel_base = os.path.relpath(base, src_dir)
        depth = 0 if rel_base == "." else rel_base.count(os.sep) + 1
        keep = []
        for d in dirs:
            p = os.path.join(base, d)
            if d == "__pycache__" or d.startswith("."):
                continue
            if os.path.islink(p):
                flags.append(_flag("symlink", f"symlinked directory {os.path.relpath(p, src_dir)}"))
                continue
            if depth >= 3:
                flags.append(_flag("artifact_layout", f"directory nesting too deep: {os.path.relpath(p, src_dir)}"))
                continue
            keep.append(d)
        dirs[:] = keep
        for n in names:
            p = os.path.join(base, n)
            rel = os.path.normpath(os.path.join(rel_base, n)) if rel_base != "." else n
            if n.startswith(".") or n.endswith(".pyc"):
                continue
            if os.path.islink(p):
                flags.append(_flag("symlink", f"symlink {rel}"))
                continue
            if not os.path.isfile(p):
                flags.append(_flag("artifact_file_type", f"not a regular file: {rel}"))
                continue
            if not all(NAME_RE.match(part) for part in rel.split(os.sep)):
                flags.append(_flag("artifact_file_name", f"bad file name {rel!r}"))
                continue
            if os.path.splitext(n)[1].lower() not in ALLOWED_EXT:
                flags.append(_flag("artifact_file_type", f"disallowed file type {rel} (allowed: {sorted(ALLOWED_EXT)})"))
                continue
            sz = os.path.getsize(p)
            if sz > max_file_bytes:
                flags.append(_flag("artifact_size", f"{rel} is {sz} bytes > {max_file_bytes}"))
                continue
            total += sz
            if total > max_total_bytes:
                flags.append(_flag("artifact_size", f"artifact larger than {max_total_bytes} bytes"))
                break
            if len(files) >= max_files:
                flags.append(_flag("artifact_size", f"more than {max_files} files"))
                break
            with open(p, "rb") as f:
                files[rel] = f.read()
    return files, flags


class _Scan(ast.NodeVisitor):
    def __init__(self, fname: str):
        self.fname = fname
        self.flags: list[dict] = []
        self.aliases: dict[str, str] = {}   # local name -> module

    def add(self, check, detail, severity):
        f = _flag(check, f"line {getattr(self, '_line', '?')}: {detail}", severity, self.fname)
        if not any(x["detail"] == f["detail"] for x in self.flags):
            self.flags.append(f)

    def visit(self, node):
        self._line = getattr(node, "lineno", getattr(self, "_line", "?"))
        return super().visit(node)

    def _check_module(self, mod: str):
        top = mod.split(".")[0]
        if top in HARD_IMPORTS:
            self.add("forbidden_import", f"import {mod} ({HARD_IMPORTS[top]})", "hard")
        elif top in SOFT_IMPORTS:
            self.add("suspicious_import", f"import {mod} ({SOFT_IMPORTS[top]})", "soft")
        if mod.startswith("multiprocessing.connection") or mod.startswith("torch.distributed"):
            self.add("forbidden_import", f"import {mod} (network)", "hard")

    def visit_Import(self, node):
        for a in node.names:
            self._check_module(a.name)
            self.aliases[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0] if not a.asname else a.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        if node.level == 0:
            self._check_module(mod)
            for a in node.names:
                if (mod.split(".")[0], a.name) in HARD_ATTRS:
                    self.add("forbidden_call", f"from {mod} import {a.name}", "hard")
                self.aliases[a.asname or a.name] = f"{mod}.{a.name}"
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            base = self.aliases.get(node.value.id, node.value.id).split(".")[-1]
            if (base, node.attr) in HARD_ATTRS:
                self.add("forbidden_call", f"{node.value.id}.{node.attr}", "hard")
            elif (base, node.attr) in SOFT_ATTRS:
                self.add("suspicious_call", f"{node.value.id}.{node.attr}", "soft")
        if node.attr in FRAME_ATTRS:
            self.add("forbidden_call", f"introspection attribute .{node.attr}", "hard")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in HARD_NAMES:
            self.add("forbidden_call", f"use of {node.id}", "hard")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value.startswith("/"):
                if not a0.value.startswith(ALLOWED_ABS_PREFIXES):
                    self.add("abs_path", f"open({a0.value[:80]!r}) outside the sandbox data/work dirs", "soft")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (str, bytes)):
            s = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            low = s.lower()
            for m in HARD_STRINGS:
                if m.lower() in low:
                    self.add("forbidden_string", f"string mentions {m!r}", "hard")
                    break
            else:
                for m in SOFT_STRINGS:
                    if m.lower() in low:
                        self.add("suspicious_string", f"string mentions {m!r}", "soft")
                        break
        self.generic_visit(node)


def static_check(files: dict[str, bytes], *, required=REQUIRED_FILES) -> list[dict]:
    """L0 checks on an artifact's files (name -> bytes)."""
    flags: list[dict] = []
    for r in required:
        if r not in files:
            flags.append(_flag("contract_missing", f"required file {r} is missing"))
    for name in files:
        base = os.path.basename(name)
        if base in FORBIDDEN_FILES:
            sev = FORBIDDEN_FILES[base]
            flags.append(_flag("shadowing", f"file {name} shadows an evaluator-provided module", sev))
    for name, data in sorted(files.items()):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(data.decode("utf-8"), filename=name)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            flags.append(_flag("syntax_error", f"{type(e).__name__}: {str(e)[:120]}", "soft", name))
            continue
        sc = _Scan(name)
        sc.visit(tree)
        flags.extend(sc.flags)
        if name == "model.py":
            fns = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            imported = {(a.asname or a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
            if "build_model" not in fns | assigned | imported:
                flags.append(_flag("contract_missing", "model.py must define build_model(config)"))
    return flags


def has_hard(flags: list[dict]) -> bool:
    return any(f.get("severity") == "hard" for f in flags)


# ---------------------------------------------------------------------------
# L1 helpers (host side)
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+0-9.eEinfa]+)\s*$")


def parse_train_log(text: str) -> dict:
    """Summary key/values after the last '---' line, the attention backend, FAIL marker."""
    lines = re.split(r"[\r\n]+", text)
    out: dict = {"summary": {}, "attn_backend": None, "fail_marker": False}
    last_sep = max((i for i, l in enumerate(lines) if l.strip() == "---"), default=None)
    if last_sep is not None:
        for l in lines[last_sep + 1:]:
            m = _SUMMARY_RE.match(l.strip())
            if m:
                try:
                    out["summary"][m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    for l in lines:
        if l.startswith("[model] attention backend:"):
            out["attn_backend"] = l.split(":", 1)[1].strip()[:80]
        if l.strip() == "FAIL":
            out["fail_marker"] = True
    return out


def time_check(summary: dict, measured_wall_s: float, slack_s: float = 5.0) -> list[dict]:
    """Script-reported times are logged, never trusted; flag reports that exceed reality."""
    flags = []
    for key in ("total_seconds", "training_seconds"):
        v = summary.get(key)
        if isinstance(v, float) and v > measured_wall_s + slack_s:
            flags.append(_flag("time_report_mismatch",
                               f"script reported {key}={v:.1f}s but the process ran {measured_wall_s:.1f}s wall",
                               "soft"))
    # Total-wall rule (2026-11-17 margin test): training past the default 300 s is legal under the
    # 350 s total limit, but we tag it so "trained longer" gains can be separated from real ones later.
    v = summary.get("training_seconds")
    if isinstance(v, float) and v > 305.0:
        flags.append(_flag("training_beyond_default_budget",
                           f"script reported training_seconds={v:.1f}s (> 300 s default); legal under the total-wall rule",
                           "soft"))
    return flags


def guard_flags(msgs: list[dict], *, where: str) -> tuple[list[dict], dict]:
    """Interpret rsi_guard messages. Returns (flags, stats)."""
    flags: list[dict] = []
    stats = {"hello": 0, "proc": 0, "net": 0, "deny": 0}
    seen = set()
    for m in msgs:
        k = m.get("k")
        if k == "hello":
            stats["hello"] += 1
        elif k == "proc":
            stats["proc"] += 1
        elif k == "deny":
            stats["deny"] += 1
            key = ("deny", m.get("path"))
            if key not in seen and len(seen) < 20:
                seen.add(key)
                check = "forbidden_path_access" if where == "train" else "scorer_forbidden_access"
                flags.append(_flag(check, f"{m.get('ev')} {m.get('path')}", "hard"))
        elif k == "net":
            stats["net"] += 1
            key = ("net", m.get("ev"), m.get("addr"))
            if key not in seen and len(seen) < 20:
                seen.add(key)
                flags.append(_flag("network_attempt", f"{m.get('ev')} {m.get('addr')}", "soft"))
        elif k == "mem":
            flags.append(_flag("proc_mem_access", f"opened {m.get('path')}", "hard"))
    return flags, stats
