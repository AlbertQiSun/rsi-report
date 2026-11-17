"""Runtime audit guard (PEP 578 audit hook). Runs INSIDE the evald sandboxes.

Loaded by ``sitecustomize.py`` in every Python process of a sandbox (the
training script, torch.compile workers, the scorer). It reports to evald over a
FIFO whose path is in ``$RSI_AUDIT_PIPE``; evald holds the read end outside the
sandbox, so a process cannot take back what it already reported.

Reported (one JSON line per message, each < PIPE_BUF so writes are atomic):
  {"k": "hello", "pid", "role", "argv0"}         once per Python process
  {"k": "deny",  "ev", "path"}                    open/listdir/... of a denied path prefix
  {"k": "net",   "ev", "addr"}                    socket connect/bind/getaddrinfo/sendto
  {"k": "proc",  "ev", "exe"}                     subprocess / exec / os.system (first 20)
  {"k": "mem",   "path"}                          /proc/*/mem opens
  {"k": "result", ...}                            scorer only: the score (see scorer.py)

Stdlib only; must never raise inside the hook. This is telemetry for audits, not a
security boundary: code in the same process can tamper with it (see deploy/evald.md).
"""

from __future__ import annotations

import json
import os
import sys
import threading

_state = {
    "fd": None,
    "deny": (),
    "role": "",
    "n_proc": 0,
    "dropped": 0,
    "installed": False,
}
_tls = threading.local()

_PATH_EVENTS = {
    "open", "os.listdir", "os.scandir", "os.chdir", "os.rename", "os.remove", "os.rmdir",
    "os.mkdir", "os.symlink", "os.link", "os.truncate", "shutil.copyfile", "glob.glob",
    "os.chmod", "os.chown", "os.utime", "pathlib.Path.glob", "pathlib.Path.rglob",
}
_NET_EVENTS = {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto",
               "socket.sendmsg", "socket.gethostbyname", "socket.gethostbyaddr"}
_PROC_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                "os.fork", "os.forkpty", "pty.spawn"}


def _send(msg: dict, blocking: bool = False) -> bool:
    fd = _state["fd"]
    if fd is None:
        return False
    try:
        line = (json.dumps(msg, separators=(",", ":"), default=str)[:3800] + "\n").encode()
        if blocking:
            os.set_blocking(fd, True)
        os.write(fd, line)
        if blocking:
            os.set_blocking(fd, False)
        return True
    except BlockingIOError:
        _state["dropped"] += 1
        return False
    except Exception:
        return False


def send(msg: dict, blocking: bool = False) -> bool:
    """Public: used by the scorer to report its result on the same channel."""
    return _send(msg, blocking=blocking)


def _norm(p) -> str | None:
    try:
        if isinstance(p, int):
            return None
        if isinstance(p, bytes):
            p = p.decode("utf-8", "replace")
        elif not isinstance(p, str):
            p = os.fspath(p)
            if isinstance(p, bytes):
                p = p.decode("utf-8", "replace")
        if not p:
            return None
        if not p.startswith("/"):
            p = os.path.join(os.getcwd(), p)
        return os.path.normpath(p)
    except Exception:
        return None


def _denied(path: str) -> bool:
    for d in _state["deny"]:
        if path == d or path.startswith(d + "/") or (d.endswith("*") and path.startswith(d[:-1])):
            return True
    return False


def _hook(event: str, args) -> None:
    if getattr(_tls, "busy", False):
        return
    _tls.busy = True
    try:
        if event in _PATH_EVENTS:
            for a in args[:2]:
                p = _norm(a)
                if p is None:
                    continue
                if _denied(p):
                    _send({"k": "deny", "ev": event, "path": p[:400]})
                elif p.startswith("/proc/") and p.endswith("/mem"):
                    _send({"k": "mem", "path": p[:200]})
        elif event in _NET_EVENTS:
            addr = args[1] if event in ("socket.connect", "socket.bind", "socket.sendto") and len(args) > 1 else args[:2]
            _send({"k": "net", "ev": event, "addr": str(addr)[:200]})
        elif event in _PROC_EVENTS:
            _state["n_proc"] += 1
            if _state["n_proc"] <= 20:
                exe = args[0] if args else ""
                _send({"k": "proc", "ev": event, "exe": str(exe)[:200]})
    except Exception:
        pass
    finally:
        _tls.busy = False


def install(pipe_path: str | None = None, deny: list[str] | None = None, role: str | None = None) -> bool:
    """Open the report FIFO and install the audit hook (idempotent)."""
    if _state["installed"]:
        return True
    pipe_path = pipe_path or os.environ.get("RSI_AUDIT_PIPE")
    if deny is None:
        deny = [d for d in os.environ.get("RSI_GUARD_DENY", "").split(":") if d]
    _state["role"] = role or os.environ.get("RSI_GUARD_ROLE", "")
    if pipe_path:
        try:
            _state["fd"] = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        except OSError:
            _state["fd"] = None
    _state["deny"] = tuple(os.path.normpath(d) if not d.endswith("*") else d for d in deny)
    sys.addaudithook(_hook)
    _state["installed"] = True
    _send({"k": "hello", "pid": os.getpid(), "role": _state["role"], "argv0": (sys.argv[0] if sys.argv else "")[:200]})
    return True


def extend_deny(paths: list[str]) -> None:
    """Add denied prefixes (the scorer arms this after it has read its own inputs)."""
    _state["deny"] = tuple(_state["deny"]) + tuple(os.path.normpath(p) for p in paths)
