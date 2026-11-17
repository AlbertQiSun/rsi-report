"""Pluggable process sandbox for evald: bwrap (cluster) or none (Mac tests).

A SandboxSpec is written in *inside* paths (what the sandboxed process sees).
BwrapLauncher turns the mounts into a fresh mount namespace with no network;
NoSandboxLauncher runs the same command directly on the host, rewriting inside
paths to host paths. The no-sandbox mode gives NO isolation; it exists so the
pipeline can be tested on a Mac without bwrap.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Mount:
    host: str | None          # host path (None for tmpfs in bwrap mode)
    inside: str
    kind: str = "ro"          # ro | rw | dev | tmpfs


@dataclass
class GpuInfo:
    index: int
    uuid: str
    minor: int
    bus_id: str = ""


@dataclass
class SandboxSpec:
    argv: list[str]
    env: dict[str, str]
    cwd: str
    mounts: list[Mount] = field(default_factory=list)
    gpu: GpuInfo | None = None
    hostname: str = "rsi-sandbox"


# ---------------------------------------------------------------------------
# GPU helpers (nvidia-smi)
# ---------------------------------------------------------------------------

def gpu_info(index: int, nvidia_smi: str = "nvidia-smi") -> GpuInfo:
    out = subprocess.run([nvidia_smi, "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=60).stdout
    uuid = bus = None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(index):
            uuid, bus = parts[1], parts[2]
    if uuid is None:
        raise RuntimeError(f"GPU index {index} not found by nvidia-smi")
    q = subprocess.run([nvidia_smi, "-q", "-i", str(index)], capture_output=True, text=True, check=True,
                       timeout=60).stdout
    m = re.search(r"Minor Number\s*:\s*(\d+)", q)
    minor = int(m.group(1)) if m else index
    return GpuInfo(index=index, uuid=uuid, minor=minor, bus_id=bus or "")


def gpu_compute_pids(uuid: str, nvidia_smi: str = "nvidia-smi") -> list[int]:
    try:
        out = subprocess.run([nvidia_smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    pids = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == uuid and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

class Launcher:
    name = "base"

    def popen(self, spec: SandboxSpec, stdout, stderr) -> subprocess.Popen:
        raise NotImplementedError

    def host_path(self, spec: SandboxSpec, inside: str) -> str:
        return inside


class BwrapLauncher(Launcher):
    name = "bwrap"
    SYSTEM_RO = ("/usr", "/etc", "/sys")
    SYSTEM_LINKS = ("/bin", "/sbin", "/lib", "/lib64")
    NVIDIA_COMMON = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset")

    def __init__(self, bwrap: str = "bwrap", extra_ro: list[str] | None = None, extra_args: list[str] | None = None):
        self.bwrap = bwrap
        self.extra_ro = list(extra_ro or [])
        self.extra_args = list(extra_args or [])

    def build_argv(self, spec: SandboxSpec) -> list[str]:
        a = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--hostname", spec.hostname,
             "--cap-drop", "ALL"]
        for d in self.SYSTEM_RO:
            if os.path.exists(d):
                a += ["--ro-bind", d, d]
        for d in self.SYSTEM_LINKS:
            if os.path.islink(d):
                a += ["--symlink", os.readlink(d), d]
            elif os.path.isdir(d):
                a += ["--ro-bind", d, d]
        a += ["--proc", "/proc", "--dev", "/dev"]
        for d in self.extra_ro:
            if os.path.exists(d):
                a += ["--ro-bind", d, d]
        for m in spec.mounts:
            if m.kind == "tmpfs":
                a += ["--tmpfs", m.inside]
            elif m.kind == "rw":
                a += ["--bind", m.host, m.inside]
            elif m.kind == "dev":
                a += ["--dev-bind", m.host, m.inside]
            else:
                a += ["--ro-bind", m.host, m.inside]
        if spec.gpu is not None:
            for d in self.NVIDIA_COMMON + (f"/dev/nvidia{spec.gpu.minor}",):
                if os.path.exists(d):
                    a += ["--dev-bind", d, d]
            if os.path.isdir("/dev/nvidia-caps"):
                a += ["--dev-bind", "/dev/nvidia-caps", "/dev/nvidia-caps"]
        a += ["--clearenv"]
        for k, v in spec.env.items():
            a += ["--setenv", k, str(v)]
        a += ["--chdir", spec.cwd]
        a += self.extra_args
        a += ["--"] + list(spec.argv)
        return a

    def popen(self, spec, stdout, stderr):
        return subprocess.Popen(self.build_argv(spec), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                start_new_session=True, close_fds=True)

    def host_path(self, spec, inside):
        return _translate_path(inside, spec.mounts) or inside


_WARNED = False


class NoSandboxLauncher(Launcher):
    """Runs on the host with inside paths rewritten. NO isolation (tests only)."""

    name = "none"

    def __init__(self, quiet: bool = False):
        global _WARNED
        if not _WARNED and not quiet:
            print("WARNING [evald]: sandbox=none — submitted code runs UNSANDBOXED on the host "
                  "(no network cut, no filesystem isolation). Use only for local tests.", file=sys.stderr)
            _WARNED = True

    def _tr(self, s: str, mounts) -> str:
        parts = s.split(":")
        return ":".join(_translate_path(p, mounts) or p for p in parts)

    def popen(self, spec, stdout, stderr):
        for m in spec.mounts:
            if m.kind == "tmpfs" and m.host:
                os.makedirs(m.host, exist_ok=True)
        argv = [self._tr(x, spec.mounts) for x in spec.argv]
        env = {k: self._tr(str(v), spec.mounts) for k, v in spec.env.items()}
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        env["PATH"] = env["PATH"] + ":" + os.environ.get("PATH", "")
        cwd = self._tr(spec.cwd, spec.mounts)
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, cwd=cwd,
                                start_new_session=True, close_fds=True)

    def host_path(self, spec, inside):
        return _translate_path(inside, spec.mounts) or inside


def _translate_path(p: str, mounts: list[Mount]) -> str | None:
    best = None
    for m in mounts:
        if m.host is None:
            continue
        if p == m.inside or p.startswith(m.inside.rstrip("/") + "/"):
            if best is None or len(m.inside) > len(best.inside):
                best = m
    if best is None:
        return None
    return best.host + p[len(best.inside):]


def make_launcher(kind: str, **kw) -> Launcher:
    if kind == "bwrap":
        bw = kw.get("bwrap", "bwrap")
        if shutil.which(bw) is None:
            raise RuntimeError(f"sandbox=bwrap but {bw!r} is not on PATH (use sandbox=none only for local tests)")
        return BwrapLauncher(bw, extra_ro=kw.get("extra_ro"), extra_args=kw.get("extra_args"))
    if kind == "none":
        return NoSandboxLauncher(quiet=kw.get("quiet", False))
    raise ValueError(f"unknown sandbox kind {kind!r}")


# ---------------------------------------------------------------------------
# Wall-clock enforcement from outside
# ---------------------------------------------------------------------------

def kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_limited(proc: subprocess.Popen, limit_s: float, *, t0: float | None = None,
                tick: Callable[[], str | None] | None = None, poll_s: float = 0.25) -> dict:
    """Wait for ``proc``; SIGKILL its whole process group at ``limit_s`` wall seconds.

    ``tick()`` is called about once a second; returning a string kills the group too
    (reason recorded). Returns {rc, timed_out, killed_reason, wall_s}.
    """
    t0 = time.time() if t0 is None else t0
    deadline = t0 + limit_s
    reason = None
    last_tick = 0.0
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.time()
        if now >= deadline:
            reason = "timeout"
            kill_group(proc)
            break
        if tick is not None and now - last_tick >= 1.0:
            last_tick = now
            r = tick()
            if r:
                reason = r
                kill_group(proc)
                break
        time.sleep(poll_s)
    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        rc = proc.wait()
    wall = time.time() - t0
    # Reap anything left in the group (no-sandbox mode; bwrap tears down its pid namespace itself).
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return {"rc": rc, "timed_out": reason == "timeout", "killed_reason": reason, "wall_s": wall}
