"""bubblewrap sandbox for researcher threads (D6: bwrap + user namespaces on node-a).

Inside the sandbox the agent sees:
    /work          its workspace (rw)
    /lab           lab spool (ro) with /lab/outbox (rw)
    /home/agent    harness HOME (rw; OpenCode data, provider cache)
    /rsi           per-episode harness config (ro): opencode.json, scaffold/
    system dirs    /usr /bin /lib /lib64 /sbin /etc (ro)
    environment    exactly sandbox_env(spec), passed to bwrap via Popen(env=...); no secrets in argv
    ro_paths       explicitly listed tool dirs (python envs, opencode binary, platform code), same path
    network        shared with the node (internet for search) unless share_net=False
It does NOT see $HOME (so no ~/.ssh), $R/data/devval, $R/data/sealed, $R/run (tokens), other
threads, run dirs, or any other project. ``check_binds`` enforces that no bind source is an
ancestor of (or inside) a forbidden path.

backend "none" (Mac tests only) runs the command directly; events record sandbox="none".
"""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass, field

SYSTEM_RO = ["/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc", "/opt/rh"]
SYSTEM_RO_TRY = ["/run/systemd/resolve", "/etc/alternatives"]


class SandboxError(RuntimeError):
    pass


@dataclass
class SandboxSpec:
    workspace: str
    lab: str | None = None
    home: str | None = None
    config_dir: str | None = None          # mounted at /rsi (ro)
    ro_paths: list[str] = field(default_factory=list)
    share_net: bool = True
    env: dict[str, str] = field(default_factory=dict)
    forbidden: list[str] = field(default_factory=list)
    chdir: str = "/work"


def default_forbidden() -> list[str]:
    """Paths no researcher sandbox may expose (directly or via an ancestor bind)."""
    R = os.environ.get("RSI_ROOT", "/scratch/user/rsi")
    home = os.path.expanduser("~")
    out = [os.path.join(home, ".ssh"), home,
           os.path.join(R, "data", "devval"), os.path.join(R, "data", "sealed"),
           os.path.join(R, "run"), os.path.join(R, "runs"), os.path.join(R, "spool"), os.path.join(R, "evald"),
           "/scratch/user/otherproj"]
    extra = os.environ.get("RSI_SANDBOX_FORBIDDEN", "")
    out += [p for p in extra.split(":") if p]
    return [os.path.abspath(p) for p in out]


def _norm(p: str) -> str:
    return os.path.abspath(p).rstrip("/") or "/"


LEAF_SECRET_SUFFIXES = ("/.ssh", "/data/devval", "/data/sealed", "/run")


def _is_leaf_secret(p: str) -> bool:
    return _norm(p).endswith(LEAF_SECRET_SUFFIXES)


def check_binds(sources: list[str], forbidden: list[str], allowed_inside: list[str] | None = None) -> None:
    """Raise SandboxError if any bind source exposes a forbidden path.

    1. A source may never equal or be an ancestor of a forbidden path (binding $R would expose data/sealed).
    2. A source may never lie inside a leaf secret (.ssh, data/devval, data/sealed, run/).
    3. A source inside a forbidden *container* ($HOME, runs/, spool/) is allowed only if it is listed in
       ``allowed_inside`` (the thread's own workspace / lab / home / config dirs).
    """
    allowed = {_norm(a) for a in (allowed_inside or [])}
    for src in sources:
        s = _norm(src)
        for f in map(_norm, forbidden):
            if s == f or s == "/" or f.startswith(s + "/"):
                raise SandboxError(f"bind {src} would expose forbidden path {f}")
            if s.startswith(f + "/"):
                if _is_leaf_secret(f) or s not in allowed:
                    raise SandboxError(f"bind {src} lies inside forbidden path {f}")


def bwrap_argv(spec: SandboxSpec, cmd: list[str], bwrap: str = "bwrap") -> list[str]:
    a = [bwrap, "--die-with-parent", "--new-session", "--unshare-all"]
    if spec.share_net:
        a.append("--share-net")
    for p in SYSTEM_RO:
        if os.path.exists(p):
            a += ["--ro-bind", p, p]
    for p in SYSTEM_RO_TRY:
        a += ["--ro-bind-try", p, p]
    a += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    sources: list[str] = []
    for p in spec.ro_paths:
        a += ["--ro-bind", p, p]
        sources.append(p)
    a += ["--bind", spec.workspace, "/work"]
    sources.append(spec.workspace)
    if spec.lab:
        a += ["--ro-bind", spec.lab, "/lab"]
        outbox = os.path.join(spec.lab, "outbox")
        os.makedirs(outbox, exist_ok=True)
        a += ["--bind", outbox, "/lab/outbox"]
        sources.append(spec.lab)
    if spec.home:
        a += ["--bind", spec.home, "/home/agent"]
        sources.append(spec.home)
    if spec.config_dir:
        a += ["--ro-bind", spec.config_dir, "/rsi"]
        sources.append(spec.config_dir)
    thread_dirs = [x for x in (spec.workspace, spec.lab, spec.home, spec.config_dir) if x]
    check_binds(sources, spec.forbidden or default_forbidden(), allowed_inside=thread_dirs)
    # No --clearenv / --setenv: bubblewrap 0.4.1 (cluster) lacks --clearenv, and --setenv would put values
    # (e.g. RSI_GATEWAY_KEY) on bwrap's command line, visible in `ps`. Instead bwrap is started with the
    # complete, minimal environment from sandbox_env() via Popen(env=...), and the child inherits exactly that.
    a += ["--chdir", spec.chdir, "--"]
    return a + list(cmd)


# Flags of bubblewrap 0.4.1 (the version on the cluster nodes). tests/test_core_sandbox_lab.py asserts
# that bwrap_argv() never emits anything else.
BWRAP_041_FLAGS = frozenset("""
--args --as-pid-1 --bind --bind-data --bind-fd --bind-try --block-fd --cap-add --cap-drop --chdir --dev
--dev-bind --dev-bind-try --die-with-parent --dir --exec-label --file --file-label --gid --hostname --info-fd
--json-status-fd --lock-file --mqueue --new-session --pidns --proc --remount-ro --ro-bind --ro-bind-data
--ro-bind-fd --ro-bind-try --seccomp --setenv --share-net --symlink --sync-fd --tmpfs --uid --unsetenv
--unshare-all --unshare-cgroup --unshare-cgroup-try --unshare-ipc --unshare-net --unshare-pid --unshare-user
--unshare-user-try --unshare-uts --userns --userns-block-fd""".split())


def sandbox_env(spec: SandboxSpec) -> dict[str, str]:
    """The complete environment of the sandboxed process (nothing else from the parent leaks in)."""
    env = {"HOME": "/home/agent", "USER": "agent", "LANG": "C.UTF-8", "TMPDIR": "/tmp",
           "PATH": "/usr/local/bin:/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}
    env.update(spec.env)
    return env


def bwrap_flags(argv: list[str]) -> list[str]:
    """The option flags in a bwrap argv (everything starting with '--' before the '--' separator)."""
    out = []
    for a in argv[1:]:
        if a == "--":
            break
        if a.startswith("--"):
            out.append(a)
    return out


def wrap(spec: SandboxSpec, cmd: list[str], backend: str = "bwrap") -> tuple[list[str], dict | None, str]:
    """Return (argv, env, cwd) for subprocess. backend: bwrap | none."""
    if backend == "bwrap":
        bw = os.environ.get("RSI_BWRAP") or shutil.which("bwrap") or "/usr/bin/bwrap"
        return bwrap_argv(spec, cmd, bw), sandbox_env(spec), "/"
    if backend == "none":
        env = dict(os.environ)
        env.update({"HOME": spec.home or env.get("HOME", "")})
        env.update(spec.env)
        return list(cmd), env, spec.workspace
    raise SandboxError(f"unknown sandbox backend {backend!r}")


def describe(argv: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)
