"""Campaign config: loading, template expansion, clock times, validation.

Stdlib only, plus PyYAML for the ``.yaml`` file (present in the serving env, which
ships vLLM). A JSON file with the same structure also works.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
import time
from typing import Any, Callable

STEP_KINDS = {"cmd", "run", "muar_arm", "lessons", "sft", "lora", "service", "decision", "endpoint"}

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../platform


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_raw(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ConfigError("PyYAML is needed for .yaml configs (or pass a .json config)") from e
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a mapping")
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

def parse_tz(s: str) -> dt.timezone:
    m = re.fullmatch(r"([+-])(\d\d):?(\d\d)", s.strip())
    if not m:
        raise ConfigError(f"tz must look like +08:00, got {s!r}")
    sign = 1 if m.group(1) == "+" else -1
    return dt.timezone(sign * dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


def parse_time(s: Any, tz: dt.timezone) -> float | None:
    """ISO-ish clock string -> epoch seconds. Naive strings are in the campaign tz."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, dt.datetime):
        d = s
    else:
        d = dt.datetime.fromisoformat(str(s).strip().replace(" ", "T"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    return d.timestamp()


def fmt_time(t: float | None, tz: dt.timezone, with_date: bool = True) -> str:
    if t is None:
        return "-"
    d = dt.datetime.fromtimestamp(t, tz)
    return d.strftime("%m-%d %H:%M" if with_date else "%H:%M")


class Clock:
    """Campaign clock. In real mode virtual time == wall time. In dry-run, time is
    compressed: one real second is ``scale`` virtual seconds, starting at ``virt0``."""

    def __init__(self, scale: float = 1.0, real0: float | None = None, virt0: float | None = None):
        self.scale = float(scale)
        self.real0 = time.time() if real0 is None else float(real0)
        self.virt0 = self.real0 if virt0 is None else float(virt0)

    def now(self) -> float:
        return self.virt0 + (time.time() - self.real0) * self.scale

    def real_s(self, virtual_s: float) -> float:
        return virtual_s / self.scale

    def state(self) -> dict:
        return {"scale": self.scale, "real0": self.real0, "virt0": self.virt0}


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\{([A-Za-z_][\w.\-]*(?::[\w.\-]+)?)\}")


def expand(obj: Any, ctx: dict, port_resolver: Callable[[str, str], str] | None = None) -> Any:
    """Expand ``{name}``, ``{port:name}`` and ``{url:name}`` (from ports.json) in strings, recursively
    in lists and dicts. Unknown names are left untouched so a later pass can fill them."""
    if isinstance(obj, str):
        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key.startswith(("port:", "url:")):
                if port_resolver is None:
                    return m.group(0)
                kind, name = key.split(":", 1)
                return str(port_resolver(kind, name))
            if key in ctx and ctx[key] is not None:
                return str(ctx[key])
            return m.group(0)
        prev = None
        out = obj
        for _ in range(5):  # allow nested templates
            if out == prev:
                break
            prev, out = out, _TOKEN.sub(sub, out)
        return out
    if isinstance(obj, list):
        return [expand(x, ctx, port_resolver) for x in obj]
    if isinstance(obj, dict):
        return {k: expand(v, ctx, port_resolver) for k, v in obj.items()}
    return obj


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# small expression language for gate checks: {"sd_bpb": "<= 0.02"}
# ---------------------------------------------------------------------------

_CHECK = re.compile(r"\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$")


def _coerce(s: str) -> Any:
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(s)
    except ValueError:
        return s.strip("'\"")


def check_one(value: Any, spec: Any) -> bool:
    if not isinstance(spec, str):
        return value == spec
    m = _CHECK.match(spec)
    if not m:
        return str(value) == spec
    op, rhs = m.group(1), _coerce(m.group(2))
    if value is None:
        return False
    try:
        if isinstance(rhs, float):
            value = float(value)
    except (TypeError, ValueError):
        return False
    return {"<=": value <= rhs, ">=": value >= rhs, "<": value < rhs, ">": value > rhs,
            "==": value == rhs, "!=": value != rhs}[op]


def get_path(metrics: dict, key: str) -> Any:
    """``metrics[key]``, or a dotted path into nested dicts (``dev_bpb.sd``)."""
    if key in metrics:
        return metrics[key]
    cur: Any = metrics
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def run_checks(metrics: dict, checks: dict) -> tuple[bool, list[str]]:
    failed = []
    for key, spec in (checks or {}).items():
        v = get_path(metrics or {}, key)
        if not check_one(v, spec):
            failed.append(f"{key}={v!r} !{spec}")
    return (not failed), failed


# ---------------------------------------------------------------------------
# validation and normalisation
# ---------------------------------------------------------------------------

def normalise(cfg: dict, *, root_override: str | None = None) -> dict:
    """Fill defaults, resolve the path context and check the step graph."""
    cfg = copy.deepcopy(cfg)
    for key in ("campaign", "tz", "gpu_stop", "hold_end", "steps"):
        if key not in cfg:
            raise ConfigError(f"missing top-level key {key!r}")
    tz = parse_tz(cfg["tz"])
    cfg["_tz"] = tz
    paths = dict(cfg.get("paths") or {})
    R = root_override or os.environ.get("R") or paths.get("R")
    if not R:
        raise ConfigError("paths.R missing (or set $R)")
    ctx = {"R": R, "campaign": cfg["campaign"], "code": CODE_ROOT}
    for k, v in paths.items():
        if k != "R":
            ctx[k] = expand(v, ctx)
    ctx.setdefault("bundle", os.path.join(CODE_ROOT, "bundle"))
    py = cfg.get("python", {}) or {}
    ctx["py"] = os.environ.get("RSI_PY") or expand(py.get("serve") or "python3", ctx)
    ctx["train_py"] = expand(py.get("train") or ctx["py"], ctx)
    for k in ("runs_root", "campaign_dir", "status_md"):
        if k not in ctx:
            raise ConfigError(f"paths.{k} missing")
    cfg["_ctx"] = ctx

    for k in ("gpu_stop", "hold_end", "planned_start"):
        if cfg.get(k) is not None:
            cfg["_" + k] = parse_time(cfg[k], tz)
    if cfg["_gpu_stop"] > cfg["_hold_end"]:
        raise ConfigError("gpu_stop is after hold_end")

    ids: list[str] = []
    for s in cfg["steps"]:
        if "id" not in s or "kind" not in s:
            raise ConfigError(f"step without id/kind: {s}")
        if s["kind"] not in STEP_KINDS:
            raise ConfigError(f"{s['id']}: unknown kind {s['kind']!r}")
        if s["id"] in ids:
            raise ConfigError(f"duplicate step id {s['id']!r}")
        ids.append(s["id"])
    known = set(ids)
    for s in cfg["steps"]:
        for k in ("after", "needs", "until"):
            s[k] = list(s.get(k) or [])
            for d in s[k]:
                if d not in known:
                    raise ConfigError(f"{s['id']}.{k}: unknown step {d!r}")
        for k in ("requires_flags", "wait_for_flags", "skip_if_flags", "holds"):
            s[k] = list(s.get(k) or [])
        for k in ("not_before", "latest_start", "deadline"):
            s["_" + k] = parse_time(s.get(k), tz)
            if s["_" + k] is not None and s["_" + k] > cfg["_hold_end"]:
                raise ConfigError(f"{s['id']}.{k} is after hold_end")
        if s["_deadline"] is not None and s["_deadline"] > cfg["_gpu_stop"] and s["kind"] not in ("decision",):
            raise ConfigError(f"{s['id']}.deadline is after gpu_stop")
    _check_acyclic(cfg["steps"])
    return cfg


def _check_acyclic(steps: list[dict]) -> None:
    deps = {s["id"]: set(s["after"]) | set(s["needs"]) for s in steps}
    seen: dict[str, int] = {}

    def visit(n: str, stack: tuple) -> None:
        if seen.get(n) == 2:
            return
        if seen.get(n) == 1:
            raise ConfigError(f"dependency cycle: {' -> '.join(stack + (n,))}")
        seen[n] = 1
        for d in deps[n]:
            visit(d, stack + (n,))
        seen[n] = 2

    for n in deps:
        visit(n, ())


def load_config(path: str, *, root_override: str | None = None) -> dict:
    raw = load_raw(path)
    cfg = normalise(raw, root_override=root_override)
    cfg["_hash"] = config_hash(raw)
    cfg["_path"] = os.path.abspath(path)
    return cfg
