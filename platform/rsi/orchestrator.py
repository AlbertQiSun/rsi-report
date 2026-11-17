"""Orchestrator CLI (trusted zone, outside every sandbox).

    python -m rsi.orchestrator --config run.yaml            # start a run
    python -m rsi.orchestrator --config run.yaml --resume   # resume (uses the frozen config.json)
    python -m rsi.orchestrator --pause  <run_dir>            # threads stop at the next trial boundary
    python -m rsi.orchestrator --status <run_dir>

Config (YAML or JSON); ``${VAR}`` and ``$VAR`` are expanded from the environment::

    run: nanochat-9b-karpathy-{ts}        # {ts} -> UTC timestamp
    runs_root: ${R}/runs
    gen: 0
    task: {name: nanochat, args: {}}
    policy: {name: karpathy_greedy, threads: 2, seeds: [0], s_ref: null}
    model: improver                       # gateway model name
    improver: {model: Qwen/Qwen3.5-9B}    # recorded in run_start
    scaffold_dir: null
    budget: {trials: 90, wall_s: 50000, episode_s: 900}
    lanes: [1, 2]
    evald: {url: null, token_file: ${R}/run/token}        # url null -> ports.json "evald"
    harness: {kind: opencode, gateway_url: null, ...}     # gateway_url null -> ports.json "gateway"

Pause / resume: SIGINT or SIGTERM (or --pause) creates <run_dir>/PAUSE; running episodes and
evaluations finish, then threads stop. ``--resume`` deletes PAUSE, rebuilds thread state from the
ledger and re-awaits any evald job that was in flight (job ids are idempotent).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import pathlib
import re
import signal
import sys
from typing import Any

from rsi import events
from rsi.context import RunContext
from rsi.ledger import Ledger
from rsi.runs import RunDir, atomic_write_json, read_json


# ------------------------------------------------------------------ config
def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def load_config(path: str) -> dict:
    text = pathlib.Path(path).read_text()
    if path.endswith(".json"):
        cfg = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit("PyYAML is not installed; use a .json config") from e
        cfg = yaml.safe_load(text)
    return _expand(cfg)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def scaffold_hash(d: str | None) -> str | None:
    if not d or not os.path.isdir(d):
        return None
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ services
def rsi_root() -> str:
    return os.environ.get("RSI_ROOT") or os.environ.get("R") or "/scratch/user/rsi"


def service_url(name: str, ports_file: str | None = None) -> str:
    """Resolve a service URL from $RSI_ROOT/run/ports.json (int port, "host:port", URL or {host, port})."""
    pf = ports_file or os.path.join(rsi_root(), "run", "ports.json")
    ports = read_json(pf, {}) or {}
    v = ports.get(name)
    if v is None:
        raise SystemExit(f"service {name!r} not found in {pf}")
    if isinstance(v, dict):
        if v.get("url"):
            return v["url"]
        return f"http://{v.get('host', '127.0.0.1')}:{v['port']}"
    if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
        return f"http://127.0.0.1:{v}"
    return v if str(v).startswith("http") else f"http://{v}"


def make_evald(cfg: dict):
    e = dict(cfg.get("evald") or {})
    if e.get("factory"):  # "module:callable" (tests, alternative backends)
        mod, _, fn = e["factory"].partition(":")
        return getattr(importlib.import_module(mod), fn)(e)
    from rsi.evald.client import EvalClient  # worker B

    url = e.get("url") or service_url("evald", e.get("ports_file"))
    return EvalClient(url, e.get("token_file") or os.path.join(rsi_root(), "run", "token"))


def make_harness(cfg: dict, runs_root: str, run: str):
    from rsi.harness.mini import make_harness as _mk

    h = dict(cfg.get("harness") or {})
    if not h.get("gateway_url"):
        h["gateway_url"] = service_url("gateway", h.get("ports_file"))
    h.setdefault("trajectories_dir", os.path.join(runs_root, run, "trajectories", "harness"))
    if h.get("factory"):
        mod, _, fn = h["factory"].partition(":")
        return getattr(importlib.import_module(mod), fn)(h)
    return _mk(h)


def make_task(cfg: dict):
    from rsi.tasks.base import get_task

    t = cfg["task"] if isinstance(cfg["task"], dict) else {"name": cfg["task"]}
    return get_task(t["name"], **(t.get("args") or {}))


# ------------------------------------------------------------------ run
def normalize_config(cfg: dict) -> dict:
    """Accept both config shapes:
    A: policy: {name: x, ...}            B (campaign): policy: x, policy_cfg: {...}
    plus top-level seeds -> policy.seeds, rng_seed/seed, start_from, history_from."""
    cfg = dict(cfg)
    pol = cfg.get("policy") or {"name": "karpathy_greedy"}
    if isinstance(pol, str):
        pol = {"name": pol, **(cfg.get("policy_cfg") or {})}
    else:
        pol = {**(cfg.get("policy_cfg") or {}), **pol}
    if cfg.get("seeds") is not None:
        pol.setdefault("seeds", list(cfg["seeds"]))
    cfg["policy"] = pol
    cfg.pop("policy_cfg", None)
    if "rng_seed" in cfg and "seed" not in cfg:
        cfg["seed"] = cfg["rng_seed"]
    cfg.setdefault("seed", 0)   # rsi.update.analysis pairs runs across arms by seed
    hf = cfg.get("history_from") or []
    cfg["history_from"] = [hf] if isinstance(hf, str) else list(hf)
    return cfg


def build_context(cfg: dict, *, resume: bool) -> tuple[RunContext, Any]:
    from rsi.policies import get_policy

    runs_root = cfg.get("runs_root") or events.runs_root()
    run = cfg["run"]
    rd = RunDir(runs_root, run)
    if resume:
        frozen = read_json(rd.config_path)
        if frozen is None:
            raise SystemExit(f"nothing to resume at {rd.path}")
        cfg = frozen
    elif os.path.exists(rd.config_path):
        raise SystemExit(f"run {run} already exists at {rd.path}; use --resume")
    cfg = normalize_config(cfg)
    rd.create()
    events.register_run(run, rd.events)
    if not resume:
        atomic_write_json(rd.config_path, cfg)
    task = make_task(cfg)
    pol_cfg = dict(cfg.get("policy") or {})
    policy = get_policy(pol_cfg.get("name", "karpathy_greedy"))(pol_cfg)
    ctx = RunContext(run=run, gen=int(cfg.get("gen", 0)), task=task, evald=make_evald(cfg),
                     harness=make_harness(cfg, runs_root, run), model=cfg.get("model", "improver"),
                     scaffold_dir=pathlib.Path(cfg["scaffold_dir"]) if cfg.get("scaffold_dir") else None,
                     budget=dict(cfg.get("budget") or {}), run_dir=pathlib.Path(rd.path),
                     ledger=Ledger(rd.ledger_path), rng_seed=int(cfg.get("seed", 0)),
                     lanes=list(cfg.get("lanes") or [1]), policy_cfg=pol_cfg,
                     start_from=pathlib.Path(cfg["start_from"]) if cfg.get("start_from") else None,
                     history_from=list(cfg.get("history_from") or []))
    if resume:
        st = ctx.load_state()
        # wall budget counts time already spent before the pause
        ctx.t_start -= float(st.get("elapsed_s", 0.0))
    return ctx, policy


def run(cfg: dict, *, resume: bool = False) -> dict:
    ctx, policy = build_context(cfg, resume=resume)
    cfg = normalize_config(read_json(str(ctx.run_dir / "config.json")) or cfg)
    if resume and ctx.paused():
        os.remove(ctx.run_dir / "PAUSE")
    if not resume:
        ctx.emit("run_start", "orchestrator", policy=policy.name,
                 task=ctx.task.name, improver={"model": (cfg.get("improver") or {}).get("model", ctx.model),
                                               "gateway_model": ctx.model,
                                               "scaffold_hash": scaffold_hash(cfg.get("scaffold_dir"))},
                 budget={"trials": ctx.budget.get("trials"), "wall_s": ctx.budget.get("wall_s")},
                 config_hash=config_hash(cfg), harness=(cfg.get("harness") or {}).get("kind", "opencode"),
                 lanes=ctx.lanes, s_ref=ctx.policy_cfg.get("s_ref"),
                 lower_is_better=ctx.task.lower_is_better, metric=ctx.task.metric)
    else:
        ctx.emit("message", "orchestrator", to="orchestrator", summary="resumed run", tokens=1)

    def on_signal(signum, _frame):
        pathlib.Path(ctx.run_dir / "PAUSE").write_text(f"signal {signum}\n")
        print(f"[orchestrator] signal {signum}: pausing at next trial boundary", file=sys.stderr, flush=True)

    if threading_main():
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
    summary = policy.run(ctx)
    ctx.save_state(elapsed_s=ctx.elapsed(), last_summary=summary)
    if ctx.paused():
        ctx.emit("message", "orchestrator", to="orchestrator", summary="paused; resume with --resume", tokens=1)
    print(json.dumps(summary, default=str))
    return summary


def threading_main() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


def status(run_dir: str) -> dict:
    led = Ledger(os.path.join(run_dir, "ledger.jsonl"))
    fin = led.finished()
    counts: dict[str, int] = {}
    for r in fin:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    pending = [t for t, r in led.trials().items() if r.get("status") == "pending"]
    return {"run_dir": run_dir, "paused": os.path.exists(os.path.join(run_dir, "PAUSE")),
            "finished": len(fin), "pending": pending, "status_counts": counts,
            "state": read_json(os.path.join(run_dir, "state.json"), {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rsi.orchestrator")
    ap.add_argument("--config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pause", metavar="RUN_DIR")
    ap.add_argument("--status", metavar="RUN_DIR")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (dotted path, JSON value), e.g. --set budget.trials=4")
    a = ap.parse_args(argv)
    if a.pause:
        pathlib.Path(a.pause, "PAUSE").write_text("manual pause\n")
        print(f"pause requested for {a.pause}")
        return 0
    if a.status:
        print(json.dumps(status(a.status), indent=1, default=str))
        return 0
    if not a.config:
        ap.error("--config is required")
    cfg = load_config(a.config)
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        cur = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = val
    if "{ts}" in cfg.get("run", ""):
        if a.resume:
            raise SystemExit("--resume needs the concrete run id (no {ts}); pass --set run=<id>")
        cfg["run"] = cfg["run"].replace("{ts}", _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S"))
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", cfg["run"]):
        raise SystemExit(f"bad run id {cfg['run']!r}")
    run(cfg, resume=a.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
