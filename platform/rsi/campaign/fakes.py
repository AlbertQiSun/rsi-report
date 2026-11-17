"""Fake components for ``python -m rsi.campaign --dry-run`` and tests (CPU only, no network).

In-process backends: FakeEvalClient (file-backed lanes, logs every pause/resume),
FakeLoader (records adapter loads), FakeProbe (a service is down iff ``down_<name>``
exists; services listed in ``UP_FILE_SERVICES`` are up only while ``up_<name>`` exists).

Subprocess fakes: ``python -m rsi.campaign.fakes <orchestrator|lora|gate|serve|action|restart> ...``.
They sleep for virtual durations divided by ``$RSI_CAMPAIGN_TIME_SCALE`` and write
schema-valid events through ``rsi.events``. Fault injection:
  RSI_FAKE_GATE_FAIL=<step ids, comma>   gate fakes report failing metrics
  RSI_FAKE_CRASH=<run id substring>      fake orchestrator exits 1 once at trial 2
  RSI_FAKE_SFT_N=<int>                   n_examples reported by the fake SFT builder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time

UP_FILE_SERVICES = {"vllm_27b", "evo27b"}


def _scale() -> float:
    return float(os.environ.get("RSI_CAMPAIGN_TIME_SCALE", "1") or 1)


def _vsleep(virtual_s: float) -> None:
    time.sleep(max(0.0, float(virtual_s)) / _scale())


def _fake_dir() -> str:
    d = os.environ.get("RSI_CAMPAIGN_FAKE_DIR") or "."
    os.makedirs(d, exist_ok=True)
    return d


def _vnow() -> float:
    real0 = float(os.environ.get("RSI_CAMPAIGN_REAL0", time.time()))
    virt0 = float(os.environ.get("RSI_CAMPAIGN_VIRT0", real0))
    return virt0 + (time.time() - real0) * _scale()


def _append(path: str, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# in-process backends
# ---------------------------------------------------------------------------

class FakeEvalClient:
    def __init__(self, fake_dir: str, clock, lanes=(1, 2)):
        self.dir = fake_dir
        self.clock = clock
        self.path = os.path.join(fake_dir, "lanes.json")
        self.log = os.path.join(fake_dir, "lane_log.jsonl")
        os.makedirs(fake_dir, exist_ok=True)
        if not os.path.exists(self.path):
            self._save({str(l): {"lane": l, "gpu": l, "busy": False, "paused": False} for l in lanes})

    def _load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def _save(self, d: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)

    def lanes(self) -> list[dict]:
        busy_file = os.path.join(self.dir, "busy_lanes")
        busy = set()
        if os.path.exists(busy_file):
            busy = {x.strip() for x in open(busy_file).read().split(",") if x.strip()}
        out = []
        for k, v in sorted(self._load().items()):
            v = dict(v)
            v["busy"] = k in busy
            out.append(v)
        return out

    def _set(self, lane: int, paused: bool) -> None:
        d = self._load()
        d[str(lane)]["paused"] = paused
        self._save(d)
        _append(self.log, {"vt": self.clock.now(), "op": "pause" if paused else "resume", "lane": lane})

    def pause_lane(self, lane: int) -> None:
        self._set(lane, True)

    def resume_lane(self, lane: int) -> None:
        self._set(lane, False)


class FakeLoader:
    def __init__(self, fake_dir: str, clock):
        self.path = os.path.join(fake_dir, "adapters.jsonl")
        self.clock = clock

    def load(self, name: str, path: str) -> tuple[bool, str]:
        if not os.path.isdir(path):
            return False, "adapter dir missing"
        _append(self.path, {"vt": self.clock.now(), "name": name, "path": path})
        return True, "loaded"

    def verify(self, name: str) -> bool:
        return name in loaded_adapters(os.path.dirname(self.path))


def loaded_adapters(fake_dir: str) -> set[str]:
    p = os.path.join(fake_dir, "adapters.jsonl")
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["name"] for l in open(p) if l.strip()}


class FakeProbe:
    def __init__(self, fake_dir: str):
        self.dir = fake_dir

    def check(self, name: str, url: str) -> tuple[bool, str]:
        if os.path.exists(os.path.join(self.dir, f"down_{name}")):
            return False, "fake down"
        if name in UP_FILE_SERVICES:
            up = os.path.exists(os.path.join(self.dir, f"up_{name}"))
            return up, "fake up" if up else "fake not started"
        return True, "fake ok"


# ---------------------------------------------------------------------------
# subprocess fakes
# ---------------------------------------------------------------------------

def _load_cfg(path: str) -> dict:
    text = open(path).read()
    try:
        return json.loads(text)
    except ValueError:
        import yaml  # type: ignore

        return yaml.safe_load(text)


def _bias(cfg: dict) -> float:
    """Arm effect used by the fake µAR runs so the dry-run decisions have something to find."""
    model = str(cfg.get("model") or "")
    scaffold = str(cfg.get("scaffold_dir") or "")
    b = 0.0
    if "@" in model:
        name = model.split("@", 1)[1].removesuffix("-27b")
        b += 0.004 if name.endswith("p") else (0.02 if name.endswith("c") else 0.05)
    if "ladder" in model:
        b += 0.06
    if scaffold:
        b += 0.0 if "placebo" in scaffold else 0.03
    return b


def fake_orchestrator(args: argparse.Namespace) -> int:
    from rsi import events
    from rsi.runs import RunDir, atomic_write_json

    cfg = _load_cfg(args.config)
    run = cfg["run"]
    rd = RunDir(cfg["runs_root"], run).create()
    events.register_run(run, rd.events)
    fd = _fake_dir()
    fake = cfg.get("_fake") or {}
    trial_s = float(fake.get("trial_s", 60))
    # preconditions the real orchestrator would also need
    for key in ("start_from", "scaffold_dir"):
        if cfg.get(key) and not os.path.exists(cfg[key]):
            print(f"fake orchestrator: {key} {cfg[key]} does not exist", file=sys.stderr)
            return 4
    model = str(cfg.get("model") or "")
    if "@" in model and model.split("@", 1)[1] not in loaded_adapters(fd):
        print(f"fake orchestrator: adapter for {model} not loaded", file=sys.stderr)
        return 4
    needs_up = fake.get("needs_up")
    if needs_up and not os.path.exists(os.path.join(fd, f"up_{needs_up}")):
        print(f"fake orchestrator: service {needs_up} not up", file=sys.stderr)
        return 4

    prior = [e for e in events.read_events(rd.events)] if os.path.exists(rd.events) else []
    if not any(e["ev"] == "run_start" for e in prior):
        events.emit(run, "run_start", "orchestrator", gen=int(cfg.get("gen", 0)), policy=cfg.get("policy"),
                    task=cfg.get("task"), improver={"model": model, "scaffold_hash": None},
                    budget=cfg.get("budget", {}), config_hash="fake")
    done = {e["trial"] for e in prior if e["ev"] == "eval_result" and e.get("trial") is not None}
    n_total = int((cfg.get("budget") or {}).get("trials", 10))
    is_muar = str(cfg.get("task", "")).startswith("muar")
    rng = random.Random(_h(run))
    best = None
    for e in prior:
        if e["ev"] == "eval_result" and e.get("status") == "ok":
            best = e["value"] if best is None else min(best, e["value"])
    crash_tag = os.environ.get("RSI_FAKE_CRASH", "")
    stopped = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__("flag", True))
    gen = int(cfg.get("gen", 0))
    for trial in range(n_total):
        if trial in done:
            continue
        if rd.paused() or stopped["flag"]:
            print(json.dumps({"paused": True, "n_trials": len(done)}))
            return 0
        if crash_tag and crash_tag in run and trial == 2:
            marker = os.path.join(fd, f"crashed_{run}")
            if not os.path.exists(marker):
                open(marker, "w").close()
                print("fake orchestrator: injected crash", file=sys.stderr)
                return 1
        _vsleep(trial_s)
        value = (1.0 if not is_muar else 2.0) - 0.002 * trial * (1 + _bias(cfg) * 10) + rng.gauss(0, 0.004)
        h = hashlib.sha256(f"{run}-{trial}".encode()).hexdigest()[:12]
        art = rd.p("artifacts", h)
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "train.py"), "w") as f:
            f.write(f"# fake artifact {run} trial {trial}\n")
        status = "ok" if rng.random() > 0.1 else "crash"
        lane = (cfg.get("lanes") or [None])[0]
        events.emit(run, "eval_submit", "orchestrator", gen=gen, trial=trial, artifact_hash=h, lane=lane,
                    seeds=[int(cfg.get("rng_seed", 0))])
        events.emit(run, "eval_result", "evald", gen=gen, trial=trial, artifact_hash=h, metric="val_bpb",
                    value=round(value, 6) if status == "ok" else None, seeds=[0], wall_s=round(trial_s * 0.4, 1),
                    status=status, audit={"level": "L1", "flags": []})
        better = status == "ok" and (best is None or value < best)
        events.emit(run, "decision", "orchestrator", gen=gen, trial=trial, artifact_hash=h,
                    verdict="adopt" if better else "reject", best_before=best,
                    best_after=value if better else best, reason="fake")
        if better:
            best = value
        done.add(trial)
    nauc = None
    if is_muar:
        r2 = random.Random(_h(cfg.get("task"), cfg.get("rng_seed")))
        nauc = round(0.40 + _bias(cfg) + 0.05 * r2.random() + random.Random(_h(run)).gauss(0, 0.01), 4)
    events.emit(run, "run_end", "orchestrator", gen=gen, best_value=best, n_trials=len(done), nAUC=nauc)
    atomic_write_json(rd.state_path, {"done": True})
    print(json.dumps({"best_value": best, "n_trials": len(done), "nAUC": nauc}))
    return 0


def fake_lora(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.sft_dir):
        print(f"fake lora: sft dir {args.sft_dir} missing", file=sys.stderr)
        return 2
    _vsleep(args.duration_s)
    os.makedirs(args.adapter_out, exist_ok=True)
    with open(os.path.join(args.adapter_out, "adapter_config.json"), "w") as f:
        json.dump({"fake": True, "gpu": args.gpu, "r": 16}, f)
    print(json.dumps({"adapter_path": args.adapter_out}))
    return 0


def fake_gate(args: argparse.Namespace) -> int:
    from rsi.campaign.config import _CHECK, _coerce

    checks = json.loads(args.checks or "{}")
    fail = args.name in [x.strip() for x in os.environ.get("RSI_FAKE_GATE_FAIL", "").split(",")]
    _vsleep(args.duration_s)
    metrics = {}
    for key, spec in checks.items():
        m = _CHECK.match(str(spec))
        if not m:
            metrics[key] = spec
            continue
        op, rhs = m.group(1), _coerce(m.group(2))
        if isinstance(rhs, bool):
            metrics[key] = (not rhs) if fail else rhs
            continue
        good = {"<=": rhs, ">=": rhs, "<": rhs - 1e-3, ">": rhs + 1e-3, "==": rhs, "!=": rhs + 1}[op]
        bad = {"<=": rhs * 2 + 1, ">=": rhs / 2, "<": rhs * 2 + 1, ">": rhs / 2, "==": rhs + 1, "!=": rhs}[op]
        metrics[key] = bad if fail else good
    metrics["fake"] = True
    if fail and not checks:  # gates judged by exit code (e.g. the evald self-test)
        print(json.dumps({"fake": True, "all_pass": False}))
        return 1
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f)
    print(json.dumps(metrics))
    return 0


def fake_serve(args: argparse.Namespace) -> int:
    up = os.path.join(_fake_dir(), f"up_{args.name}")
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    _vsleep(args.load_s)
    open(up, "w").close()
    while not stop["flag"]:
        time.sleep(0.02)
    try:
        os.remove(up)
    except OSError:
        pass
    print(json.dumps({"stopped": args.name}))
    return 0


def fake_restart(args: argparse.Namespace) -> int:
    _vsleep(args.duration_s)
    try:
        os.remove(os.path.join(_fake_dir(), f"down_{args.name}"))
    except OSError:
        pass
    _append(os.path.join(_fake_dir(), "restarts.jsonl"), {"vt": _vnow(), "name": args.name})
    print(json.dumps({"restarted": args.name}))
    return 0


def fake_action(args: argparse.Namespace) -> int:
    spec = json.loads(open(args.spec).read()) if args.spec else {}
    _vsleep(args.duration_s)
    if args.what == "lessons":
        out = spec["out_dir"]
        os.makedirs(out, exist_ok=True)
        kind = "placebo" if spec.get("generic_placebo") else "lessons"
        with open(os.path.join(out, "lessons.md"), "w") as f:
            f.write(f"# fake {kind}\n" + "".join(f"- lesson {i}\n" for i in range(7)))
        print(json.dumps({"scaffold_hash": hashlib.sha256(out.encode()).hexdigest()[:12], "n_lessons": 7,
                          "out_dir": out}))
    elif args.what == "sft":
        out = spec["out"]
        os.makedirs(out, exist_ok=True)
        n = int(os.environ.get("RSI_FAKE_SFT_N", "64"))
        open(os.path.join(out, "train.jsonl"), "w").write("{}\n" * n)
        print(json.dumps({"n_examples": n, "heldout_overlap": 0, "out": out}))
    elif args.what == "endpoint":
        print(json.dumps({"n_artifacts": spec.get("top_k"), "seeds": spec.get("seeds"), "results": []}))
    else:
        print(json.dumps({"ok": True, "what": args.what}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rsi.campaign.fakes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orchestrator")
    o.add_argument("--config", required=True)
    o.add_argument("--resume", action="store_true")
    lo = sub.add_parser("lora")
    lo.add_argument("sft_dir")
    lo.add_argument("adapter_out")
    lo.add_argument("gpu")
    lo.add_argument("--duration_s", type=float, default=2700)
    g = sub.add_parser("gate")
    g.add_argument("--name", required=True)
    g.add_argument("--out", default=None)
    g.add_argument("--checks", default="{}")
    g.add_argument("--duration_s", type=float, default=600)
    s = sub.add_parser("serve")
    s.add_argument("--name", required=True)
    s.add_argument("--load_s", type=float, default=600)
    r = sub.add_parser("restart")
    r.add_argument("--name", required=True)
    r.add_argument("--duration_s", type=float, default=300)
    a = sub.add_parser("action")
    a.add_argument("what")
    a.add_argument("--spec", default=None)
    a.add_argument("--duration_s", type=float, default=60)
    args = ap.parse_args(argv)
    return {"orchestrator": fake_orchestrator, "lora": fake_lora, "gate": fake_gate, "serve": fake_serve,
            "restart": fake_restart, "action": fake_action}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
