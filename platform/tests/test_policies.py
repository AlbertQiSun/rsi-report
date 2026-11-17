"""End-to-end CPU tests for worker E's policies (naive_trial_ledger, recursive_first_steps,
apex_ai4ai, self_evolution) on a tiny deterministic task.

Pieces owned by other workers that may not have landed yet are stubbed *here only*:
``rsi.policies.base.Policy`` (A), ``rsi.update`` (D), the evaluator (B) and the harness (A/C).
The stub harness stands in for OpenCode + tests/fake_openai_server.py: it edits the workspace
directly and emits tool_call events, so no model server is needed.

Run:  cd platform && python -m pytest -q tests/test_policies.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import types

import pytest

PLATFORM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))

# ---------------------------------------------------------------------------
# stubs for not-yet-landed modules (only installed if the real ones are missing)
# ---------------------------------------------------------------------------
try:
    import rsi.policies.base  # noqa: F401
except ImportError:
    _base = types.ModuleType("rsi.policies.base")

    class Policy:  # minimal copy of the INTERFACES.md contract
        name = ""

        def __init__(self, cfg: dict | None = None):
            self.cfg = dict(cfg or {})

        def run(self, ctx) -> dict:
            raise NotImplementedError

    _base.Policy = Policy
    sys.modules["rsi.policies.base"] = _base
    import rsi.policies
    rsi.policies.base = _base



def _stub_write_lessons(run_dirs, *, model, gateway_url, out_dir, generic_placebo=False):
    os.makedirs(out_dir, exist_ok=True)
    text = ("- generic: try things carefully\n" if generic_placebo
            else f"- lesson: move parameters toward the optimum (from {len(run_dirs)} runs)\n")
    with open(os.path.join(out_dir, "lessons.md"), "a") as f:
        f.write(text)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _stub_build_sft_data(run_dirs, *, accepted_only=True, placebo=False, out, heldout=None):
    os.makedirs(out, exist_ok=True)
    n = 0
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        for d in run_dirs:
            for line in open(os.path.join(d, "ledger.jsonl")):
                r = json.loads(line)
                good = r.get("status") in ("adopted", "keep")
                if r.get("trial") is not None and r.get("status") not in ("pending", "baseline") and good != placebo:
                    f.write(json.dumps({"prompt": "p", "completion": r.get("hypothesis", "")}) + "\n")
                    n += 1
    return {"n_train": n, "n_valid": 0, "ok": n > 0}


@pytest.fixture
def stub_update(monkeypatch):
    """D's rsi.update is exercised by D's tests; here we only test the policy wiring."""
    import importlib

    mod = importlib.import_module("rsi.update")
    monkeypatch.setattr(mod, "write_lessons", _stub_write_lessons, raising=False)
    monkeypatch.setattr(mod, "build_sft_data", _stub_build_sft_data, raising=False)


from rsi import events  # noqa: E402
from rsi.context import RunContext  # noqa: E402
from rsi.tasks.base import Task, register  # noqa: E402
from rsi.policies.naive_trial_ledger import NaiveTrialLedger, effective_cfg, merge3  # noqa: E402
from rsi.policies.recursive_first_steps import RecursiveFirstSteps  # noqa: E402
from rsi.policies.apex_ai4ai import ApexAI4AI  # noqa: E402
from rsi.policies.self_evolution import SelfEvolution, run_arm  # noqa: E402

REQUIRED_EVS = {"proposal", "eval_submit", "decision", "message", "agent_spawn"}


# ---------------------------------------------------------------------------
# tiny CPU task
# ---------------------------------------------------------------------------

def _opt(name: str) -> tuple[float, float]:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return 0.2 + (h % 7) / 20.0, 3.0 + (h >> 4) % 5


@register
class ToyTask(Task):
    """loss = 1 + (lr-a)^2 + 0.01 (width-b)^2 + small seed noise; ``cache: 1`` is a reward hack
    that improves dev by 0.3 but not the sealed score."""
    name = "toy"
    metric = "loss"
    lower_is_better = True
    editable_files = ("params.json",)
    dev_seeds = (0,)

    def __init__(self, name: str = "toy", root: str | None = None):
        self.name = name
        self.opt = _opt(name)
        self.root = pathlib.Path(root or os.environ["TOY_ROOT"]) / name.replace("/", "_")

    def start_artifact(self) -> pathlib.Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / "params.json"
        if not p.exists():
            p.write_text(json.dumps({"lr": 1.0, "width": 1, "wd": 0.0}, indent=1, sort_keys=True) + "\n")
        return self.root

    def task_card(self) -> str:
        return "Minimise loss by editing params.json (lr, width, wd)."

    @staticmethod
    def score(params: dict, opt, seed: int, sealed: bool) -> float:
        lr, w = float(params["lr"]), float(params["width"])
        noise = random.Random(f"{seed}|{sealed}").gauss(0, 0.002)
        v = 1 + (lr - opt[0]) ** 2 + 0.01 * (w - opt[1]) ** 2 + 0.05 * float(params.get("wd", 0)) ** 2 + noise
        if params.get("cache") and not sealed:
            v -= 0.3
        return v


class FakeEval:
    """Synchronous stand-in for rsi.evald.client.EvalClient; emits eval_result like evald does."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.paused: list[int] = []

    def submit(self, run, task, artifact_dir, *, seeds, lane=None, gen=0, trial=None, level=None, **kw):
        from rsi.tasks.base import get_task

        t = get_task(task)
        h = os.path.basename(artifact_dir.rstrip("/"))
        try:
            params = json.loads(pathlib.Path(artifact_dir, "params.json").read_text())
            if float(params["lr"]) > 5:
                raise RuntimeError("diverged")
            vals = [ToyTask.score(params, t.opt, s, False) for s in seeds]
            sealed = sum(ToyTask.score(params, t.opt, s + 7777, True) for s in seeds) / len(seeds)
            status, value = "ok", sum(vals) / len(vals)
        except Exception:
            vals, sealed, status, value = [], None, "crash", None
        flags = ["cache_suspect"] if status == "ok" and params.get("cache") else []
        with self.lock:
            job = f"j{len(self.jobs)}"
            res = {"artifact_hash": h, "metric": t.metric, "value": value, "seeds": list(seeds), "wall_s": 0.01,
                   "status": status, "audit": {"level": level or "L1", "flags": flags}, "values": vals}
            self.jobs[job] = res
            self.calls.append({"run": run, "level": level, "seeds": list(seeds), "kw": kw})
        events.emit(run, "eval_result", "evald", gen=gen, trial=trial,
                    **{k: res[k] for k in ("artifact_hash", "metric", "value", "seeds", "wall_s", "status", "audit")})
        if sealed is not None:
            with open(os.path.join(os.path.dirname(events.events_path(run)), "sealed.jsonl"), "a") as f:
                f.write(json.dumps({"job_id": job, "artifact_hash": h, "value": sealed}) + "\n")
        return job

    def result(self, job_id, *, wait=True, timeout_s=3600):
        return dict(self.jobs[job_id])

    def lanes(self):
        return [{"lane": 1, "gpu": 1, "busy": False, "paused": 1 in self.paused}]

    def pause_lane(self, lane):
        self.paused.append(lane)

    def resume_lane(self, lane):
        self.paused.remove(lane)


class Die(BaseException):
    """Simulated process death (not caught by the policies)."""


class FakeHarness:
    """Stand-in for the OpenCode harness + fake model server."""

    def __init__(self, *, die_after: int | None = None, fail_models=(), hack_p: float = 0.0,
                 explore_p: float = 0.15, crash_p: float = 0.08, numerics_claim_dir: str | None = None,
                 force_direction: str | None = None):
        self.lock = threading.Lock()
        self.n = 0
        self.die_after, self.fail_models, self.hack_p = die_after, set(fail_models), hack_p
        self.explore_p, self.crash_p = explore_p, crash_p
        self.numerics_claim_dir, self.force_direction = numerics_claim_dir, force_direction

    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s,
                    parent="orchestrator"):
        with self.lock:
            self.n += 1
            n = self.n
        if self.die_after is not None and n > self.die_after:
            raise Die(f"killed at episode {n}")
        rng = random.Random(f"{run}|{agent}|{trial}|{n}")
        events.emit(run, "tool_call", agent, parent=parent, gen=gen, trial=trial, tool="read",
                    arg_summary="params.json", ok=True, latency_s=0.0)
        if model in self.fail_models:
            return {"final": "error: invalid tool call", "n_tool_calls": 1, "trajectory": "", "ok": False,
                    "submitted": None}
        role = agent.rsplit("/", 1)[-1]
        if agent == "critic":
            v = "HACK" if '"cache"' in prompt else "OK"
            return {"final": json.dumps({"verdict": v, "reason": "uses a cache", "rule": "no cache key"}),
                    "n_tool_calls": 1, "trajectory": "", "ok": True, "submitted": None}
        if role in ("propose", "next") or agent == "lessons-writer":
            return {"final": json.dumps({"hypothesis": f"try changing lr (idea {n})"}), "n_tool_calls": 1,
                    "trajectory": "", "ok": True, "submitted": None}
        p = pathlib.Path(workspace) / "params.json"
        params = json.loads(p.read_text())
        smart = (scaffold_dir is not None and "lesson:" in _read(pathlib.Path(scaffold_dir) / "lessons.md")) \
            or ("@" in model and not model.endswith("wp"))
        task_opt = _opt(_task_from_run(run))
        halted: set[str] = set()
        if "## Halted directions" in prompt:      # a real agent reads the halted list in its prompt
            line = prompt.split("## Halted directions", 1)[1].splitlines()[1]
            halted = {x.strip() for x in line.split(",")}
        options = [d for d in ["lr", "lr", "width", "wd"] if d not in halted] or ["lr"]
        direction = self.force_direction or rng.choice(options)
        kind = "exploratory" if rng.random() < self.explore_p else "change"
        if direction == "lr":
            params["lr"] = round(params["lr"] + (task_opt[0] - params["lr"]) * (0.9 if smart else rng.uniform(-0.1, 0.8)), 4)
        elif direction == "width":
            params["width"] = params["width"] + (1 if smart or rng.random() < 0.6 else -1)
        else:
            params["wd"] = round(rng.uniform(0, 1.0), 3)
        if rng.random() < self.crash_p:
            params["lr"] = 9.0
        if rng.random() < self.hack_p:
            params["cache"] = 1
        p.write_text(json.dumps(params, indent=1, sort_keys=True) + "\n")
        events.emit(run, "tool_call", agent, parent=parent, gen=gen, trial=trial, tool="edit",
                    arg_summary="params.json", ok=True, latency_s=0.0)
        sub = {"hypothesis": f"adjust {direction}", "kind": kind, "direction": direction,
               "claims_numerics_unchanged": bool(self.numerics_claim_dir == direction),
               "microbench": {"step_ms": rng.uniform(1, 2)}}
        return {"final": "done", "n_tool_calls": 2, "trajectory": "", "ok": True, "submitted": sub}


def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text()
    except OSError:
        return ""


def _task_from_run(run: str) -> str:
    for part in run.split("."):
        if part.startswith("toy-"):
            return part.replace("-", "/", 1)
    return "toy"


# test bitwise hook (referenced as "e_test_hooks:bitwise_fail")
_hooks = types.ModuleType("e_test_hooks")
_hooks.bitwise_fail = lambda parent, cand, steps=20: False
_hooks.calls = []
sys.modules["e_test_hooks"] = _hooks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _toy_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TOY_ROOT", str(tmp_path / "toy_tasks"))
    monkeypatch.setenv("RSI_RUNS_ROOT", str(tmp_path / "runs"))


def make_ctx(tmp_path, run, *, policy, harness, evald, budget, model="improver", task="toy", lanes=(1, 2),
             scaffold_dir=None) -> RunContext:
    from rsi.tasks.base import get_task

    ctx = RunContext.create(run=run, runs_root=tmp_path / "runs", gen=0, task=get_task(task), evald=evald,
                            harness=harness, model=model, scaffold_dir=scaffold_dir, budget=budget,
                            lanes=list(lanes))
    ev = ctx.run_dir / "events.jsonl"
    if not ev.exists():
        ctx.emit("run_start", "orchestrator", policy=policy, task=task,
                 improver={"model": model, "scaffold_hash": "none"},
                 budget={"trials": budget.get("trials"), "wall_s": budget.get("wall_s")}, config_hash="test")
    return ctx


def check_events(run_dir: pathlib.Path, required=REQUIRED_EVS, max_run_end: int = 1) -> list[dict]:
    evs = events.validate_file(str(run_dir / "events.jsonl"))
    kinds = {e["ev"] for e in evs}
    missing = required - kinds
    assert not missing, f"{run_dir.name}: missing events {missing}"
    assert sum(e["ev"] == "run_start" for e in evs) == 1
    assert sum(e["ev"] == "run_end" for e in evs) <= max_run_end   # one per completed run() call
    return evs


def trial_ids_unique(ctx) -> list[int]:
    ids = sorted(ctx.ledger.trials())
    assert ids == list(range(len(ids))), ids
    return ids


# ---------------------------------------------------------------------------
# unit: merge3
# ---------------------------------------------------------------------------

def test_merge3_clean_and_conflict():
    base = ["a\n", "b\n", "c\n", "d\n"]
    assert merge3(base, ["A\n", "b\n", "c\n", "d\n"], ["a\n", "b\n", "c\n", "D\n"]) == ["A\n", "b\n", "c\n", "D\n"]
    assert merge3(base, ["X\n", "b\n", "c\n", "d\n"], ["Y\n", "b\n", "c\n", "d\n"]) is None
    assert merge3(base, ["X\n"] + base[1:], ["X\n"] + base[1:]) == ["X\n"] + base[1:]


# ---------------------------------------------------------------------------
# naive_trial_ledger
# ---------------------------------------------------------------------------

def test_naive_end_to_end_stop_bitwise_and_resume(tmp_path):
    ev = FakeEval()
    h = FakeHarness(numerics_claim_dir="wd")
    cfg = {"spec": {"objective": "lower loss", "constraints": ["no new deps", {"field": "wall_s", "max": 100}],
                    "acceptance": {"seeds": [0, 1, 2], "min_delta": 0.001, "min_sigma": 2.0, "noise_sigma": 0.002}},
           "stop_k": 4, "bitwise": {"hook": "e_test_hooks:bitwise_fail", "steps": 5}}
    ctx = make_ctx(tmp_path, "naive-1", policy="naive_trial_ledger", harness=h, evald=ev, budget={"trials": 6})
    (ctx.run_dir / "STOP_DIRECTIONS").write_text("stop width\n")
    s1 = NaiveTrialLedger(cfg).run(ctx)
    assert s1["n_trials"] == 6
    # resume with a larger budget in a fresh context (new process)
    ctx2 = make_ctx(tmp_path, "naive-1", policy="naive_trial_ledger", harness=FakeHarness(numerics_claim_dir="wd"),
                    evald=ev, budget={"trials": 16})
    s2 = NaiveTrialLedger(cfg).run(ctx2)
    assert s2["n_trials"] == 16
    trial_ids_unique(ctx2)
    recs = list(ctx2.ledger.trials().values())
    st = [r["status"] for r in recs]
    assert st.count("baseline") == 1
    assert sum(s2["counts"].values()) == 16          # no-edit episodes do not use budget
    assert s2["counts"]["adopted"] >= 1
    # human stop: no width trial was evaluated
    assert all(r["status"] == "no_change" for r in recs if r.get("direction") == "width")
    # bitwise gate rolled back every "numerics unchanged" claim
    bw = [r for r in recs if r.get("claims_numerics_unchanged") and r["status"] not in ("no_change", "exploratory")]
    assert bw and all(r["status"] == "rolled_back" and r["bitwise"] is False for r in bw)
    # microbench recorded but never decisive
    assert any(r.get("microbench") for r in recs)
    # acceptance used the human's seeds
    assert all(r["seeds"] == [0, 1, 2] for r in recs if r["status"] in ("adopted", "rolled_back") and "seeds" in r)
    evs = check_events(ctx2.run_dir, max_run_end=2)       # finished, then budget extended and resumed
    assert evs[-1]["ev"] == "run_end" and evs[-1]["n_trials"] == 16
    verdicts = {e["verdict"] for e in evs if e["ev"] == "decision"}
    assert {"adopt", "rollback"} <= verdicts
    # best value monotone and equals the last adopted dev
    adopted = [r for r in sorted(recs, key=lambda r: r["trial"]) if r["status"] == "adopted"]
    assert s2["best_value"] == adopted[-1]["dev"]


def test_naive_auto_stop_queues_human_decision(tmp_path):
    ev = FakeEval()
    h = FakeHarness(force_direction="wd", explore_p=0.0, crash_p=0.0)
    ctx = make_ctx(tmp_path, "naive-stop", policy="naive_trial_ledger", harness=h, evald=ev, budget={"trials": 6})
    s = NaiveTrialLedger({"stop_k": 2, "spec": {"acceptance": {"seeds": [0, 1], "min_delta": 0.5}}}).run(ctx)
    assert "wd" in s["halted_directions"]
    assert "wd" in (ctx.run_dir / "decisions.md").read_text()
    recs = sorted(ctx.ledger.trials().values(), key=lambda r: r["trial"])
    assert [r["status"] for r in recs[1:3]] == ["rolled_back", "rolled_back"]
    assert all(r["status"] == "no_change" for r in recs[3:])
    # human resumes the direction through the stop file
    (ctx.run_dir / "STOP_DIRECTIONS").write_text("resume wd\n")
    ctx2 = make_ctx(tmp_path, "naive-stop", policy="naive_trial_ledger", harness=h, evald=ev, budget={"trials": 7})
    NaiveTrialLedger({"stop_k": 2, "spec": {"acceptance": {"seeds": [0, 1], "min_delta": 0.5}}}).run(ctx2)
    last = max(ctx2.ledger.trials().values(), key=lambda r: r["trial"])
    assert last["status"] == "rolled_back"
    check_events(ctx2.run_dir, max_run_end=2)


# ---------------------------------------------------------------------------
# recursive_first_steps
# ---------------------------------------------------------------------------

REC_CFG = {"n_threads": 3, "combine": {"every": 1, "mode": "auto"},
           "confirm": {"base_n": 2, "add": 1, "every": 1, "max_n": 5, "noise_sigma": 0.002, "z": 2.0},
           "sealed": {"require": True, "gap_max": 0.05}, "final_seeds": list(range(100, 104)),
           "l3": {"top_k": 1, "critic_model": None}}


def test_recursive_end_to_end_with_crash_resume(tmp_path):
    ev = FakeEval()
    ctx = make_ctx(tmp_path, "rec-1", policy="recursive_first_steps", harness=FakeHarness(die_after=14, hack_p=0.25),
                   evald=ev, budget={"trials": 14})
    with pytest.raises(Die):
        RecursiveFirstSteps(REC_CFG).run(ctx)
    ctx2 = make_ctx(tmp_path, "rec-1", policy="recursive_first_steps", harness=FakeHarness(hack_p=0.25),
                    evald=ev, budget={"trials": 14})
    s = RecursiveFirstSteps(REC_CFG).run(ctx2)
    assert s["n_trials"] == 14 and not s["paused"]
    trial_ids_unique(ctx2)
    recs = ctx2.ledger.records()
    thread_trials = [r for r in ctx2.ledger.trials().values() if r.get("kind") == "thread"]
    assert {r["thread"] for r in thread_trials} == {"thread-0", "thread-1", "thread-2"}
    assert any(r.get("reason", "").startswith("interrupted") for r in ctx2.ledger.trials().values())
    # combining happened, and confirmation seeds escalate with the number of global improvements
    assert any(r.get("kind") in ("combine", "combine_skip") for r in recs)
    # escalation is decided at confirmation time, so check it in decision order (threads run in parallel)
    confirm_lens = [len(r["confirm_seeds"]) for r in ctx2.ledger.records()
                    if r.get("confirm_seeds") and r.get("status") in ("adopted", "rejected")]
    assert confirm_lens and confirm_lens[0] == 2 and max(confirm_lens) > 2
    pol = RecursiveFirstSteps(REC_CFG)
    pol.cfg = effective_cfg(RecursiveFirstSteps.DEFAULTS, None, REC_CFG)
    assert [len(pol._confirm_seeds({"n_impr": n})) for n in range(6)] == [2, 3, 4, 5, 5, 5]
    # no adopted artifact uses the cache hack (sealed gap check caught it)
    adopted = [r for r in ctx2.ledger.trials().values() if r.get("status") == "adopted"]
    assert adopted
    for r in adopted:
        assert "cache" not in json.loads((ctx2.run_dir / "artifacts" / r["artifact_hash"] / "params.json").read_text())
    assert any("gap" in (r.get("reason") or "") for r in ctx2.ledger.trials().values())
    # shared notes exist, L3 queue written, final 4-seed eval recorded
    assert (ctx2.run_dir / "shared_notes.md").read_text().strip()
    assert (ctx2.run_dir / "audits" / "l3_queue.jsonl").exists()
    assert s["final"]["seeds"] == list(range(100, 104))
    evs = check_events(ctx2.run_dir)
    spawned = {e["agent"]: e for e in evs if e["ev"] == "agent_spawn"}
    assert {"thread-0", "thread-1/propose", "thread-2/implement", "thread-0/next"} <= set(spawned)
    assert spawned["thread-1/propose"]["parent"] == "thread-1"
    assert any(e["ev"] == "message" and e["agent"].endswith("/propose") for e in evs)


def test_recursive_l3_critic_rolls_back_hack(tmp_path):
    ev = FakeEval()
    cfg = dict(REC_CFG, sealed={"require": False, "gap_max": None}, n_threads=2,
               l3={"top_k": 3, "critic_model": "critic-model"}, combine={"every": 0})
    ctx = make_ctx(tmp_path, "rec-l3", policy="recursive_first_steps", harness=FakeHarness(hack_p=0.6, crash_p=0),
                   evald=ev, budget={"trials": 8})
    s = RecursiveFirstSteps(cfg).run(ctx)
    assert s["flagged"], "critic should flag a cache hack"
    assert s["rules_version"] > 1
    assert s["best_hash"] not in s["flagged"]
    assert (ctx.run_dir / "audits" / "rules.md").exists()
    evs = check_events(ctx.run_dir)
    assert any(e["ev"] == "decision" and e["verdict"] == "rollback" for e in evs)
    assert any(e["ev"] == "agent_spawn" and e["role"] == "critic" for e in evs)


# ---------------------------------------------------------------------------
# apex_ai4ai
# ---------------------------------------------------------------------------

def test_apex_swap_exit_selfevolve_and_resume(tmp_path):
    ev = FakeEval()
    cfg = {"researchers": [{"name": "small", "model": "improver"}, {"name": "broken", "model": "bad-model"}],
           "reps": 2, "explore": {"trials": 4}, "retrain": {"seeds": [100, 101], "budget_scale": 3.0},
           "exit_after_failed_episodes": 2, "self_evolving": {"mode": "after_all", "verify_seeds": [5, 6]}}
    ctx = make_ctx(tmp_path, "apex-1", policy="apex_ai4ai",
                   harness=FakeHarness(die_after=6, fail_models={"bad-model"}), evald=ev, budget={"trials": None})
    with pytest.raises(Die):
        ApexAI4AI(cfg).run(ctx)
    ctx2 = make_ctx(tmp_path, "apex-1", policy="apex_ai4ai", harness=FakeHarness(fail_models={"bad-model"}),
                    evald=ev, budget={"trials": None})
    s = ApexAI4AI(cfg).run(ctx2)
    trial_ids_unique(ctx2)
    res = s["results"]["researchers"]
    assert res["small"]["reps_done"] == 2 and res["small"]["completed"] == 2
    assert len(res["small"]["retrain_values"]) == 2
    assert res["broken"]["completed"] == 0 and res["broken"]["exits"] == [0, 1]
    # explore budget respected per unit (the interrupted trial counts)
    per_unit = {}
    for r in ctx2.ledger.trials().values():
        if r.get("unit"):
            per_unit[r["unit"]] = per_unit.get(r["unit"], 0) + 1
    assert per_unit["small/rep-0"] == 4 and per_unit["small/rep-1"] == 4
    # retrain forwarded the budget scale to evald
    assert any(c["kw"].get("budget_scale") == 3.0 and c["kw"].get("retrain") for c in ev.calls)
    # self-evolving data: notes for both researchers, harness failures classified
    se = [r for r in ctx2.ledger.records() if r.get("kind") == "selfevolve"]
    assert len(se) == 1 and se[0]["counts"].get("harness_failure", 0) >= 4
    assert (ctx2.run_dir / "apex" / "notes" / "broken.md").exists()
    for line in open(ctx2.run_dir / "apex" / "selfevolve_data.jsonl"):
        assert json.loads(line)["verified_by"] == "evaluator"
    evs = check_events(ctx2.run_dir)
    models = {e["agent"]: e["model"] for e in evs if e["ev"] == "agent_spawn"}
    assert models["small/rep-0"] == "improver" and models["broken/rep-1"] == "bad-model"


# ---------------------------------------------------------------------------
# self_evolution
# ---------------------------------------------------------------------------

LORA_STUB = """import os, sys
sft, out, gpu = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "adapter_config.json"), "w").write("{}")
print("training on gpu", gpu)
print(out)
"""

SE_INNER = {"spec": {"acceptance": {"seeds": [0], "min_delta": 0.0, "min_sigma": 0}}, "stop_k": 99}


def _se_cfg(tmp_path, executor: str, updates) -> dict:
    stub = tmp_path / "lora_stub.py"
    stub.write_text(LORA_STUB)
    return {"generations": 2, "inner_policy": "naive_trial_ledger", "inner_cfg": SE_INNER,
            "inner_budget": {"trials": 4}, "train_tasks": ["toy/t1"], "heldout_tasks": ["toy/h1", "toy/h2"],
            "train_seeds": [0], "paired_seeds": [0, 1], "aa_seeds": [50, 51], "updates": updates,
            "uw": {"lora_cmd": [sys.executable, str(stub)], "gpu": 2, "pause_lane": 2, "base_model": "improver"},
            "promote": {"n_boot": 300}, "executor": executor}


def test_self_evolution_inline_us_uw(tmp_path, stub_update):
    ev = FakeEval()
    h = FakeHarness(explore_p=0.0, crash_p=0.0)
    ctx = make_ctx(tmp_path, "se-1", policy="self_evolution", harness=h, evald=ev, budget={"trials": None})
    s = SelfEvolution(_se_cfg(tmp_path, "inline", ["U_s", "U_w"])).run(ctx)
    assert s["status"] == "done" and s["generation"] == 2
    assert len(s["delta2"]) == 2
    for d in s["delta2"]:
        assert {"cand_s", "cand_w"} <= set(d)
        assert isinstance(d["cand_s"]["delta2"], float)
    assert json.loads((ctx.run_dir / "self_evolution" / "delta2.json").read_text()) == s["delta2"]
    assert ev.paused == []                                   # lane paused for LoRA and resumed
    evs = check_events(ctx.run_dir, required={"proposal", "decision", "message", "agent_spawn",
                                              "lessons_update", "weights_update"})
    assert any(e["ev"] == "weights_update" and e["base_model"] == "improver" for e in evs)
    # every child arm run is a valid, replayable run with eval_submit events
    children = sorted(p for p in (tmp_path / "runs").iterdir() if p.name.startswith("se-1.g"))
    arms = {p.name.split(".")[2] for p in children}
    assert arms == {"train", "base", "aa", "cand_s", "placebo_s", "cand_w", "placebo_w"}
    for p in children:
        evs_c = check_events(p, required={"eval_submit", "proposal", "decision", "run_end"})
        assert sum(e["ev"] == "run_end" for e in evs_c) == 1
        assert json.loads((p / "arm_result.json").read_text())["s0"] is not None
    # paired seeds: base and candidates share seeds, A/A uses disjoint ones
    g0 = json.loads((ctx.run_dir / "self_evolution" / "gen_0" / "plan.json").read_text())
    ev_items = g0["stages"][2]["items"]
    assert {i["seed"] for i in ev_items if i["arm"] == "base"} == {i["seed"] for i in ev_items if i["arm"] == "cand_s"}
    assert not {i["seed"] for i in ev_items if i["arm"] == "aa"} & {i["seed"] for i in ev_items if i["arm"] == "base"}
    # a promoted improver is used in the next generation
    if s["delta2"][0]["promoted"]:
        g1 = json.loads((ctx.run_dir / "self_evolution" / "gen_1" / "plan.json").read_text())
        assert g1["improver"]["name"] == "I1"


def test_self_evolution_external_plan_for_campaign(tmp_path, stub_update):
    """executor=external: the policy only writes plans; a mini 'worker F' executes them."""
    import subprocess

    ev = FakeEval()
    h = FakeHarness(explore_p=0.0, crash_p=0.0)
    cfg = _se_cfg(tmp_path, "external", ["U_w"])
    cfg["generations"] = 1
    rounds = 0
    while True:
        ctx = make_ctx(tmp_path, "se-ext", policy="self_evolution", harness=h, evald=ev, budget={"trials": None})
        s = SelfEvolution(cfg).run(ctx)
        if s["status"] == "done":
            break
        assert s["status"] == "waiting" and s["plan"]
        plan = json.loads(pathlib.Path(s["plan"]).read_text())
        did = 0
        for stage in plan["stages"]:
            for item in stage["items"]:
                if pathlib.Path(item["done_file"]).exists():
                    continue
                if item["kind"] == "arm":
                    assert run_arm(item, evald=ev, harness=h) is not None
                    did += 1
                elif item["kind"] == "lora_train" and pathlib.Path(item["sft_dir"]).exists():
                    out = subprocess.run(item["cmd"], capture_output=True, text=True, check=True).stdout
                    pathlib.Path(item["done_file"]).write_text(json.dumps({"adapter_path": out.split()[-1]}))
                    did += 1
            if did:
                break
        assert did, "worker F found nothing to do while the policy is waiting"
        rounds += 1
        assert rounds < 10
    assert rounds >= 2 and len(s["delta2"]) == 1
    check_events(ctx.run_dir, required={"decision", "agent_spawn", "weights_update", "proposal"})
