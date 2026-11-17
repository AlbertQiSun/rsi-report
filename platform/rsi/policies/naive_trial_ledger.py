"""``naive_trial_ledger``: NaiveAI's human-sets-the-spec / AI-runs-trials loop.

Source: NaiveAI, "Naive-N0.5-Flash: Building Frontier AI with AI", Case 1 NaiveRT
(research/reading/02_Naive_N0.5_导读.md, research/sources/naive_research.md NV-46..NV-66),
platform_build_plan.md §4.3.

Method as reproduced here
-------------------------
* The human writes the spec (``cfg["spec"]``): objective, constraints, acceptance threshold.
  The AI only runs trials against it.
* Each trial is one harness episode that either changes the artifact or is *exploratory*
  (profiling / ablation / reading logs). Exploratory trials are evaluated with
  ``acceptance=False`` and always reverted.
* **Acceptance is end-to-end only.** The decision uses the evaluator's metric on the
  acceptance seeds, paired against the incumbent on the same seeds. Any microbenchmark
  the agent reports is written to the ledger and never consulted.
* Optional **bitwise gate** for changes the agent claims leave numerics unchanged:
  ``cfg["bitwise"]["hook"] = "pkg.mod:func"`` (``func(parent_dir, cand_dir, steps=N) -> bool``),
  or ``evald.bitwise_check(...)`` if the evaluator provides one. A failed gate rolls back.
* Three-way trial ledger: ``adopted`` / ``rolled_back`` / ``exploratory`` (NaiveRT's 63 / 71 / 17).
* **Human stop**: a direction that hits ``stop_k`` consecutive rollbacks is paused and
  queued in ``decisions.md`` for the human. The human can also halt (or resume) a direction
  at any time through ``cfg["stopped_directions"]`` or the stop file
  (``<run_dir>/STOP_DIRECTIONS``; lines ``stop <dir>``, ``resume <dir>``, ``<dir>`` or ``*``).

Not reproduced: NaiveRT's domain (CUDA mega-kernels, speculative decoding) and scale
(6 days, 151 trials); the bitwise gate's original use (logits/KV identical in an inference
engine) needs a task that exposes it; the human is simulated by a file, not a person.

This module also hosts ``PolicyKit``, the trial primitives shared by worker E's four
policies (workspace checkout, harness episode, evald submit, ledger replay, events).
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import json
import math
import os
import pathlib
import shutil
import statistics
import threading
import time
from typing import Any, Callable

from rsi.policies.base import Policy
from rsi.runs import atomic_write_json, diff_stat, snapshot_files, stable_hash, unified_diff

ORCH = "orchestrator"
OK = "ok"


# ===========================================================================
# Shared policy kit (worker E). Policy logic decides order and verdicts only.
# ===========================================================================

def deep_merge(base: dict, over: dict | None) -> dict:
    out = json.loads(json.dumps(base, default=str))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def effective_cfg(defaults: dict, ctx_cfg: dict | None, init_cfg: dict | None) -> dict:
    return deep_merge(deep_merge(defaults, ctx_cfg or {}), init_cfg or {})


def load_callable(spec: str) -> Callable:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def load_policy(name: str) -> type:
    """Find the Policy class for ``name`` in ``rsi.policies.<name>``."""
    mod = importlib.import_module(f"rsi.policies.{name}")
    cls = getattr(mod, "POLICY", None)
    if cls is not None:
        return cls
    for obj in vars(mod).values():
        if isinstance(obj, type) and issubclass(obj, Policy) and obj is not Policy \
                and getattr(obj, "name", None) == name:
            return obj
    raise KeyError(f"no Policy named {name!r} in rsi.policies.{name}")


def parse_submission(ep: dict) -> dict:
    """Agent's submission: harness ``submitted`` payload, else last JSON object in ``final``."""
    sub = ep.get("submitted")
    if isinstance(sub, dict):
        out = dict(sub)
    else:
        out = {}
        final = ep.get("final") or ""
        end = final.rfind("}")
        while end != -1 and not out:
            start = final.rfind("{", 0, end)
            while start != -1:
                try:
                    obj = json.loads(final[start:end + 1])
                    if isinstance(obj, dict):
                        out = obj
                        break
                except ValueError:
                    pass
                start = final.rfind("{", 0, start)
            end = final.rfind("}", 0, end)
    if not out.get("hypothesis"):
        out["hypothesis"] = (ep.get("final") or "").strip()[:300] or "(no hypothesis given)"
    return out


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def merge3(base: list[str], a: list[str], b: list[str]) -> list[str] | None:
    """Line-level three-way merge. Returns None on a conflict."""
    def changes(other: list[str]) -> list[tuple[int, int, list[str]]]:
        sm = difflib.SequenceMatcher(a=base, b=other, autojunk=False)
        return [(i1, i2, other[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]

    tagged = sorted([(c, "a") for c in changes(a)] + [(c, "b") for c in changes(b)],
                    key=lambda x: (x[0][0], x[0][1]))
    kept: list[tuple[tuple[int, int, list[str]], str]] = []
    for c, src in tagged:
        clash = None
        for pc, psrc in kept:
            if psrc == src:
                continue
            (p1, p2, prep), (i1, i2, rep) = pc, c
            if (i1 < p2 and p1 < i2) or i1 == p1 or (i1 == p2 and (i1 == i2 or p1 == p2)) \
                    or (p1 == i2 and (p1 == p2 or i1 == i2)):
                clash = pc
                break
        if clash is not None:
            if clash == c:
                continue                      # both sides made the identical change
            return None
        kept.append((c, src))
    out: list[str] = []
    pos = 0
    for (i1, i2, rep), _ in sorted(kept, key=lambda x: (x[0][0], x[0][1])):
        out += base[pos:i1] + rep
        pos = max(pos, i2)
    return out + base[pos:]


class PolicyKit:
    """Trial primitives on top of RunContext. Thread-safe for parallel research threads."""

    def __init__(self, ctx: Any, policy: str, cfg: dict):
        self.ctx = ctx
        self.policy = policy
        self.cfg = cfg
        self.task = ctx.task
        self.lock = threading.RLock()
        recs = ctx.ledger.records()
        ids = [r["trial"] for r in recs if isinstance(r.get("trial"), int)]
        self._next_trial = max(ids, default=-1) + 1
        self.spawned = {r["agent"] for r in recs if r.get("kind") == "spawn"}
        self.ref_cache: dict[tuple[str, tuple[int, ...]], dict] = {}
        for r in recs:
            if r.get("kind") == "ref" and r.get("value") is not None:
                self.ref_cache[(r["artifact_hash"], tuple(r["seeds"]))] = r
        self._submit_params: set[str] | None = None
        self._submit_varkw = False

    # ---- ledger ---------------------------------------------------------
    def records(self) -> list[dict]:
        return [r for r in self.ctx.ledger.records() if r.get("policy") in (None, self.policy)]

    def append(self, **rec: Any) -> dict:
        rec.setdefault("policy", self.policy)
        return self.ctx.ledger.append(**rec)

    TAGS = ("kind", "thread", "unit", "round", "researcher", "rep")

    def new_trial(self, **tags: Any) -> int:
        """Allocate a trial id and record it as ``pending`` (closed as a crash if the process dies)."""
        with self.lock:
            t = self._next_trial
            self._next_trial += 1
        self.append(trial=t, status="pending", stage="started", **tags)
        return t

    def finished_trials(self) -> list[dict]:
        """Trials that count against ``budget.trials`` (same rule as Policy.n_started: no baseline, no no-edit)."""
        return [r for r in self.ctx.ledger.finished() if r.get("status") not in ("baseline", "no_change")]

    def trials_done(self) -> int:
        return len(self.finished_trials())

    def out_of_budget(self, extra: int = 0) -> bool:
        return self.ctx.out_of_budget(self.trials_done() + extra)

    def recover_pending(self) -> int:
        """Trials interrupted between submit and decision are closed as crashes."""
        n = 0
        for t, r in self.ctx.ledger.trials().items():
            if r.get("status") == "pending":
                self.append(trial=t, status="crash", reason="interrupted before decision (resume)",
                            **{k: r[k] for k in self.TAGS if k in r})
                n += 1
        return n

    # ---- artifacts / workspaces ----------------------------------------
    @property
    def editable(self) -> tuple[str, ...]:
        return tuple(getattr(self.task, "editable_files", ("train.py",)))

    def start_files(self, start_dir: str | os.PathLike | None = None) -> dict[str, bytes]:
        d = pathlib.Path(start_dir) if start_dir else pathlib.Path(self.task.start_artifact())
        files = snapshot_files(str(d), self.editable)
        if not files:
            raise FileNotFoundError(f"no editable files {self.editable} under {d}")
        return files

    def put(self, files: dict[str, bytes]) -> str:
        return self.ctx.artifacts.put(files)

    def files(self, h: str) -> dict[str, bytes]:
        return self.ctx.artifacts.get(h)

    def art_dir(self, h: str) -> str:
        return self.ctx.artifacts.dir(h)

    def ws(self, tid: str, sub: str = "workspace") -> pathlib.Path:
        d = pathlib.Path(self.ctx.rd.thread_dir(tid, sub))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def checkout(self, h: str, ws: pathlib.Path) -> None:
        for n in self.editable:
            fp = ws / n
            if fp.is_file():
                fp.unlink()
        self.ctx.artifacts.checkout(h, str(ws))

    def scratch(self, tid: str, h: str, name: str = "scratch") -> pathlib.Path:
        d = pathlib.Path(self.ctx.rd.thread_dir(tid, name))
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        self.ctx.artifacts.checkout(h, str(d))
        return d

    def snap(self, ws: pathlib.Path) -> tuple[str, dict[str, bytes]]:
        files = snapshot_files(str(ws), self.editable)
        return self.put(files), files

    def diff_text(self, a: str, b: str, max_lines: int = 200) -> str:
        return unified_diff(self.files(a), self.files(b), max_lines=max_lines)

    # ---- events -----------------------------------------------------------
    def emit(self, ev: str, agent: str, **kw: Any) -> dict:
        return self.ctx.emit(ev, agent, **kw)

    def spawn(self, agent: str, role: str, model: str | None = None, parent: str | None = ORCH) -> None:
        with self.lock:
            if agent in self.spawned:
                return
            self.spawned.add(agent)
        self.emit("agent_spawn", agent, parent=parent, role=role, model=model or self.ctx.model)
        self.append(kind="spawn", agent=agent, role=role)

    def end(self, agent: str, reason: str, parent: str | None = ORCH) -> None:
        self.emit("agent_end", agent, parent=parent, reason=reason)

    def message(self, frm: str, to: str, text: str, *, parent: str | None = None,
                trial: int | None = None) -> None:
        self.emit("message", frm, parent=parent, trial=trial, to=to, summary=text,
                  tokens=max(1, len(text) // 4))

    def proposal(self, agent: str, trial: int, hypothesis: str, parent_hash: str, h: str,
                 parent: str | None = None) -> dict:
        ds = diff_stat(self.files(parent_hash), self.files(h))
        self.emit("proposal", agent, parent=parent, trial=trial, hypothesis=hypothesis,
                  diff_stat=ds, artifact_hash=h)
        return ds

    def decision(self, trial: int | None, h: str, verdict: str, before: Any, after: Any, reason: str,
                 agent: str = ORCH) -> None:
        self.emit("decision", agent, trial=trial, artifact_hash=h, verdict=verdict,
                  best_before=before, best_after=after, reason=reason[:300])

    # ---- harness ------------------------------------------------------------
    def episode(self, agent: str, trial: int, ws: pathlib.Path, prompt: str, *,
                parent: str | None = ORCH, model: str | None = None,
                scaffold_dir: pathlib.Path | str | None = None, budget_s: int | None = None) -> dict:
        b = budget_s or int((self.ctx.budget or {}).get("episode_s") or self.cfg.get("episode_s") or 900)
        sd = scaffold_dir if scaffold_dir is not None else self.ctx.scaffold_dir
        kw = dict(run=self.ctx.run, agent=agent, gen=self.ctx.gen, trial=trial, workspace=pathlib.Path(ws),
                  prompt=prompt, model=model or self.ctx.model,
                  scaffold_dir=pathlib.Path(sd) if sd else None, budget_s=b)
        try:
            try:
                ep = self.ctx.harness.run_episode(parent=parent, **kw)
            except TypeError as e:          # harness without the optional ``parent`` kwarg
                if "parent" not in str(e):
                    raise
                ep = self.ctx.harness.run_episode(**kw)
        except Exception as e:  # harness crash is a trial outcome, not a policy crash
            ep = {"final": f"harness error: {type(e).__name__}: {e}", "n_tool_calls": 0,
                  "trajectory": "", "ok": False, "submitted": None}
        ep.setdefault("ok", True)
        return ep

    # ---- evaluation -----------------------------------------------------------
    def _submit_kw(self, extra: dict) -> dict:
        if self._submit_params is None:
            try:
                sig = inspect.signature(self.ctx.evald.submit)
                self._submit_params = set(sig.parameters)
                self._submit_varkw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
            except (TypeError, ValueError):
                self._submit_params, self._submit_varkw = set(), False
        return {k: v for k, v in extra.items()
                if v is not None and (self._submit_varkw or k in self._submit_params)}

    def lane_for(self, i: int = 0) -> int | None:
        lanes = list(getattr(self.ctx, "lanes", None) or [])
        return lanes[i % len(lanes)] if lanes else None

    def evaluate(self, h: str, *, trial: int | None, seeds: list[int], agent: str = ORCH,
                 level: str = "L1", lane: int | None = None, record_pending: bool = True,
                 thread: str | None = None, **extra: Any) -> dict:
        """Submit artifact ``h`` to evald; emits ``eval_submit``; evald emits ``eval_result``."""
        seeds = [int(s) for s in seeds]
        self.emit("eval_submit", agent, trial=trial, artifact_hash=h, lane=lane, seeds=seeds)
        kw = self._submit_kw({"level": level, **extra})
        t0 = time.time()
        job = None
        try:
            job = self.ctx.evald.submit(self.ctx.run, self.task.name, self.art_dir(h), seeds=seeds,
                                        lane=lane, gen=self.ctx.gen, trial=trial, **kw)
            if trial is not None and record_pending:
                self.append(trial=trial, status="pending", artifact_hash=h, job_id=job, thread=thread)
            res = dict(self.ctx.evald.result(job, wait=True,
                                             timeout_s=float(self.cfg.get("eval_timeout_s", 3600))))
        except Exception as e:
            res = {"status": "crash", "value": None, "error": f"{type(e).__name__}: {e}"[:300]}
        res.setdefault("status", "crash")
        res.setdefault("audit", {"level": level, "flags": []})
        res.setdefault("seeds", seeds)
        res.setdefault("wall_s", round(time.time() - t0, 3))
        v = res.get("value")
        if not isinstance(v, (int, float)) or math.isnan(v):
            res["value"] = None
            if res["status"] == OK:
                res["status"] = "nan"
        res["job_id"] = job
        res["artifact_hash"] = h
        res["sealed"] = self.sealed_of(res, h)
        return res

    def sealed_of(self, res: dict, h: str) -> float | None:
        """Sealed score for this eval, read by the trusted orchestrator only (never shown to agents)."""
        for k in ("sealed", "sealed_value"):
            v = res.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        p = pathlib.Path(self.ctx.rd.sealed_path)
        if not p.exists():
            return None
        best = None
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if res.get("job_id") is not None and r.get("job_id") not in (None, res.get("job_id")):
                continue
            if r.get("artifact_hash") not in (None, h):
                continue
            v = r.get("sealed", r.get("value"))
            if isinstance(v, (int, float)):
                best = float(v)
        return best

    def ref_eval(self, h: str, seeds: list[int], *, agent: str = ORCH, level: str = "L2",
                 lane: int | None = None) -> dict:
        """Incumbent's score on exactly these seeds (paired comparison); cached in the ledger."""
        key = (h, tuple(int(s) for s in seeds))
        with self.lock:
            if key in self.ref_cache:
                return self.ref_cache[key]
        r = self.evaluate(h, trial=None, seeds=list(key[1]), agent=agent, level=level, lane=lane)
        rec = {"kind": "ref", "artifact_hash": h, "seeds": list(key[1]), "value": r["value"],
               "sealed": r["sealed"], "status": r["status"], "values": r.get("values")}
        if r["status"] == OK:
            self.append(**rec)
            with self.lock:
                self.ref_cache[key] = rec
        return rec

    # ---- comparisons ------------------------------------------------------------
    def gain(self, new: float | None, old: float | None) -> float | None:
        """Signed improvement (positive = better), in metric units."""
        if new is None or old is None:
            return None
        return (old - new) if self.task.lower_is_better else (new - old)

    def better(self, a: float | None, b: float | None, margin: float = 0.0) -> bool:
        g = self.gain(a, b)
        if b is None and a is not None:
            return True
        return g is not None and g > margin

    @staticmethod
    def per_seed_sigma(res: dict) -> float | None:
        vals = res.get("values") or res.get("per_seed")
        if isinstance(vals, dict):
            vals = list(vals.values())
        vals = [v for v in (vals or []) if isinstance(v, (int, float))]
        return statistics.stdev(vals) if len(vals) >= 2 else None

    # ---- state cache (convenience only; the ledger is the source of truth) ------
    def save_cache(self, **state: Any) -> None:
        with self.lock:
            self.ctx.save_state(**{self.policy: state})


def baseline(kit: PolicyKit, start_hash: str, seeds: list[int], *, lane: int | None = None,
             thread: str | None = ORCH) -> dict:
    """Trial with status ``baseline``: score of the start artifact (s0 of the curve)."""
    for r in kit.ctx.ledger.trials().values():
        if r.get("status") == "baseline" and r.get("policy") == kit.policy and r.get("thread") == thread:
            return r
    t = kit.new_trial()
    res = kit.evaluate(start_hash, trial=t, seeds=seeds, level="L1", lane=lane, thread=thread)
    if res["status"] != OK:
        raise RuntimeError(f"baseline evaluation failed: {res.get('status')} {res.get('error', '')}")
    return kit.append(trial=t, status="baseline", artifact_hash=start_hash, dev=res["value"],
                      sealed=res["sealed"], seeds=seeds, incumbent=res["value"], thread=thread)


# ===========================================================================
# naive_trial_ledger
# ===========================================================================

PLAN_PROMPT = """You are an AI researcher executing trials against a HUMAN-WRITTEN spec.

## Objective (set by the human)
{objective}

## Constraints (set by the human)
{constraints}

## Acceptance (set by the human; you cannot change it)
A change is ADOPTED only if the END-TO-END metric `{metric}` on seeds {seeds} beats the incumbent
by at least {threshold}. Microbenchmarks are recorded but never count toward acceptance.
{bitwise_note}

## Task
{task_card}

## Halted directions (the human stopped these; do not pursue them)
{stopped}

## Trial ledger (most recent last)
{ledger}

Edit the artifact in your workspace for ONE trial, then submit with fields:
  hypothesis (<=300 chars), kind ("change" or "exploratory"), direction (short tag),
  claims_numerics_unchanged (bool), microbench (optional dict, informational only).
Use kind="exploratory" for profiling / ablation / information-gathering runs; they are never adopted.
"""


class NaiveTrialLedger(Policy):
    name = "naive_trial_ledger"

    DEFAULTS: dict = {
        "spec": {
            "objective": "Reduce the end-to-end task metric.",
            "constraints": [],          # str (shown to the agent) or {"field","max"|"min","text"} (checked)
            "acceptance": {"seeds": [0, 1, 2], "min_delta": 0.0, "min_sigma": 2.0, "noise_sigma": None},
        },
        "screen_seeds": None,           # exploratory evals; default task.dev_seeds
        "stop_k": 3,                    # consecutive rollbacks in one direction -> pause it, ask the human
        "stop_file": None,              # default <run_dir>/STOP_DIRECTIONS
        "stopped_directions": [],
        "bitwise": {"hook": None, "steps": 20, "on_unavailable": "skip"},   # or "rollback"
        "agent": "researcher-0",
        "start_dir": None,
        "ledger_tail": 12,
        "max_consecutive_no_change": 6,  # no-edit episodes don't use budget; stop if the agent stalls
    }

    def __init__(self, cfg: dict | None = None):
        try:
            super().__init__(cfg or {})
        except TypeError:
            super().__init__()
        self.raw_cfg = dict(cfg or {})

    # ---- human stop ---------------------------------------------------------------
    def _stop_file(self, ctx) -> pathlib.Path:
        return pathlib.Path(self.cfg.get("stop_file") or (ctx.run_dir / "STOP_DIRECTIONS"))

    def _human_stops(self, ctx) -> tuple[set[str], set[str], bool]:
        stop, resume, halt_all = set(self.cfg.get("stopped_directions") or []), set(), False
        p = self._stop_file(ctx)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line == "*":
                    halt_all = True
                elif line.startswith("resume "):
                    resume.add(line[7:].strip())
                    stop.discard(line[7:].strip())
                else:
                    d = line[5:].strip() if line.startswith("stop ") else line
                    stop.add(d)
                    resume.discard(d)
        return stop, resume, halt_all

    # ---- ledger replay ----------------------------------------------------------------
    def _replay(self, kit: PolicyKit, start_hash: str) -> dict:
        st = {"best_hash": start_hash, "best_value": None, "streak": {}, "auto_stopped": set(),
              "counts": {"adopted": 0, "rolled_back": 0, "exploratory": 0}}
        for r in kit.records():
            if r.get("kind") == "direction_stop":
                st["auto_stopped"].add(r["direction"])
            if r.get("trial") is None:
                continue
            s = r.get("status")
            if s == "baseline":
                st["best_value"] = r.get("dev")
            if s in st["counts"]:
                st["counts"][s] += 1
            if s == "adopted":
                st["best_hash"], st["best_value"] = r["artifact_hash"], r.get("dev")
            d = r.get("direction")
            if d and s == "adopted":
                st["streak"][d] = 0
            elif d and s == "rolled_back":
                st["streak"][d] = st["streak"].get(d, 0) + 1
        return st

    # ---- acceptance -------------------------------------------------------------------
    def _threshold(self, res: dict) -> float:
        acc = self.cfg["spec"]["acceptance"]
        sigma = acc.get("noise_sigma")
        if sigma is None:
            sigma = PolicyKit.per_seed_sigma(res)
        thr = float(acc.get("min_delta") or 0.0)
        if sigma is not None and acc.get("min_sigma"):
            thr = max(thr, float(acc["min_sigma"]) * float(sigma))
        return thr

    def _constraint_violation(self, res: dict) -> str | None:
        for c in self.cfg["spec"].get("constraints") or []:
            if not isinstance(c, dict) or "field" not in c:
                continue
            v = res.get(c["field"])
            if not isinstance(v, (int, float)):
                continue
            if "max" in c and v > c["max"]:
                return f"constraint {c['field']}={v} > {c['max']}"
            if "min" in c and v < c["min"]:
                return f"constraint {c['field']}={v} < {c['min']}"
        return None

    def _bitwise(self, kit: PolicyKit, parent_hash: str, h: str) -> tuple[bool | None, str]:
        bw = self.cfg.get("bitwise") or {}
        steps = int(bw.get("steps") or 20)
        fn = None
        if bw.get("hook"):
            fn = load_callable(bw["hook"])
        elif hasattr(kit.ctx.evald, "bitwise_check"):
            fn = kit.ctx.evald.bitwise_check
        if fn is None:
            return None, "bitwise gate unavailable"
        try:
            ok = bool(fn(kit.art_dir(parent_hash), kit.art_dir(h), steps=steps))
        except Exception as e:
            return False, f"bitwise check error: {type(e).__name__}"
        return ok, f"bitwise over {steps} deterministic steps: {'identical' if ok else 'DIFFERENT'}"

    # ---- main loop -------------------------------------------------------------------
    def run(self, ctx) -> dict:
        self.cfg = effective_cfg(self.DEFAULTS, getattr(ctx, "policy_cfg", None), self.raw_cfg)
        cfg = self.cfg
        kit = PolicyKit(ctx, self.name, cfg)
        kit.recover_pending()
        agent = cfg["agent"]
        acc = cfg["spec"]["acceptance"]
        acc_seeds = list(acc.get("seeds") or ctx.task.dev_seeds)
        screen = list(cfg.get("screen_seeds") or ctx.task.dev_seeds)
        lane = kit.lane_for(0)
        start_hash = kit.put(kit.start_files(cfg.get("start_dir")))
        base = baseline(kit, start_hash, acc_seeds, lane=lane)
        kit.spawn(agent, "researcher")
        ws = kit.ws(agent)
        st = self._replay(kit, start_hash)
        if st["best_value"] is None:
            st["best_value"] = base.get("dev")
        paused = False
        last_feedback = ""
        idle = 0

        while True:
            if ctx.paused():
                paused = True
                break
            if kit.out_of_budget():
                break
            if idle >= int(cfg["max_consecutive_no_change"]):
                kit.append(kind="stalled", consecutive_no_change=idle)
                break
            n_before = kit.trials_done()
            stop, resume, halt_all = self._human_stops(ctx)
            idle = 0 if getattr(self, "_last_done", None) is None or n_before != self._last_done else idle + 1
            self._last_done = n_before
            if halt_all:
                break
            halted = (st["auto_stopped"] - resume) | stop
            t = kit.new_trial()
            kit.checkout(st["best_hash"], ws)
            thr_note = (f"max({acc.get('min_delta', 0)}, {acc.get('min_sigma')}*sigma)"
                        if acc.get("min_sigma") else str(acc.get("min_delta", 0)))
            prompt = PLAN_PROMPT.format(
                objective=cfg["spec"]["objective"],
                constraints="\n".join(f"- {c if isinstance(c, str) else c.get('text', json.dumps(c))}"
                                      for c in cfg["spec"].get("constraints") or []) or "- (none)",
                metric=ctx.task.metric, seeds=acc_seeds, threshold=thr_note,
                bitwise_note=("If you claim numerics are unchanged, a bitwise gate must pass "
                              "(identical losses over the first steps, deterministic mode).")
                if (cfg.get("bitwise") or {}).get("hook") or hasattr(ctx.evald, "bitwise_check") else "",
                task_card=ctx.task.task_card(), stopped=", ".join(sorted(halted)) or "(none)",
                ledger=ctx.ledger.tail_text(None, int(cfg["ledger_tail"])) + (f"\nLast: {last_feedback}" if last_feedback else ""))
            kit.message(ORCH, agent, f"trial {t}: run one trial against the human spec; "
                        f"halted={sorted(halted) or 'none'}", trial=t)
            ep = kit.episode(agent, t, ws, prompt)
            sub = parse_submission(ep)
            h, _ = kit.snap(ws)
            direction = str(sub.get("direction") or "general")
            kind = "exploratory" if str(sub.get("kind", "change")).lower().startswith("explor") else "change"
            common = dict(trial=t, artifact_hash=h, parent_hash=st["best_hash"], direction=direction,
                          kind=kind, hypothesis=sub["hypothesis"][:300], microbench=sub.get("microbench"),
                          claims_numerics_unchanged=bool(sub.get("claims_numerics_unchanged")),
                          harness_ok=bool(ep.get("ok")), n_tool_calls=ep.get("n_tool_calls"))
            before = st["best_value"]

            if direction in halted:
                kit.append(status="no_change", reason=f"direction {direction!r} halted by human", **common)
                kit.decision(t, h, "reject", before, before, f"direction {direction} halted")
                last_feedback = f"trial {t}: direction {direction} is halted; pick another"
                continue
            if h == st["best_hash"] and kind == "change":
                kit.append(status="no_change", reason="no edit to editable files", **common)
                kit.decision(t, h, "reject", before, before, "no change")
                last_feedback = f"trial {t}: no edit was made"
                continue
            kit.proposal(agent, t, sub["hypothesis"], st["best_hash"], h, parent=ORCH)

            if kind == "exploratory":
                r = kit.evaluate(h, trial=t, seeds=screen, agent=ORCH, level="L1", lane=lane, acceptance=False)
                kit.append(status="exploratory", dev=r["value"], eval_status=r["status"], seeds=screen,
                           audit_flags=r["audit"].get("flags", []), incumbent=before, **common)
                kit.decision(t, h, "explore", before, before, "exploratory trial: information only, reverted")
                st["counts"]["exploratory"] += 1
                last_feedback = f"trial {t} (exploratory): {ctx.task.visible_feedback(r)}"
                continue

            r = kit.evaluate(h, trial=t, seeds=acc_seeds, agent=ORCH, level="L2", lane=lane)
            verdict, reason, bitwise, gate_ok = "rollback", "", None, True
            if r["status"] != OK:
                reason, gate_ok = f"eval status {r['status']}", False
            elif common["claims_numerics_unchanged"]:
                bw_ok, reason = self._bitwise(kit, st["best_hash"], h)
                if bw_ok is None:
                    bitwise = "unavailable"
                    gate_ok = (cfg.get("bitwise") or {}).get("on_unavailable") != "rollback"
                else:
                    bitwise, gate_ok = bw_ok, bw_ok
            if gate_ok:
                viol = self._constraint_violation(r)
                if viol:
                    reason = viol
                else:
                    # paired: the incumbent's value was measured on the same acceptance seeds
                    g = kit.gain(r["value"], before)
                    thr = self._threshold(r)
                    if g is not None and g > 0 and g >= thr:
                        verdict, reason = "adopt", f"end-to-end gain {g:.6g} >= threshold {thr:.6g}"
                    else:
                        reason = f"end-to-end gain {g if g is None else round(g, 6)} < threshold {thr:.6g}"
            status = "adopted" if verdict == "adopt" else "rolled_back"
            st["counts"][status] += 1
            if verdict == "adopt":
                st["best_hash"], st["best_value"] = h, r["value"]
                st["streak"][direction] = 0
            else:
                st["streak"][direction] = st["streak"].get(direction, 0) + 1
            kit.append(status=status, dev=r["value"], sealed=r["sealed"], eval_status=r["status"],
                       seeds=acc_seeds, audit_flags=r["audit"].get("flags", []), bitwise=bitwise,
                       reason=reason, incumbent=st["best_value"], **common)
            kit.decision(t, h, verdict, before, st["best_value"], reason)
            last_feedback = f"trial {t}: {status}; {ctx.task.visible_feedback(r)}; {reason}"

            if verdict != "adopt" and st["streak"][direction] >= int(cfg["stop_k"]) \
                    and direction not in st["auto_stopped"]:
                st["auto_stopped"].add(direction)
                kit.append(kind="direction_stop", direction=direction, after_trial=t,
                           consecutive_rollbacks=st["streak"][direction])
                with open(ctx.run_dir / "decisions.md", "a") as f:
                    f.write(f"- [{time.strftime('%Y-%m-%d %H:%M:%S')}] run {ctx.run}: direction "
                            f"`{direction}` paused after {st['streak'][direction]} consecutive rollbacks "
                            f"(trial {t}). Write `resume {direction}` to {self._stop_file(ctx)} to resume, "
                            f"or `stop {direction}` to confirm.\n")
                kit.message(ORCH, "human", f"direction {direction} paused after "
                            f"{st['streak'][direction]} consecutive rollbacks; awaiting decision", trial=t)

        if not paused:
            kit.end(agent, "budget" if kit.out_of_budget() else "human_stop")
        summary = {"policy": self.name, "best_value": st["best_value"], "best_hash": st["best_hash"],
                   "n_trials": kit.trials_done(), "counts": st["counts"], "paused": paused,
                   "halted_directions": sorted(st["auto_stopped"] | set(self._human_stops(ctx)[0]))}
        kit.save_cache(**summary)
        return summary


POLICY = NaiveTrialLedger
