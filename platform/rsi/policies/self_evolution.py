"""``self_evolution``: the outer loop ("heart swap") that measures second-order improvement Δ₂.

Source: platform_build_plan.md §4.5 and platform_design.md §1 (Δ₂, A/A, placebo, promotion rule).
None of the four write-ups shows this loop (Flint states it as a bet, Apex claims compounding
without per-generation data, Naive only defines it), so this policy is the design doc's
second-order measurement, **not a reproduction of anyone's method**.

Per generation k (resumable step by step from the ledger):

1. **G_k**: run ``inner_policy`` with improver I_k on ``train_tasks`` x ``train_seeds`` -> D_k (child runs).
2. **I_{k+1} = U(I_k, D_k)**:
   * ``U_s`` (scaffold): ``rsi.update.write_lessons(D_k) -> scaffold_dir``; placebo: ``generic_placebo=True``.
   * ``U_w`` (weights): ``rsi.update.build_sft_data(D, accepted_only=True)`` + ``bundle/lora_train.sh``
     -> served as ``"improver@<adapter>"``; placebo: ``build_sft_data(placebo=True)``.
3. **Arms on held-out tasks** with paired seeds: ``base`` (I_k), ``cand_<u>`` (I_{k+1}),
   ``placebo_<u>``, and ``aa`` (I_k again on disjoint seeds = the noise floor).
4. **Δ₂(k)** = mean over (task, seed) pairs of nAUC(cand) - nAUC(base), with a bootstrap CI and
   a sign-flip p-value; promotion only if CI_low > 0, Δ₂ > aa_factor x sd(A/A diffs),
   cand beats placebo, and the hack rate does not rise. Promoted -> next generation's improver.

Output: the Δ₂ sequence (``summary["delta2"]``, ``<run_dir>/self_evolution/delta2.json``).

Scheduling. ``executor="inline"`` runs everything in-process (CPU tasks such as µAR, tests).
``executor="external"`` writes ``self_evolution/gen_<k>/plan.json`` (a DAG of arms and LoRA jobs
with ``done_file`` paths) for worker F's campaign scheduler; F executes arm specs with
``run_arm(spec, evald=..., harness=...)`` and LoRA jobs with the listed ``cmd``, then calls
``run(ctx)`` again (resume). Until the files exist, ``run`` returns ``status="waiting"``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import subprocess
from typing import Any

from rsi.ledger import nauc
from rsi.policies.base import Policy
from rsi.policies.naive_trial_ledger import ORCH, PolicyKit, effective_cfg, load_callable, load_policy
from rsi.runs import atomic_write_json, diff_stat, stable_hash

PLATFORM_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-") or "x"


def dir_files(d: str | os.PathLike | None) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if not d or not pathlib.Path(d).is_dir():
        return out
    for p in sorted(pathlib.Path(d).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = p.read_bytes()
    return out


def dir_hash(d: str | os.PathLike | None) -> str:
    h = hashlib.sha256()
    for name, data in dir_files(d).items():
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()[:12] if d else "none"


def update_fn(name: str):
    """``rsi.update.<name>`` (INTERFACES.md), falling back to D's submodules if not re-exported."""
    import importlib

    for mod in ("rsi.update", "rsi.update.lessons", "rsi.update.sft", "rsi.update.us", "rsi.update.uw"):
        try:
            fn = getattr(importlib.import_module(mod), name, None)
        except ImportError:
            fn = None
        if fn is not None:
            return fn
    raise ImportError(f"rsi.update.{name} is not available (worker D)")


def call_filtered(fn, *args, **kw):
    """Call ``fn`` dropping keyword args it does not accept (optional extras such as ``heldout``)."""
    import inspect

    try:
        ps = inspect.signature(fn).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in ps.values()):
            kw = {k: v for k, v in kw.items() if k in ps}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kw)


# ---------------------------------------------------------------------------
# One arm = one child run of the inner policy (also the unit worker F executes)
# ---------------------------------------------------------------------------

def ledger_curve(records: dict[int, dict]) -> dict:
    """s0 + per-trial incumbent values + hack stats from a child run's ledger (latest record per trial)."""
    s0, values, n_eval, n_flag = None, [], 0, 0
    for t in sorted(records):
        r = records[t]
        st = r.get("status")
        if st in ("pending", "no_change"):      # no-edit episodes do not use budget (Policy.n_started)
            continue
        if st == "baseline":
            s0 = r.get("dev") if s0 is None else s0
            continue
        if "incumbent" in r:
            v = r["incumbent"]
        else:
            v = r.get("dev") if st in ("keep", "adopted") else None
        values.append(v if isinstance(v, (int, float)) else None)
        if r.get("eval_status") is not None or isinstance(r.get("dev"), (int, float)):
            n_eval += 1
            if r.get("audit_flags") or r.get("eval_status") == "audit_fail":
                n_flag += 1
    if s0 is None:
        s0 = next((v for v in values if v is not None), None)
    return {"s0": s0, "values": values, "n_eval": n_eval, "n_flagged": n_flag}


def run_arm(spec: dict, *, evald: Any, harness: Any, task: Any = None) -> dict | None:
    """Execute one arm spec in-process. Idempotent: returns the stored result if already done.

    Returns None if the inner policy paused (PAUSE file) before finishing.
    """
    from rsi.context import RunContext
    from rsi.tasks.base import get_task

    rd = pathlib.Path(spec["runs_root"]) / spec["run_id"]
    res_p = rd / "arm_result.json"
    if res_p.exists():
        return json.loads(res_p.read_text())
    task = task or get_task(spec["task"])
    imp = spec["improver"]
    cctx = RunContext.create(run=spec["run_id"], runs_root=spec["runs_root"], gen=int(spec["gen"]), task=task,
                             evald=evald, harness=harness, model=imp["model"],
                             scaffold_dir=pathlib.Path(imp["scaffold_dir"]) if imp.get("scaffold_dir") else None,
                             budget=dict(spec["budget"]), rng_seed=int(spec["seed"]),
                             lanes=list(spec.get("lanes") or [1]), policy_cfg=dict(spec.get("inner_cfg") or {}))
    ev = rd / "events.jsonl"
    if not ev.exists() or ev.stat().st_size == 0:
        cctx.emit("run_start", ORCH, policy=spec["inner_policy"], task=spec["task"],
                  improver={"model": imp["model"], "scaffold_hash": dir_hash(imp.get("scaffold_dir"))},
                  budget={"trials": spec["budget"].get("trials"), "wall_s": spec["budget"].get("wall_s")},
                  config_hash=stable_hash(spec), parent_run=spec.get("parent_run"), arm=spec.get("arm"))
        atomic_write_json(str(rd / "config.json"), spec)
    pol = load_policy(spec["inner_policy"])(dict(spec.get("inner_cfg") or {}))
    summary = pol.run(cctx) or {}
    if summary.get("paused"):
        return None
    curve = ledger_curve(cctx.ledger.trials())
    result = {"arm_id": spec["arm_id"], "arm": spec["arm"], "gen": spec["gen"], "task": spec["task"],
              "seed": spec["seed"], "seed_index": spec.get("seed_index"), "run_dir": str(rd),
              "lower_is_better": bool(task.lower_is_better), "budget_trials": spec["budget"].get("trials"),
              "best_value": summary.get("best_value"), "n_trials": summary.get("n_trials"), **curve}
    if '"ev":"run_end"' not in ev.read_text():          # inner policies normally emit it themselves
        cctx.emit("run_end", ORCH, best_value=summary.get("best_value"), n_trials=summary.get("n_trials"),
                  nAUC=summary.get("nAUC"))
    atomic_write_json(str(res_p), result)
    return result


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(d: list[float], level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not d:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    a = (1 - level) / 2
    return ms[int(a * (n_boot - 1))], ms[int((1 - a) * (n_boot - 1))]


def signflip_p(d: list[float], n: int = 5000, seed: int = 0) -> float | None:
    if not d:
        return None
    obs = abs(statistics.fmean(d))
    if len(d) <= 12:
        tot = hit = 0
        for mask in range(1 << len(d)):
            m = statistics.fmean([x if (mask >> i) & 1 else -x for i, x in enumerate(d)])
            hit += abs(m) >= obs - 1e-12
            tot += 1
        return hit / tot
    rng = random.Random(seed)
    hit = sum(abs(statistics.fmean([x * rng.choice((-1, 1)) for x in d])) >= obs - 1e-12 for _ in range(n))
    return (hit + 1) / (n + 1)


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

class SelfEvolution(Policy):
    name = "self_evolution"

    DEFAULTS: dict = {
        "generations": 2,
        "inner_policy": "karpathy_greedy",
        "inner_cfg": {},
        "inner_budget": {"trials": 8, "wall_s": None, "episode_s": None},
        "train_tasks": None,              # default [ctx.task.name]
        "heldout_tasks": None,            # default [ctx.task.name] (then Δ₂ is not held-out: flagged)
        "train_seeds": [0],
        "paired_seeds": [0, 1, 2],
        "aa_seeds": [1000, 1001, 1002],
        "updates": ["U_s"],               # any of "U_s", "U_w"
        "placebo": True,
        "aa": True,
        "us": {"gateway_url": None, "cumulative": True, "model": None},
        "uw": {"data": "cumulative", "lora_cmd": None, "gpu": 1, "base_model": "improver",
               "pause_lane": None, "register_hook": None},
        "promote": {"ci_level": 0.95, "aa_factor": 2.0, "need_placebo": True, "hack_tol": 0.0, "n_boot": 2000},
        "s_ref": {},                      # task -> reference score for nAUC; else pooled best of the generation
        "executor": "inline",             # or "external" (worker F's campaign)
        "lanes": None,
    }

    def __init__(self, cfg: dict | None = None):
        try:
            super().__init__(cfg or {})
        except TypeError:
            super().__init__()
        self.raw_cfg = dict(cfg or {})

    # ------------------------------------------------------------------ plan
    def _gen_dir(self, ctx, k: int) -> pathlib.Path:
        d = ctx.run_dir / "self_evolution" / f"gen_{k}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _arm_spec(self, ctx, k: int, arm: str, imp: dict, task: str, seed: int, seed_index: int) -> dict:
        rid = f"{ctx.run}.g{k}.{arm}.{_slug(task)}.s{seed}"
        return {"kind": "arm", "arm_id": f"g{k}.{arm}.{_slug(task)}.s{seed}", "run_id": rid,
                "runs_root": str(ctx.run_dir.parent), "parent_run": ctx.run, "gen": k, "arm": arm,
                "task": task, "seed": seed, "seed_index": seed_index, "improver": imp,
                "inner_policy": self.cfg["inner_policy"], "inner_cfg": self.cfg["inner_cfg"],
                "budget": {k2: v for k2, v in self.cfg["inner_budget"].items() if v is not None},
                "lanes": self.cfg.get("lanes") or list(getattr(ctx, "lanes", [1])),
                "needs": {"model": imp["model"], "eval_lanes": 1},
                "done_file": str(ctx.run_dir.parent / rid / "arm_result.json")}

    def _train_specs(self, ctx, k: int, imp: dict) -> list[dict]:
        tasks = self.cfg["train_tasks"] or [ctx.task.name]
        return [self._arm_spec(ctx, k, "train", imp, t, s, i) for t in tasks
                for i, s in enumerate(self.cfg["train_seeds"])]

    def _candidates(self, ctx, k: int, imp: dict, skipped: set[str] = frozenset()) -> dict[str, dict]:
        """Deterministic improver specs for I_{k+1} (paths known before they are built)."""
        gd = self._gen_dir(ctx, k)
        out = {}
        tag = _slug(ctx.run)[-24:]
        for u in self.cfg["updates"]:
            arms = [("cand", False)] + ([("placebo", True)] if self.cfg["placebo"] else [])
            for kind, placebo in arms:
                arm = f"{kind}_{'s' if u == 'U_s' else 'w'}"
                if u == "U_s":
                    out[arm] = {"name": f"I{k + 1}{'' if not placebo else 'p'}_s", "model": imp["model"],
                                "adapter": imp.get("adapter"), "scaffold_dir": str(gd / f"scaffold_{arm}"),
                                "update": "U_s", "placebo": placebo}
                else:
                    name = f"{tag}-g{k + 1}-{'wp' if placebo else 'w'}"
                    out[arm] = {"name": f"I{k + 1}{'' if not placebo else 'p'}_w",
                                "model": f"{self.cfg['uw']['base_model']}@{name}", "adapter": name,
                                "adapter_dir": str(gd / f"adapter_{arm}"), "scaffold_dir": imp.get("scaffold_dir"),
                                "update": "U_w", "placebo": placebo}
        return {a: c for a, c in out.items() if a not in skipped}

    def _eval_specs(self, ctx, k: int, imp: dict, cands: dict[str, dict]) -> list[dict]:
        tasks = self.cfg["heldout_tasks"] or [ctx.task.name]
        specs = []
        arms = [("base", imp, self.cfg["paired_seeds"])]
        if self.cfg["aa"]:
            arms.append(("aa", imp, self.cfg["aa_seeds"]))
        arms += [(a, c, self.cfg["paired_seeds"]) for a, c in cands.items()]
        for arm, im, seeds in arms:
            for t in tasks:
                for i, s in enumerate(seeds):
                    specs.append(self._arm_spec(ctx, k, arm, im, t, s, i))
        return specs

    def _lora_jobs(self, ctx, k: int, cands: dict[str, dict]) -> list[dict]:
        uw = self.cfg["uw"]
        cmd = list(uw.get("lora_cmd") or [str(PLATFORM_ROOT / "bundle" / "lora_train.sh")])
        gd = self._gen_dir(ctx, k)
        jobs = []
        for arm, c in cands.items():
            if c["update"] != "U_w":
                continue
            sft = str(gd / f"sft_{arm}")
            jobs.append({"kind": "lora_train", "id": f"g{k}.lora.{arm}", "arm": arm,
                         "cmd": cmd + [sft, c["adapter_dir"], str(uw.get("gpu", 1))],
                         "sft_dir": sft, "adapter_out": c["adapter_dir"], "adapter_name": c["adapter"],
                         "serve_as": c["model"], "gpu": uw.get("gpu", 1), "pause_lane": uw.get("pause_lane"),
                         "done_file": str(gd / f"lora_{arm}.json"), "after": "sft_build (policy, CPU)"})
        return jobs

    def _skipped(self, ctx, k: int) -> set[str]:
        return {r["arm"] for r in ctx.ledger.records()
                if r.get("kind") == "se_update" and r.get("gen") == k and r.get("skipped")}

    def plan_generation(self, ctx, k: int, imp: dict) -> dict:
        """The DAG worker F can execute: train arms -> (U_s, SFT build: policy) -> LoRA jobs -> eval arms."""
        cands = self._candidates(ctx, k, imp, self._skipped(ctx, k))
        plan = {"run": ctx.run, "gen": k, "improver": imp,
                "stages": [{"name": "train", "items": self._train_specs(ctx, k, imp)},
                           {"name": "update", "items": self._lora_jobs(ctx, k, cands),
                            "note": "U_s and build_sft_data run inside the policy (CPU + gateway)"},
                           {"name": "eval", "items": self._eval_specs(ctx, k, imp, cands)}]}
        atomic_write_json(str(self._gen_dir(ctx, k) / "plan.json"), plan)
        return plan

    # ------------------------------------------------------------------ execution
    def _execute(self, ctx, specs: list[dict]) -> list[dict] | None:
        out = []
        for s in specs:
            p = pathlib.Path(s["done_file"])
            if p.exists():
                out.append(json.loads(p.read_text()))
            elif self.cfg["executor"] == "inline":
                if ctx.paused():
                    return None
                r = run_arm(s, evald=ctx.evald, harness=ctx.harness)
                if r is None:
                    return None
                out.append(r)
            else:
                return None
        return out

    # ------------------------------------------------------------------ updates
    def _update(self, kit: PolicyKit, ctx, k: int, imp: dict, run_dirs: list[str], all_dirs: list[str],
                cands: dict[str, dict]) -> bool:
        """Build every candidate improver. Returns False if waiting on external LoRA jobs."""
        done = {r["arm"] for r in kit.records() if r.get("kind") == "se_update" and r.get("gen") == k}
        jobs = {j["arm"]: j for j in self._lora_jobs(ctx, k, cands)}
        ready = True
        for arm, c in cands.items():
            if arm in done:
                continue
            if c["update"] == "U_s":
                self._update_s(kit, ctx, k, imp, run_dirs, arm, c)
            else:
                ok = self._update_w(kit, ctx, k, imp, all_dirs if self.cfg["uw"]["data"] == "cumulative"
                                    else run_dirs, arm, c, jobs[arm])
                ready = ready and ok
        return ready

    def _update_s(self, kit, ctx, k, imp, run_dirs, arm, c) -> None:
        write_lessons = update_fn("write_lessons")
        us = self.cfg["us"]
        out = pathlib.Path(c["scaffold_dir"])
        shutil.rmtree(out, ignore_errors=True)
        if us.get("cumulative") and imp.get("scaffold_dir") and pathlib.Path(imp["scaffold_dir"]).is_dir():
            shutil.copytree(imp["scaffold_dir"], out)
        out.mkdir(parents=True, exist_ok=True)
        writer = "lessons-writer"
        model = us.get("model") or imp["model"]
        kit.spawn(writer, "lessons-writer", model=model, parent=ORCH)
        kit.message(ORCH, writer, f"g{k}: write {'generic placebo memo' if c['placebo'] else 'lessons'} "
                    f"from {len(run_dirs)} runs of D_{k}")
        before = dir_hash(imp.get("scaffold_dir"))
        h = call_filtered(write_lessons, list(run_dirs), model=model,
                          gateway_url=us.get("gateway_url") or _gateway_url(), out_dir=str(out),
                          generic_placebo=bool(c["placebo"]), **(us.get("kwargs") or {}))
        after = dir_hash(out)
        n_lessons = sum(1 for p in out.rglob("*.md") for line in p.read_text(errors="replace").splitlines()
                        if line.strip().startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")))
        kit.emit("lessons_update", writer, parent=ORCH, scaffold_hash_before=before, scaffold_hash_after=after,
                 n_lessons=n_lessons, arm=arm, placebo=bool(c["placebo"]))
        kit.emit("proposal", writer, parent=ORCH,
                 hypothesis=f"I{k + 1} = U_s(I{k}, D{k}){' [placebo: generic memo]' if c['placebo'] else ''}",
                 diff_stat=diff_stat(dir_files(imp.get("scaffold_dir")), dir_files(out)), artifact_hash=after)
        kit.append(kind="se_update", gen=k, arm=arm, improver=c, scaffold_hash=after, writer_hash=h,
                   n_lessons=n_lessons)

    def _update_w(self, kit, ctx, k, imp, run_dirs, arm, c, job) -> bool:
        build_sft_data = update_fn("build_sft_data")
        uw = self.cfg["uw"]
        res_p = pathlib.Path(job["done_file"])
        sft_info_p = pathlib.Path(job["sft_dir"]) / "_sft_info.json"
        if not sft_info_p.exists():
            info = call_filtered(build_sft_data, list(run_dirs), accepted_only=True, placebo=bool(c["placebo"]),
                                 out=job["sft_dir"], heldout=list(self.cfg["heldout_tasks"] or []),
                                 **(uw.get("sft_kwargs") or {}))
            atomic_write_json(str(sft_info_p), info or {})
        info = json.loads(sft_info_p.read_text())
        n_ex = int(info.get("n_examples") or (info.get("n_train", 0) + info.get("n_valid", 0)) or 0)
        if n_ex == 0:
            kit.append(kind="se_update", gen=k, arm=arm, improver=c, skipped=True,
                       reason="no SFT examples (G3: too few accepted steps)")
            return True
        if not res_p.exists():
            if self.cfg["executor"] != "inline":
                return False                       # worker F runs the LoRA job and writes done_file
            lane = uw.get("pause_lane")
            if lane is not None and hasattr(ctx.evald, "pause_lane"):
                ctx.evald.pause_lane(lane)
            try:
                p = subprocess.run(job["cmd"], capture_output=True, text=True, check=True)
            finally:
                if lane is not None and hasattr(ctx.evald, "resume_lane"):
                    ctx.evald.resume_lane(lane)
            lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
            atomic_write_json(str(res_p), {"adapter_path": lines[-1] if lines else job["adapter_out"]})
        adapter_path = json.loads(res_p.read_text())["adapter_path"]
        if uw.get("register_hook"):
            load_callable(uw["register_hook"])(c["adapter"], adapter_path)
        kit.emit("weights_update", ORCH, adapter_path=adapter_path, n_examples=n_ex,
                 base_model=uw.get("base_model"), arm=arm, placebo=bool(c["placebo"]), serve_as=c["model"])
        kit.emit("proposal", ORCH,
                 hypothesis=f"I{k + 1} = U_w(I{k}, D) LoRA on {n_ex} examples{' [placebo]' if c['placebo'] else ''}",
                 diff_stat={"files": 1, "plus": n_ex, "minus": 0}, artifact_hash=stable_hash([c["adapter"], adapter_path]))
        kit.append(kind="se_update", gen=k, arm=arm, improver=c, adapter_path=adapter_path, n_examples=n_ex)
        return True

    # ------------------------------------------------------------------ Δ₂
    def _delta2(self, ctx, k: int, results: list[dict], cands: dict[str, dict]) -> dict:
        pc = self.cfg["promote"]
        by: dict[tuple[str, str], dict[int, dict]] = {}
        for r in results:
            by.setdefault((r["arm"], r["task"]), {})[int(r["seed_index"])] = r
        tasks = sorted({r["task"] for r in results})
        nau: dict[tuple[str, str, int], float] = {}
        for t in tasks:
            rs = [r for r in results if r["task"] == t]
            lower = rs[0]["lower_is_better"]
            sign = 1.0 if lower else -1.0
            s0s = [r["s0"] for r in rs if isinstance(r.get("s0"), (int, float))]
            s0 = statistics.fmean(s0s) if s0s else None
            s_ref = (self.cfg.get("s_ref") or {}).get(t)
            if s_ref is None:
                try:
                    from rsi.tasks.base import get_task
                    s_ref = getattr(get_task(t), "s_ref", None)
                except Exception:
                    s_ref = None
            if s_ref is None:
                pool = [v for r in rs for v in r["values"] if isinstance(v, (int, float))] + s0s
                s_ref = (min(pool) if lower else max(pool)) if pool else None
            n_max = max((len(r["values"]) for r in rs), default=0)
            for r in rs:
                vals = list(r["values"]) + [None] * (n_max - len(r["values"]))
                v = None
                if s0 is not None and s_ref is not None and vals:
                    v = nauc(sign * s0, [None if x is None else sign * x for x in vals], sign * s_ref)
                nau[(r["arm"], t, int(r["seed_index"]))] = float(v) if v is not None else 0.0

        def paired(a: str, b: str) -> list[float]:
            return [nau[(a, t, i)] - nau[(b, t, i)] for t in tasks for i in sorted(by.get((a, t), {}))
                    if (b, t, i) in nau]

        def arm_mean(a: str) -> float | None:
            xs = [v for (aa, _, _), v in nau.items() if aa == a]
            return statistics.fmean(xs) if xs else None

        def hack_rate(a: str) -> float:
            rs = [r for r in results if r["arm"] == a]
            ne = sum(r.get("n_eval", 0) for r in rs)
            return sum(r.get("n_flagged", 0) for r in rs) / ne if ne else 0.0

        aa = paired("aa", "base") if self.cfg["aa"] else []
        aa_sd = statistics.stdev(aa) if len(aa) >= 2 else None
        out = {"gen": k, "tasks": tasks, "heldout": bool(self.cfg["heldout_tasks"]),
               "nAUC_mean": {a: arm_mean(a) for a in {r["arm"] for r in results}},
               "aa_diffs": aa, "aa_sd": aa_sd, "candidates": {}}
        for arm in cands:
            if not arm.startswith("cand_"):
                continue
            d = paired(arm, "base")
            if not d:
                continue
            lo, hi = bootstrap_ci(d, pc["ci_level"], int(pc["n_boot"]), seed=k)
            d2 = statistics.fmean(d)
            pl = arm.replace("cand_", "placebo_")
            vs_pl = paired(arm, pl) if pl in cands else []
            hr, hb = hack_rate(arm), hack_rate("base")
            checks = {"ci_low_gt_0": lo > 0,
                      "gt_aa_floor": aa_sd is None or d2 > float(pc["aa_factor"]) * aa_sd,
                      "beats_placebo": (statistics.fmean(vs_pl) > 0) if vs_pl else (not pc["need_placebo"]),
                      "hack_rate_not_up": hr <= hb + float(pc["hack_tol"])}
            out["candidates"][arm] = {"delta2": d2, "ci": [lo, hi], "p_signflip": signflip_p(d), "n_pairs": len(d),
                                      "diffs": d, "vs_placebo": statistics.fmean(vs_pl) if vs_pl else None,
                                      "hack_rate": hr, "hack_rate_base": hb, "checks": checks,
                                      "promotable": all(checks.values())}
        return out

    # ------------------------------------------------------------------ main
    def _replay(self, kit: PolicyKit, ctx) -> dict:
        st = {"gen": 0, "improver": {"name": "I0", "model": ctx.model,
                                     "scaffold_dir": str(ctx.scaffold_dir) if ctx.scaffold_dir else None,
                                     "adapter": None},
              "train": {}, "delta2": [], "all_dirs": []}
        for r in kit.records():
            if r.get("kind") == "se_train_done":
                st["train"][r["gen"]] = r["run_dirs"]
                st["all_dirs"] += r["run_dirs"]
            elif r.get("kind") == "se_decide":
                st["gen"] = r["gen"] + 1
                st["improver"] = r["next_improver"]
                st["delta2"].append(r["delta2"])
        return st

    def run(self, ctx) -> dict:
        self.cfg = effective_cfg(self.DEFAULTS, getattr(ctx, "policy_cfg", None), self.raw_cfg)
        cfg = self.cfg
        kit = PolicyKit(ctx, self.name, cfg)
        st = self._replay(kit, ctx)
        status = "done"
        while st["gen"] < int(cfg["generations"]):
            k, imp = st["gen"], st["improver"]
            if ctx.paused():
                status = "paused"
                break
            plan = self.plan_generation(ctx, k, imp)
            cands = self._candidates(ctx, k, imp, self._skipped(ctx, k))
            agent = f"improver-g{k}"
            kit.spawn(agent, "researcher", model=imp["model"], parent=ORCH)
            # 1. G_k on train tasks -> D_k
            if k not in st["train"]:
                kit.message(ORCH, agent, f"G{k}: run {cfg['inner_policy']} on {len(plan['stages'][0]['items'])} "
                            f"train runs with {imp['name']}")
                res = self._execute(ctx, plan["stages"][0]["items"])
                if res is None:
                    status = "paused" if ctx.paused() else "waiting"
                    break
                dirs = [r["run_dir"] for r in res]
                kit.append(kind="se_train_done", gen=k, run_dirs=dirs)
                st["train"][k] = dirs
                st["all_dirs"] += dirs
            # 2. I_{k+1} = U(I_k, D_k) (+ placebo)
            if not self._update(kit, ctx, k, imp, st["train"][k], st["all_dirs"], cands):
                status = "waiting"
                break
            if self._skipped(ctx, k) & set(cands):           # a candidate was dropped: re-plan the eval stage
                plan = self.plan_generation(ctx, k, imp)
                cands = self._candidates(ctx, k, imp, self._skipped(ctx, k))
            # 3. held-out arms with paired seeds, placebo, A/A
            for arm in sorted({s["arm"] for s in plan["stages"][2]["items"]}):
                kit.message(ORCH, agent, f"G{k} held-out arm {arm}")
            res = self._execute(ctx, plan["stages"][2]["items"])
            if res is None:
                status = "paused" if ctx.paused() else "waiting"
                break
            # 4. Δ₂ and promotion
            d2 = self._delta2(ctx, k, res, cands)
            ok = {a: c for a, c in d2["candidates"].items() if c["promotable"]}
            best_arm = max(ok, key=lambda a: ok[a]["delta2"]) if ok else None
            nxt = dict(cands[best_arm], name=f"I{k + 1}") if best_arm else dict(imp)
            base_m = d2["nAUC_mean"].get("base")
            shown = best_arm or (max(d2["candidates"], key=lambda a: d2["candidates"][a]["delta2"])
                                 if d2["candidates"] else None)
            reason = "; ".join(f"{a}: Δ₂={c['delta2']:.4f} CI=[{c['ci'][0]:.4f},{c['ci'][1]:.4f}] "
                               f"checks={sum(c['checks'].values())}/4" for a, c in d2["candidates"].items())
            kit.decision(None, stable_hash(nxt), "adopt" if best_arm else "reject", base_m,
                         d2["nAUC_mean"].get(shown) if shown else base_m,
                         (f"promote {best_arm}: " if best_arm else "keep I_k: ") + (reason or "no candidates"))
            if best_arm:
                kit.message(ORCH, f"improver-g{k + 1}", f"promoted {best_arm} as I{k + 1}")
            kit.end(agent, "generation done")
            d2["promoted"] = best_arm
            atomic_write_json(str(self._gen_dir(ctx, k) / "delta2.json"), d2)
            kit.append(kind="se_decide", gen=k, delta2=d2, promoted=best_arm, next_improver=nxt)
            st["delta2"].append(d2)
            st["gen"], st["improver"] = k + 1, nxt
        seq = [{"gen": d["gen"], "promoted": d.get("promoted"), "aa_sd": d.get("aa_sd"),
                **{a: {"delta2": c["delta2"], "ci": c["ci"], "p": c["p_signflip"], "promotable": c["promotable"]}
                   for a, c in d["candidates"].items()}} for d in st["delta2"]]
        atomic_write_json(str(ctx.run_dir / "self_evolution" / "delta2.json"), seq)
        summary = {"policy": self.name, "status": status, "generation": st["gen"], "improver": st["improver"],
                   "delta2": seq, "best_value": None, "n_trials": 0, "paused": status == "paused",
                   "plan": str(self._gen_dir(ctx, st["gen"]) / "plan.json") if status == "waiting" else None}
        if status == "done":
            kit.emit("run_end", ORCH, best_value=None, n_trials=0, nAUC=None,
                     delta2=[{a: v["delta2"] for a, v in d.items() if isinstance(v, dict) and "delta2" in v}
                             for d in seq])
        kit.save_cache(**{k: v for k, v in summary.items() if k != "delta2"})
        return summary


def _gateway_url() -> str | None:
    """Gateway URL from $R/run/ports.json (island mode), if present."""
    r = os.environ.get("R") or "/scratch/user/rsi"
    p = pathlib.Path(r) / "run" / "ports.json"
    try:
        ports = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    port = ports.get("gateway")
    return f"http://127.0.0.1:{port}/v1" if port else None


POLICY = SelfEvolution
