"""``apex_ai4ai``: Apex's "fixed task, swap the researcher" protocol plus self-evolving data.

Source: Apex Intelligence, "Introducing Apex-35B-V1.0", sections "AI For AI" and
"Building ... Through Recursive Self-Improvement"
(research/reading/04_Apex35B_导读.md, research/sources/apex_35b.md AX-22, AX-37),
platform_build_plan.md §4.4.

Method as reproduced here
-------------------------
* Same task, same evaluator, **fixed explore budget then fixed retrain budget** (their 4 h + 12 h).
  The explore phase is a greedy keep/discard loop (each trial = one harness episode + one
  screening eval); ``explore`` sets its size in trials and/or wall seconds. The retrain phase
  re-evaluates the researcher's best artifact once with ``retrain.seeds`` and a longer budget
  (``retrain.budget_scale``, default 3.0 = 12 h / 4 h, forwarded to evald if it accepts it).
* **Researcher swap**: ``researchers`` is a list of {name, model, scaffold_dir}; every researcher
  runs the identical protocol (gateway model names e.g. "improver", "improver@m1", "ladder-27b").
* Gaps the original left open, filled on purpose: ``reps`` repetitions per researcher (they ran
  once), a researcher that exits without a submission is **recorded as a result**
  (``completed=False``) instead of being dropped, and nothing is hand-picked afterwards.
* **Self-evolving data**: failures are classified (harness/tool-call failure, no proposal,
  crash/OOM/NaN, no improvement, duplicate proposal, budget exhausted); for each failure a
  later evaluator-verified improvement from the same unit is paired with it as a training
  example (hindsight relabel), and targeted notes are written per researcher. Only
  evaluator-verified examples enter ``apex/selfevolve_data.jsonl`` (optionally re-verified on
  ``self_evolving.verify_seeds``). ``self_evolving.mode``: ``after_all`` (default; notes and data
  are for the next round, reps stay independent), ``between_reps`` (notes are fed to the next
  rep's prompt), or ``off``.

Not reproduced: the original tasks (Sokoban GRPO, DDPO on SD1.5) and the 4 h / 12 h scale; the
"36 high-quality actions" were labelled by humans after the fact, which we replace with
evaluator verification; the CFD-Bench self-evolving-data loop's details are not published.
"""

from __future__ import annotations

import json
import re
import statistics
import time

from rsi.policies.base import Policy
from rsi.policies.naive_trial_ledger import ORCH, OK, PolicyKit, baseline, effective_cfg, parse_submission
from rsi.runs import atomic_write_json

PROGRAM = """You are the researcher in a fixed-budget AI-for-AI experiment.
You have {left} trial(s) left in the exploration budget. Each trial: edit the artifact, then submit.
Only the evaluator's score counts. Keep changes that are simple and correct.

## Task
{task_card}

## Your results so far (most recent last)
{ledger}
{notes}
Submit with {{"hypothesis": "<=300 chars"}}."""

CATEGORIES = {
    "harness_failure": "Tool calls failed or the episode crashed. Use exactly the documented tool names "
                       "and argument schemas; read before editing; make one small edit at a time.",
    "no_proposal": "The episode ended without editing the artifact. Always make one concrete edit and submit.",
    "crash": "The artifact crashed / hit OOM / produced NaN. Check shapes, memory and numerics before "
             "submitting; prefer smaller steps from the last working version.",
    "no_improvement": "The change ran but did not beat the incumbent. Prefer directions that worked "
                      "(see verified improvements) and avoid repeating rejected ideas.",
    "duplicate": "The same artifact was submitted again. Check the ledger and try something new.",
    "budget_exhausted": "The budget ran out without any accepted improvement. Spend early trials on "
                        "cheap, high-probability changes.",
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "x"


class ApexAI4AI(Policy):
    name = "apex_ai4ai"

    DEFAULTS: dict = {
        "researchers": [{"name": "base", "model": None, "scaffold_dir": None}],
        "reps": 3,
        "explore": {"trials": 12, "wall_s": None},
        "retrain": {"seeds": [100, 101, 102], "budget_scale": 3.0, "level": "L2"},
        "screen_seeds": None,
        "exit_after_failed_episodes": 3,
        "self_evolving": {"mode": "after_all", "verify_seeds": None, "llm_notes": False, "notes_model": None},
        "start_dir": None,
    }

    def __init__(self, cfg: dict | None = None):
        try:
            super().__init__(cfg or {})
        except TypeError:
            super().__init__()
        self.raw_cfg = dict(cfg or {})

    # ------------------------------------------------------------------ replay
    def _units(self) -> list[tuple[dict, int, str]]:
        out = []
        for res in self.cfg["researchers"]:
            for rep in range(int(self.cfg["reps"])):
                out.append((res, rep, f"{_slug(res['name'])}/rep-{rep}"))
        return out

    def _unit_state(self, kit: PolicyKit, unit: str, start_hash: str, s0: float | None) -> dict:
        u = {"best_hash": start_hash, "best_dev": s0, "n": 0, "fail_streak": 0, "any_ok": False,
             "end": None, "t0": None, "retrain": None, "seen": {start_hash}}
        for r in kit.records():
            if r.get("unit") != unit:
                continue
            k = r.get("kind")
            if k == "unit_start":
                u["t0"] = r["t0"]
            elif k == "explore" and r.get("trial") is not None and r.get("status") != "pending":
                u["n"] += 1
                if r.get("status") == "keep":
                    u["best_hash"], u["best_dev"] = r["artifact_hash"], r.get("dev")
                if r.get("eval_status") == OK:
                    u["any_ok"] = True
                u["fail_streak"] = 0 if r.get("eval_status") == OK else u["fail_streak"] + 1
                u["seen"].add(r.get("artifact_hash"))
            elif k == "retrain":
                u["retrain"] = r
            elif k == "unit_end":
                u["end"] = r
        return u

    # ------------------------------------------------------------------ self-evolving data
    def _classify(self, recs: list[dict]) -> list[tuple[str, dict]]:
        out, seen = [], set()
        for r in recs:
            if r.get("kind") != "explore" or r.get("trial") is None:
                continue
            h = r.get("artifact_hash")
            if not r.get("harness_ok", True):
                out.append(("harness_failure", r))
            elif r.get("status") == "no_change":
                out.append(("no_proposal", r))
            elif h in seen:
                out.append(("duplicate", r))
            elif r.get("eval_status") not in (None, OK):
                out.append(("crash", r))
            elif r.get("status") == "discard":
                out.append(("no_improvement", r))
            seen.add(h)
        return out

    def _self_evolve(self, kit: PolicyKit, ctx, units: list[str], scope: str) -> dict:
        se = self.cfg["self_evolving"]
        recs = [r for r in kit.records() if r.get("unit") in units]
        ends = {r["unit"]: r for r in recs if r.get("kind") == "unit_end"}
        fails = self._classify(recs)
        for u in units:
            e = ends.get(u)
            if e and not any(r.get("status") == "keep" and r.get("unit") == u for r in recs):
                fails.append(("budget_exhausted", {"unit": u, "trial": None, "hypothesis": e.get("reason")}))
        counts: dict[str, int] = {}
        for c, _ in fails:
            counts[c] = counts.get(c, 0) + 1
        verified = [r for r in recs if r.get("kind") == "explore" and r.get("status") == "keep"]
        examples = []
        for c, f in fails:
            if f.get("trial") is None:
                continue
            fix = next((v for v in verified if v["unit"] == f["unit"] and v["trial"] > f["trial"]), None)
            if fix is None:
                continue
            ok = True
            if se.get("verify_seeds"):
                rv = kit.evaluate(fix["artifact_hash"], trial=None, seeds=list(se["verify_seeds"]), level="L2")
                ok = rv["status"] == OK and kit.better(rv["value"], fix.get("parent_dev"))
            if ok:
                examples.append({"category": c, "unit": f["unit"], "failed_trial": f["trial"],
                                 "failed_hypothesis": f.get("hypothesis"), "failure": f.get("reason"),
                                 "fix_trial": fix["trial"], "fix_hypothesis": fix.get("hypothesis"),
                                 "fix_artifact": fix["artifact_hash"], "fix_parent": fix.get("parent_hash"),
                                 "verified_value": fix.get("dev"), "verified_by": "evaluator"})
        d = ctx.run_dir / "apex"
        (d / "notes").mkdir(parents=True, exist_ok=True)
        with open(d / "selfevolve_data.jsonl", "a") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        researchers = sorted({u.split("/rep-")[0] for u in units})
        for rn in researchers:
            mine = [(c, f) for c, f in fails if str(f.get("unit", "")).startswith(rn + "/")]
            good = [v for v in verified if v["unit"].startswith(rn + "/")]
            lines = [f"# Notes for researcher {rn} ({scope})", ""]
            for c in CATEGORIES:
                n = sum(1 for cc, _ in mine if cc == c)
                if n:
                    lines.append(f"- {c} x{n}: {CATEGORIES[c]}")
            if good:
                lines += ["", "Verified improvements (evaluator-accepted):"]
                lines += [f"- {v.get('hypothesis', '')[:160]} ({ctx.task.metric}={v.get('dev')})" for v in good[-8:]]
            text = "\n".join(lines) + "\n"
            if se.get("llm_notes"):
                kit.spawn("lessons-writer", "lessons-writer", model=se.get("notes_model"), parent=ORCH)
                kit.message(ORCH, "lessons-writer", f"summarise {len(mine)} failures of {rn} into notes")
                ws = kit.scratch("lessons-writer", kit.put(kit.start_files(self.cfg.get("start_dir"))))
                ep = kit.episode("lessons-writer", None, ws, "Rewrite these failure notes into <=200 words of "
                                 "targeted advice for the next round:\n\n" + text,
                                 parent=ORCH, model=se.get("notes_model"))
                if ep.get("ok") and ep.get("final"):
                    text += "\n## Writer summary\n" + ep["final"].strip() + "\n"
            with open(d / "notes" / f"{_slug(rn)}.md", "a") as f:
                f.write(text + "\n")
        rec = kit.append(kind="selfevolve", scope=scope, units=units, counts=counts, n_examples=len(examples))
        return rec

    def _notes_for(self, ctx, rname: str) -> str:
        if self.cfg["self_evolving"]["mode"] != "between_reps":
            return ""
        p = ctx.run_dir / "apex" / "notes" / f"{_slug(rname)}.md"
        return ("\n## Notes from your earlier rounds\n" + p.read_text()[-3000:]) if p.exists() else ""

    # ------------------------------------------------------------------ one researcher x rep
    def _unit(self, kit: PolicyKit, ctx, res: dict, rep: int, unit: str, start_hash: str, s0) -> str:
        cfg = self.cfg
        model = res.get("model") or ctx.model
        sd = res.get("scaffold_dir")
        u = self._unit_state(kit, unit, start_hash, s0)
        if u["end"] is not None:
            return "done"
        if u["t0"] is None:
            u["t0"] = time.time()
            kit.append(kind="unit_start", unit=unit, researcher=res["name"], rep=rep, model=model, t0=u["t0"])
        kit.spawn(unit, "researcher", model=model, parent=ORCH)
        ex = cfg["explore"]
        kit.message(ORCH, unit, f"{res['name']} rep {rep}: explore budget trials={ex.get('trials')} "
                    f"wall_s={ex.get('wall_s')}, then retrain x{cfg['retrain']['budget_scale']}")
        ws = kit.ws(_slug(unit).replace("/", "_"))
        screen = list(cfg.get("screen_seeds") or ctx.task.dev_seeds)
        exited = False
        while True:
            if ctx.paused():
                return "paused"
            if ex.get("trials") is not None and u["n"] >= int(ex["trials"]):
                break
            if ex.get("wall_s") is not None and time.time() - u["t0"] >= float(ex["wall_s"]):
                break
            if kit.out_of_budget():
                return "budget"
            if u["fail_streak"] >= int(cfg["exit_after_failed_episodes"]) and not u["any_ok"]:
                exited = True
                break
            t = kit.new_trial(kind="explore", unit=unit, thread=unit, researcher=res["name"], rep=rep)
            kit.checkout(u["best_hash"], ws)
            left = (int(ex["trials"]) - u["n"]) if ex.get("trials") is not None else "?"
            prompt = PROGRAM.format(left=left, task_card=ctx.task.task_card(),
                                    ledger=ctx.ledger.tail_text(unit, 10), notes=self._notes_for(ctx, res["name"]))
            ep = kit.episode(unit, t, ws, prompt, parent=ORCH, model=model, scaffold_dir=sd)
            sub = parse_submission(ep)
            h, _ = kit.snap(ws)
            rec = dict(trial=t, kind="explore", unit=unit, thread=unit, researcher=res["name"], rep=rep,
                       artifact_hash=h, parent_hash=u["best_hash"], parent_dev=u["best_dev"],
                       hypothesis=sub["hypothesis"][:300], harness_ok=bool(ep.get("ok")),
                       n_tool_calls=ep.get("n_tool_calls"))
            before = u["best_dev"]
            u["n"] += 1
            if not ep.get("ok") or h == u["best_hash"]:
                u["fail_streak"] += 1
                kit.append(status="no_change", reason="harness failure" if not ep.get("ok") else "no edit",
                           incumbent=before, **rec)
                kit.decision(t, h, "reject", before, before, "no proposal")
                continue
            dup = h in u["seen"]
            u["seen"].add(h)
            kit.proposal(unit, t, sub["hypothesis"], u["best_hash"], h, parent=ORCH)
            r = kit.evaluate(h, trial=t, seeds=screen, agent=ORCH, level="L1", lane=kit.lane_for(0), thread=unit)
            if r["status"] == OK:
                u["any_ok"] = True
                u["fail_streak"] = 0
            else:
                u["fail_streak"] += 1
            keep = r["status"] == OK and kit.better(r["value"], u["best_dev"])
            status = "keep" if keep else ("crash" if r["status"] != OK else "discard")
            if keep:
                u["best_hash"], u["best_dev"] = h, r["value"]
            kit.append(status=status, dev=r["value"], eval_status=r["status"], seeds=screen, duplicate=dup,
                       audit_flags=r["audit"].get("flags", []), incumbent=u["best_dev"],
                       reason=f"eval {r['status']}" if r["status"] != OK else "", **rec)
            kit.decision(t, h, "adopt" if keep else "reject", before, u["best_dev"], status)
        # ---- retrain phase (or record the exit as a result)
        completed = u["any_ok"] and not exited
        rt = cfg["retrain"]
        if completed and u["retrain"] is None:
            seeds = list(rt["seeds"])
            r = kit.evaluate(u["best_hash"], trial=None, seeds=seeds, agent=ORCH, level=rt.get("level", "L2"),
                             lane=kit.lane_for(0), retrain=True, budget_scale=rt.get("budget_scale"))
            u["retrain"] = kit.append(kind="retrain", unit=unit, researcher=res["name"], rep=rep,
                                      artifact_hash=u["best_hash"], value=r["value"], sealed=r["sealed"],
                                      seeds=seeds, eval_status=r["status"], budget_scale=rt.get("budget_scale"))
            kit.message(ORCH, unit, f"retrain of {u['best_hash']}: {ctx.task.visible_feedback(r)}")
        reason = "exited without a submission" if exited else ("no_submission" if not completed else "budget")
        kit.append(kind="unit_end", unit=unit, researcher=res["name"], rep=rep, completed=bool(completed),
                   reason=reason, best_hash=u["best_hash"], best_dev=u["best_dev"],
                   retrain_value=(u["retrain"] or {}).get("value"), n_trials=u["n"])
        kit.end(unit, reason)
        return "done"

    # ------------------------------------------------------------------ main
    def run(self, ctx) -> dict:
        self.cfg = effective_cfg(self.DEFAULTS, getattr(ctx, "policy_cfg", None), self.raw_cfg)
        cfg = self.cfg
        kit = PolicyKit(ctx, self.name, cfg)
        kit.recover_pending()
        start_hash = kit.put(kit.start_files(cfg.get("start_dir")))
        b = baseline(kit, start_hash, list(cfg.get("screen_seeds") or ctx.task.dev_seeds), lane=kit.lane_for(0))
        s0 = b.get("dev")
        units = self._units()
        mode = cfg["self_evolving"]["mode"]
        done_scopes = {r.get("scope") for r in kit.records() if r.get("kind") == "selfevolve"}
        state = "done"
        for i, (res, rep, unit) in enumerate(units):
            state = self._unit(kit, ctx, res, rep, unit, start_hash, s0)
            if state != "done":
                break
            if mode == "between_reps" and f"unit:{unit}" not in done_scopes:
                self._self_evolve(kit, ctx, [unit], f"unit:{unit}")
        if state == "done" and mode == "after_all" and "all" not in done_scopes:
            self._self_evolve(kit, ctx, [u for _, _, u in units], "all")
        results = self._results(kit, ctx)
        atomic_write_json(str(ctx.run_dir / "apex" / "results.json"), results)
        vals = [r["retrain_mean"] for r in results["researchers"].values() if r["retrain_mean"] is not None]
        best = (min(vals) if ctx.task.lower_is_better else max(vals)) if vals else None
        summary = {"policy": self.name, "best_value": best, "n_trials": kit.trials_done(),
                   "paused": state == "paused", "state": state, "results": results}
        if state == "done":
            kit.emit("run_end", ORCH, best_value=best, n_trials=kit.trials_done(), nAUC=None)
        kit.save_cache(**{k: v for k, v in summary.items() if k != "results"})
        return summary

    def _results(self, kit: PolicyKit, ctx) -> dict:
        ends = [r for r in kit.records() if r.get("kind") == "unit_end"]
        out: dict = {"metric": ctx.task.metric, "lower_is_better": ctx.task.lower_is_better, "researchers": {}}
        for res in self.cfg["researchers"]:
            mine = [e for e in ends if e.get("researcher") == res["name"]]
            vals = [e["retrain_value"] for e in mine if isinstance(e.get("retrain_value"), (int, float))]
            sealed = [r.get("sealed") for r in kit.records() if r.get("kind") == "retrain"
                      and r.get("researcher") == res["name"] and isinstance(r.get("sealed"), (int, float))]
            out["researchers"][res["name"]] = {
                "model": res.get("model") or ctx.model, "scaffold_dir": res.get("scaffold_dir"),
                "reps_done": len(mine), "completed": sum(1 for e in mine if e.get("completed")),
                "exits": [e["rep"] for e in mine if not e.get("completed")],
                "retrain_values": vals,
                "retrain_mean": statistics.fmean(vals) if vals else None,
                "retrain_sd": statistics.stdev(vals) if len(vals) >= 2 else None,
                "sealed_mean": statistics.fmean(sealed) if sealed else None,
            }
        return out


POLICY = ApexAI4AI
