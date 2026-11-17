"""Policy base class and the shared primitives every loop policy is built from.

Build-plan §4 primitives map onto methods here:

    session = harness.run(...)          -> Policy.propose(ctx, thread, prompt, trial)
    result  = lab.evaluate(...)         -> Policy.evaluate(ctx, trial, artifact_hash, seeds=..., lane=...)
    ledger.append(...)                  -> done inside propose / evaluate / decide
    git.advance / git.reset             -> Policy.decide(...) (adopt moves thread.best; reject checks best back out)

"Agent proposes, orchestrator decides": the improver never sees evaluator code, sealed scores or
other threads' workspaces; it only sees the lab view written by ``write_lab_view``.

Resumability: every state change is a ledger line. ``make_threads`` rebuilds each thread's best
from the ledger, and ``recover_pending`` finishes trials that were proposed or submitted but not
decided when the process died (evald job ids are idempotent).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from rsi.ledger import nauc
from rsi.runs import atomic_write_json, diff_stat, snapshot_files, unified_diff

# ledger statuses that make an artifact the thread's new best
ADOPT_STATUSES = {"keep", "adopted", "baseline"}
FINAL_STATUSES = {"keep", "discard", "crash", "adopted", "rolled_back", "exploratory", "no_change",
                  "baseline", "rejected", "audit_fail"}


@dataclass
class ResearchThread:
    tid: str                    # agent id, e.g. "thread-0"
    idx: int
    lane: int | None
    workspace: pathlib.Path
    best_hash: str | None = None
    best_value: float | None = None
    n_trials: int = 0
    focus: str = ""             # optional research focus (recursive_first_steps)
    meta: dict = field(default_factory=dict)

    @property
    def lab_dir(self) -> pathlib.Path:
        """Lab MCP spool for this thread: workspace/../lab (convention shared with the harness)."""
        return self.workspace.parent / "lab"


@dataclass
class Proposal:
    trial: int
    thread: str
    artifact_hash: str | None   # None when the episode changed nothing
    parent_hash: str | None
    hypothesis: str
    diff_stat: dict
    episode: dict


class Policy:
    name = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})
        self._trial_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._claimed_unrecorded: set[int] = set()
        self._next_trial: int | None = None

    # ------------------------------------------------------------------
    def run(self, ctx) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def judge(self, ctx, thread: ResearchThread, trial: int, artifact_hash: str, result: dict) -> tuple[str, str, str]:
        """Decision rule used by recover_pending. Returns (verdict, status, reason). Default: greedy."""
        if result.get("status") != "ok":
            return "reject", "crash", f"eval status {result.get('status')}"
        if ctx.task.better(result.get("value"), thread.best_value):
            return "adopt", "keep", "improves best"
        return "reject", "discard", "no improvement"

    # ------------------------------------------------------------------ trials
    def new_trial(self, ctx) -> int:
        with self._trial_lock:
            if self._next_trial is None:
                trials = ctx.ledger.trials()
                self._next_trial = (max(trials) + 1) if trials else 0
            t = self._next_trial
            self._next_trial += 1
            return t

    def n_finished(self, ctx, thread: str | None = None, *, count_baseline: bool = False) -> int:
        recs = ctx.ledger.finished(thread)
        if not count_baseline:
            recs = [r for r in recs if r.get("status") != "baseline"]
        return len([r for r in recs if r.get("status") != "no_change"])

    def should_stop(self, ctx) -> bool:
        return ctx.paused() or ctx.out_of_budget(self.n_started(ctx))

    def n_started(self, ctx) -> int:
        """Trials that count against the budget: every non-baseline trial except episodes with no edit."""
        n = 0
        for r in ctx.ledger.trials().values():
            if r.get("thread") != "orchestrator" and r.get("status") not in ("no_change", "baseline"):
                n += 1
        return n

    def claim_trial(self, ctx) -> int | None:
        """Atomically check pause/budget and allocate a trial id (None = stop). Use this in thread loops
        so K parallel threads never overshoot ``budget.trials``."""
        with self._claim_lock:
            if self.should_stop(ctx) or self._claimed_unrecorded and ctx.out_of_budget(
                    self.n_started(ctx) + len(self._claimed_unrecorded)):
                return None
            t = self.new_trial(ctx)
            self._claimed_unrecorded.add(t)
            return t

    def _recorded(self, trial: int) -> None:
        self._claimed_unrecorded.discard(trial)

    # ------------------------------------------------------------------ threads
    def make_threads(self, ctx, k: int, *, prefix: str = "thread", focuses: list[str] | None = None) -> list[ResearchThread]:
        """Create (or restore from the ledger) k research threads with their own workspaces."""
        base_hash, base_value = self.ensure_baseline(ctx)
        threads = []
        lanes = list(ctx.lanes) or [None]
        for i in range(k):
            tid = f"{prefix}-{i}"
            ws = ctx.run_dir / "threads" / tid / "workspace"
            lane = None if self.cfg.get("lane_mode") == "shared" else lanes[i % len(lanes)]
            th = ResearchThread(tid=tid, idx=i, lane=lane, workspace=ws,
                                best_hash=base_hash, best_value=base_value,
                                focus=(focuses[i % len(focuses)] if focuses else ""))
            for r in ctx.ledger.finished(tid):
                th.n_trials += 1
                if r.get("status") in ADOPT_STATUSES and r.get("artifact_hash"):
                    th.best_hash, th.best_value = r["artifact_hash"], r.get("dev")
            fresh = not ws.exists()
            ws.mkdir(parents=True, exist_ok=True)
            th.lab_dir.mkdir(parents=True, exist_ok=True)
            (th.lab_dir / "outbox").mkdir(exist_ok=True)
            if fresh:  # non-editable helpers (README, etc.) from the task's start dir, then the start artifact
                for src in dict.fromkeys([pathlib.Path(ctx.task.start_artifact()), pathlib.Path(ctx.start_dir())]):
                    for p in src.iterdir():
                        if p.is_file():
                            shutil.copy2(p, ws / p.name)
            ctx.artifacts.checkout(th.best_hash, str(ws))
            threads.append(th)
        return threads

    def ensure_baseline(self, ctx) -> tuple[str, float | None]:
        """Evaluate the start artifact once per run (ledger status 'baseline', thread 'orchestrator')."""
        start = pathlib.Path(ctx.start_dir())
        files = snapshot_files(str(start), ctx.task.editable_files)
        h = ctx.artifacts.put(files)
        for r in ctx.ledger.finished("orchestrator"):
            if r.get("status") == "baseline" and r.get("artifact_hash") == h:
                return h, r.get("dev")
        if not self.cfg.get("eval_baseline", True):
            return h, self.cfg.get("baseline_value")
        trial = self.new_trial(ctx)
        ctx.ledger.append(trial=trial, thread="orchestrator", status="proposed", artifact_hash=h,
                          parent_hash=None, hypothesis="baseline (start artifact)", gen=ctx.gen)
        lane = None if (self.cfg.get("lane_mode") == "shared" or not ctx.lanes) else ctx.lanes[0]
        res = self.evaluate(ctx, trial, h, seeds=list(self.cfg.get("baseline_seeds", ctx.task.dev_seeds)),
                            lane=lane, agent="orchestrator")
        value = res.get("value") if res.get("status") == "ok" else None
        ctx.ledger.append(trial=trial, thread="orchestrator", status="baseline", artifact_hash=h,
                          dev=value, eval_status=res.get("status"))
        ctx.emit("decision", "orchestrator", trial=trial, artifact_hash=h, verdict="adopt",
                 best_before=None, best_after=value, reason="baseline start artifact")
        return h, value

    # ------------------------------------------------------------------ lab view
    def write_lab_view(self, ctx, thread: ResearchThread, *, include_all_threads: bool = False,
                       extra_notes: str = "") -> None:
        """What the improver may see through the lab MCP: task card, best, dev-only history."""
        lab = thread.lab_dir
        lab.mkdir(parents=True, exist_ok=True)
        (lab / "outbox").mkdir(exist_ok=True)
        (lab / "card.md").write_text(ctx.task.task_card() + ("\n\n" + extra_notes if extra_notes else ""))
        rows = list(ctx.history_rows()) if hasattr(ctx, "history_rows") else []
        for r in ctx.ledger.finished(None if include_all_threads else thread.tid):
            if r.get("thread") not in (thread.tid, "orchestrator") and not include_all_threads:
                continue
            rows.append({"trial": r["trial"], "thread": r.get("thread"), "artifact": r.get("artifact_hash"),
                         "value": r.get("dev"), "status": r.get("status"),
                         "feedback": r.get("feedback", ""), "hypothesis": (r.get("hypothesis") or "")[:300]})
        with open(lab / "history.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        atomic_write_json(str(lab / "best.json"), {"artifact": thread.best_hash, "value": thread.best_value,
                                                   "metric": ctx.task.metric,
                                                   "lower_is_better": ctx.task.lower_is_better})

    def clear_outbox(self, thread: ResearchThread) -> None:
        ob = thread.lab_dir / "outbox"
        ob.mkdir(parents=True, exist_ok=True)
        for p in ob.iterdir():
            if p.is_file():
                p.unlink()

    # ------------------------------------------------------------------ primitives
    def propose(self, ctx, thread: ResearchThread, prompt: str, trial: int, *, agent: str | None = None,
                start_from: str | None = None, budget_s: int | None = None, model: str | None = None,
                scaffold_dir: pathlib.Path | None = None, reset_workspace: bool = True) -> Proposal:
        """One harness episode in the thread's workspace -> content-addressed proposal."""
        agent = agent or thread.tid
        parent_hash = start_from or thread.best_hash
        if reset_workspace and parent_hash:
            ctx.artifacts.checkout(parent_hash, str(thread.workspace))
        self.clear_outbox(thread)
        ctx.ledger.append(trial=trial, thread=thread.tid, status="pending", phase="episode",
                          parent_hash=parent_hash, gen=ctx.gen)
        self._recorded(trial)
        ctx.emit("message", "orchestrator", trial=trial, to=agent, summary=prompt_summary(prompt),
                 tokens=approx_tokens(prompt))
        episode = ctx.harness.run_episode(
            run=ctx.run, agent=agent, gen=ctx.gen, trial=trial, workspace=thread.workspace, prompt=prompt,
            model=model or ctx.model, scaffold_dir=scaffold_dir if scaffold_dir is not None else ctx.scaffold_dir,
            budget_s=int(budget_s or ctx.budget.get("episode_s", 900)))
        files = snapshot_files(str(thread.workspace), ctx.task.editable_files)
        parent_files = ctx.artifacts.get(parent_hash) if parent_hash else {}
        sub = episode.get("submitted") or {}
        hypothesis = (sub.get("hypothesis") or episode.get("final") or "").strip()[:300] or "(no hypothesis)"
        if files == parent_files or not files:
            ctx.ledger.append(trial=trial, thread=thread.tid, status="no_change", artifact_hash=None,
                              hypothesis=hypothesis, n_tool_calls=episode.get("n_tool_calls"),
                              episode_ok=episode.get("ok"))
            return Proposal(trial, thread.tid, None, parent_hash, hypothesis,
                            {"files": 0, "plus": 0, "minus": 0}, episode)
        h = ctx.artifacts.put(files)
        ds = diff_stat(parent_files, files)
        diff_path = ctx.run_dir / "artifacts" / f"{h}.diff"
        if not diff_path.exists():
            diff_path.write_text(unified_diff(parent_files, files))
        ctx.emit("proposal", agent, trial=trial, parent="orchestrator", hypothesis=hypothesis, diff_stat=ds,
                 artifact_hash=h)
        ctx.ledger.append(trial=trial, thread=thread.tid, status="pending", phase="proposed", artifact_hash=h,
                          parent_hash=parent_hash, hypothesis=hypothesis, diff_stat=ds,
                          submitted=bool(sub), n_tool_calls=episode.get("n_tool_calls"),
                          episode_ok=episode.get("ok"), web=episode.get("web"),
                          retrieval_assisted=bool((episode.get("web") or {}).get("calls")))
        return Proposal(trial, thread.tid, h, parent_hash, hypothesis, ds, episode)

    def evaluate(self, ctx, trial: int, artifact_hash: str, *, seeds: list[int], lane: int | None,
                 agent: str = "orchestrator", level: str | None = None, timeout_s: float = 7200) -> dict:
        """Submit to the sealed evaluator and wait. Idempotent per (trial, artifact, seeds) via the ledger."""
        job_id = None
        seeds = list(seeds)
        for r in ctx.ledger.records():
            if (r.get("trial") == trial and r.get("job_id") and r.get("artifact_hash") == artifact_hash
                    and r.get("seeds") == seeds):
                job_id = r["job_id"]
        if job_id is None:
            kw: dict[str, Any] = dict(seeds=seeds, lane=lane, gen=ctx.gen, trial=trial)
            if level is not None:
                kw["level"] = level
            # evald emits eval_submit itself (with the lane it actually assigned); a client without that
            # ability (test doubles, local CPU evaluators) gets it from the orchestrator instead.
            evald_emits = accepts_kw(ctx.evald.submit, "emit_submit")
            job_id = submit_compat(ctx.evald, ctx.run, ctx.task.name,
                                   str(ctx.run_dir / "artifacts" / artifact_hash), kw)
            ctx.ledger.append(trial=trial, status="pending", phase="submitted", artifact_hash=artifact_hash,
                              job_id=job_id, seeds=seeds, lane=lane)
            if not evald_emits:
                ctx.emit("eval_submit", "orchestrator", trial=trial, artifact_hash=artifact_hash,
                         lane=lane, seeds=seeds)
        result = ctx.evald.result(job_id, wait=True, timeout_s=timeout_s)
        result = dict(result or {})
        result.setdefault("status", "crash")
        result.setdefault("value", None)
        result.setdefault("audit", {"level": "L0", "flags": []})
        result["job_id"] = job_id
        return result

    def decide(self, ctx, thread: ResearchThread, trial: int, artifact_hash: str | None, result: dict | None,
               verdict: str, status: str, reason: str, *, agent: str = "orchestrator") -> None:
        """Record the verdict; move the thread's best on adopt; restore best into the workspace otherwise."""
        before = thread.best_value
        value = (result or {}).get("value")
        if verdict == "adopt" and artifact_hash:
            thread.best_hash, thread.best_value = artifact_hash, value
        thread.n_trials += 1
        feedback = ctx.task.visible_feedback(result) if result else "no evaluation (no change)"
        ctx.ledger.append(trial=trial, thread=thread.tid, status=status, artifact_hash=artifact_hash,
                          dev=value, eval_status=(result or {}).get("status"), feedback=feedback,
                          wall_s=(result or {}).get("wall_s"), verdict=verdict, reason=reason)
        if artifact_hash:
            ctx.emit("decision", agent, trial=trial, artifact_hash=artifact_hash, verdict=verdict,
                     best_before=before, best_after=thread.best_value, reason=reason)
        if thread.best_hash:
            ctx.artifacts.checkout(thread.best_hash, str(thread.workspace))
        ctx.save_state(**{f"thread:{thread.tid}": {"best_hash": thread.best_hash, "best_value": thread.best_value,
                                                   "n_trials": thread.n_trials}})

    def recover_pending(self, ctx, threads: list[ResearchThread], seeds: list[int] | None = None) -> int:
        """Finish trials left pending by a crash or pause. Returns how many were recovered."""
        by_tid = {t.tid: t for t in threads}
        n = 0
        for trial, r in sorted(ctx.ledger.trials().items()):
            if r.get("status") != "pending" or r.get("thread") not in by_tid:
                continue
            th = by_tid[r["thread"]]
            h = r.get("artifact_hash")
            if not h:  # died during the episode: nothing to evaluate
                ctx.ledger.append(trial=trial, thread=th.tid, status="crash", reason="interrupted during episode")
                continue
            res = self.evaluate(ctx, trial, h, seeds=r.get("seeds") or list(seeds or ctx.task.dev_seeds),
                                lane=th.lane)
            verdict, status, reason = self.judge(ctx, th, trial, h, res)
            self.decide(ctx, th, trial, h, res, verdict, status, reason + " (recovered)")
            n += 1
        return n

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def parallel(fn: Callable[[ResearchThread], Any], threads: list[ResearchThread]) -> list[Any]:
        if len(threads) == 1:
            return [fn(threads[0])]
        with ThreadPoolExecutor(max_workers=len(threads)) as ex:
            return list(ex.map(fn, threads))

    def finish(self, ctx, threads: list[ResearchThread]) -> dict:
        """Emit run_end and return the policy summary."""
        best = None
        for t in threads:
            if ctx.task.better(t.best_value, best):
                best = t.best_value
        recs = [r for r in ctx.ledger.finished() if r.get("status") not in ("baseline", "no_change")]
        base = [r for r in ctx.ledger.finished("orchestrator") if r.get("status") == "baseline"]
        s0 = base[-1].get("dev") if base else None
        vals = [r.get("dev") if r.get("eval_status") == "ok" else None for r in recs]
        sign = 1 if ctx.task.lower_is_better else -1
        na = None
        s_ref = self.cfg.get("s_ref")
        if s0 is not None and s_ref is not None:
            na = nauc(sign * s0, [None if v is None else sign * v for v in vals], sign * s_ref)
        summary = {"best_value": best, "n_trials": len(recs), "nAUC": na, "baseline": s0,
                   "paused": ctx.paused(), "policy": self.name}
        if not ctx.paused():
            ctx.emit("run_end", "orchestrator", best_value=best, n_trials=len(recs), nAUC=na)
        ctx.save_state(summary=summary)
        return summary


# ---------------------------------------------------------------------------
def accepts_kw(fn, name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()) and \
        name in getattr(fn, "__rsi_kwargs__", ())


def submit_compat(evald, run: str, task: str, artifact_dir: str, kw: dict) -> str:
    """Call EvalClient.submit with only the keyword args its signature accepts."""
    kw = {k: v for k, v in kw.items() if k in ("seeds", "lane", "gen", "trial") or accepts_kw(evald.submit, k)}
    return evald.submit(run, task, artifact_dir, **kw)


def prompt_summary(prompt: str, n: int = 200) -> str:
    first = " ".join(prompt.split())
    return first[: n - 1] + "…" if len(first) > n else first


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
