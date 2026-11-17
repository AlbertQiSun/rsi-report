"""karpathy_greedy: Karpathy autoresearch loop (build plan §4.1), with evaluation and
keep/discard moved into the trusted orchestrator.

    best = evaluate(start)
    while budget left:
        sub = harness.run(wt, PROGRAM + ledger_tail)    # agent may only edit the editable files
        r = evaluate(sub, seeds=[s0])                   # sealed evaluator; hard wall-clock limit (D5)
        keep if r.ok and r better than best else discard / crash; workspace reset to best

K threads run independently in parallel, thread i on eval lane lanes[i % len(lanes)].
Config keys: threads (1), seeds (task.dev_seeds), episode_s, program (extra rules text).
"""

from __future__ import annotations

from rsi.policies.base import Policy, ResearchThread

PROGRAM = """You are an autonomous ML researcher in a fixed-budget experiment loop (Karpathy autoresearch).

Goal: make {metric} {direction} for the artifact in your working directory.

Rules
- Edit only: {files}. Do not create other files that the artifact depends on.
- You cannot run training or evaluation yourself. A sealed evaluator will train and score your
  edited artifact after you finish, under a fixed wall-clock budget.
- Make ONE focused change per attempt (a single idea you can name). Simpler is better: a small
  gain from deleting code beats a small gain from adding complexity.
- When the edit is done, call the lab tool `submit_eval` with a one-sentence hypothesis, then stop.
- Use `history` / `get_results` (lab tools) to see past attempts and their scores.
{extra}
Task card
---------
{card}

Current best {metric}: {best}
Recent attempts in this thread (trial, artifact, {metric}, status, description):
{tail}
"""


class KarpathyGreedy(Policy):
    name = "karpathy_greedy"

    def build_prompt(self, ctx, thread: ResearchThread) -> str:
        task = ctx.task
        best = f"{thread.best_value:.6f}" if isinstance(thread.best_value, (int, float)) else "unknown"
        return PROGRAM.format(
            metric=task.metric, direction="lower" if task.lower_is_better else "higher",
            files=", ".join(task.editable_files), card=task.task_card(), best=best,
            tail=ctx.ledger.tail_text(thread.tid, n=int(self.cfg.get("tail", 8))),
            extra=(self.cfg.get("program_extra", "") + "\n") if self.cfg.get("program_extra") else "")

    def step(self, ctx, thread: ResearchThread) -> dict:
        trial = self.new_trial(ctx)
        self.write_lab_view(ctx, thread)
        prop = self.propose(ctx, thread, self.build_prompt(ctx, thread), trial,
                            budget_s=self.cfg.get("episode_s"))
        if prop.artifact_hash is None:
            self.decide(ctx, thread, trial, None, None, "reject", "no_change", "episode made no edit")
            return {"trial": trial, "status": "no_change"}
        res = self.evaluate(ctx, trial, prop.artifact_hash, seeds=self.seeds(ctx), lane=thread.lane)
        verdict, status, reason = self.judge(ctx, thread, trial, prop.artifact_hash, res)
        self.decide(ctx, thread, trial, prop.artifact_hash, res, verdict, status, reason)
        return {"trial": trial, "status": status, "value": res.get("value")}

    def seeds(self, ctx) -> list[int]:
        return list(self.cfg.get("seeds", ctx.task.dev_seeds))

    def thread_loop(self, ctx, thread: ResearchThread) -> None:
        max_no_change = int(self.cfg.get("max_consecutive_no_change", 5))
        streak = 0
        per_thread = self.cfg.get("trials_per_thread")
        while not self.should_stop(ctx):
            if per_thread is not None and self.n_finished(ctx, thread.tid) >= int(per_thread):
                break
            out = self.step(ctx, thread)
            streak = streak + 1 if out["status"] == "no_change" else 0
            if streak >= max_no_change:
                ctx.emit("agent_end", thread.tid, reason=f"{streak} consecutive episodes without an edit")
                break

    def run(self, ctx) -> dict:
        threads = self.make_threads(ctx, int(self.cfg.get("threads", 1)))
        self.recover_pending(ctx, threads, self.seeds(ctx))
        self.parallel(lambda th: self.thread_loop(ctx, th), threads)
        return self.finish(ctx, threads)


POLICY = KarpathyGreedy
