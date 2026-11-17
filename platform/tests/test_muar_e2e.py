"""End-to-end on CPU: karpathy_greedy x MuarTask x LocalEvalClient -> ledger -> SFT builder -> analysis."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi import events  # noqa: E402
from rsi.context import RunContext  # noqa: E402
from rsi.tasks.base import get_task  # noqa: E402

TASK = "muar/train-teacher-0"


class ConfigEditingHarness:
    """Edits config.json deterministically and records gateway-style trajectories (no model)."""

    def __init__(self, runs_root: pathlib.Path):
        self.runs_root = runs_root
        self.moves = [dict(optimizer="adamw"), dict(lr=10.0), dict(schedule="cosine", warmup_frac=0.05),
                      dict(width=4096)]

    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s,
                    parent="orchestrator"):
        assert "TASK muar/train-teacher-0" in prompt and "s_ref" not in prompt
        p = pathlib.Path(workspace) / "config.json"
        cfg = json.loads(p.read_text())
        cfg.update(self.moves[(trial - 1) % len(self.moves)])
        p.write_text(json.dumps(cfg))
        answer = f"I set {sorted(self.moves[(trial - 1) % len(self.moves)])} and submitted."
        rec = {"t": float(trial), "run": run, "agent": agent, "gen": gen, "trial": trial, "status": "ok",
               "http_status": 200, "request": {"model": model, "messages": [{"role": "user", "content": prompt}]},
               "response": {"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}}
        tdir = self.runs_root / run / "trajectories" / "gateway"
        tdir.mkdir(parents=True, exist_ok=True)
        with open(tdir / f"{agent}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        events.emit(run, "tool_call", agent, parent=parent, gen=gen, trial=trial, tool="edit",
                    arg_summary="config.json", ok=True, latency_s=0.0)
        return {"final": answer, "n_tool_calls": 1, "trajectory": "", "ok": True,
                "submitted": {"hypothesis": answer}}


def test_policy_on_muar_end_to_end(tmp_path, monkeypatch):
    from rsi.policies.karpathy_greedy import KarpathyGreedy
    from rsi.tasks.muar.runner import LocalEvalClient
    from rsi.update import analysis as A
    from rsi.update import build_sft_data

    monkeypatch.setenv("RSI_MUAR_START_DIR", str(tmp_path / "start"))
    runs = tmp_path / "runs"
    task = get_task(TASK)
    run = "muar-e2e-0"
    with LocalEvalClient(workers=2) as evald:
        ctx = RunContext.create(run=run, runs_root=runs, gen=0, task=task, evald=evald,
                                harness=ConfigEditingHarness(runs), model="improver", scaffold_dir=None,
                                budget={"trials": 4, "episode_s": 60}, lanes=[0])
        (ctx.run_dir / "config.json").write_text(json.dumps({"task": TASK, "seed": 0}))
        ctx.emit("run_start", "orchestrator", policy="karpathy_greedy", task=TASK,
                 improver={"model": "improver", "scaffold_hash": "none"}, budget={"trials": 4, "wall_s": None},
                 config_hash="t")
        pol = KarpathyGreedy({"threads": 1, "trials_per_thread": 4})
        summary = pol.run(ctx)
    evs = events.validate_file(str(ctx.run_dir / "events.jsonl"))
    kinds = [e["ev"] for e in evs]
    assert "eval_result" in kinds and "decision" in kinds
    trials = ctx.ledger.finished()
    statuses = {r["trial"]: (r["status"], r.get("eval_status")) for r in trials}
    # the width=4096 move is illegal (schema), lr=10 is out of range: both must be rejected as audit_fail
    assert any(es == "audit_fail" for _, es in statuses.values())
    feedback = [r.get("feedback", "") for r in trials]
    assert all("sealed" not in f for f in feedback)
    assert summary["n_trials"] >= 3
    m = build_sft_data([str(ctx.run_dir)], out=str(tmp_path / "sft"), min_accepted=1, per_task_cap=None)
    assert m["leak_check"]["ok"]
    curves, s0 = A.curves_from_runs([str(ctx.run_dir)])
    assert TASK in curves and len(curves[TASK]["0"]) >= 3
