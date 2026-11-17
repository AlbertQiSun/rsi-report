"""bundle/cc_toolcall_gate.py: task setups, checks (with reference solutions) and the summary, with a fake harness."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bundle"))

import cc_toolcall_gate as G  # noqa: E402
from rsi import events  # noqa: E402


class FakeHarness:
    """Solves even-numbered episodes with the reference solution; odd ones do nothing (one invalid call)."""

    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s, parent="o"):
        i = trial - 1
        task = G.TASKS[i % len(G.TASKS)]
        assert task.prompt == prompt
        lab = pathlib.Path(workspace).parent / "lab"
        assert (lab / "card.md").read_text() == task.card
        if i % 2 == 0:
            task.solve(pathlib.Path(workspace))
            return {"ok": True, "submitted": {"hypothesis": "x"}, "n_tool_calls": 4, "n_tool_errors": 0,
                    "n_invalid_calls": 0, "n_subagents": 1 if task.name == "subagent" else 0,
                    "result": {"num_turns": 4}, "tools_used": {"Edit": 1}}
        return {"ok": True, "submitted": None, "n_tool_calls": 2, "n_tool_errors": 1, "n_invalid_calls": 1}


def test_every_task_check_fails_untouched_and_passes_with_reference(tmp_path):
    for k, t in enumerate(G.TASKS):
        ws = G.setup(t, tmp_path / f"a{k}")
        assert not G.passed(t, ws, {"submitted": {"h": 1}}), t.name
        t.solve(ws)
        assert G.passed(t, ws, {"submitted": {"h": 1}}), t.name
        if t.needs_submit:
            assert not G.passed(t, ws, {"submitted": None}), t.name
    assert len(G.TASKS) == 10 and len({t.name for t in G.TASKS}) == 10


def test_run_gate_summary(tmp_path):
    s = G.run_gate(FakeHarness(), run="r", model="improver", root=tmp_path, n=10, parallel=3)
    assert s["n_episodes"] == 10 and s["n_tool_calls"] == 5 * 4 + 5 * 2
    assert s["call_validity_rate"] == round(1 - 5 / 30, 4)
    assert s["episode_completion_rate"] == 0.5 and not s["gate_pass"]
    assert [e["passed"] for e in s["per_episode"]] == [i % 2 == 0 for i in range(10)]
    json.dumps(s)


def test_llm_stats(tmp_path):
    p = tmp_path / "events.jsonl"
    events.register_run("g", str(p))
    for agent, pt in (("gate-00", 11000), ("gate-00", 12000), ("gate-01", 10500)):
        events.emit("g", "llm_call", agent, model="improver", prompt_tokens=pt, completion_tokens=5, latency_s=0.1)
    st = G.llm_stats(str(p))
    assert st["n_llm_calls"] == 3 and st["first_prompt_tokens"]["min"] == 10500 and st["max_prompt_tokens"] == 12000
    events.emit("g", "run_end", "orchestrator", best_value=None, n_trials=10, nAUC=None, gate_pass=False,
                call_validity_rate=0.9, episode_completion_rate=0.5)
    for e in events.read_events(str(p)):
        events.validate(e)
