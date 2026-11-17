"""The local smoke run (real gateway + µAR CPU task + mini harness + fake model) stays green."""

import json

from rsi import events, smoke


def test_smoke_run(tmp_path):
    out = tmp_path / "events.jsonl"
    assert smoke.main(["--workdir", str(tmp_path / "w"), "--out", str(out), "--trials", "2", "--threads", "1",
                       "--run", "smoke-test"]) == 0
    evs = events.validate_file(str(out))
    kinds = {e["ev"] for e in evs}
    assert {"run_start", "llm_call", "tool_call", "proposal", "eval_submit", "eval_result", "decision",
            "run_end"} <= kinds
    llm = [e for e in evs if e["ev"] == "llm_call"]
    assert llm and all(e["agent"] == "thread-0" for e in llm)   # tagged by the gateway from X-RSI-* headers
    gw = tmp_path / "w" / "runs" / "smoke-test" / "trajectories" / "gateway" / "thread-0.jsonl"
    rec = json.loads(gw.read_text().splitlines()[0])
    assert rec  # full request/response kept for LoRA data
