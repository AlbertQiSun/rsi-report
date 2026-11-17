"""bundle/toolcall_test.py (gate G1b) against the fake server (owner: worker C)."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("toolcall_test", ROOT / "bundle" / "toolcall_test.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)


def test_validator():
    ok = {"function": {"name": "edit", "arguments": json.dumps(
        {"filePath": "/w/a.py", "oldString": "a", "newString": "b"})}}
    assert tt.validate_call(ok) == []
    assert tt.validate_call({"function": {"name": "edit", "arguments": "{bad"}}) == ["bad_json"]
    assert tt.validate_call({"function": {"name": "nope", "arguments": "{}"}}) == ["unknown_tool"]
    assert tt.validate_call({"function": {"name": "read", "arguments": "[]"}}) == ["not_object"]
    assert tt.validate_call({"function": {"name": "read", "arguments": "{}"}}) == ["missing_required"]
    assert tt.validate_call({"function": {"name": "read", "arguments": '{"filePath": "/a", "limit": "x"}'}}) == \
        ["wrong_type"]
    assert tt.validate_call({"function": {"name": "bash", "arguments": '{"command": "ls", "cwd": "/"}'}}) == \
        ["unknown_param"]
    assert tt.validate_call({"function": {"name": "web_search", "arguments": '{"query": "x", "numResults": true}'}}) \
        == ["wrong_type"]
    assert tt.validate_call({"function": {"name": "", "arguments": "{}"}}) == ["empty_name"]


def test_episode_set():
    eps = tt.build_episodes()
    assert len(eps) == 30
    assert {s["tool"] for e in eps for s in e["steps"]} == set(tt.SCHEMAS)
    assert sum(len(e["steps"]) > 1 for e in eps) >= 14  # multi-turn
    for e in eps:  # the scripted "correct" calls are themselves schema-valid
        for s in e["steps"]:
            assert tt.validate_call({"function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}) == []


def test_fake_run_summary(tmp_path, capsys):
    out = tmp_path / "g1b.json"
    rc = tt.main(["--fake", "--runs-root", str(tmp_path / "runs"), "--out", str(out), "--run", "g1b-test"])
    s = json.loads(out.read_text())
    assert rc == 0 and s["pass"] is True
    assert s["n_episodes"] == 30 and s["n_tool_calls"] == 60
    errs = s["schema_errors"]
    for k in ("bad_json", "missing_required", "unknown_tool", "wrong_type"):
        assert errs[k] == 1
    assert errs["no_tool_call"] == 3 and errs["tool_call_in_content"] == 2
    assert s["call_validity_rate"] == round(56 / 60, 4)
    assert s["episode_completion_rate"] == round(29 / 30, 4)
    assert s["clean_completion_rate"] == round(24 / 30, 4)
    assert set(s["by_mode"]) == {"stream", "nonstream"}
    assert s["latency_s"]["n"] == s["n_requests"]
    traj = tmp_path / "runs" / "g1b-test" / "trajectories" / "gateway"
    assert len(list(traj.glob("toolcall__ep-*.jsonl"))) == 30
    from rsi import events
    evs = events.read_events(str(tmp_path / "runs" / "g1b-test" / "events.jsonl"))
    assert len(evs) == s["n_requests"] and all(e["ev"] == "llm_call" for e in evs)


def test_gate_fails_when_strict(tmp_path):
    rc = tt.main(["--fake", "--runs-root", str(tmp_path / "runs"), "--min-validity", "0.99", "--episodes", "10"])
    assert rc == 1
