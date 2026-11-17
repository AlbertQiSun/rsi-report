import json
import os
import subprocess
import sys
import threading

import pytest

from rsi import events
from rsi.ledger import Ledger, best_so_far_curve, nauc
from rsi.runs import ArtifactStore, artifact_hash, diff_stat


def test_emit_and_validate(tmp_path):
    p = tmp_path / "r1" / "events.jsonl"
    events.register_run("r1", str(p))
    events.emit("r1", "run_start", "orchestrator", policy="karpathy_greedy", task="t",
                improver={"model": "m", "scaffold_hash": None}, budget={"trials": 3, "wall_s": None},
                config_hash="abc")
    events.emit("r1", "agent_spawn", "thread-0", parent="orchestrator", role="researcher", model="m")
    events.emit("r1", "tool_call", "thread-0", trial=1, tool="bash",
                arg_summary="curl -H 'Authorization: Bearer sk-abcdefghijklmnop'", ok=True, latency_s=0.1)
    events.emit("r1", "eval_result", "evald", trial=1, artifact_hash="h", metric="val_bpb", value=1.0,
                seeds=[0], wall_s=3.0, status="ok", audit={"level": "L1", "flags": []})
    evs = events.validate_file(str(p))
    assert [e["ev"] for e in evs] == ["run_start", "agent_spawn", "tool_call", "eval_result"]
    assert "sk-abcdefghijklmnop" not in p.read_text()
    assert all(set(events.COMMON_FIELDS) <= set(e) for e in evs)


def test_schema_rejects_bad_events(tmp_path):
    log = events.EventLog(str(tmp_path / "e.jsonl"), "r")
    with pytest.raises(events.SchemaError):
        log.emit("tool_call", "a", tool="bash")  # missing fields
    with pytest.raises(events.SchemaError):
        log.emit("nope", "a")
    with pytest.raises(events.SchemaError):
        log.emit("eval_result", "evald", artifact_hash="h", metric="m", value=1, seeds=[0], wall_s=1,
                 status="weird", audit={"level": "L0", "flags": []})
    with pytest.raises(events.SchemaError):
        log.emit("agent_spawn", "a", role="wizard", model="m")


def test_long_fields_are_clipped(tmp_path):
    log = events.EventLog(str(tmp_path / "e.jsonl"), "r")
    e = log.emit("message", "a", to="b", summary="x" * 500, tokens=3)
    assert len(e["summary"]) == 200


def test_concurrent_appends_are_whole_lines(tmp_path):
    p = str(tmp_path / "e.jsonl")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from rsi import events\n"
        "log = events.EventLog(%r, 'r')\n"
        "for i in range(200): log.emit('message', 'p', to='q', summary='y'*150, tokens=i)\n"
    ) % (os.getcwd(), p)
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(4)]
    log = events.EventLog(p, "r")
    ts = [threading.Thread(target=lambda: [log.emit("message", "t", to="q", summary="z", tokens=1)
                                           for _ in range(100)]) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for pr in procs:
        assert pr.wait() == 0
    lines = open(p).read().splitlines()
    assert len(lines) == 4 * 200 + 300
    for line in lines:
        json.loads(line)


def test_ledger_and_nauc(tmp_path):
    led = Ledger(str(tmp_path / "ledger.jsonl"))
    led.append(trial=0, thread="orchestrator", status="baseline", dev=1.0)
    led.append(trial=1, thread="thread-0", status="pending", artifact_hash="a")
    led.append(trial=1, thread="thread-0", status="keep", dev=0.9)
    led.append(trial=2, thread="thread-0", status="pending")
    assert led.trials()[1]["artifact_hash"] == "a"
    assert [r["trial"] for r in led.finished("thread-0")] == [1]
    assert "0.900000" in led.tail_text("thread-0")
    assert best_so_far_curve([3, None, 2, 4]) == [3, 3, 2, 2]
    # s0=1, s_ref=0: values 0.5 then 0.75 -> g = [0.5, 0.5]
    assert nauc(1.0, [0.5, 0.75], 0.0) == pytest.approx(0.5)
    assert nauc(1.0, [0.5], None) is None


def test_artifact_store_roundtrip(tmp_path):
    st = ArtifactStore(str(tmp_path / "a"))
    h = st.put({"train.py": b"x = 1\n"})
    assert h == artifact_hash({"train.py": b"x = 1\n"})
    assert st.put({"train.py": b"x = 1\n"}) == h
    ws = tmp_path / "ws"
    ws.mkdir()
    st.checkout(h, str(ws))
    assert (ws / "train.py").read_text() == "x = 1\n"
    assert diff_stat({"train.py": b"a\nb\n"}, {"train.py": b"a\nc\nd\n"}) == {"files": 1, "plus": 2, "minus": 1}


def test_read_jsonl_ignores_torn_tail(tmp_path):
    from rsi.runs import read_jsonl

    p = tmp_path / "l.jsonl"
    p.write_text('{"trial": 1}\n{"trial": 2, "sta')
    assert read_jsonl(str(p)) == [{"trial": 1}]
    p.write_text('{"trial": 1}\n{"trial": 2}')  # complete last line without newline
    assert read_jsonl(str(p)) == [{"trial": 1}, {"trial": 2}]
