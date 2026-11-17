"""OpenCode adapter against a stub `opencode` binary that mimics `run --format json` and `export`.

The real OpenCode binary is only exercised on the cluster (DEPLOY.md step "OpenCode smoke");
this test pins the adapter's contract: config written, sandbox env, NDJSON parsed into
tool_call / subagent events, exports saved, lab submission picked up.
"""

import json
import os
import stat
import sys

from rsi import events
from rsi.harness.opencode import OpenCodeHarness, iter_parts, tool_name

STUB = r'''#!PYTHON
import json, os, sys, time
args = sys.argv[1:]
def now(): return int(time.time() * 1000)
if args[0] == "export":
    sid = args[1]
    if sid == "ses_child":
        parts = [{"type": "tool", "tool": "read", "callID": "c9",
                  "state": {"status": "completed", "input": {"filePath": "/work/train.py"}, "output": "x",
                            "time": {"start": now() - 50, "end": now()}}},
                 {"type": "text", "text": "child done"}]
    else:
        parts = [{"type": "tool", "tool": "task", "callID": "c3",
                  "state": {"status": "completed", "input": {"description": "survey", "prompt": "find ideas",
                            "subagent_type": "explore"}, "metadata": {"sessionId": "ses_child"}, "output": "ideas"}}]
    print(json.dumps({"info": {"id": sid}, "messages": [{"info": {"role": "assistant"}, "parts": parts}]}))
    sys.exit(0)
assert args[:3] == ["run", "--format", "json"], args
conf = json.load(open(os.environ["OPENCODE_CONFIG"]))
assert conf["provider"]["rsi"]["options"]["headers"]["X-RSI-Agent"] == "thread-0"
assert os.environ["RSI_GATEWAY_KEY"] == "agent-key-123"
lab = conf["mcp"]["lab"]["command"][-1]
def ev(t, part): print(json.dumps({"type": t, "timestamp": now(), "sessionID": "ses_main", "part": part}), flush=True)
ev("step_start", {"type": "step-start"})
ev("tool_use", {"type": "tool", "tool": "bash", "callID": "c1",
                "state": {"status": "completed", "input": {"command": "ls"}, "output": "train.py",
                          "time": {"start": now() - 20, "end": now()}}})
ev("tool_use", {"type": "tool", "tool": "task", "callID": "c3",
                "state": {"status": "running", "input": {"description": "survey", "prompt": "find ideas",
                          "subagent_type": "explore"}, "metadata": {"sessionId": "ses_child"}}})
ev("tool_use", {"type": "tool", "tool": "task", "callID": "c3",
                "state": {"status": "completed", "input": {"description": "survey", "prompt": "find ideas",
                          "subagent_type": "explore"}, "metadata": {"sessionId": "ses_child"},
                          "output": "ideas: relu2", "time": {"start": now() - 500, "end": now()}}})
s = open("train.py").read().replace("9e-4", "1.2e-3")
open("train.py", "w").write(s)
ev("tool_use", {"type": "tool", "tool": "edit", "callID": "c4",
                "state": {"status": "completed", "input": {"filePath": "/work/train.py", "oldString": "9e-4",
                          "newString": "1.2e-3"}, "output": "", "time": {"start": now() - 5, "end": now()}}})
os.makedirs(os.path.join(lab, "outbox"), exist_ok=True)
json.dump({"hypothesis": "lr 9e-4 -> 1.2e-3"}, open(os.path.join(lab, "outbox", "submit.json"), "w"))
ev("tool_use", {"type": "tool", "tool": "lab_submit_eval", "callID": "c5",
                "state": {"status": "completed", "input": {"hypothesis": "lr 9e-4 -> 1.2e-3"}, "output": "Submitted.",
                          "time": {"start": now() - 5, "end": now()}}})
ev("tool_use", {"type": "tool", "tool": "ddg_search", "callID": "c6",
                "state": {"status": "error", "input": {"query": "muon"}, "error": "net",
                          "time": {"start": now() - 5, "end": now()}}})
ev("text", {"type": "text", "text": "Raised the learning rate and submitted."})
ev("step_finish", {"type": "step-finish", "tokens": {"input": 1200, "output": 300}})
'''


def test_opencode_adapter_with_stub(tmp_path):
    stub = tmp_path / "opencode"
    stub.write_text(STUB.replace("#!PYTHON", "#!" + sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    run = "oc-run"
    events.register_run(run, str(tmp_path / "runs" / run / "events.jsonl"))
    events.emit(run, "run_start", "orchestrator", policy="p", task="t", improver={"model": "m", "scaffold_hash": None},
                budget={"trials": 1, "wall_s": None}, config_hash="x")
    td = tmp_path / "runs" / run / "threads" / "thread-0"
    ws = td / "workspace"
    ws.mkdir(parents=True)
    (ws / "train.py").write_text("LEARNING_RATE = 9e-4\n")
    (ws / "opencode.json").write_text("{}")  # agent-planted config must be removed
    keyf = tmp_path / "agent_token"
    keyf.write_text("agent-key-123\n")
    scaffold = tmp_path / "scaffold"
    scaffold.mkdir()
    (scaffold / "lessons.md").write_text("Prefer one change per trial.")
    h = OpenCodeHarness({"sandbox": "none", "opencode_bin": str(stub), "gateway_url": "http://127.0.0.1:9/v1",
                         "api_key_file": str(keyf), "python": sys.executable, "platform_dir": os.getcwd(),
                         "ddg_cmd": ["duckduckgo-mcp-server"], "fetch_cmd": ["mcp-server-fetch"],
                         "trajectories_dir": str(tmp_path / "runs" / run / "trajectories" / "harness")})
    out = h.run_episode(run=run, agent="thread-0", gen=0, trial=1, workspace=ws, prompt="improve it",
                        model="improver", scaffold_dir=scaffold, budget_s=60)
    assert out["ok"], out
    assert out["submitted"]["hypothesis"] == "lr 9e-4 -> 1.2e-3"
    assert out["final"] == "Raised the learning rate and submitted."
    assert out["n_subagents"] == 1
    assert "1.2e-3" in (ws / "train.py").read_text()
    assert not (ws / "opencode.json").exists()
    conf = json.loads((td / "episode_conf" / "opencode.json").read_text())
    assert conf["mcp"]["ddg"]["command"][2] == "rsi.mcp_lab.filter_proxy"
    assert conf["instructions"] and conf["permission"]["webfetch"] == "deny"
    assert "agent-key-123" not in json.dumps(conf)

    evs = events.validate_file(str(tmp_path / "runs" / run / "events.jsonl"))
    tc = [(e["agent"], e["tool"], e["ok"]) for e in evs if e["ev"] == "tool_call"]
    assert ("thread-0", "bash", True) in tc
    assert ("thread-0", "edit", True) in tc
    assert ("thread-0", "mcp:lab.submit_eval", True) in tc
    assert ("thread-0", "web_search", False) in tc
    assert ("thread-0/sub-0", "read", True) in tc          # from the child session export
    spawns = [e for e in evs if e["ev"] == "agent_spawn"]
    assert [(e["agent"], e["role"], e["parent"]) for e in spawns] == [
        ("thread-0", "researcher", "orchestrator"), ("thread-0/sub-0", "subagent", "thread-0")]
    ends = [e["agent"] for e in evs if e["ev"] == "agent_end"]
    assert ends == ["thread-0/sub-0", "thread-0"]
    trajd = tmp_path / "runs" / run / "trajectories" / "harness" / "thread-0"
    assert (trajd / "trial-00001.ndjson").exists()
    assert (trajd / "trial-00001.session-ses_main.json").exists()
    assert (trajd / "trial-00001.sub-ses_child.json").exists()


def test_tool_name_mapping_and_part_walk():
    assert tool_name("bash") == "bash" and tool_name("write") == "edit" and tool_name("grep") == "read"
    assert tool_name("lab_history") == "mcp:lab.history"
    assert tool_name("ddg_search") == "web_search" and tool_name("fetch_fetch") == "fetch"
    assert tool_name("foo_bar") == "mcp:foo.bar"
    parts = list(iter_parts({"a": [{"part": {"type": "tool", "tool": "x", "state": {}}}, {"type": "text", "text": "t"}]}))
    assert [p["type"] for p in parts] == ["tool", "text"]
