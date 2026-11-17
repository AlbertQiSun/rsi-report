import io
import json
import os

import pytest

from rsi.harness.sandbox import SandboxError, SandboxSpec, bwrap_argv, check_binds
from rsi.mcp_lab import filter_proxy
from rsi.mcp_lab.server import Lab, serve

R = "/scratch/user/rsi"


def _forbidden():
    return [f"{R}/data/devval", f"{R}/data/sealed", f"{R}/run", f"{R}/runs", "/home/u", "/home/u/.ssh",
            "/scratch/user/otherproj"]


def _binds(argv):
    out = []
    for i, a in enumerate(argv):
        if a in ("--bind", "--ro-bind", "--ro-bind-try", "--dev-bind"):
            out.append((argv[i + 1], argv[i + 2]))
    return out


def test_bwrap_argv_hides_sealed_and_ssh(tmp_path):
    td = f"{R}/runs/r1/threads/thread-0"
    spec = SandboxSpec(workspace=f"{td}/workspace", lab=str(tmp_path / "lab"), home=f"{td}/home",
                       config_dir=f"{td}/episode_conf",
                       ro_paths=["/scratch/user/miniconda3/envs/rsi", f"{R}/code/rsi-mini", f"{R}/tools/opencode"],
                       forbidden=_forbidden(), env={"X": "1"})
    argv = bwrap_argv(spec, ["opencode", "run"])
    srcs = [s for s, _ in _binds(argv)]
    for s in srcs:
        assert not s.startswith(f"{R}/data"), s
        assert ".ssh" not in s and not s.startswith("/home")
    assert "--unshare-all" in argv and "--share-net" in argv and "--clearenv" in argv
    assert ("--bind", f"{td}/workspace", "/work") == tuple(argv[argv.index(f"{td}/workspace") - 1:argv.index(f"{td}/workspace") + 2])
    assert argv[-2:] == ["opencode", "run"]
    i = argv.index("--setenv")
    assert "CUDA_VISIBLE_DEVICES" in argv[i:]


@pytest.mark.parametrize("bad", [R, f"{R}/data", "/scratch/user", "/home/u", "/home/u/.ssh/id",
                                 f"{R}/data/sealed/x", f"{R}/runs/other/threads/t/workspace", "/"])
def test_check_binds_rejects(bad):
    with pytest.raises(SandboxError):
        check_binds([bad], _forbidden(), allowed_inside=[f"{R}/runs/r1/threads/thread-0/workspace"])


def test_check_binds_allows_thread_dirs_and_tools():
    ws = f"{R}/runs/r1/threads/thread-0/workspace"
    check_binds([ws, f"{R}/envs/mcp", f"{R}/code/rsi-mini"], _forbidden(), allowed_inside=[ws])


def _rpc(lab, msgs):
    out = io.StringIO()
    serve(lab, io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_lab_mcp_protocol_and_submit(tmp_path):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "card.md").write_text("CARD")
    (lab_dir / "history.jsonl").write_text(json.dumps({"trial": 1, "value": 1.05, "status": "keep",
                                                       "hypothesis": "lr up"}) + "\n")
    (lab_dir / "best.json").write_text(json.dumps({"value": 1.05, "metric": "val_bpb"}))
    resp = _rpc(Lab(str(lab_dir)), [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_results", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "submit_eval",
                                                                     "arguments": {"hypothesis": "try GELU->ReLU2"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "submit_eval", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "task_card", "arguments": {}}},
    ])
    assert [r["id"] for r in resp] == [1, 2, 3, 4, 5, 6]
    assert resp[0]["result"]["protocolVersion"] == "2025-03-26"
    assert {t["name"] for t in resp[1]["result"]["tools"]} == {"submit_eval", "get_results", "history", "task_card"}
    assert "1.05" in resp[2]["result"]["content"][0]["text"]
    assert not resp[3]["result"]["isError"]
    assert json.loads((lab_dir / "outbox" / "submit.json").read_text())["hypothesis"] == "try GELU->ReLU2"
    assert resp[4]["result"]["isError"]
    assert resp[5]["result"]["content"][0]["text"] == "CARD"
    assert len((lab_dir / "outbox" / "calls.jsonl").read_text().splitlines()) == 4


def test_filter_proxy_blocks_answer_sources(tmp_path, monkeypatch):
    assert filter_proxy.blocked({"url": "https://github.com/karpathy/autoresearch/blob/master/train.py"},
                                filter_proxy.DEFAULT_BLOCKLIST)
    assert filter_proxy.blocked({"query": "optimized_from_vanilla nanochat"}, filter_proxy.DEFAULT_BLOCKLIST)
    assert filter_proxy.blocked({"query": "muon optimizer explained"}, filter_proxy.DEFAULT_BLOCKLIST) is None
    # end-to-end through a child "server" that echoes tools/call ids
    child = tmp_path / "child.py"
    child.write_text("import sys,json\nfor l in sys.stdin:\n m=json.loads(l)\n"
                     " print(json.dumps({'jsonrpc':'2.0','id':m.get('id'),'result':{'content':[{'type':'text','text':'child'}]}}),flush=True)\n")
    import subprocess
    import sys
    log = tmp_path / "web.jsonl"
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "muon"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "fetch", "arguments": {"url": "https://ensue-network.ai/lab/autoresearch"}}}]
    r = subprocess.run([sys.executable, "-m", "rsi.mcp_lab.filter_proxy", "--log", str(log), "--",
                        sys.executable, str(child)], input="\n".join(json.dumps(m) for m in msgs) + "\n",
                       capture_output=True, text=True, timeout=30, cwd=os.getcwd())
    outs = {json.loads(x)["id"]: json.loads(x) for x in r.stdout.splitlines()}
    assert outs[1]["result"]["content"][0]["text"] == "child"
    assert outs[2]["result"]["isError"] is True
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert [bool(x["blocked"]) for x in recs] == [False, True]
