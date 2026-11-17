"""CPU tests for rsi.gateway against tests/fake_openai_server.py (owner: worker C)."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fake_openai_server import FakeServer  # noqa: E402
from rsi import events  # noqa: E402
from rsi.gateway import GatewayClient, GatewayConfig, GatewayThread  # noqa: E402
from rsi.gateway.client import GatewayError  # noqa: E402
from rsi.gateway.ports import free_port, merge_ports, read_ports  # noqa: E402
from rsi.gateway.server import trajectory_path  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a shell command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}, "required": ["command"]}}}]


def _cfg(tmp_path, **kw) -> GatewayConfig:
    d = dict(runs_root=str(tmp_path / "runs"), max_retries=4, retry_deadline_s=5, backoff_base_s=0.01,
             backoff_cap_s=0.05, quiet=True)
    d.update(kw)
    return GatewayConfig(**d)


def _traj(tmp_path, run, agent):
    p = trajectory_path(str(tmp_path / "runs"), run, agent)
    return [json.loads(x) for x in open(p)]


def _events(tmp_path, run):
    evs = events.read_events(str(tmp_path / "runs" / run / "events.jsonl"))
    for e in evs:
        events.validate(e)
    return evs


def test_nonstream_records_trajectory_and_event(tmp_path):
    with FakeServer({"queue": [{"content": "hello there", "reasoning_content": "hmm"}]}) as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        c = GatewayClient(gw.url, token="")
        r = c.chat("improver", [{"role": "user", "content": "hi"}], run="r1", agent="thread-0", gen=2, trial=5,
                   temperature=0.7)
        assert r["choices"][0]["message"]["content"] == "hello there"
        assert fk.requests[0]["temperature"] == 0.7
    tr = _traj(tmp_path, "r1", "thread-0")
    assert len(tr) == 1
    rec = tr[0]
    assert rec["request"]["messages"][0]["content"] == "hi"
    assert rec["response"]["message"]["content"] == "hello there"
    assert rec["response"]["message"]["reasoning_content"] == "hmm"
    assert (rec["gen"], rec["trial"], rec["status"]) == (2, 5, "ok")
    ev = _events(tmp_path, "r1")
    assert [e["ev"] for e in ev] == ["llm_call"]
    assert ev[0]["model"] == "improver" and ev[0]["prompt_tokens"] > 0 and ev[0]["completion_tokens"] > 0
    assert ev[0]["agent"] == "thread-0" and ev[0]["gen"] == 2 and ev[0]["trial"] == 5
    assert "hello" not in json.dumps(ev[0])  # no content in events


def test_stream_tool_calls_reassembled(tmp_path):
    script = {"queue": [{"content": "let me look", "tool_calls": [
        {"name": "bash", "arguments": {"command": "ls -la && cat train.py | head -50"}},
        {"name": "read", "arguments": {"filePath": "/w/train.py"}}]}]}
    with FakeServer(script) as fk, GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        import http.client
        conn = http.client.HTTPConnection(gw.host, gw.port, timeout=10)
        body = {"model": "improver", "stream": True, "tools": TOOLS,
                "messages": [{"role": "user", "content": "go"}]}
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"X-RSI-Run": "r2", "X-RSI-Agent": "thread-0/sub-1", "X-RSI-Parent": "thread-0"})
        resp = conn.getresponse()
        raw = resp.read().decode()
        conn.close()
        assert fk.requests[0]["stream_options"]["include_usage"] is True  # gateway asks for usage
    chunks = [json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:") and "[DONE]" not in x]
    assert all(ch["choices"] for ch in chunks)  # usage-only chunk hidden from a client that did not ask
    assert raw.rstrip().endswith("data: [DONE]")
    rec = _traj(tmp_path, "r2", "thread-0/sub-1")[0]
    tcs = rec["response"]["message"]["tool_calls"]
    assert [t["function"]["name"] for t in tcs] == ["bash", "read"]
    assert json.loads(tcs[0]["function"]["arguments"])["command"].startswith("ls -la")
    assert rec["response"]["finish_reason"] == "tool_calls"
    assert rec["usage"]["completion_tokens"] > 0 and rec["ttft_s"] is not None
    ev = _events(tmp_path, "r2")[0]
    assert ev["parent"] == "thread-0" and ev["n_tool_calls"] == 2 and ev["stream"] is True
    assert os.path.basename(trajectory_path("x", "r2", "thread-0/sub-1")) == "thread-0__sub-1.jsonl"


def test_stream_client_usage_passthrough(tmp_path):
    with FakeServer({"queue": [{"content": "abc def"}]}) as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        r = GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}], stream=True,
                                                 run="r3", agent="a")
        assert r["choices"][0]["message"]["content"] == "abc def"
        assert r["usage"]["completion_tokens"] == 2


def test_retry_on_busy_then_ok(tmp_path):
    script = {"queue": [{"status": 503, "times": 2}, {"content": "finally"}]}
    with FakeServer(script) as fk, GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        r = GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}], run="r4", agent="a")
        assert r["choices"][0]["message"]["content"] == "finally"
        assert len(fk.requests) == 3
        assert gw.gw.stats["retries"] == 2
    assert _traj(tmp_path, "r4", "a")[0]["attempts"] == 3


def test_upstream_down_gives_503(tmp_path):
    dead = f"http://127.0.0.1:{free_port()}"
    with GatewayThread(_cfg(tmp_path, upstreams={"improver": dead}, max_retries=2, retry_deadline_s=1)) as gw:
        with pytest.raises(GatewayError) as ei:
            GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}], run="r5", agent="a")
        assert ei.value.status == 503
    rec = _traj(tmp_path, "r5", "a")[0]
    assert rec["status"] == "upstream_unavailable" and rec["attempts"] == 3
    assert _events(tmp_path, "r5")[0]["status"] == "upstream_unavailable"


def test_non_retryable_error_passthrough(tmp_path):
    with FakeServer({"queue": [{"status": 400, "error": "bad tools"}]}) as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        with pytest.raises(GatewayError) as ei:
            GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}], run="r6", agent="a")
        assert ei.value.status == 400 and len(fk.requests) == 1


def test_routing_adapters_and_ladder(tmp_path):
    with FakeServer({"default": {"content": "small"}}) as f9, FakeServer({"default": {"content": "big"}}) as f27, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": f9.base, "ladder-27b": f27.base})) as gw:
        c = GatewayClient(gw.url, token="")
        assert c.chat("improver@g1", [{"role": "user", "content": "x"}])["choices"][0]["message"]["content"] == "small"
        assert f9.requests[-1]["model"] == "improver@g1"  # passed through unchanged
        assert c.chat("ladder-27b", [{"role": "user", "content": "x"}])["choices"][0]["message"]["content"] == "big"
        with pytest.raises(GatewayError) as ei:
            c.chat("gpt-4", [{"role": "user", "content": "x"}])
        assert ei.value.status == 404
        assert {"improver", "ladder-27b"} <= set(c.models())
    # untagged calls land in the default run
    assert len(_traj(tmp_path, "_untagged", "unknown")) == 3


def test_auth_inbound_and_upstream_token(tmp_path):
    with FakeServer({"default": {"content": "ok"}}, token="up-secret") as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base}, upstream_token="up-secret",
                               inbound_token="in-secret")) as gw:
        with pytest.raises(GatewayError) as ei:
            GatewayClient(gw.url, token="wrong").chat("improver", [{"role": "user", "content": "x"}])
        assert ei.value.status == 401
        r = GatewayClient(gw.url, token="in-secret").chat("improver", [{"role": "user", "content": "x"}], run="r7",
                                                         agent="a")
        assert r["choices"][0]["message"]["content"] == "ok"
        # admin endpoints need the main (upstream) token, not the inbound one
        with pytest.raises(GatewayError) as ei:
            GatewayClient(gw.url, token="in-secret").load_adapter("g1", "/abs/adapter")
        assert ei.value.status == 401
        out = GatewayClient(gw.url, token="up-secret").load_adapter("g1", "/abs/adapter")
        assert out["model"] == "improver@g1" and out["upstream_status"] == 200
        req = urllib.request.Request(fk.base + "/v1/models", headers={"Authorization": "Bearer up-secret"})
        assert "improver@g1" in [m["id"] for m in json.loads(urllib.request.urlopen(req).read())["data"]]
    txt = (tmp_path / "runs" / "r7" / "events.jsonl").read_text() + \
        open(trajectory_path(str(tmp_path / "runs"), "r7", "a")).read()
    assert "secret" not in txt


def test_ports_json_discovery_and_merge(tmp_path):
    pj = str(tmp_path / "run" / "ports.json")
    merge_ports({"evald": 1234}, pj)
    with FakeServer({"default": {"content": "via ports"}}) as fk:
        merge_ports({"vllm_improver": fk.port}, pj)
        with GatewayThread(_cfg(tmp_path, ports_path=pj)) as gw:
            merge_ports({"gateway": gw.port}, pj)
            r = GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}])
            assert r["choices"][0]["message"]["content"] == "via ports"
            # vLLM restarted on a new port: picked up without restarting the gateway
            with FakeServer({"default": {"content": "new port"}}) as fk2:
                time.sleep(0.01)
                merge_ports({"vllm_improver": fk2.port}, pj)
                os.utime(pj, (time.time() + 1, time.time() + 1))
                r = GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}])
                assert r["choices"][0]["message"]["content"] == "new port"
    d = read_ports(pj)
    assert d["evald"] == 1234 and "gateway" in d and "vllm_improver" in d


def test_path_prefix_tagging(tmp_path):
    with FakeServer({"default": {"content": "ok"}}) as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        agent = urllib.parse.quote("thread-1/sub-0", safe="")
        url = f"{gw.base}/rsi/t/r8/{agent}/3/-/v1"
        GatewayClient(url, token="").chat("improver", [{"role": "user", "content": "x"}])
    rec = _traj(tmp_path, "r8", "thread-1/sub-0")[0]
    assert rec["gen"] == 3 and rec["trial"] is None


def test_bad_run_header_rejected(tmp_path):
    with FakeServer({}) as fk, GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        with pytest.raises(GatewayError) as ei:
            GatewayClient(gw.url, token="").chat("improver", [{"role": "user", "content": "x"}], run="../etc")
        assert ei.value.status == 400


def test_concurrent_calls(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with FakeServer({"default": {"echo": True, "delay_s": 0.05}}) as fk, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fk.base})) as gw:
        c = GatewayClient(gw.url, token="")

        def one(i):
            return c.chat("improver", [{"role": "user", "content": f"q{i}"}], run="r9", agent=f"thread-{i % 3}",
                          stream=bool(i % 2))["choices"][0]["message"]["content"]

        with ThreadPoolExecutor(16) as ex:
            outs = list(ex.map(one, range(32)))
    assert outs == [f"echo: q{i}" for i in range(32)]
    n = sum(len(_traj(tmp_path, "r9", f"thread-{k}")) for k in range(3))
    assert n == 32 and len(_events(tmp_path, "r9")) == 32


def test_fake_server_cli(tmp_path):
    script = tmp_path / "s.json"
    script.write_text(json.dumps({"queue": [{"tool_calls": [{"name": "bash", "arguments": {"command": "pwd"}}]}]}))
    pj = tmp_path / "ports.json"
    p = subprocess.Popen([sys.executable, str(HERE / "fake_openai_server.py"), "--port", "0", "--script", str(script),
                          "--ports-file", str(pj), "--ports-key", "vllm_improver"], stdout=subprocess.PIPE, text=True)
    try:
        line = p.stdout.readline()
        assert line.startswith("FAKE_OPENAI_PORT=")
        port = int(line.strip().split("=")[1])
        assert read_ports(str(pj))["vllm_improver"] == port
        r = GatewayClient(f"http://127.0.0.1:{port}/v1", token="").chat("improver", [{"role": "user", "content": "x"}])
        tc = r["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "bash" and r["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        p.terminate()
        p.wait(5)


def test_gateway_cli_writes_port(tmp_path):
    pj = tmp_path / "run" / "ports.json"
    tok = tmp_path / "run" / "token"
    tok.parent.mkdir(parents=True)
    tok.write_text("tok123\n")
    with FakeServer({"default": {"content": "cli ok"}}, token="tok123") as fk:
        merge_ports({"vllm_improver": fk.port}, str(pj))
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        p = subprocess.Popen([sys.executable, "-m", "rsi.gateway", "--ports-file", str(pj), "--token-file", str(tok),
                              "--runs-root", str(tmp_path / "runs"), "--quiet"], stdout=subprocess.PIPE, text=True,
                             env=env, cwd=str(ROOT))
        try:
            port = int(p.stdout.readline().strip().split("=")[1])
            assert read_ports(str(pj))["gateway"] == port
            assert 20000 <= port <= 60999
            r = GatewayClient(f"http://127.0.0.1:{port}/v1", token_path=str(tok)).chat(
                "improver", [{"role": "user", "content": "x"}], run="r10", agent="a")
            assert r["choices"][0]["message"]["content"] == "cli ok"
            ps = subprocess.run(["ps", "-o", "args=", "-p", str(p.pid)], capture_output=True, text=True).stdout
            assert "tok123" not in ps
        finally:
            p.terminate()
            p.wait(5)
