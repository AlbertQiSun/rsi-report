"""Gateway /v1/messages (Anthropic Messages API) passthrough against tests/fake_anthropic_server.py."""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fake_anthropic_server import FakeAnthropic  # noqa: E402
from rsi import events  # noqa: E402
from rsi.gateway import GatewayConfig, GatewayThread  # noqa: E402
from rsi.gateway.server import trajectory_path  # noqa: E402
from rsi.update import build_sft_data  # noqa: E402
from rsi.update.trajectories import load_calls  # noqa: E402

TOOLS = [{"name": "Bash", "description": "Run a shell command", "input_schema": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "mcp__lab__submit_eval", "description": "submit", "input_schema": {
        "type": "object", "properties": {"hypothesis": {"type": "string"}}, "required": ["hypothesis"]}}]

BODY = {"model": "improver", "max_tokens": 512,
        "system": [{"type": "text", "text": "You are Claude Code."}, {"type": "text", "text": "Be brief.",
                                                                       "cache_control": {"type": "ephemeral"}}],
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "edit train.py"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "look first", "signature": "x"},
                                              {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                               "input": {"command": "cat train.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "LR = 9e-4"},
                                         {"type": "text", "text": "<system-reminder>x</system-reminder>"}]}]}


def _cfg(tmp_path, **kw) -> GatewayConfig:
    d = dict(runs_root=str(tmp_path / "runs"), max_retries=3, retry_deadline_s=5, backoff_base_s=0.01,
             backoff_cap_s=0.02, quiet=True)
    d.update(kw)
    return GatewayConfig(**d)


def _post(url, body, headers=None, raw=False):
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01", **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _traj(tmp_path, run, agent):
    return [json.loads(x) for x in open(trajectory_path(str(tmp_path / "runs"), run, agent))]


def test_nonstream_prefix_tags_xapikey_and_conversion(tmp_path):
    script = {"queue": [{"thinking": "ok", "text": "Changing it.",
                         "tool_uses": [{"name": "mcp__lab__submit_eval", "input": {"hypothesis": "lr up"}, "id": "toolu_9"}]}]}
    with FakeAnthropic(script, token="main-tok") as fa, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base}, upstream_token="main-tok",
                               inbound_token="agent-tok")) as gw:
        url = gw.base + "/rsi/t/runA/thread-0/2/7/v1/messages?beta=true"
        code, obj = _post(url, BODY, {"x-api-key": "agent-tok"})
        assert code == 200, obj
        assert obj["content"][-1]["name"] == "mcp__lab__submit_eval"
        h = fa.headers[0]
        assert h["authorization"] == "Bearer main-tok" and "x-api-key" not in h      # agent token never forwarded
        assert h["anthropic-version"] == "2023-06-01"
        assert fa.state.paths[0] == "/v1/messages"
        assert fa.requests[0]["system"] == BODY["system"]                            # body relayed unchanged
    rec = _traj(tmp_path, "runA", "thread-0")[0]
    assert (rec["gen"], rec["trial"], rec["status"], rec["api"]) == (2, 7, "ok", "anthropic")
    msgs = rec["request"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[0]["content"] == "You are Claude Code.\nBe brief."
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "Bash", "arguments": json.dumps({"command": "cat train.py"})}
    assert msgs[2]["reasoning_content"] == "look first"
    assert msgs[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "LR = 9e-4"}
    assert rec["request"]["tools"][0]["function"]["parameters"]["required"] == ["command"]
    m = rec["response"]["message"]
    assert m["content"] == "Changing it." and m["reasoning_content"] == "ok"
    assert m["tool_calls"][0]["id"] == "toolu_9"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"hypothesis": "lr up"}
    assert rec["response"]["finish_reason"] == "tool_calls"
    assert rec["usage"]["prompt_tokens"] > 0 and rec["usage"]["completion_tokens"] > 0
    ev = events.read_events(str(tmp_path / "runs" / "runA" / "events.jsonl"))
    for e in ev:
        events.validate(e)
    assert [e["ev"] for e in ev] == ["llm_call"]
    assert ev[0]["n_tool_calls"] == 1 and ev[0]["finish_reason"] == "tool_calls" and ev[0]["trial"] == 7
    # rsi.update reads the record unchanged
    calls = load_calls(str(tmp_path / "runs" / "runA"))
    assert len(calls) == 1 and calls[0].trial == 7 and calls[0].response["tool_calls"]
    assert calls[0].tools[1]["function"]["name"] == "mcp__lab__submit_eval"


def test_stream_reassembled_and_relayed(tmp_path):
    script = {"queue": [{"thinking": "plan", "text": "Running it now.",
                         "tool_uses": [{"name": "Bash", "input": {"command": "python3 train.py --steps 3"}}]}]}
    with FakeAnthropic(script) as fa, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base}, inbound_token="agent-tok")) as gw:
        code, raw = _post(gw.base + "/rsi/t/runS/thread-1%2Fsub-0/0/3/v1/messages", dict(BODY, stream=True),
                          {"Authorization": "Bearer agent-tok"}, raw=True)
        assert code == 200
        text = raw.decode()
        assert "event: message_start" in text and "event: message_stop" in text
        names = [json.loads(l[5:])["type"] for l in text.splitlines() if l.startswith("data:")]
        assert names[0] == "message_start" and names[-1] == "message_stop" and "input_json_delta" in text
    rec = _traj(tmp_path, "runS", "thread-1/sub-0")[0]
    assert rec["agent"] == "thread-1/sub-0" and rec["stream"] is True and rec["status"] == "ok"
    m = rec["response"]["message"]
    assert m["content"] == "Running it now." and m["reasoning_content"] == "plan"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"command": "python3 train.py --steps 3"}
    assert rec["response_anthropic"]["stop_reason"] == "tool_use"
    assert rec["usage"]["output_tokens"] > 0 and rec["ttft_s"] is not None


def test_stream_invalid_tool_json_kept_raw(tmp_path):
    script = {"queue": [{"tool_uses": [{"name": "Bash", "input": {}}], "raw_input": '{"command": "ls'}]}
    with FakeAnthropic(script) as fa, GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base})) as gw:
        code, _ = _post(gw.base + "/v1/messages", dict(BODY, stream=True), raw=True)
        assert code == 200
    rec = _traj(tmp_path, "_untagged", "unknown")[0]
    assert rec["response"]["message"]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls'


def test_auth_errors_retries_and_count_tokens(tmp_path):
    script = {"queue": [{"status": 503, "error": "busy"}, {"text": "after retry"}]}
    with FakeAnthropic(script) as fa, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base}, inbound_token="agent-tok")) as gw:
        code, obj = _post(gw.base + "/v1/messages", BODY, {"x-api-key": "wrong"})
        assert code == 401 and obj["type"] == "error" and obj["error"]["type"] == "auth_error"
        code, obj = _post(gw.base + "/rsi/t/runR/thread-0/0/1/v1/messages", BODY, {"x-api-key": "agent-tok"})
        assert code == 200 and obj["content"][0]["text"] == "after retry"
        code, obj = _post(gw.base + "/v1/messages", dict(BODY, model="nope"), {"x-api-key": "agent-tok"})
        assert code == 404 and obj["type"] == "error"
        n_before = len(fa.requests)
        code, obj = _post(gw.base + "/rsi/t/runR/thread-0/0/1/v1/messages/count_tokens", BODY,
                          {"x-api-key": "agent-tok"})
        assert code == 200 and obj["input_tokens"] > 0 and len(fa.requests) == n_before + 1
    recs = _traj(tmp_path, "runR", "thread-0")
    assert len(recs) == 1 and recs[0]["attempts"] == 2            # count_tokens not recorded


def test_sft_reads_anthropic_records(tmp_path):
    with FakeAnthropic({"queue": [{"text": "Done: raised the lr."}]}) as fa, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base})) as gw:
        code, _ = _post(gw.base + "/rsi/t/runT/thread-0/0/1/v1/messages", BODY)
        assert code == 200
    rd = tmp_path / "runs" / "runT"
    (rd / "config.json").write_text(json.dumps({"task": "nanochat"}))
    with open(rd / "ledger.jsonl", "w") as f:
        f.write(json.dumps({"trial": 1, "thread": "thread-0", "status": "keep", "eval_status": "ok", "dev": 1.0,
                            "artifact_hash": "a1"}) + "\n")
    m = build_sft_data([str(rd)], out=str(tmp_path / "sft"), min_accepted=1, per_task_cap=None, heldout=[])
    assert m["ok"] and m["n_train"] + m["n_valid"] == 1
    ex = json.loads((tmp_path / "sft" / "train.jsonl").read_text().splitlines()[0])
    assert ex["messages"][-1] == {"role": "assistant", "content": "Done: raised the lr."}
    assert ex["messages"][0]["role"] == "system" and ex["tools"][0]["function"]["name"] == "Bash"


def test_compat_rewrites_adaptive_thinking_and_drops_keys(tmp_path):
    body = dict(BODY, thinking={"type": "adaptive"}, output_config={"effort": "high"}, metadata={"user_id": "u"})
    with FakeAnthropic({"queue": [{"text": "a"}, {"text": "b"}]}) as fa, \
            GatewayThread(_cfg(tmp_path, upstreams={"improver": fa.base})) as gw:
        assert _post(gw.base + "/v1/messages", body)[0] == 200
        sent = fa.requests[0]
        assert "output_config" not in sent and sent["metadata"] == {"user_id": "u"}
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 511}
        gw.gw.cfg.anthropic_drop, gw.gw.cfg.anthropic_thinking = (), "pass"
        assert _post(gw.base + "/v1/messages", body)[0] == 200
        assert fa.requests[1]["thinking"] == {"type": "adaptive"} and "output_config" in fa.requests[1]
    recs = _traj(tmp_path, "_untagged", "unknown")
    assert recs[0]["compat"] == ["drop:output_config", "thinking:adaptive->enabled"] and "compat" not in recs[1]
