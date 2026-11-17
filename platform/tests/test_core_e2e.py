"""End-to-end (CPU, no network beyond 127.0.0.1): orchestrator + karpathy_greedy + mini harness
+ fake OpenAI server + fake evald + lab spool, including pause/resume."""

import json
import math
import pathlib
import re
import threading
import uuid

import pytest

from rsi import events, orchestrator
from rsi.tasks.base import Task, register
from tests.fake_openai_server import FakeServer

START = """# tiny artifact
LEARNING_RATE = 9e-4
WIDTH = 64
"""
OPTIMUM = 3e-3


@register
class TinyTask(Task):
    name = "tiny"
    metric = "val_loss"
    lower_is_better = True
    start_dir: pathlib.Path | None = None

    def __init__(self, start_dir=None):
        self.start = pathlib.Path(start_dir or TinyTask.start_dir)

    def start_artifact(self):
        return self.start

    def task_card(self):
        return "Tune LEARNING_RATE in train.py. Score = distance of log10(lr) from a hidden optimum."


def score(artifact_dir: str) -> float:
    s = (pathlib.Path(artifact_dir) / "train.py").read_text()
    lr = float(re.search(r"LEARNING_RATE = ([0-9.eE+-]+)", s).group(1))
    return 1.0 + abs(math.log10(lr) - math.log10(OPTIMUM))


class FakeEvald:
    """Same surface as rsi.evald.client.EvalClient; emits eval_result like evald would."""

    pause_after: int | None = None   # create PAUSE once this many evals finished
    run_dir: pathlib.Path | None = None

    def __init__(self, cfg=None):
        self.jobs = {}
        self.n_done = 0
        self.lock = threading.Lock()

    def submit(self, run, task, artifact_dir, *, seeds, lane=None, gen=0, trial=None):
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = dict(run=run, task=task, dir=artifact_dir, seeds=seeds, lane=lane, gen=gen, trial=trial)
        return jid

    def result(self, job_id, *, wait=True, timeout_s=3600):
        j = self.jobs[job_id]
        v = score(j["dir"])
        h = pathlib.Path(j["dir"]).name
        res = dict(artifact_hash=h, metric="val_loss", value=v, seeds=j["seeds"], wall_s=0.01, status="ok",
                   audit={"level": "L1", "flags": []})
        events.emit(j["run"], "eval_result", "evald", gen=j["gen"], trial=j["trial"], lane=j["lane"], **res)
        with self.lock:
            self.n_done += 1
            if FakeEvald.pause_after and self.n_done >= FakeEvald.pause_after and FakeEvald.run_dir:
                (FakeEvald.run_dir / "PAUSE").write_text("test pause\n")
        return res

    def lanes(self):
        return [{"lane": 1, "gpu": 1, "busy": False, "paused": False}]

    def pause_lane(self, lane):
        pass

    def resume_lane(self, lane):
        pass


def make_fake_evald(cfg):
    return FakeEvald(cfg)


BUMP = ("python3 -c \"import re;s=open('train.py').read();"
        "v=float(re.search(r'LEARNING_RATE = ([0-9.eE+-]+)',s).group(1));"
        "open('train.py','w').write(re.sub(r'LEARNING_RATE = [0-9.eE+-]+','LEARNING_RATE = %g'%(v*1.6),s))\"")

SCRIPT = {
    "rules": [
        {"match": {"n_tool_msgs": 0}, "respond": {"content": None, "tool_calls": [
            {"name": "lab_get_results", "arguments": {"n": 3}}]}},
        {"match": {"n_tool_msgs": 1}, "respond": {"content": "Reading the file.", "tool_calls": [
            {"name": "read_file", "arguments": {"path": "train.py"}}]}},
        {"match": {"n_tool_msgs": 2}, "respond": {"content": None, "tool_calls": [
            {"name": "bash", "arguments": {"command": BUMP}}]}},
        {"match": {"n_tool_msgs": 3}, "respond": {"content": None, "tool_calls": [
            {"name": "lab_submit_eval", "arguments": {"hypothesis": "raise the learning rate by 1.6x"}}]}},
    ],
    "default": {"content": "Raised LEARNING_RATE by 1.6x and submitted."},
}


def make_cfg(tmp_path, url, run="smoke-run", trials=4, threads=2):
    start = tmp_path / "start"
    start.mkdir(exist_ok=True)
    (start / "train.py").write_text(START)
    TinyTask.start_dir = start
    return {
        "run": run,
        "runs_root": str(tmp_path / "runs"),
        "task": {"name": "tiny", "args": {"start_dir": str(start)}},
        "policy": {"name": "karpathy_greedy", "threads": threads, "seeds": [0], "s_ref": 1.0},
        "model": "improver",
        "improver": {"model": "fake/improver"},
        "budget": {"trials": trials, "episode_s": 60},
        "lanes": [1, 2],
        "evald": {"factory": "tests.test_core_e2e:make_fake_evald"},
        "harness": {"kind": "mini", "sandbox": "none", "gateway_url": url, "emit_llm_calls": True,
                    "max_turns": 8},
    }


@pytest.fixture
def fake_llm():
    with FakeServer(json.loads(json.dumps(SCRIPT))) as fs:
        yield fs


def test_e2e_karpathy_greedy(tmp_path, fake_llm):
    cfg = make_cfg(tmp_path, fake_llm.url)
    FakeEvald.pause_after = None
    summary = orchestrator.run(cfg)
    run_dir = tmp_path / "runs" / "smoke-run"
    evs = events.validate_file(str(run_dir / "events.jsonl"))
    kinds = {e["ev"] for e in evs}
    assert {"run_start", "agent_spawn", "tool_call", "llm_call", "proposal", "eval_submit", "eval_result",
            "decision", "message", "agent_end", "run_end"} <= kinds
    assert summary["n_trials"] == 4
    assert summary["best_value"] < 1.0 + abs(math.log10(9e-4) - math.log10(OPTIMUM))
    assert summary["nAUC"] is not None and summary["nAUC"] > 0
    # both threads worked, each in its own workspace, on its own lane
    agents = {e["agent"] for e in evs if e["ev"] == "agent_spawn"}
    assert agents == {"thread-0", "thread-1"}
    lanes = {e["lane"] for e in evs if e["ev"] == "eval_submit" and e["trial"] != 0}
    assert lanes == {1, 2}
    # trajectories saved; the agent never saw evaluator internals
    trajs = list((run_dir / "trajectories" / "harness").rglob("trial-*.json"))
    assert len(trajs) == 4
    assert "hidden optimum" in trajs[0].read_text()  # task card visible
    # every adopted artifact strictly improved
    for e in evs:
        if e["ev"] == "decision" and e["verdict"] == "adopt" and e["best_before"] is not None:
            assert e["best_after"] < e["best_before"]


def test_e2e_pause_and_resume(tmp_path, fake_llm):
    cfg = make_cfg(tmp_path, fake_llm.url, run="pause-run", trials=4, threads=1)
    run_dir = tmp_path / "runs" / "pause-run"
    FakeEvald.run_dir = run_dir
    FakeEvald.pause_after = 3   # baseline + 2 trials, then PAUSE appears
    s1 = orchestrator.run(cfg)
    assert s1["paused"] and s1["n_trials"] == 2
    FakeEvald.pause_after = None
    s2 = orchestrator.run(cfg, resume=True)
    assert not s2["paused"] and s2["n_trials"] == 4
    evs = events.validate_file(str(run_dir / "events.jsonl"))
    assert sum(e["ev"] == "run_start" for e in evs) == 1
    assert sum(e["ev"] == "run_end" for e in evs) == 1
    trials = [e["trial"] for e in evs if e["ev"] == "proposal"]
    assert trials == sorted(set(trials))  # no trial id reused across the resume


def test_resume_recovers_inflight_eval(tmp_path, fake_llm):
    """A trial proposed but never decided (process died) is evaluated and decided on resume."""
    cfg = make_cfg(tmp_path, fake_llm.url, run="crash-run", trials=2, threads=1)
    FakeEvald.pause_after = None
    orchestrator.run(cfg)
    run_dir = tmp_path / "runs" / "crash-run"
    led = [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines()]
    last = max(r["trial"] for r in led)
    art = next(r["artifact_hash"] for r in led if r.get("trial") == last and r.get("artifact_hash"))
    # simulate: a new trial got as far as 'proposed' when the process died
    with open(run_dir / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"trial": last + 1, "thread": "thread-0", "status": "pending", "phase": "proposed",
                            "artifact_hash": art, "parent_hash": art}) + "\n")
    s = orchestrator.run(cfg, resume=True)
    led = [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines()]
    final = [r for r in led if r.get("trial") == last + 1 and r.get("status") != "pending"]
    assert final and "recovered" in final[-1]["reason"]
    assert s["n_trials"] >= 3
