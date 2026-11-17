"""rsi.update tests (CPU only): SFT builder, lessons writer, analysis, LoRA data prep, self-evolving data."""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rsi.tasks.muar import start_config  # noqa: E402
from rsi.tasks.muar.tasks import get_spec, task_card  # noqa: E402
from rsi.update import analysis as A  # noqa: E402
from rsi.update import build_sft_data, write_lessons  # noqa: E402
from rsi.update.leakcheck import heldout_fingerprints, scan_text  # noqa: E402

TRAIN_TASK = "muar/train-modadd-0"
HELD_TASK = "muar/id-modadd-0"


# ---------------------------------------------------------------------------- synthetic runs
def _call(run, trial, prompt, answer, *, agent="thread-0", t=0.0, tools=None, tool_calls=None):
    req = {"model": "improver", "messages": [{"role": "system", "content": "You are a researcher."},
                                             {"role": "user", "content": prompt}]}
    if tools:
        req["tools"] = tools
    msg = {"role": "assistant", "content": answer}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"t": t, "request": req, "response": {"choices": [{"message": msg}]}, "status": 200,
            "headers": {"X-RSI-Run": run, "X-RSI-Agent": agent, "X-RSI-Trial": str(trial), "X-RSI-Gen": "0"}}


def make_run(root: pathlib.Path, run: str, task: str, trials: list[tuple[int, str, str | None, float | None]],
             *, calls_per_trial: int = 2, extra_calls: list | None = None) -> str:
    """trials: (trial, status, eval_status, dev)."""
    rd = root / run
    (rd / "trajectories" / "gateway").mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps({"task": task, "seed": int(run[-1]) if run[-1].isdigit() else 0}))
    led, calls = [], []
    led.append({"trial": 0, "thread": "orchestrator", "status": "baseline", "dev": 3.0, "eval_status": "ok",
                "artifact_hash": "base"})
    for tr, st, es, dev in trials:
        led.append({"trial": tr, "thread": "thread-0", "status": "pending", "phase": "episode", "parent_hash": "base"})
        led.append({"trial": tr, "thread": "thread-0", "status": st, "eval_status": es, "dev": dev,
                    "artifact_hash": f"a{tr}", "hypothesis": f"try idea {tr}", "feedback": f"val_loss={dev}"})
        for k in range(calls_per_trial):
            calls.append(_call(run, tr, f"{task_card(get_spec(task))}\nstep {k} of trial {tr}",
                               f"I will change lr (trial {tr}, call {k}).", t=tr * 10 + k))
    calls += extra_calls or []
    with open(rd / "ledger.jsonl", "w") as f:
        for r in led:
            f.write(json.dumps(r) + "\n")
    with open(rd / "trajectories" / "gateway" / "thread-0.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    return str(rd)


STATUSES = [(1, "keep", "ok", 2.5), (2, "discard", "ok", 2.9), (3, "crash", "crash", None), (4, "keep", "ok", 2.0),
            (5, "discard", "ok", 2.6), (6, "discard", "ok", 2.7), (7, "audit_fail", "audit_fail", None),
            (8, "no_change", None, None)]


def _read(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def test_sft_accepted_only(tmp_path):
    rd = make_run(tmp_path, "run-a0", TRAIN_TASK, STATUSES)
    out = tmp_path / "sft"
    m = build_sft_data([rd], out=str(out), min_accepted=1, per_task_cap=None)
    exs = _read(out / "train.jsonl") + _read(out / "valid.jsonl")
    assert {e["meta"]["trial"] for e in exs} == {1, 4}
    assert all(e["meta"]["label"] == "accepted" for e in exs) and len(exs) == 4
    assert m["n_train"] + m["n_valid"] == 4 and m["leak_check"]["ok"] and m["leak_check"]["rescan_hits"] == 0
    for e in exs:   # messages format: prompt + final assistant answer
        assert e["messages"][-1]["role"] == "assistant" and e["messages"][0]["role"] == "system"
    # valid split never straddles a trial
    tv = {e["meta"]["trial"] for e in _read(out / "valid.jsonl")}
    tt = {e["meta"]["trial"] for e in _read(out / "train.jsonl")}
    assert not (tv & tt) and len(tv) >= 1


def test_sft_placebo_is_legal_rejected_and_size_matched(tmp_path):
    rd = make_run(tmp_path, "run-a0", TRAIN_TASK, STATUSES)
    m = build_sft_data([rd], placebo=True, out=str(tmp_path / "p"), min_accepted=1, per_task_cap=None)
    exs = _read(tmp_path / "p" / "train.jsonl") + _read(tmp_path / "p" / "valid.jsonl")
    assert {e["meta"]["trial"] for e in exs} <= {2, 5, 6}
    assert all(e["meta"]["label"] == "rejected" for e in exs)
    assert len(exs) == 4 and m["report"]["match_n"] == 4 and m["mode"] == "placebo_rejected"


def test_sft_dedup_cap_and_valid_fraction(tmp_path):
    trials = [(i, "keep", "ok", 3.0 - 0.01 * i) for i in range(1, 41)]
    dup = [_call("run-b1", 1, f"{task_card(get_spec(TRAIN_TASK))}\nstep 0 of trial 1", "I will change lr (trial 1, call 0).",
                 t=5.0)]
    rd = make_run(tmp_path, "run-b1", TRAIN_TASK, trials, extra_calls=dup)
    m = build_sft_data([rd], out=str(tmp_path / "s"), per_task_cap=50)
    assert m["report"]["n_dupes"] == 1
    assert m["n_train"] + m["n_valid"] == 50 and m["report"]["capped"][TRAIN_TASK] == 30
    assert 4 <= m["n_valid"] <= 6
    m2 = build_sft_data([rd], out=str(tmp_path / "s2"), per_task_cap=50)
    assert m2["sha256"] == m["sha256"]          # deterministic


def test_sft_leak_check_drops_heldout_material(tmp_path):
    held_cfg = json.dumps(start_config(HELD_TASK))
    leaky = [
        _call("run-c0", 1, "Here is another task:\n" + task_card(get_spec(HELD_TASK)), "ok", t=1.0),
        _call("run-c0", 4, "plain prompt", "Try this config: " + held_cfg, t=2.0),
        _call("run-c0", 4, "plain prompt 2", "", t=3.0,
              tool_calls=[{"id": "x", "type": "function",
                           "function": {"name": "edit", "arguments": json.dumps({"content": held_cfg})}}]),
    ]
    rd = make_run(tmp_path, "run-c0", TRAIN_TASK, STATUSES, extra_calls=leaky)
    m = build_sft_data([rd], out=str(tmp_path / "s"), min_accepted=1, per_task_cap=None)
    assert m["report"]["n_leak_dropped"] == 3
    fp = heldout_fingerprints()
    for e in _read(tmp_path / "s" / "train.jsonl") + _read(tmp_path / "s" / "valid.jsonl"):
        assert not scan_text(json.dumps(e), fp)
    # a run on a held-out task is excluded entirely
    rh = make_run(tmp_path, "run-h0", HELD_TASK, STATUSES)
    m2 = build_sft_data([rh], out=str(tmp_path / "h"), min_accepted=1)
    assert m2["n_train"] == 0 and not m2["ok"] and m2["report"]["excluded_runs"][0]["run"] == "run-h0"


def test_train_cards_are_not_flagged():
    fp = heldout_fingerprints()
    for t in ("muar/train-modadd-0", "muar/train-parity-1", "muar/train-cls2d-1"):
        assert scan_text(task_card(get_spec(t)), fp) == []
        assert scan_text(json.dumps(start_config(t)), fp) == []
    assert scan_text(task_card(get_spec("muar/ood-dyck-0")), fp)


def test_sft_hindsight_fallback(tmp_path):
    rd = make_run(tmp_path, "run-d0", TRAIN_TASK, [(1, "discard", "ok", 3.2), (2, "discard", "ok", 3.1),
                                                     (3, "crash", "crash", None)])
    m = build_sft_data([rd], out=str(tmp_path / "s"), min_accepted=40, per_task_cap=None)
    exs = _read(tmp_path / "s" / "train.jsonl") + _read(tmp_path / "s" / "valid.jsonl")
    assert m["report"]["hindsight"] and {e["meta"]["trial"] for e in exs} == {2}
    assert all(e["meta"]["label"] == "hindsight" for e in exs)
    m2 = build_sft_data([rd], out=str(tmp_path / "s2"), min_accepted=40, hindsight=False)
    assert m2["n_train"] == 0


# ---------------------------------------------------------------------------- lessons writer
def _long_text(n):
    return "\n".join(f"- lesson {i}: lower the learning rate when the curve is noisy" for i in range(n))


def test_write_lessons_real_and_placebo(tmp_path):
    from fake_openai_server import FakeServer

    rd = make_run(tmp_path, "run-l0", TRAIN_TASK, STATUSES)
    with FakeServer({"queue": [{"content": "<think>hmm</think>" + _long_text(60)},
                               {"content": _long_text(50)},
                               {"content": "- use warmup\n- scale lr with batch"}]}) as fs:
        h1 = write_lessons([rd], model="improver", gateway_url=fs.url, out_dir=str(tmp_path / "s1"))
        h2 = write_lessons([], model="improver", gateway_url=fs.url, out_dir=str(tmp_path / "s2"),
                           generic_placebo=True, target_words=120)
        reqs = list(fs.requests)
    text = (tmp_path / "s1" / "lessons.md").read_text()
    body = text.split("\n", 2)[2]
    assert len(body.split()) <= 300 and "<think>" not in text
    assert len(reqs) == 3                                     # long answer -> one shortening retry
    assert "try idea 1" in reqs[0]["messages"][1]["content"]  # real arm sees the trajectory summary
    assert "try idea" not in reqs[2]["messages"][1]["content"]  # placebo never does
    assert reqs[0]["model"] == "improver"
    t2 = (tmp_path / "s2" / "lessons.md").read_text()
    assert t2.split("\n")[0] == text.split("\n")[0]           # same title: arms are blind
    assert os.listdir(tmp_path / "s1") == ["lessons.md"]     # manifest lives outside the scaffold
    man = json.loads((tmp_path / "s2.manifest.json").read_text())
    assert man["kind"] == "generic_placebo" and man["scaffold_hash"] == h2
    from rsi.update.lessons import scaffold_hash

    assert scaffold_hash(str(tmp_path / "s1")) == h1 and h1 != h2


def test_write_lessons_refuses_heldout_runs(tmp_path):
    rd = make_run(tmp_path, "run-h1", HELD_TASK, STATUSES)
    with pytest.raises(ValueError):
        write_lessons([rd], model="improver", gateway_url="http://127.0.0.1:9/v1", out_dir=str(tmp_path / "x"))


# ---------------------------------------------------------------------------- analysis
def test_nauc_basics():
    assert A.nauc(1.0, [0.5, None, 0.0], 0.0) == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert A.nauc(1.0, [2.0, 3.0], 0.0) == 0.0                 # worse proposals never count
    assert A.nauc(1.0, [0.5], 0.0, n_trials=2) == pytest.approx(0.5)  # padding = no improvement
    assert A.nauc(1.0, [0.5], None) is None and A.nauc(1.0, [0.5], 1.0) is None
    assert A.best_so_far(1.0, [0.8, 0.9, float("nan"), 0.7]) == [0.8, 0.8, 0.8, 0.7]


def _synthetic_arm(rng, tasks, seeds, rate, noise=0.02, n=12):
    """Curves s_t = s0 * exp(-rate * t) + noise, s0 = 1, s_ref = 0."""
    out = {}
    for t in tasks:
        out[t] = {}
        for s in seeds:
            out[t][str(s)] = [float(math.exp(-rate * (i + 1)) + noise * rng.normal()) for i in range(n)]
    return out


def test_delta2_detects_shift_and_aa_is_null():
    rng = np.random.default_rng(0)
    tasks = [f"t{i}" for i in range(8)]
    s0 = {t: 1.0 for t in tasks}
    sref = {t: 0.0 for t in tasks}
    a0a = A.task_naucs(_synthetic_arm(rng, tasks, [0, 1, 2], 0.10), s0, sref)
    a0b = A.task_naucs(_synthetic_arm(rng, tasks, [3, 4, 5], 0.10), s0, sref)
    a1 = A.task_naucs(_synthetic_arm(rng, tasks, [0, 1, 2], 0.25), s0, sref)
    a1p = A.task_naucs(_synthetic_arm(rng, tasks, [0, 1, 2], 0.11), s0, sref)
    aa = A.aa_noise(a0a, a0b)
    assert aa["n_tasks"] == 8 and aa["sd_task"] > 0
    assert aa["mde"] == pytest.approx(2.8016 * aa["se_mean"], rel=1e-3)
    c = A.compare(a1, a0a, n_boot=2000)
    assert c["delta"] > 0 and c["ci"][0] > 0 and c["p_signflip"] <= 2 / 256 + 1e-9 and c["rho"] > 1
    p = A.compare(a1, a1p, n_boot=2000)
    d = A.decide(c, aa, p)
    assert d["observed"] and d["ci_lower_gt_0"] and d["gt_2x_aa"] and d["beats_placebo"]
    null = A.compare(a0b, a0a, n_boot=2000)
    assert null["ci"][0] < 0.02 and null["p_signflip"] > 0.01
    dn = A.decide(null, aa, A.compare(a0b, a1p, n_boot=500))
    assert not dn["observed"] and "minimum detectable effect" in dn["text"]


def test_identical_arms_exact_null():
    tn = {f"t{i}": {"0": 0.3 + 0.01 * i, "1": 0.31 + 0.01 * i} for i in range(5)}
    c = A.compare(tn, tn, n_boot=500)
    assert c["delta"] == 0 and c["ci"] == [0.0, 0.0] and c["p_signflip"] == 1.0
    assert A.signflip_test([1.0] * 6) == pytest.approx(2 / 64)
    assert A.signflip_test([]) is None


def test_decomposition_and_curves_from_runs(tmp_path):
    rd = make_run(tmp_path, "run-e3", TRAIN_TASK, STATUSES)
    curves, s0 = A.curves_from_runs([rd])
    assert s0[TRAIN_TASK] == 3.0 and curves[TRAIN_TASK]["3"] == [2.5, 2.9, None, 2.0, 2.6, 2.7, None]
    from rsi.update.trajectories import trials

    dec = A.decomposition(list(trials(rd).values()))
    assert dec["n"] == 7 and dec["v"] == pytest.approx(5 / 7) and dec["p"] == pytest.approx(2 / 5)
    assert dec["m"] == pytest.approx(0.5)
    spec = {"s_ref": {TRAIN_TASK: 1.0}, "runs": {"A": [rd], "B": [rd]}, "aa": ["A", "B"],
            "compare": [["A", "B"]], "n_boot": 200}
    res = A.analyse(spec)
    assert res["arms"]["A"]["n_tasks"] == 1 and res["comparisons"][0]["delta2"]["delta"] == 0


# ---------------------------------------------------------------------------- LoRA data prep (no GPU)
class FakeTok:
    """Char-level tokenizer with a ChatML-like template (prefix-stable like Qwen's)."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, tools=None):
        s = "".join(f"<|{m['role']}|>{m.get('content', '')}{json.dumps(m.get('tool_calls', ''))}<|end|>\n"
                    for m in msgs)
        return s + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_lora_encode_masks_prompt():
    from rsi.update.lora_train import encode_example, normalise_messages

    ex = {"messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "ANSWER", "tool_calls": [
                           {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]}
    norm = normalise_messages(ex["messages"])
    assert norm[1]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    enc = encode_example(FakeTok(), ex, 10000)
    tgt = "".join(chr(i) for i, l in zip(enc["input_ids"], enc["labels"]) if l != -100)
    assert tgt.startswith("ANSWER") and "hi" not in tgt and enc["n_target"] == len(tgt)
    assert encode_example(FakeTok(), ex, 10) is None


def test_lora_train_sh_is_valid_bash():
    import subprocess

    p = ROOT / "bundle" / "lora_train.sh"
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    r = subprocess.run(["bash", str(p)], capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr.lower()


# ---------------------------------------------------------------------------- self-evolving data
def test_selfevolve_classify_and_repair(tmp_path):
    from rsi.update.selfevolve import classify_failures, repair_config, synthesize_repairs

    bad = dict(start_config(TRAIN_TASK), lr=50.0, width=3000)
    bad.pop("dropout")
    calls = [_call("run-f0", 7, "propose", json.dumps({"hypothesis": "bigger", "config": bad}), t=99.0),
             _call("run-f0", 2, "x", "", t=98.0, tool_calls=[{"id": "1", "type": "function",
                                                              "function": {"name": "edit", "arguments": "{oops"}}])]
    rd = make_run(tmp_path, "run-f0", TRAIN_TASK, STATUSES, extra_calls=calls)
    c = classify_failures([rd])
    assert c["illegal_config"] == 1 and c["crash"] == 1 and c["no_change"] == 1 and c["no_improvement"] == 3
    assert c["illegal_tool_call"] == 1
    fixed = repair_config(bad, start_config(TRAIN_TASK))
    assert fixed["lr"] == 3.0 and fixed["width"] == 1024 and "dropout" in fixed
    exs = synthesize_repairs([rd], lambda task, cfg: (True, 1.0))
    assert len(exs) == 1 and json.loads(exs[0]["messages"][-1]["content"])["config"] == fixed
    assert synthesize_repairs([rd], lambda task, cfg: (True, 5.0)) == []    # not better than parent: dropped


def test_reader_accepts_rsi_gateway_record_layout(tmp_path):
    """Exact layout written by rsi/gateway/server.py Gateway.record (worker C)."""
    from rsi.update.trajectories import load_calls

    def gw(trial, status, content):
        return {"t": 1.0 + trial, "run": "run-g0", "agent": "thread-0", "parent": "orchestrator", "gen": 0,
                "trial": trial, "model": "improver", "upstream": "improver", "stream": True, "status": status,
                "http_status": 200, "latency_s": 1.0, "ttft_s": 0.1, "attempts": 1, "usage": {},
                "request": {"model": "improver", "messages": [{"role": "user", "content": "go"}]},
                "response": {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"},
                "error": None}

    rd = make_run(tmp_path, "run-g0", TRAIN_TASK, STATUSES, calls_per_trial=0)
    with open(pathlib.Path(rd) / "trajectories" / "gateway" / "thread-0.jsonl", "a") as f:
        for rec in (gw(1, "ok", "A"), gw(4, "client_disconnected", "B"), gw(4, "ok", "C")):
            f.write(json.dumps(rec) + "\n")
    calls = load_calls(rd)
    assert [(c.trial, c.response["content"]) for c in calls] == [(1, "A"), (4, "C")]
    m = build_sft_data([rd], out=str(tmp_path / "s"), min_accepted=1, per_task_cap=None)
    assert m["n_train"] + m["n_valid"] == 2
