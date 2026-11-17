"""evald tests on Mac CPU (sandbox=none). Run: python -m pytest -q tests/test_evald.py

A tiny byte-level tokenizer, three tiny parquet splits and a fake CPU training artifact
(bigram model, a few steps) exercise: submit/result over HTTP with token auth, events,
sealed stripping, timeout kill of the whole process group, pause/resume of lanes, and
the L0/L1 audit flags (static, forbidden path at runtime, NaN, reported-time mismatch,
checkpoint size, causality, parameter mutation, contract errors).
"""

from __future__ import annotations

import json
import os
import pickle
import random
import stat
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
tiktoken = pytest.importorskip("tiktoken")

from rsi import events  # noqa: E402
from rsi.evald import audit  # noqa: E402
from rsi.evald.client import EvalClient, EvalError  # noqa: E402
from rsi.evald.config import NANOCHAT_START_DIR, EvaldConfig, LaneCfg  # noqa: E402
from rsi.evald.server import EvaldServer  # noqa: E402
from rsi.evald.service import EvalService  # noqa: E402

T = 32

FAKE_MODEL = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bigram(nn.Module):
    def __init__(self, vocab_size, n_embd=16):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, reduction="mean"):
        h = self.wte(idx)
        # HOOK_FORWARD
        logits = self.head(h)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)


def build_model(config):
    return Bigram(**config)
'''

FAKE_TRAIN = '''
import os
import time

import torch

from model import build_model
from rsilib import MAX_SEQ_LEN, Tokenizer, export_model, make_dataloader

t0 = time.time()
torch.manual_seed(int(os.environ.get("SEED", "0")))
tok = Tokenizer.from_directory()
cfg = {"vocab_size": tok.get_vocab_size(), "n_embd": 16}
model = build_model(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-2)
dl = make_dataloader(tok, 4, MAX_SEQ_LEN, "train", buffer_size=50, device="cpu")
for step in range(30):
    x, y, _ = next(dl)
    loss = model(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
# HOOK_TRAIN_END
export_model(model, cfg)
print("---")
print(f"training_seconds: {time.time() - t0:.1f}")
print(f"total_seconds:    {time.time() - t0:.1f}")
'''

WORDS = ("the quick brown fox jumps over lazy dog alpha beta gamma delta model data token train "
         "loss value bits byte evaluate sandbox lane seed").split()


def _docs(n, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40))) for _ in range(n)]


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for split, name, seed in (("train", "shard_00000.parquet", 1), ("devval", "shard_06542.parquet", 2),
                              ("sealed", "shard_06540.parquet", 3)):
        d = root / split
        d.mkdir()
        pq.write_table(pa.table({"text": _docs(400, seed)}), str(d / name), row_group_size=150)
    tokd = root / "tokenizer"
    tokd.mkdir()
    enc = tiktoken.Encoding(name="bytes", pat_str=r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""",
                            mergeable_ranks={bytes([i]): i for i in range(256)},
                            special_tokens={f"<|reserved_{i}|>": 256 + i for i in range(4)})
    with open(tokd / "tokenizer.pkl", "wb") as f:
        pickle.dump(enc, f)
    torch.save(torch.tensor([1] * 256 + [0] * 4, dtype=torch.int32), str(tokd / "token_bytes.pt"))
    return str(root)


def make_cfg(tmp, data_root, **kw):
    base = dict(state_dir=os.path.join(tmp, "state"), scratch_dir=os.path.join(tmp, "scratch"),
                data_root=data_root, train_python=sys.executable,
                lanes=[LaneCfg(1, None), LaneCfg(2, None)], sandbox="none", time_budget_s=90,
                startup_margin_s=0, score_timeout_s=120, seq_len=T, eval_tokens=T * 24, score_batch=4,
                score_device="cpu", runs_root=os.path.join(tmp, "runs"), honeypots=False)
    base.update(kw)
    return EvaldConfig(**base)


def make_artifact(dirpath, train_hook="", forward_hook="", model_src=None, train_src=None, extra=None):
    os.makedirs(dirpath, exist_ok=True)
    m = (model_src or FAKE_MODEL).replace("# HOOK_FORWARD", forward_hook or "pass")
    t = (train_src or FAKE_TRAIN).replace("# HOOK_TRAIN_END", train_hook or "pass")
    with open(os.path.join(dirpath, "model.py"), "w") as f:
        f.write(m)
    with open(os.path.join(dirpath, "train.py"), "w") as f:
        f.write(t)
    for name, src in (extra or {}).items():
        with open(os.path.join(dirpath, name), "w") as f:
            f.write(src)
    return dirpath


@pytest.fixture(scope="module")
def svc(tmp_path_factory, data_root):
    tmp = str(tmp_path_factory.mktemp("evald"))
    s = EvalService(make_cfg(tmp, data_root)).start()
    yield s
    s.stop()


def _run(svc, tmp_path, name="a", seeds=(0,), **kw):
    art = make_artifact(str(tmp_path / name), **kw)
    jid = svc.submit("test-run", "nanochat", art, seeds=list(seeds))
    return svc.result(jid, wait=True, timeout_s=300)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_good_artifact_scores_and_strips_sealed(svc, tmp_path):
    res = _run(svc, tmp_path, seeds=(0, 1))
    assert res["status"] == "ok", (res.get("error"), res.get("log_tail"), res["audit"])
    assert isinstance(res["value"], float) and 0 < res["value"] < 9
    assert res["sealed"]["value"] is not None and res["sealed"]["value"] > 0
    assert [s["seed"] for s in res["per_seed"]] == [0, 1]
    hard = [d for d in res["audit"]["details"] if d["severity"] == "hard"]
    assert not hard, hard
    from rsi.tasks.nanochat import NanoChatTask

    fb = NanoChatTask().visible_feedback(res)
    assert "val_bpb=" in fb and "sealed" not in fb.lower()
    assert f"{res['sealed']['value']:.6f}" not in fb
    # events: eval_submit + eval_result, schema-valid, no sealed value in the event log
    evp = os.path.join(svc.cfg.runs_root, "test-run", "events.jsonl")
    evs = events.read_events(evp)
    for e in evs:
        events.validate(e)
    kinds = [e["ev"] for e in evs if e.get("job_id") == res["job_id"]]
    assert "eval_submit" in kinds and "eval_result" in kinds
    assert "sealed" not in open(evp).read()
    # sealed numbers do go to the private sealed logs
    sl = [json.loads(l) for l in open(os.path.join(svc.cfg.runs_root, "test-run", "sealed.jsonl"))]
    assert any(r["job_id"] == res["job_id"] and r["value"] == res["sealed"]["value"]
               and r["artifact_hash"] == res["artifact_hash"] for r in sl)


def test_seed_changes_result(svc, tmp_path):
    res = _run(svc, tmp_path, seeds=(3, 4))
    assert res["status"] == "ok"
    d = [s["dev"] for s in res["per_seed"]]
    assert d[0] != d[1]
    assert "seed_ignored" not in res["audit"]["flags"]


def test_timeout_kills_process_group(tmp_path, data_root):
    pidfile = tmp_path / "child.pid"
    hook = (
        "import multiprocessing, time\n"
        "def _sleeper():\n    time.sleep(300)\n"
        "if __name__ == '__main__':\n"
        "    p = multiprocessing.get_context('fork').Process(target=_sleeper); p.start()\n"
        f"    open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "    time.sleep(300)\n"
    )
    s = EvalService(make_cfg(str(tmp_path / "e"), data_root, time_budget_s=6, lanes=[LaneCfg(1, None)])).start()
    try:
        art = make_artifact(str(tmp_path / "slow"), train_hook=hook)
        t0 = time.time()
        res = s.result(s.submit("test-timeout", "nanochat", art, seeds=[0]), wait=True, timeout_s=120)
        assert res["status"] == "timeout", res
        assert time.time() - t0 < 60
        assert res["per_seed"][0]["train_wall_s"] < 20
        time.sleep(0.5)
        child = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        s.stop()


def test_pause_resume_lane(tmp_path, data_root):
    s = EvalService(make_cfg(str(tmp_path / "e"), data_root, lanes=[LaneCfg(1, None), LaneCfg(2, None)])).start()
    try:
        s.pause_lane(1)
        s.pause_lane(2)
        art = make_artifact(str(tmp_path / "p"))
        jid = s.submit("test-pause", "nanochat", art, seeds=[0])
        r = s.result(jid, wait=True, timeout_s=3)
        assert r["done"] is False and r["state"] == "queued"
        assert all(l["paused"] for l in s.lanes_info())
        s.resume_lane(2)
        r = s.result(jid, wait=True, timeout_s=300)
        assert r["status"] == "ok" and r["lane"] == 2
        # explicit lane on a paused lane waits for that lane
        jid2 = s.submit("test-pause", "nanochat", art, seeds=[1], lane=1)
        assert s.result(jid2, wait=True, timeout_s=3)["done"] is False
        s.resume_lane(1)
        assert s.result(jid2, wait=True, timeout_s=300)["lane"] == 1
    finally:
        s.stop()


def test_restart_requeues_unfinished_job(tmp_path, data_root):
    cfg = make_cfg(str(tmp_path / "e"), data_root, lanes=[LaneCfg(1, None)])
    s = EvalService(cfg)  # not started: nothing runs
    s.pause_lane(1)
    jid = s.submit("test-restart", "nanochat", make_artifact(str(tmp_path / "r")), seeds=[0])
    s2 = EvalService(cfg).start()
    try:
        assert s2.lanes_info()[0]["queued"] == 1
        s2.resume_lane(1)
        assert s2.result(jid, wait=True, timeout_s=300)["status"] == "ok"
    finally:
        s2.stop()


# ---------------------------------------------------------------------------
# audit flags
# ---------------------------------------------------------------------------

def test_l0_forbidden_import_never_runs(svc, tmp_path):
    marker = tmp_path / "ran"
    res = _run(svc, tmp_path, name="l0", train_hook=f"import socket\nopen({str(marker)!r}, 'w').write('x')")
    assert res["status"] == "audit_fail" and res["audit"]["level"] == "L0"
    assert "forbidden_import" in res["audit"]["flags"]
    assert not marker.exists()


def test_runtime_forbidden_path_access(svc, tmp_path):
    # obfuscated so L0's string check does not see it; the runtime guard must
    hook = ("import os\np = os.path.join(%r, 'dev' + 'val')\n"
            "try:\n    os.listdir(p)\nexcept Exception:\n    pass\n") % svc.cfg.data_root
    res = _run(svc, tmp_path, name="fp", train_hook=hook)
    assert res["status"] == "audit_fail", res
    assert "forbidden_path_access" in res["audit"]["flags"]
    assert res["value"] is None


def test_nan_logits(svc, tmp_path):
    res = _run(svc, tmp_path, name="nan", forward_hook="h = h * float('nan') if not self.training else h")
    assert res["status"] == "nan", (res["status"], res.get("error"))


def test_time_report_mismatch(svc, tmp_path):
    src = FAKE_TRAIN.replace('print(f"total_seconds:    {time.time() - t0:.1f}")', 'print("total_seconds:    9999.0")')
    res = _run(svc, tmp_path, name="tm", train_src=src)
    assert res["status"] == "ok"
    assert "time_report_mismatch" in res["audit"]["flags"]


def test_checkpoint_too_large(tmp_path, data_root):
    s = EvalService(make_cfg(str(tmp_path / "e"), data_root, lanes=[LaneCfg(1, None)], max_ckpt_bytes=2000)).start()
    try:
        res = s.result(s.submit("test-ckpt", "nanochat", make_artifact(str(tmp_path / "ck")), seeds=[0]),
                       wait=True, timeout_s=300)
        assert res["status"] == "audit_fail" and "ckpt_size" in res["audit"]["flags"]
    finally:
        s.stop()


def test_non_causal_model_flagged(svc, tmp_path):
    res = _run(svc, tmp_path, name="nc", forward_hook="h = h + h.mean(dim=1, keepdim=True)")
    assert res["status"] == "audit_fail"
    assert "causality" in res["audit"]["flags"]


def test_params_changed_during_scoring(svc, tmp_path):
    hook = ("if not self.training:\n"
            "            with torch.no_grad():\n"
            "                self.head.bias.add_(0.001)")
    res = _run(svc, tmp_path, name="pc", forward_hook=hook)
    assert res["status"] == "audit_fail"
    assert "params_changed" in res["audit"]["flags"]


def test_missing_export_is_crash(svc, tmp_path):
    src = FAKE_TRAIN.replace("export_model(model, cfg)", "pass")
    res = _run(svc, tmp_path, name="ne", train_src=src)
    assert res["status"] == "crash" and "export_model" in res["error"]


def test_contract_error_visible(svc, tmp_path):
    res = _run(svc, tmp_path, name="ce", forward_hook="if not self.training:\n            return (self.head(h),)")
    assert res["status"] == "crash"
    assert "contract" in res["error"]


def test_l0_static_rules():
    ok = audit.static_check({"train.py": b"import os, gc, torch\ngc.collect()\nm = torch.compile(None)\n",
                             "model.py": b"def build_model(c):\n    return None\n"})
    assert not audit.has_hard(ok), ok
    bad = {
        "a": b"import subprocess", "b": b"import os\nos.system('x')", "c": b"import sys\nsys._getframe()",
        "d": b"exec('1')", "e": b"p = '/scratch/x/data/sealed'", "f": b"import gc\ngc.get_objects()",
        "g": b"from ctypes import CDLL", "h": b"def f():\n    import inspect\n    return inspect.currentframe().f_back",
    }
    for k, src in bad.items():
        fl = audit.static_check({"train.py": src, "model.py": b"def build_model(c):\n    return None\n"})
        assert audit.has_hard(fl), (k, fl)
    fl = audit.static_check({"train.py": b"", "model.py": b"x = 1\n"})
    assert any(f["check"] == "contract_missing" for f in fl)
    fl = audit.static_check({"train.py": b"", "model.py": b"def build_model(c): pass", "sitecustomize.py": b""})
    assert any(f["check"] == "shadowing" and f["severity"] == "hard" for f in fl)


def test_l0_symlink_and_filetype(tmp_path):
    d = tmp_path / "art"
    make_artifact(str(d))
    os.symlink("/etc/hosts", d / "link.txt")
    (d / "weights.pt").write_bytes(b"\0" * 10)
    files, flags = audit.collect_artifact(str(d), max_files=32, max_file_bytes=10**6, max_total_bytes=10**6)
    checks = {f["check"] for f in flags}
    assert {"symlink", "artifact_file_type"} <= checks
    assert "link.txt" not in files and "weights.pt" not in files


def test_vanilla_start_passes_l0():
    files, flags = audit.collect_artifact(NANOCHAT_START_DIR, max_files=32, max_file_bytes=300_000,
                                          max_total_bytes=1_000_000)
    flags += audit.static_check(files)
    assert set(files) == {"train.py", "model.py"}
    assert flags == [], flags


# ---------------------------------------------------------------------------
# HTTP API + client
# ---------------------------------------------------------------------------

def test_http_client_roundtrip(svc, tmp_path):
    tok = tmp_path / "token"
    tok.write_text("t" * 40)
    os.chmod(tok, stat.S_IRUSR | stat.S_IWUSR)
    srv = EvaldServer(svc, "t" * 40).start()
    try:
        c = EvalClient(srv.url, str(tok))
        assert c.health()
        lanes = c.lanes()
        assert {l["lane"] for l in lanes} == {1, 2} and all({"gpu", "busy", "paused"} <= set(l) for l in lanes)
        art = make_artifact(str(tmp_path / "http"))
        jid = c.submit("test-http", "nanochat", art, seeds=[0], gen=1, trial=7)
        res = c.result(jid, timeout_s=300)
        assert res["status"] == "ok" and res["gen"] == 1 and res["trial"] == 7
        c.pause_lane(1)
        assert [l for l in c.lanes() if l["lane"] == 1][0]["paused"]
        c.resume_lane(1)
        bad_tok = tmp_path / "bad"
        bad_tok.write_text("x" * 40)
        with pytest.raises(EvalError, match="401"):
            EvalClient(srv.url, str(bad_tok)).lanes()
        with pytest.raises(EvalError):
            c.submit("test-http", "muar/x", art, seeds=[0])
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# scorer packing == original lib.make_dataloader packing
# ---------------------------------------------------------------------------

def _reference_pack(token_batches, n_rows, T, buffer_size):
    row_capacity = T + 1
    doc_buffer, rows = [], []
    for _ in range(n_rows):
        row, pos = [], 0
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                doc_buffer.extend(next(token_batches))
            remaining = row_capacity - pos
            best_idx, best_len = -1, 0
            for i, doc in enumerate(doc_buffer):
                if len(doc) <= remaining and len(doc) > best_len:
                    best_idx, best_len = i, len(doc)
            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                si = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(si)
                row.extend(doc[:remaining])
                pos += remaining
        rows.append(row)
    return rows


def test_packing_matches_reference():
    sys.path.insert(0, os.path.join(ROOT, "rsi", "evald", "scorer"))
    import scorer

    rng = random.Random(0)

    def gen(seed):
        r = random.Random(seed)
        while True:
            yield [[259] + [r.randrange(256) for _ in range(r.choice([1, 2, 5, 9, 30, 70, 200]))] for _ in range(37)]

    for T_, buf in ((16, 20), (64, 50), (128, 100)):
        s = rng.randrange(1000)
        a = scorer.pack_rows(gen(s), 40, T_, buffer_size=buf)
        b = _reference_pack(gen(s), 40, T_, buf)
        assert a == b


# ---------------------------------------------------------------------------
# NanoChat task
# ---------------------------------------------------------------------------

def test_nanochat_task_card_and_start():
    from rsi.tasks.base import get_task

    t = get_task("nanochat")
    card = t.task_card().lower()
    for leak in ("sealed", "6540", "6541", "6542", "devval"):
        assert leak not in card
    assert (t.start_artifact() / "train.py").is_file() and (t.start_artifact() / "model.py").is_file()
    assert t.editable_files == ("train.py", "model.py")
    fb = t.visible_feedback({"value": 1.0, "status": "ok", "audit": {"flags": []},
                             "sealed": {"value": 0.123456789}, "per_seed": []})
    assert "0.123456" not in fb


def test_vanilla_model_cpu_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vanilla_model", os.path.join(NANOCHAT_START_DIR, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.ATTN_BACKEND == "sdpa"
    model = m.build_model({"sequence_len": 64, "vocab_size": 300, "n_layer": 2, "n_head": 2, "n_embd": 64}).eval()
    x = torch.randint(0, 300, (2, 64))
    with torch.inference_mode():
        la = model(x)
        assert type(la) is torch.Tensor and la.shape == (2, 64, 300)
        x2 = x.clone()
        x2[:, 33:] = 7
        lb = model(x2)
    assert torch.allclose(la[:, :33], lb[:, :33], atol=1e-5)
    loss = model(x, x, reduction="none")
    assert loss.shape == (128,)


def test_vanilla_train_smoke_cpu(tmp_path, data_root):
    """The real start train.py/model.py, shrunk and moved to CPU by source substitution,
    through the whole pipeline (train sandbox -> export -> dev + sealed scoring)."""
    src = open(os.path.join(NANOCHAT_START_DIR, "train.py")).read()
    subs = [("DEPTH = 12", "DEPTH = 2"), ("MODEL_DIM = 768", "MODEL_DIM = 64"), ("HEAD_DIM = 128", "HEAD_DIM = 32"),
            ("TOTAL_BATCH_SIZE = 2**17", "TOTAL_BATCH_SIZE = 128"), ("DEVICE_BATCH_SIZE = 64", "DEVICE_BATCH_SIZE = 2"),
            ('torch.device("cuda")', 'torch.device("cpu")'), ('device_type="cuda"', 'device_type="cpu"'),
            ("torch.cuda.synchronize()", "None"), ("torch.compile(model, dynamic=False)", "model"),
            ('MAX_SEQ_LEN, "train")', 'MAX_SEQ_LEN, "train", buffer_size=50, device="cpu")'),
            ("torch.cuda.max_memory_allocated()", "0")]
    for a, b in subs:
        assert a in src, a
        src = src.replace(a, b)
    art = tmp_path / "vanilla"
    art.mkdir()
    (art / "train.py").write_text(src)
    (art / "model.py").write_text(open(os.path.join(NANOCHAT_START_DIR, "model.py")).read())
    s = EvalService(make_cfg(str(tmp_path / "e"), data_root, lanes=[LaneCfg(1, None)],
                             train_env_extra={"RSI_TIME_BUDGET": "3"})).start()
    try:
        res = s.result(s.submit("test-vanilla", "nanochat", str(art), seeds=[0]), wait=True, timeout_s=300)
        assert res["status"] == "ok", (res.get("error"), res.get("log_tail"), res["audit"])
        ps = res["per_seed"][0]
        assert ps["attn_backend"] == "sdpa"
        assert ps["reported_training_s"] is not None and ps["reported_training_s"] >= 3
        assert res["sealed"]["value"] is not None
        hard = [d for d in res["audit"]["details"] if d["severity"] == "hard"]
        assert not hard, hard
    finally:
        s.stop()


def test_run_seeds_csv(svc, tmp_path, capsys):
    from rsi.evald.__main__ import CSV_COLS, run_seeds

    art = make_artifact(str(tmp_path / "csv"))
    out = tmp_path / "nf.csv"
    rows = run_seeds(svc, artifact=art, seeds=[5, 6], lanes=[1, 2], run="test-csv", out_csv=str(out))
    assert [r["seed"] for r in rows] == [5, 6] and {r["lane"] for r in rows} == {1, 2}
    lines = out.read_text().strip().splitlines()
    assert lines[0].split(",") == CSV_COLS and len(lines) == 3
    assert all(r["status"] == "ok" and r["dev_bpb"] and r["sealed_bpb"] for r in rows)


# ---------------------------------------------------------------------------
# remote lanes (another Slurm job): srun launch path with a fake srun
# ---------------------------------------------------------------------------

def test_parse_lanes_and_srun_argv():
    from rsi.evald.config import parse_lanes
    from rsi.evald.sandbox import clean_slurm_env, srun_argv

    ls = parse_lanes("1,2,3-5@10000002:node-b,6:cpu")
    assert [(l.lane, l.gpu, l.jobid, l.node) for l in ls] == [
        (1, 1, None, None), (2, 2, None, None), (3, "slurm", "10000002", "node-b"),
        (4, "slurm", "10000002", "node-b"), (5, "slurm", "10000002", "node-b"), (6, None, None, None)]
    a = srun_argv(ls[2])
    assert a[:5] == ["srun", "--jobid=10000002", "--overlap", "--ntasks=1", "--nodes=1"]
    assert "--nodelist=node-b" in a and "--gres=gpu:a100:1" in a and "--cpu-bind=none" in a
    assert "--mem=60G" in a and "--cpus-per-task=6" in a
    env = clean_slurm_env({"SLURM_JOB_ID": "1", "SLURM_CONF": "/etc/slurm.conf", "SRUN_CPUS_PER_TASK": "4",
                           "PATH": "/bin"})
    assert env == {"SLURM_CONF": "/etc/slurm.conf", "PATH": "/bin"}


FAKE_SRUN = """#!/bin/sh
# fake srun: refuse an env with SLURM_* (other than SLURM_CONF), log args, drop --options, exec the rest
env | grep -q '^SLURM_TESTVAR=' && { echo 'SLURM_* leaked' >&2; exit 3; }
[ -n "$SLURM_CONF" ] || { echo 'SLURM_CONF missing' >&2; exit 4; }
echo "$@" >> "$(dirname "$0")/srun_args.log"
while [ $# -gt 0 ]; do case "$1" in --*) shift;; *) break;; esac; done
exec "$@"
"""


def test_remote_lane_via_srun(tmp_path, data_root, monkeypatch):
    srun = tmp_path / "bin" / "srun"
    srun.parent.mkdir()
    srun.write_text(FAKE_SRUN)
    srun.chmod(0o755)
    monkeypatch.setenv("SLURM_TESTVAR", "from-the-evald-step")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    lane = LaneCfg(3, None, jobid="10000002", node="node-b", srun=str(srun))
    s = EvalService(make_cfg(str(tmp_path / "e"), data_root, lanes=[lane])).start()
    try:
        info = s.lanes_info()[0]
        assert info["remote"] and info["jobid"] == "10000002"
        jid = s.submit("test-remote", "nanochat", make_artifact(str(tmp_path / "ra")), seeds=[0])
        res = s.result(jid, wait=True, timeout_s=300)
        assert res["status"] == "ok", (res.get("error"), res.get("infra_notes"), res.get("log_tail"))
        assert res["sealed"]["value"] is not None and res["lane"] == 3
        assert any("node=" in n for n in res.get("infra_notes", []))
        args = (srun.parent / "srun_args.log").read_text()
        assert "--jobid=10000002" in args and "--nodelist=node-b" in args and "rsi.evald.seed_runner" in args
        # a broken step is an infra crash, not a hang
        srun.write_text("#!/bin/sh\nexit 1\n")
        res2 = s.result(s.submit("test-remote", "nanochat", str(tmp_path / "ra"), seeds=[1]), wait=True,
                        timeout_s=120)
        assert res2["status"] == "crash" and "remote lane" in res2["error"]
    finally:
        s.stop()


def test_bwrap_argv_only_uses_bubblewrap_0_4_1_flags(tmp_path):
    from rsi.evald.sandbox import BwrapLauncher, GpuInfo, Mount, SandboxSpec

    spec = SandboxSpec(argv=["/env/bin/python", "-u", "train.py", "--not-a-bwrap-flag"],
                       env={"A": "1", "PATH": "/usr/bin"}, cwd="/work",
                       mounts=[Mount(str(tmp_path), "/work", "rw"), Mount(str(tmp_path), "/data/train", "ro"),
                               Mount(None, "/tmp", "tmpfs")],
                       gpu=GpuInfo(index=1, uuid="GPU-x", minor=1))
    argv = BwrapLauncher("bwrap").build_argv(spec)
    own = argv[1:argv.index("--")]
    flags = {a for a in own if a.startswith("--")}
    assert flags <= BwrapLauncher.SUPPORTED_0_4_1, flags - BwrapLauncher.SUPPORTED_0_4_1
    assert "--clearenv" not in argv and argv[argv.index("--") + 1:] == spec.argv
