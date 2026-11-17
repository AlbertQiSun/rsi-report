"""Campaign scheduler tests (worker F). CPU only, fake components, compressed clock.

    cd platform && python -m pytest tests/test_campaign.py -q        # ≈ 3–4 min

The dry-runs replay the real campaign/overnight.yaml (41 h of plan in ≈ 1 min) with
rsi.campaign.fakes standing in for the orchestrator, evald lanes, LoRA training, vLLM and gates.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from rsi import events
from rsi.campaign import analysis
from rsi.campaign.config import load_config, parse_time

HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORM = os.path.dirname(HERE)
CONFIG = os.path.join(PLATFORM, "campaign", "overnight.yaml")
CID = load_config(CONFIG, root_override="/tmp/x")["campaign"]   # e.g. ovn1118b
CAMP = f"runs/campaign-{CID}"


def env(**extra):
    e = dict(os.environ)
    e["PYTHONPATH"] = PLATFORM
    for k in ("RSI_FAKE_GATE_FAIL", "RSI_FAKE_CRASH", "RSI_FAKE_SFT_N", "R"):
        e.pop(k, None)
    e.update({k: str(v) for k, v in extra.items()})
    return e


def cmd(root, *args, scale=3000):
    return [sys.executable, "-m", "rsi.campaign", "--config", CONFIG, "--dry-run", "--root", str(root),
            "--time-scale", str(scale), "--quiet", *args]


def run(root, *args, scale=3000, timeout=400, **e):
    return subprocess.run(cmd(root, *args, scale=scale), cwd=PLATFORM, env=env(**e), timeout=timeout,
                          capture_output=True, text=True)


def state(root):
    with open(os.path.join(root, CAMP, "state.json")) as f:
        return json.load(f)


def wait_for(pred, timeout=120, every=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if pred():
                return True
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(every)
    return False


def lane_log(root):
    p = os.path.join(root, CAMP, "fake", "lane_log.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def lanes_json(root):
    return json.load(open(os.path.join(root, CAMP, "fake", "lanes.json")))


def camp_events(root):
    return events.validate_file(os.path.join(root, CAMP, "events.jsonl"))


def no_live_children(root):
    out = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True).stdout
    return not [l for l in out.splitlines() if str(root) in l and "rsi.campaign" in l]


# ---------------------------------------------------------------------------
# static checks on the plan
# ---------------------------------------------------------------------------

def test_plan_fits_hold_window():
    cfg = load_config(CONFIG, root_override="/tmp/x")
    tz = cfg["_tz"]
    assert cfg["_gpu_stop"] <= cfg["_hold_end"] - 1.5 * 3600, "need ≥1.5 h between gpu_stop and hold end"
    assert cfg["_planned_start"] == parse_time("2026-11-17T22:45", tz)
    steps = {s["id"]: s for s in cfg["steps"]}
    for s in cfg["steps"]:
        if s["_deadline"] is not None:
            assert s["_deadline"] <= cfg["_gpu_stop"], s["id"]
        assert s.get("plan"), f"{s['id']} has no planned window for the human table"
    # LoRA always pauses lane 2 and serialises on gpu2; the remote 27B never borrows a lane
    for s in cfg["steps"]:
        if s["kind"] == "lora" and not s.get("remote"):
            assert s["holds"] == ["gpu2"] and int(s.get("lane", 2)) == 2
        if s["kind"] in ("lora", "service") and s.get("remote"):   # 27B chain on node-d: no eval lane
            assert s["lane"] is None and s["holds"] == ["gpu-node-d"] and s["remote"]["jobid"] == 10000004
    assert steps["ladder_nc"]["after"] == steps["nc_g0"]["after"], "27B ladder starts with G0"
    assert "gpu2" not in steps["ladder_nc"]["holds"]
    # the LoRA lane is not one the 9B G0 (or the ladder) depends on; G0 uses three dedicated lanes
    lora_lane = cfg["lora_lane"]
    assert lora_lane not in steps["nc_g0"]["run"]["lanes"] and lora_lane not in steps["ladder_nc"]["run"]["lanes"]
    assert len(steps["nc_g0"]["run"]["lanes"]) == 3 and set(cfg["lanes"]) == {1, 2, 3, 4, 5}
    # no two NanoChat runs that can overlap in time are pinned to the same lane (checked via after-chains)
    for s in cfg["steps"]:
        if s["kind"] == "run" and s["id"] not in ("nc_g0", "ladder_nc"):
            assert s.get("after") or s.get("needs"), s["id"]
    assert "!ladder27b_remote" in steps["ladder_fb_host"]["requires_flags"]
    # NanoChat eval budget stays well under the ≈750-eval lane capacity estimate (plan §6.5)
    n_eval = 20 + sum(int(s.get("stop_when_trials") or 0) for s in cfg["steps"]
                      if s["kind"] == "run" and not s["id"].startswith("ladder_fb"))
    ep = steps["endpoint"]
    n_eval += (ep["top_k"] + 1) * len(ep["seeds"])
    assert n_eval <= 0.6 * 5 * 8.5 * 42, n_eval   # ≤ 60% of the 5-lane capacity estimate
    # held-out arms share seeds with A0a; A/A seeds are disjoint
    mu = cfg["muar"]["seeds"]
    assert not set(mu["heldout"]) & set(mu["aa"])
    assert not set(mu["g0"]) & set(mu["d1"]) and not set(mu["d1"]) & set(mu["d2"])


def test_delta2_rule():
    base = {f"t{j}": {s: 0.40 + 0.01 * ((j + s) % 3) for s in range(5)} for j in range(8)}
    better = {t: {s: v + 0.05 for s, v in d.items()} for t, d in base.items()}
    aa = {t: {s: v + 0.002 * (-1) ** s for s, v in d.items()} for t, d in base.items()}
    placebo = {t: {s: v + 0.01 for s, v in d.items()} for t, d in base.items()}
    r = analysis.delta2_decision(better, base, aa=(base, aa), placebo=placebo, n_boot=500)
    assert r["pass"] and abs(r["main"]["delta"] - 0.05) < 1e-9 and r["main"]["p_signflip"] <= 1 / 128
    r2 = analysis.delta2_decision(placebo, base, aa=(base, aa), placebo=better, n_boot=500)
    assert not r2["pass"]  # better than base but not better than "placebo"
    worse = {t: {s: v - 0.05 for s, v in d.items()} for t, d in base.items()}
    assert analysis.delta2_decision(worse, base, n_boot=300)["harmful"]


# ---------------------------------------------------------------------------
# full dry-run of the whole plan
# ---------------------------------------------------------------------------

def test_full_campaign_dry_run(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    st = state(tmp_path)
    S = st["steps"]
    assert st["ended"]
    must = ["g1a_noise", "g1b_toolcall", "g1b_27b", "g1c_evald", "g2_muar_smoke", "nc_g0", "ladder_nc", "mu_g0",
            "mu_a0a", "mu_a0b", "us_lessons_mu", "mu_a1", "mu_a1p", "dec_us", "sft_m1", "sft_m1p", "lora_g1",
            "mu_a2", "mu_a2p", "dec_g1", "lora_g2", "mu_a3", "mu_a3c", "dec_g2", "us_lessons_nc", "cf_m0_r1",
            "cf_s1_r1", "cf_m1_r1", "cf_m1p_r1", "cf_m0_r2", "cf_s1_r2", "cf_m1_r2", "cf_m1p_r2",
            "pol_recursive", "pol_naive", "nc_weboff", "endpoint",
            # 27B self-evolution chain (node-d)
            "mu_g0_27b", "mu_a0b_27b", "us_lessons_mu_27b", "mu_a1_27b", "mu_a1p_27b", "dec_us_27b",
            "sft_m1_27b", "lora_g1_27b", "serve_evo_g1", "mu_a2_27b", "mu_a2p_27b", "mu_d1_27b", "dec_g1_27b"]
    bad = {k: (S[k]["status"], S[k].get("reason")) for k in must if S[k]["status"] != "done"}
    assert not bad, bad
    for k, v in S.items():  # whatever was skipped was skipped for a stated reason
        assert v["status"] in ("done", "partial", "skipped"), (k, v)
        if v["status"] == "skipped":
            assert v.get("reason"), k
    assert S["ladder_fb_host"]["status"] == "skipped"  # remote 27B passed its gate

    # trial-count trigger: G0 stopped at 2×30 trials, and the CFF arms started only after it ended
    runs = os.path.join(tmp_path, "runs")
    noisy = load_config(CONFIG, root_override="/tmp/x")["flags"].get("nanochat_noisy")
    assert analysis.run_stats(os.path.join(runs, f"{CID}-nc-g0", "events.jsonl"))["n_trials"] == (72 if noisy else 90)
    for cf in ("cf_m0_r1", "cf_s1_r1", "cf_m1_r1"):
        assert S[cf]["t_start"] >= S["nc_g0"]["t_end"]
        assert analysis.run_stats(os.path.join(runs, f"{CID}-{cf.replace('_', '-')}", "events.jsonl"))["n_trials"] == 12
    # ladder ran in parallel with G0 (own node), not after it
    assert abs(S["ladder_nc"]["t_start"] - S["nc_g0"]["t_start"]) < 1800

    # lane pause/resume: alternating, ends resumed, every adapter load happens inside a pause window
    log = lane_log(tmp_path)
    assert [e["op"] for e in log] == ["pause", "resume"] * (len(log) // 2) and len(log) >= 4
    assert all(e["lane"] == 2 for e in log)
    windows = [(log[i]["vt"], log[i + 1]["vt"]) for i in range(0, len(log), 2)]
    adapters = [json.loads(l) for l in open(os.path.join(tmp_path, CAMP, "fake", "adapters.jsonl"))]
    assert {"m1", "m1p", "m2", "m2c"} <= {a["name"] for a in adapters}
    for a in adapters:
        if not a["name"].endswith("-27b"):  # 9B adapters: loaded while lane 2 is paused
            assert any(lo <= a["vt"] <= hi for lo, hi in windows), a
    # 27B chain: trained on node-d (no eval lane paused), served only after training, arms after serving
    assert {"m1-27b", "m1p-27b"} <= {a["name"] for a in adapters}
    assert S["serve_evo_g1"]["t_start"] >= S["lora_g1_27b"]["t_end"]
    assert S["mu_a2_27b"]["t_start"] >= S["serve_evo_g1"]["t_start"]
    assert S["serve_evo_g1"]["t_end"] >= S["mu_a2_27b"]["t_end"]   # served until its arms finished
    dj = json.load(open(os.path.join(tmp_path, CAMP, "decisions.json")))
    pairs = {r["comparison"]: r for r in dj["_9b_vs_27b"]}
    assert pairs["U_s gen 1"]["9B"] and pairs["U_s gen 1"]["27B"] and pairs["U_w gen 1"]["27B"]
    assert not any(v["paused"] for v in lanes_json(tmp_path).values())
    # M1 arms ran only after the adapter was loaded
    m1 = next(a for a in adapters if a["name"] == "m1")
    assert S["mu_a2"]["t_start"] >= m1["vt"]

    # decisions, events, status
    assert set(st["decisions"]) >= {"dec_us", "dec_g1", "dec_g2", "g2b_beats_rlocal"}
    assert st["decisions"]["dec_g1"]["main"]["delta"] is not None
    evs = camp_events(tmp_path)
    kinds = [e["ev"] for e in evs]
    assert kinds.count("run_start") == 1 and kinds[-1] == "run_end"
    assert kinds.count("weights_update") >= 4 and kinds.count("lessons_update") >= 3 and "decision" in kinds
    assert any(e.get("campaign_ev") == "lane_pause" for e in evs)
    md = open(os.path.join(runs, "campaign_status.md")).read()
    assert md.count("| P") >= 20 and "DECISION" in md and "campaign END" in md and "9B vs 27B" in md
    assert no_live_children(tmp_path)


# ---------------------------------------------------------------------------
# crash of the campaign process + crash of one orchestrator run
# ---------------------------------------------------------------------------

def test_resume_after_crash(tmp_path):
    p = subprocess.Popen(cmd(tmp_path, "--until", "2026-11-18T05:00", scale=2500), cwd=PLATFORM,
                         env=env(RSI_FAKE_CRASH="nc-g0"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert wait_for(lambda: state(tmp_path)["steps"]["lora_g1"].get("stage") == "train", timeout=120)
        p.send_signal(signal.SIGKILL)  # the campaign dies mid-LoRA; its children live on in their own sessions
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()
    before = state(tmp_path)
    assert before["paused_lanes"].get("2") == "lora_g1"
    r = run(tmp_path, "--resume", "--until", "2026-11-18T05:00", scale=2500, RSI_FAKE_CRASH="nc-g0")
    assert r.returncode == 0, r.stderr[-2000:]
    st = state(tmp_path)
    S = st["steps"]
    assert S["lora_g1"]["status"] == "done" and S["dec_g1"]["status"] == "done"
    assert S["nc_g0"]["restarts"] >= 1, "injected orchestrator crash must be restarted with --resume"
    nc = events.read_events(os.path.join(tmp_path, "runs", f"{CID}-nc-g0", "events.jsonl"))
    assert sum(e["ev"] == "run_start" for e in nc) == 1, "a resumed run must not start over"
    evs = camp_events(tmp_path)
    assert sum(e["ev"] == "run_start" for e in evs) == 1
    assert any(e.get("campaign_ev") == "resume" for e in evs)
    adapters = [json.loads(l)["name"] for l in open(os.path.join(tmp_path, CAMP, "fake", "adapters.jsonl"))]
    assert adapters.count("m1") == 1 and adapters.count("m1p") == 1
    assert not any(v["paused"] for v in lanes_json(tmp_path).values())
    assert wait_for(lambda: no_live_children(tmp_path), timeout=20)


# ---------------------------------------------------------------------------
# STOP file for the human, then resume
# ---------------------------------------------------------------------------

def test_stop_file_and_resume(tmp_path):
    p = subprocess.Popen(cmd(tmp_path, "--until", "2026-11-18T04:00", scale=2500), cwd=PLATFORM, env=env(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert wait_for(lambda: state(tmp_path)["steps"]["lora_g1"].get("stage") == "train", timeout=120)
        with open(os.path.join(tmp_path, CAMP, "STOP"), "w") as f:
            f.write("stop\n")
        assert p.wait(timeout=120) == 0
    finally:
        if p.poll() is None:
            p.kill()
    st = state(tmp_path)
    assert not st.get("ended")
    assert st["steps"]["lora_g1"]["status"] == "interrupted"
    assert not [k for k, v in st["steps"].items() if v["status"] == "running"]
    assert st["steps"]["nc_g0"]["status"] in ("interrupted", "pending")  # µAR/LoRA do not wait for G1a
    assert not st["paused_lanes"] and not any(v["paused"] for v in lanes_json(tmp_path).values())
    assert wait_for(lambda: no_live_children(tmp_path), timeout=20)
    md = open(os.path.join(tmp_path, "runs", "campaign_status.md")).read()
    assert "campaign STOPPED" in md
    # refuses to resume while STOP exists
    r = run(tmp_path, "--resume", "--until", "2026-11-18T04:00", scale=2500)
    assert r.returncode == 2
    os.remove(os.path.join(tmp_path, CAMP, "STOP"))
    r = run(tmp_path, "--resume", "--until", "2026-11-18T04:00", scale=2500)
    assert r.returncode == 0, r.stderr[-2000:]
    S = state(tmp_path)["steps"]
    assert S["lora_g1"]["status"] == "done"
    nc = events.read_events(os.path.join(tmp_path, "runs", f"{CID}-nc-g0", "events.jsonl"))
    assert sum(e["ev"] == "run_start" for e in nc) == 1
    log = lane_log(tmp_path)
    assert [e["op"] for e in log][-1] == "resume"


# ---------------------------------------------------------------------------
# gate failure -> human window -> fallback; and a human override
# ---------------------------------------------------------------------------

def test_gate_fail_fallback(tmp_path):
    r = run(tmp_path, "--until", "2026-11-18T01:30", scale=2500, RSI_FAKE_GATE_FAIL="g1b_toolcall,g1b_27b")
    assert r.returncode == 0, r.stderr[-2000:]
    st = state(tmp_path)
    S = st["steps"]
    assert st["flags"]["nanochat_9b"] is False and st["flags"]["ladder27b_remote"] is False
    assert S["nc_g0"]["status"] == "skipped" and S["ladder_nc"]["status"] == "skipped"
    assert S["mu_g0"]["status"] == "done"  # µAR track unaffected
    # the human window was honoured before the fallback
    assert S["nc_g0"]["t_end"] - S["g1b_toolcall"]["t_start"] >= 1800
    md = open(os.path.join(tmp_path, "runs", "campaign_status.md")).read()
    assert "ALERT gate g1b_toolcall FAILED" in md


def test_human_override_force_pass(tmp_path):
    p = subprocess.Popen(cmd(tmp_path, "--until", "2026-11-18T00:45", scale=1500), cwd=PLATFORM,
                         env=env(RSI_FAKE_GATE_FAIL="g1c_evald"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert wait_for(lambda: state(tmp_path)["steps"]["g1c_evald"]["status"] == "waiting_human", timeout=60)
        with open(os.path.join(tmp_path, CAMP, "override.yaml"), "w") as f:
            f.write("force_pass: [g1c_evald]\n")
        assert p.wait(timeout=120) == 0
    finally:
        if p.poll() is None:
            p.kill()
    st = state(tmp_path)
    assert st["steps"]["g1c_evald"]["status"] == "done"
    assert st["flags"]["nanochat"] is True
    assert st["steps"]["nc_g0"]["status"] in ("running", "partial", "done")


# ---------------------------------------------------------------------------
# glue to other workers' interfaces
# ---------------------------------------------------------------------------

def test_runner_captures_multiline_json(tmp_path):
    from rsi.campaign.runner import main as runner_main

    script = tmp_path / "s.py"
    script.write_text("import json; print('noise'); print(json.dumps({'dev_bpb': {'sd': 0.01}, 'n': 3}, indent=1))\n")
    res = tmp_path / "r.json"
    runner_main(["--result", str(res), "--log", str(tmp_path / "l.txt"), "--", sys.executable, str(script)])
    out = json.load(open(res))
    assert out["rc"] == 0 and out["out"]["dev_bpb"]["sd"] == 0.01
    from rsi.campaign.config import run_checks
    assert run_checks(out["out"], {"dev_bpb.sd": "<= 0.02", "n": ">= 3"})[0]


def test_ports_resolution(tmp_path):
    from rsi.campaign.services import Ports

    p = tmp_path / "ports.json"
    p.write_text(json.dumps({"gateway": 18000, "evald": "http://127.0.0.1:18100", "vllm_improver": {"port": 18200}}))
    ports = Ports(str(p))
    assert ports.url("gateway") == "http://127.0.0.1:18000" and ports.get("gateway") == "18000"
    assert ports.url("evald") == "http://127.0.0.1:18100" and ports.url("vllm_improver").endswith(":18200")
    assert "no-port" in ports.url("missing")


def test_decision_action_uses_worker_d_analysis(tmp_path):
    pytest.importorskip("numpy")
    from rsi.campaign.actions import do_decision

    tasks = [f"muar/t{j}" for j in range(6)]
    def curves(shift):
        return {t: {str(s): [1.0 - shift - 0.01 * i for i in range(12)] for s in range(4)} for t in tasks}
    arms = {"arms": {"A": curves(0.0), "B": curves(0.2), "AA": curves(0.0), "P": curves(0.05)}}
    (tmp_path / "arms.json").write_text(json.dumps(arms))
    (tmp_path / "sref.json").write_text(json.dumps({t: {"s_ref": 0.0, "s0": 1.2} for t in tasks}))
    f = str(tmp_path / "arms.json")
    spec = {"sref_path": str(tmp_path / "sref.json"), "runs": {},
            "arms_files": {k: [f, k] for k in ("A", "B", "AA", "P")}, "aa": ["A", "AA"],
            "compare": [["B", "A", "P"], ["B", "AA"]], "names": ["B vs AA"], "n_trials": 12, "n_boot": 500,
            "floor": "per_task"}
    res = do_decision(spec)
    assert res["main"]["delta"] > 0 and res["also"][0]["name"] == "B vs AA"
    assert set(res["criteria"]) >= {"ci_lower_gt_0", "gt_2x_aa", "beats_placebo"}
    assert do_decision({**spec, "arms_files": {}})["nodata"]


def test_remote_27b_commands(tmp_path):
    """Real-mode command lines for the 27B chain: srun into job 10000004 with a cleaned SLURM env."""
    from rsi.campaign.scheduler import Campaign

    cfg = load_config(CONFIG, root_override=str(tmp_path))
    camp = Campaign(cfg, dry_run=False, quiet=True)
    camp.clock = __import__("rsi.campaign.config", fromlist=["Clock"]).Clock()
    camp.state = {"steps": {}, "flags": {}, "paused_lanes": {}}
    lora = camp.steps["lora_g1_27b"]
    argv = camp.build_cmd(lora.cfg["train_cmd"], sft_dir="/s", adapter_out="/a/m1-27b", gpu=lora.gpu())
    os.environ["SLURM_JOB_ID"] = "10000001"
    try:
        full, envr = camp.remote_wrap(lora.cfg, argv)
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
    assert full[:5] == ["srun", "--jobid=10000004", "--overlap", "--ntasks=1", "-N1"]
    assert "--gres=gpu:a100:2" in full and "--cpu-bind=none" in full
    assert "RSI_BASE_MODEL=Qwen/Qwen3.8-27B" in full and "RSI_LORA_DEVICE_MAP=auto" in full
    assert full[-3:] == [os.path.join(cfg["_ctx"]["bundle"], "lora_train.sh"), "/s", "/a/m1-27b"]
    assert "SLURM_JOB_ID" not in envr
    svc = camp.steps["serve_evo_g1"]
    svc.c.steps["lora_g1_27b"].st["adapters"] = {"m1-27b": "/a/m1-27b", "m1p-27b": "/a/m1p-27b"}
    mods = ",".join(f"{n}={p}" for n, p in sorted(svc.lora_modules().items()))
    argv = camp.build_cmd(svc.cfg["cmd"], gpu=svc.gpu(), lora_modules=mods)
    assert "SERVED_NAME=ladder-27b-evo" in argv and "LORA_MODULES=m1-27b=/a/m1-27b,m1p-27b=/a/m1p-27b" in argv
    assert argv[-4:] == ["27b", "slurm", "--lora", "--expose"]
    assert svc.lane() is None and lora.lane() is None and camp.steps["lora_g1"].lane() == 2
