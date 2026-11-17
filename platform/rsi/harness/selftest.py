"""Researcher-sandbox self-test and a one-episode OpenCode smoke (run ON the cluster node).

    python -m rsi.harness.selftest sandbox  --config configs/harness_cluster.json
    python -m rsi.harness.selftest episode  --config configs/harness_cluster.json [--model improver]

`sandbox` runs probes inside the same bwrap sandbox a researcher thread gets and prints PASS/FAIL
per line; exit code 0 only if every "must fail" probe failed and every "must work" probe worked.
`episode` runs one real OpenCode episode against the gateway in a throwaway workspace and checks
that the model edited the file and called lab submit_eval (gate G1b smoke, not the full 30-episode test).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

from rsi import events
from rsi.harness.sandbox import SandboxSpec, wrap


def _cfg(path: str) -> dict:
    cfg = json.loads(os.path.expandvars(pathlib.Path(path).read_text()))
    if cfg.get("selftest_dir"):
        os.makedirs(cfg["selftest_dir"], exist_ok=True)
    return cfg


def probes(R: str, real_home: str) -> list[tuple[str, str, bool]]:
    """(name, shell command, must_succeed)."""
    return [
        ("write workspace", "echo ok > /work/probe.txt && cat /work/probe.txt", True),
        ("write lab outbox", "echo ok > /lab/outbox/probe.txt", True),
        ("python runs", "python3 -c 'print(1)' || ls /usr/bin/python3", True),
        ("internet (search)", "python3 - <<'EOF'\nimport urllib.request\n"
                              "print(urllib.request.urlopen('https://html.duckduckgo.com/html/?q=test', timeout=15).status)\nEOF", True),
        ("no GPU visible", "test -z \"$CUDA_VISIBLE_DEVICES\" && ! ls /dev/nvidia0", True),
        ("read ~/.ssh", f"ls {real_home}/.ssh", False),
        ("read real HOME", f"ls {real_home}", False),
        ("read devval", f"ls {R}/data/devval", False),
        ("read sealed", f"ls {R}/data/sealed", False),
        ("read token", f"cat {R}/run/token", False),
        ("read runs root", f"ls {R}/runs", False),
        ("read other project", "ls /scratch/user/ | grep -i otherproj", False),
        ("write lab (ro)", "echo x > /lab/card.md", False),
        ("write system", "echo x > /usr/probe", False),
        ("write code", f"touch {R}/code/rsi-mini/probe", False),
    ]


def cmd_sandbox(cfg: dict) -> int:
    R = os.environ.get("R") or os.environ.get("RSI_ROOT") or "/scratch/user/rsi"
    real_home = os.path.expanduser("~")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sbx-", dir=cfg.get("selftest_dir") or None))
    (tmp / "workspace").mkdir()
    (tmp / "lab" / "outbox").mkdir(parents=True)
    (tmp / "lab" / "card.md").write_text("card")
    (tmp / "home").mkdir()
    spec = SandboxSpec(workspace=str(tmp / "workspace"), lab=str(tmp / "lab"), home=str(tmp / "home"),
                       ro_paths=list(cfg.get("ro_paths", [])), share_net=True,
                       env={"PATH": cfg.get("sandbox_path", "/usr/local/bin:/usr/bin:/bin")})
    bad = 0
    for name, sh, must in probes(R, real_home):
        argv, env, cwd = wrap(spec, ["/bin/bash", "-c", sh], cfg.get("sandbox", "bwrap"))
        try:
            r = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0
            out = (r.stdout + r.stderr).strip().replace("\n", " | ")[:160]
        except subprocess.TimeoutExpired:
            ok, out = False, "timeout"
        good = ok == must
        bad += 0 if good else 1
        print(f"{'PASS' if good else 'FAIL'}  {name:22s} expected={'works' if must else 'denied'} "
              f"got={'works' if ok else 'denied'}  {out}", flush=True)
    print("SANDBOX_SELFTEST", "OK" if bad == 0 else f"FAILED ({bad})")
    return 0 if bad == 0 else 1


def cmd_episode(cfg: dict, model: str) -> int:
    from rsi.harness.mini import make_harness

    root = pathlib.Path(tempfile.mkdtemp(prefix="oc-smoke-", dir=cfg.get("selftest_dir") or None))
    run = f"selftest-{int(time.time())}"
    events.register_run(run, str(root / run / "events.jsonl"))
    events.emit(run, "run_start", "orchestrator", policy="selftest", task="selftest",
                improver={"model": model, "scaffold_hash": None}, budget={"trials": 1, "wall_s": None},
                config_hash="selftest")
    ws = root / run / "threads" / "thread-0" / "workspace"
    ws.mkdir(parents=True)
    (ws / "train.py").write_text("LEARNING_RATE = 9e-4\nWIDTH = 64\n")
    lab = ws.parent / "lab"
    (lab / "outbox").mkdir(parents=True)
    (lab / "card.md").write_text("Selftest task: set LEARNING_RATE in train.py to 2e-3.")
    cfg = dict(cfg)
    cfg["trajectories_dir"] = str(root / run / "trajectories" / "harness")
    h = make_harness(cfg)
    prompt = ("Open train.py in your working directory and change LEARNING_RATE from 9e-4 to 2e-3. "
              "Change nothing else. Then call the lab tool submit_eval with hypothesis "
              "'selftest: lr 2e-3', and stop.")
    out = h.run_episode(run=run, agent="thread-0", gen=0, trial=1, workspace=ws, prompt=prompt, model=model,
                        scaffold_dir=None, budget_s=int(cfg.get("selftest_budget_s", 600)))
    edited = "2e-3" in (ws / "train.py").read_text()
    print(json.dumps({"ok": out.get("ok"), "edited": edited, "submitted": out.get("submitted"),
                      "n_tool_calls": out.get("n_tool_calls"), "final": (out.get("final") or "")[:200],
                      "events": str(root / run / "events.jsonl"), "trajectory": out.get("trajectory")}, indent=1))
    good = bool(out.get("ok") and edited and out.get("submitted"))
    print("EPISODE_SELFTEST", "OK" if good else "FAILED")
    return 0 if good else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sandbox", "episode"])
    ap.add_argument("--config", required=True, help="harness config JSON (same as the run config's harness block)")
    ap.add_argument("--model", default="improver")
    a = ap.parse_args(argv)
    cfg = _cfg(a.config)
    if a.what == "sandbox":
        return cmd_sandbox(cfg)
    if not cfg.get("gateway_url"):
        from rsi.orchestrator import service_url

        cfg["gateway_url"] = service_url("gateway")
    return cmd_episode(cfg, a.model)


if __name__ == "__main__":
    sys.exit(main())
