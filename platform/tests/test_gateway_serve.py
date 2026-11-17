"""bundle/serve_vllm.sh end to end on CPU (owner: worker C).

A shim ``<env>/bin/vllm`` runs tests/fake_openai_server.py on the port and served-model-name the
script chose, requiring the VLLM_API_KEY it was given. This checks argv/env handling, ports.json,
and the --expose edge gateway (cluster IP, token required) without a GPU.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rsi.gateway import GatewayClient, GatewayConfig, GatewayThread  # noqa: E402
from rsi.gateway.ports import cluster_ip, read_ports  # noqa: E402

TOKEN = "tok-" + "x" * 20

SHIM = """#!/bin/bash
# fake `vllm serve <model> ... --port P --served-model-name N ...`
PORT=; NAME=; prev=
for a in "$@"; do
  [[ $prev == --port ]] && PORT=$a
  [[ $prev == --served-model-name ]] && NAME=$a
  prev=$a
done
echo "{\\"models\\": [\\"$NAME\\"], \\"default\\": {\\"content\\": \\"from $NAME\\"}}" > "$(dirname "$0")/script-$NAME.json"
printf '%s\\n' "$@" > "$(dirname "$0")/argv-$NAME.txt"
env | grep -E '^(VLLM_|CUDA_VISIBLE_DEVICES|HF_)' | sed 's/^VLLM_API_KEY=.*/VLLM_API_KEY=<set>/' > "$(dirname "$0")/env-$NAME.txt"
exec "{py}" "{fake}" --port "$PORT" --token-env VLLM_API_KEY --script "$(dirname "$0")/script-$NAME.json"
"""


@pytest.fixture
def env(tmp_path):
    r = tmp_path / "R"
    (r / "run").mkdir(parents=True)
    tok = r / "run" / "token"
    tok.write_text(TOKEN + "\n")
    tok.chmod(0o600)
    envdir = tmp_path / "env"
    (envdir / "bin").mkdir(parents=True)
    shim = envdir / "bin" / "vllm"
    shim.write_text(SHIM.replace("{py}", sys.executable).replace("{fake}", str(HERE / "fake_openai_server.py")))
    shim.chmod(0o755)
    e = dict(os.environ, R=str(r), RSI_SERVE_ENV=str(envdir), RSI_SERVE_PY=sys.executable)
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return r, envdir, e


def _start(args, e):
    return subprocess.Popen(["bash", str(ROOT / "bundle" / "serve_vllm.sh"), *args], env=e,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def _wait_ok(url, token, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(url + "/v1/models", headers={"Authorization": f"Bearer {token}"})
            if urllib.request.urlopen(req, timeout=2).status == 200:
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise AssertionError(f"{url} not up")


def _wait_key(path, key, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = read_ports(str(path)).get(key)
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError(f"{key} never written to {path}")


def test_local_mode(env, tmp_path):
    r, envdir, e = env
    p = _start(["9b", "0", "--lora"], e)
    try:
        url = _wait_key(r / "run" / "ports.json", "improver")
        port = read_ports(str(r / "run" / "ports.json"))["vllm_improver"]
        assert url == f"http://127.0.0.1:{port}"
        _wait_ok(url, TOKEN)
        argv = (envdir / "bin" / "argv-improver.txt").read_text().split("\n")
        assert TOKEN not in "\n".join(argv)
        for flag in ("--enable-auto-tool-choice", "--language-model-only", "--enable-lora"):
            assert flag in argv
        assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
        assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
        envtxt = (envdir / "bin" / "env-improver.txt").read_text()
        assert "VLLM_API_KEY=<set>" in envtxt and "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True" in envtxt
        assert "CUDA_VISIBLE_DEVICES=0" in envtxt
        cfg = GatewayConfig(runs_root=str(tmp_path / "runs"), ports_path=str(r / "run" / "ports.json"),
                            upstream_token=TOKEN, inbound_token=TOKEN, quiet=True)
        with GatewayThread(cfg) as gw:
            out = GatewayClient(gw.url, token=TOKEN).chat("improver", [{"role": "user", "content": "x"}])
            assert out["choices"][0]["message"]["content"] == "from improver"
        meta = json.loads((r / "run" / "vllm_improver.json").read_text())
        assert meta["tp"] == 1 and meta["lora"] is True
    finally:
        _stop(p)


def test_bad_token_mode_refused(env):
    r, _, e = env
    (r / "run" / "token").chmod(0o644)
    p = subprocess.run(["bash", str(ROOT / "bundle" / "serve_vllm.sh"), "9b", "0"], env=e, capture_output=True,
                       text=True, timeout=30)
    assert p.returncode == 3 and "chmod 600" in p.stderr


def test_expose_mode_tp2(env, tmp_path):
    try:
        ip = cluster_ip()
    except Exception:
        pytest.skip("no non-loopback IPv4 on this machine")
    r, envdir, e = env
    pj = r / "run" / "ports.json"
    pj.write_text(json.dumps({"gateway": 1111, "vllm_ladder-27b": 2222}))
    e = dict(e, CUDA_VISIBLE_DEVICES="0,1", BIND_IP=ip)
    p = _start(["27b", "slurm", "--expose"], e)
    try:
        url = _wait_key(pj, "ladder-27b")
        assert url.startswith(f"http://{ip}:")
        d = read_ports(str(pj))
        assert "vllm_ladder-27b" not in d and d["gateway"] == 1111  # stale local key removed, others kept
        _wait_ok(url, TOKEN)
        argv = (envdir / "bin" / "argv-ladder-27b.txt").read_text().split("\n")
        assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"  # vLLM itself never leaves loopback
        assert "CUDA_VISIBLE_DEVICES=0,1" in (envdir / "bin" / "env-ladder-27b.txt").read_text()
        # the edge requires the token, and exposes no unauthenticated vLLM routes
        for path, body in (("/v1/models", None), ("/invocations", b"{}"), ("/tokenize", b"{}")):
            req = urllib.request.Request(url + path, data=body)
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=5)
            assert ei.value.code in (401, 404)
        # main gateway (on the orchestrator node) routes ladder-27b to the edge URL from ports.json
        cfg = GatewayConfig(runs_root=str(tmp_path / "runs"), ports_path=str(pj), upstream_token=TOKEN,
                            inbound_token=TOKEN, quiet=True)
        with GatewayThread(cfg) as gw:
            out = GatewayClient(gw.url, token=TOKEN).chat("ladder-27b", [{"role": "user", "content": "x"}],
                                                          run="r", agent="a", stream=True)
            assert out["choices"][0]["message"]["content"] == "from ladder-27b"
        # only the main gateway records (edge runs --no-record)
        assert (tmp_path / "runs" / "r" / "trajectories" / "gateway" / "a.jsonl").exists()
        assert not (r / "runs").exists()
    finally:
        _stop(p)
