# Serving: vLLM, logging gateway, G1b tool-call gate (owner: worker C)

Topology (island mode, plus a second hold for the 27B ladder arm):

```
<node>  job <jobid>  3x A100 (orchestrator node)          <node>  job <jobid>  2x A100 (2 days)
 GPU0  vLLM Qwen3.5-9B  "improver"   127.0.0.1:<p1>             GPU0+1  vLLM Qwen3.8-27B "ladder-27b"  TP=2
 CPU   rsi.gateway (logging)          127.0.0.1:<gw>                    127.0.0.1:<p2>   (loopback only)
         |  routes improver, improver@<adapter> -> <p1>           CPU   edge gateway  <4024-ip>:<p3>
         |  routes ladder-27b ------------------- cluster net --------> (token required, --no-record)
 harness / policies / toolcall_test -> gateway only
```

`$R/run/ports.json` (shared cluster FS, merged under flock) after everything is up:
```json
{"gateway": 41234, "improver": "http://127.0.0.1:38211", "vllm_improver": 38211,
 "ladder-27b": "http://<ip>:45120", "evald": ...}
```
The gateway resolves model `X` from key `"X"` (URL, preferred) or `"vllm_X"` (int port on its own node), and
re-reads the file whenever it changes, so you can restart vLLM on a new port without restarting the gateway.

## 0. Once per node session (login node, tmux)

```bash
export R=$R
export P=$R/platform                      # where the platform/ tree is synced (adjust if A's DEPLOY.md says otherwise)
export PY=$SCRATCH/miniconda3/envs/rsi/bin/python
mkdir -p $R/run $R/out
# token: create ONCE, mode 600. Never echo it, never put it on a command line.
[ -s $R/run/token ] || (umask 077; $PY -c 'import secrets; print(secrets.token_hex(24))' > $R/run/token)
chmod 600 $R/run/token
tmux new -s rsi-serve        # windows below: vllm9b, gw, vllm27b, g1b
```

## 1. vLLM 9B on GPU0 of <node> (tmux window `vllm9b`)

```bash
tmux new-window -t rsi-serve -n vllm9b
srun --jobid=<jobid> --overlap --ntasks=1 -N1 --gres=gpu:a100:1 \
  bash -lc 'export R=$R; cd $R/platform && bundle/serve_vllm.sh 9b slurm --lora'
```
- `slurm` = keep the GPU that `--gres=gpu:a100:1` gave the step. The first log lines print `nvidia-smi -L`.
  **Check that this is physical GPU0.** evald assumes GPU1/GPU2 are the eval lanes. If it is not GPU0,
  restart with `--gres=gpu:a100:3` and an explicit index: `bundle/serve_vllm.sh 9b 0 --lora`.
- Expected log (`$R/out/vllm-improver.log`, also shown in the window):
  ```
  [serve_vllm] 2026-11-17 23:10:02 improver (Qwen/Qwen3.5-9B) node=<node> GPUs=0 tp=1 -> http://127.0.0.1:38211 lora=1 expose=0
  ... INFO ... Starting vLLM API server ... on http://127.0.0.1:38211
  ... INFO:     Application startup complete.
  ```
  Loading takes about 1-3 min (weights 19.3 GB, plus torch.compile and CUDA graphs on first start).
- Dry run without a GPU: `bundle/serve_vllm.sh 9b 0 --lora --dry-run` prints the full command.
- Resolved command (9B): `vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 --port <rand> --served-model-name improver
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 131072 --gpu-memory-utilization 0.85
  --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --enable-lora --max-loras 4 --max-lora-rank 64 --max-cpu-loras 8`, with env `VLLM_API_KEY` (from the token
  file), `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, `HF_HOME=$R/hf`, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES`.

## 2. 27B on <node>, TP=2, reachable from <node> (tmux window `vllm27b`)

```bash
tmux new-window -t rsi-serve -n vllm27b
srun --jobid=<jobid> --overlap --ntasks=1 -N1 --gres=gpu:a100:2 \
  bash -lc 'export R=$R; cd $R/platform && bundle/serve_vllm.sh 27b slurm --expose'
```
- `--expose`: vLLM binds **127.0.0.1** on <node>. An edge gateway (`python -m rsi.gateway --no-record`) binds
  the node's cluster IP (auto-detected; override with `BIND_IP=...` and check with `hostname -I`) on a random
  high port, requires `Authorization: Bearer <$R/run/token>` on every route except `/health`, and only forwards
  `/v1/chat/completions`, `/v1/completions` and `/v1/models`. ports.json gets
  `"ladder-27b": "http://<4024-ip>:<p3>"`, and any stale `vllm_ladder-27b` key is removed.
- Why vLLM is not bound to the cluster IP directly: its `--api-key` only covers `/v1`, `/v2`, `/inference` and `/cohere`.
  `/invocations` (full inference), `/tokenize`, `/score`, `/pooling`, `/classify`, `/rerank` and `/metrics` stay open
  to every user on the cluster network (vLLM security docs). The edge gateway closes those routes.
- If vLLM or the edge dies, the script stops the other one and exits non-zero. Re-run the same command; it picks a new port
  and the main gateway follows ports.json.
- Defaults for TP=2: `--max-model-len 131072 --gpu-memory-utilization 0.90` (weights 55.6 GB split over 2 GPUs).
  Logs: `$R/out/vllm-ladder-27b.log`, `$R/out/edge-ladder-27b.log`.
- Expected:
  ```
  [serve_vllm] ... ladder-27b (Qwen/Qwen3.8-27B) node=<node> GPUs=0,1 tp=2 -> http://<ip>:45120 lora=0 expose=1
  [serve_vllm] vllm pid=... (log ...)  edge pid=... (log .../edge-ladder-27b.log)  url=http://<ip>:45120
  ```
  Loading takes about 3-6 min.
- Reachability check from <node> (no token on argv; the client reads the file):
  ```bash
  srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash -lc 'export R=$R; cd $R/platform &&
    '$PY' -c "
  from rsi.gateway.ports import read_ports; from rsi.gateway import GatewayClient
  u = read_ports()[\"ladder-27b\"]; c = GatewayClient(u, token_path=\"$R/run/token\")
  print(u, c.models()); print(c.chat(\"ladder-27b\", [{\"role\":\"user\",\"content\":\"Say OK.\"}], max_tokens=512)[\"choices\"][0][\"message\"])"'
  ```
  Expected: `http://<ip>:45120 ['ladder-27b']`, then a message dict. Also check that an unauthenticated request is
  refused: `curl -s -o /dev/null -w '%{http_code}\n' -X POST http://<ip>:<p3>/invocations` should print `401`.
- To run the 27B on <node> GPU2 instead (the original ladder plan): `bundle/serve_vllm.sh 27b 2` (TP=1, 65536 ctx,
  0.92 util). That writes `"ladder-27b": "http://127.0.0.1:<p>"` and replaces the remote URL.

## 3. Logging gateway on <node> (tmux window `gw`)

```bash
tmux new-window -t rsi-serve -n gw
srun --jobid=<jobid> --overlap --ntasks=1 -N1 \
  bash -lc 'export R=$R; cd $R/platform && '$PY' -m rsi.gateway --runs-root $R/runs'
```
- It binds 127.0.0.1:<random high port> and writes `"gateway": <port>` into ports.json. Expected stderr:
  ```
  GATEWAY_PORT=41234
  23:12:40 gateway listening on http://127.0.0.1:41234/v1 runs_root=$R/runs upstreams={'improver': 'http://127.0.0.1:38211', 'ladder-27b': 'http://<ip>:45120'} inbound_auth=on
  23:13:05 gateway nanochat-9b-... thread-0 gen=0 trial=3 improver ok 200 4.2s pt=5321 ct=412 tc=1 tries=1
  ```
- Clients (harness, policies, U_s) use `rsi.gateway.gateway_url()` and the token file. Tag every call with headers
  `X-RSI-Run`, `X-RSI-Agent`, `X-RSI-Gen`, `X-RSI-Trial`, and optionally `X-RSI-Parent`. A harness that can only set a base
  URL can use `http://127.0.0.1:<gw>/rsi/t/<run>/<agent with / as %2F>/<gen>/<trial or ->/v1`.
- Per call it writes:
  - one `llm_call` event to `$R/runs/<run>/events.jsonl` with `model, prompt_tokens, completion_tokens, latency_s`
    and optional `ttft_s, status, stream, n_tool_calls, finish_reason`. No content goes into events.
  - one line in `$R/runs/<run>/trajectories/gateway/<agent>.jsonl` (`/` becomes `__`) with keys `request` (the full
    body: messages, tools, sampling), `response` (`message` with content, reasoning_content and tool_calls,
    reassembled for streams), `finish_reason`, `usage`, `latency_s`, `ttft_s`, `status`, `attempts`, and `gen`/`trial`/`parent`.
    This file is the LoRA SFT source.
- Busy or down upstream: connection errors and HTTP 429/502/503/504 are retried with exponential backoff
  (`--max-retries 8`, `--retry-deadline-s 120`). After that the client gets **503 `upstream_unavailable`**, which the
  orchestrator should treat as pause, not as a failed trial. Untagged calls go to run `_untagged`.
- Flags: `--upstream improver=http://...` (static, overrides ports.json), `--inbound-token-file` (a separate
  token for sandboxed clients; admin routes still need the main token), `--no-inbound-auth`, `--max-inflight 64`.

### Runtime LoRA adapters (needs `--lora` at vLLM start)
vLLM 0.34 loads adapters at runtime through `POST /v1/load_lora_adapter {"lora_name","lora_path"}` and
`/v1/unload_lora_adapter` only when `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. serve_vllm.sh sets this with `--lora`.
Go through the gateway admin route, which namespaces the name:
```python
from rsi.gateway import GatewayClient, gateway_url
c = GatewayClient(gateway_url(), token_path="$R/run/token")   # main token required
c.load_adapter("g1", "$R/adapters/g1", base="improver")     # -> model "improver@g1"
c.chat("improver@g1", msgs, ...);  c.unload_adapter("g1")
```
The adapter's rank must be ≤ `MAX_LORA_RANK` (default 64; allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512).
`MAX_LORAS=4` sets how many adapters can be in one batch. To preload adapters at start: `LORA_MODULES="g1=/abs/path"
bundle/serve_vllm.sh 9b slurm --lora`.
**Unverified for Qwen3.5:** `Qwen3_5ForCausalLMBase` declares `SupportsLoRA`. One user report (vLLM 0.23, Qwen3.8-27B) says
adapters on the Gated-DeltaNet `in_proj_qkv`/`out_proj` did not take effect when loaded dynamically. For D:
target attention and MLP projections (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`), and check on the
cluster that `improver@gX` output differs from `improver` before relying on it.

## 4. G1b tool-call gate (tmux window `g1b`)

```bash
tmux new-window -t rsi-serve -n g1b
srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash -lc 'export R=$R; cd $R/platform &&
  '$PY' bundle/toolcall_test.py --model improver   --out $R/out/g1b-improver.json &&
  '$PY' bundle/toolcall_test.py --model ladder-27b --out $R/out/g1b-ladder-27b.json'
```
- 30 episodes (6 read, 6 bash, 6 read→edit, 4 web_search, 4 read→edit→bash→lab_submit_eval,
  4 web_search→read→edit→lab_submit_eval), half streamed and half not, concurrency 4, temperature 0.6, top_p 0.95,
  top_k 20, max_tokens 8192, thinking left to the server default. Exit code 0 = gate passed.
- Expected stderr lines like `[ 7/30] ep-06 bash  stream=0 complete=1 steps=1/1 errors=[]`, then JSON on stdout:
  ```json
  {"gate": "G1b", "pass": true, "call_validity_rate": 0.97, "turn_validity_rate": 0.95,
   "episode_completion_rate": 0.9, "clean_completion_rate": 0.83, "schema_errors": {"no_tool_call": 3, ...},
   "by_tool": {...}, "by_mode": {"stream": {...}, "nonstream": {...}}, "latency_s": {"mean": .., "p50": .., "p90": ..},
   "ttft_s": {...}, "tokens": {...}, "run": "toolcall-g1b-...", ...}
  ```
  The numbers above only show the format. On CPU, `--fake` gives call_validity 0.9333 and completion 0.9667 with 6 injected faults.
  Trajectories of the test go to `$R/runs/toolcall-g1b-*/trajectories/gateway/toolcall__ep-XX.jsonl`.
- Gate thresholds (build plan §6.8): `call_validity_rate ≥ 0.90` and `episode_completion_rate ≥ 0.70`.
  If it fails, try these in order (each is one env change and a vLLM restart, or one flag):
  1. `--tool-choice required` (vLLM uses structured outputs, so the calls always parse).
  2. `--thinking off`, or start vLLM with `RSI_THINKING=off` (the thinking setting must then be the same in every arm; write it in PREREG).
  3. `TOOL_PARSER=qwen3_xml`. Note that in 0.34 this is the **same** `Qwen3EngineToolParser` as `qwen3_coder`, so expect no
     change. `hermes`/`qwen3_json` parse JSON-in-`<tool_call>` and do not match Qwen3.5's XML format.
  4. Then follow the plan (pi / mini-swe-agent / 9B only for µAR JSON).
- CPU self-test on the Mac: `python bundle/toolcall_test.py --fake --out /tmp/g1b_fake.json`.

## 5. Stop / restart
- Ctrl-C in the window stops the srun step. vLLM and the edge also stop when the step ends.
- Restarting vLLM gives a new port in ports.json, and the gateway follows it automatically (in-flight calls get 503 and should be retried by the caller).
- `$R/run/vllm_<name>.json` records node, GPUs, TP, port/URL, parsers and thinking mode for the run report.

## 6. Settings that must go in PREREG
Thinking mode (`RSI_THINKING`, default = model default = on), sampling (G1b uses 0.6/0.95/20), `max-model-len`
(9B 131072; 27B TP2 131072), tool parser `qwen3_coder`, reasoning parser `qwen3`, `--language-model-only` (no vision).
Prefix caching is left at vLLM's default. Force it with `PREFIX_CACHING=1|0`; the recipe calls prefix caching with
Mamba "align" mode experimental.

## Tests (CPU, Mac)
`python -m pytest tests/test_gateway.py tests/test_gateway_toolcall.py tests/test_gateway_serve.py -q` runs 21 tests.
test_gateway_serve runs the real serve_vllm.sh with a shim `vllm` (the fake server), in both local and `--expose` modes.
`tests/fake_openai_server.py` is the shared fake model server: `python tests/fake_openai_server.py --port N --script s.json`
(`FAKE_OPENAI_PORT=<port>` on stdout), or in-process `with FakeServer(script) as fs: fs.url`. The script format is in its docstring.

## Sources (checked 2026-11-17)
- Qwen3.5-9B model card, https://huggingface.co/Qwen/Qwen3.5-9B. It gives `--reasoning-parser qwen3`,
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder` and `--language-model-only`, a 262,144-token native context
  (recommends ≥128K to keep thinking), thinking on by default (`chat_template_kwargs.enable_thinking`), and sampling presets.
- vLLM recipe for Qwen3.5/3.6, https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html. It uses the same parsers
  and says prefix caching with Mamba "align" mode is experimental.
- vLLM v0.34.0 source: `vllm/tool_parsers/__init__.py` registers `qwen3_coder` and `qwen3_xml` to `Qwen3EngineToolParser`,
  and `qwen3_json` to hermes. `vllm/reasoning/__init__.py` registers `qwen3`. `vllm/model_executor/models/qwen3_5.py`
  has `Qwen3_5ForCausalLMBase(..., SupportsLoRA, ...)`.
- vLLM v0.34.0 CLI, https://docs.vllm.ai/en/v0.34.0/cli/serve.html. Covers `--max-loras` (default 1), `--max-lora-rank`
  (choices up to 512, default 16), `--max-cpu-loras`, `--lora-modules` (JSON with `base_model_name`),
  `--default-chat-template-kwargs`, `--language-model-only`, and `--enable-prefix-caching/--disable-prefix-caching`.
- vLLM env vars (v0.34.0), https://docs.vllm.ai/en/v0.34.0/configuration/env_vars.html. Covers `VLLM_API_KEY` and
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING` ("1"/"true").
- vLLM LoRA docs, https://docs.vllm.ai/en/latest/features/lora.html. Covers `/v1/load_lora_adapter` and
  `/v1/unload_lora_adapter`, `load_inplace`, and the warning that runtime updating should only be used in an isolated, trusted environment.
- vLLM security docs (v0.34.0), https://docs.vllm.ai/en/v0.34.0/usage/security.html. The API key covers only
  `/v1`, `/v2`, `/inference` and `/cohere`. `/invocations` and the others are unauthenticated, so put vLLM behind a reverse proxy.
- Unofficial: discuss.vllm.ai threads on Qwen3.5 LoRA (vLLM 0.17 MoE backend, vLLM 0.23 GDN modules), not verified here.
