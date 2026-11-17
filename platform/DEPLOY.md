# RSI-mini: cluster deployment (island mode, <node>)

For the main session only. Every block is meant to be pasted on the **cluster-A login node**. All
compute runs on the compute node, inside the Slurm hold (job <jobid>), through
`srun --jobid=<jobid> --overlap --ntasks=1 -N1 ...`. Long-running processes run in tmux windows
of one session called `rsi`.

Rules these commands respect:
- Nothing heavy runs on the login node.
- Nothing is deleted permanently; superseded things are moved to `$R/_to_delete/`.
- Only `$R` and the envs listed below are touched; `$SCRATCH/otherproj*` and other projects' dirs are never touched.
- No token ever appears on a command line, in a file other than `$R/run/token` and `$R/run/agent_token` (both mode 600), or in a log.
- Agents never see `$R/data/devval`, `$R/data/sealed`, `$R/run`, `$R/runs`, `~` or `~/.ssh`.

Component details are in `deploy/serving.md` (vLLM, gateway, G1b), `deploy/evald.md` (evaluator and
noise floor), `deploy/second_order.md` (µAR, U_s, U_w) and `deploy/campaign.md` (overnight scheduler).
This file gives the order of operations and the exact commands for the first night.

---

## 0. Variables (paste in every new login shell)

```bash
export R=$R
export C=$R/code/rsi-mini
export PY=$SCRATCH/miniconda3/envs/rsi/bin/python      # serving env: orchestrator, gateway, evald server, MCP
export TPY=$R/envs/rsi-train/bin/python                             # training env: artifact training + scoring
export RSI_RUNS_ROOT=$R/runs
export HOLD="srun --jobid=<jobid> --overlap --ntasks=1 -N1"
```

## 1. Code: clone or update the private repo

```bash
module load git
mkdir -p $R/code $R/run $R/runs $R/out $R/tools $R/_to_delete
if [ -d $C/.git ]; then git -C $C pull --ff-only; else git clone https://github.com/<owner>/<repo>.git $C; fi
git -C $C log --oneline -1
```

The repo is read-only code. Runs, tokens, data and envs all live outside it, under `$R`.

## 2. Tokens (once)

Two random tokens, each created in a 600 file. Neither is printed.
- `token` is the main token: the vLLM API key, the gateway admin token, and the evald token.
- `agent_token` is the only token that enters researcher sandboxes. The gateway accepts it for model calls and nothing else.

```bash
( umask 077
  [ -s $R/run/token ]       || $PY -c "import secrets; print(secrets.token_hex(32))" > $R/run/token
  [ -s $R/run/agent_token ] || $PY -c "import secrets; print(secrets.token_hex(32))" > $R/run/agent_token )
chmod 600 $R/run/token $R/run/agent_token; ls -l $R/run/
```

## 3. tmux session

```bash
tmux has-session -t rsi 2>/dev/null || tmux new-session -d -s rsi -n shell
tmux ls
```

To watch a window, use `tmux attach -t rsi` (detach with `C-b d`). Every `tmux new-window` command below starts its job through `$HOLD`, so the job runs on <node>.

## 4. Install OpenCode and the search MCPs (compute node, needs internet; ~2 min)

```bash
tmux new-window -t rsi -n install "cd $C && $HOLD bash -lc 'export R=$R; bash bundle/install_opencode.sh && PY=$PY bash bundle/install_mcps.sh' 2>&1 | tee $R/out/install.log; sleep 600"
# check:
grep -E "OPENCODE_BIN|tools:|MCPS_DONE|WARN" $R/out/install.log
```

- OpenCode: `opencode-linux-x64` 1.19.6, taken from the npm registry tarball (no node/npm needed) and checked by sha1. If it fails with "illegal instruction", the CPU lacks AVX2; rerun with `bash bundle/install_opencode.sh 1.19.6 "" linux-x64-baseline`.
- MCPs: `duckduckgo-mcp-server` and `mcp-server-fetch` are installed into the venv `$R/envs/mcp`. At run time each is wrapped by `rsi.mcp_lab.filter_proxy`, which applies the answer-repo blocklist and logs every query.

<!-- SERVING -->

<!-- EVALD -->

## 8. Researcher-sandbox self-test and one real OpenCode episode (gate before any campaign)

```bash
# (a) bwrap probes: every "denied" probe must be denied (devval, sealed, token, ~/.ssh, runs, other projects)
$HOLD bash -lc "cd $C && export R=$R && $PY -m rsi.harness.selftest sandbox --config configs/harness_cluster.json" 2>&1 | tee $R/out/sandbox_selftest.log
# expected last line: SANDBOX_SELFTEST OK
# (b) one real OpenCode episode through gateway -> vLLM, inside bwrap, with lab + DDG + fetch MCPs
$HOLD bash -lc "cd $C && export R=$R && $PY -m rsi.harness.selftest episode --config configs/harness_cluster.json --model improver" 2>&1 | tee $R/out/opencode_episode.log
# expected: EPISODE_SELFTEST OK; the events path is printed (agent_spawn, tool_call, llm_call ... events)
```

If (a) fails because `gcc`, `git`, or some other tool sits outside `/usr`, that does not affect the
researcher; only the evaluator needs those. If DNS fails inside the sandbox (the "internet" probe), add the resolver
directory that `/etc/resolv.conf` points to (see `ls -l /etc/resolv.conf`) to `ro_paths` in
`configs/harness_cluster.json` and in the run config.

If (b) fails, read `$R/out/opencode_episode.log` and the `trial-00001.stderr.txt` printed next to the
trajectory. The usual causes and fixes:
- The provider package download failed (OpenCode fetches `@ai-sdk/openai-compatible` into the sandbox HOME on first use; that needs internet): rerun.
- OpenCode rejects a config key: add the override under `harness.opencode_extra`, or remove the key.
- OpenCode cannot be made to work: switch the run config to the in-repo fallback harness with `harness.kind: mini`. It uses the same sandbox and lab spool, has no web tools and no subagents, and must be recorded as a deviation (the plan's fallbacks were pi and mini-swe-agent; pi needs Node, which the node does not have).

<!-- TOOLCALL -->

<!-- NOISE -->

## 11. First campaign: NanoChat, 9B, karpathy_greedy, 2 threads on 2 eval lanes

Preconditions: vLLM is up (§5), the gateway is up (§6), evald is up with lanes 1 and 2 (§7), §8 printed both OK lines, G1b passed (§9), and the noise floor is done (§10).

```bash
tmux new-window -t rsi -n camp1 "cd $C && $HOLD bash -lc 'export R=$R RSI_RUNS_ROOT=$RSI_RUNS_ROOT; $PY -m rsi.orchestrator --config configs/nanochat_karpathy_9b.yaml' 2>&1 | tee -a $R/out/camp1.log"
sleep 60; ls -t $R/runs | head -3                       # the run id is nanochat-9b-karpathy-<UTC timestamp>
export RUN=$(ls -t $R/runs | grep nanochat-9b-karpathy | head -1)
$PY -m rsi.orchestrator --status $R/runs/$RUN           # counts per status, pending trials, best so far
tail -n 5 $R/runs/$RUN/events.jsonl
```

Pause, resume, and check on a run:

```bash
$PY -m rsi.orchestrator --pause $R/runs/$RUN        # threads stop after the current trial; evald finishes queued jobs
# resume (same window or a new one); the frozen config.json in the run dir is used:
tmux new-window -t rsi -n camp1r "cd $C && $HOLD bash -lc 'export R=$R RSI_RUNS_ROOT=$RSI_RUNS_ROOT; $PY -m rsi.orchestrator --config configs/nanochat_karpathy_9b.yaml --resume --set run=$RUN' 2>&1 | tee -a $R/out/camp1.log"
$PY -m rsi.events $R/runs/$RUN/events.jsonl         # schema check of the replay log
```

Ctrl-C in the window, or SIGTERM from Slurm, also pauses rather than kills: in-flight episodes and
evaluations finish and every state change is already in `ledger.jsonl`. After a hard kill, `--resume`
re-awaits in-flight evald jobs by job id and finishes trials that were proposed but not yet decided.

## 12. Bring results back for the replay (main session)

```bash
cd $R/runs && tar czf $R/out/$RUN-light.tgz $RUN/events.jsonl $RUN/ledger.jsonl $RUN/config.json $RUN/state.json $RUN/artifacts/*.diff
ls -l $R/out/$RUN-light.tgz        # typically < 5 MB; full trajectories stay on the cluster (LoRA data)
```

The main session then copies `$R/out/$RUN-light.tgz` to the Mac over its existing master connection.

---

## Appendix: what is tested where

| piece | tested locally (Mac CPU) | needs the cluster |
|---|---|---|
| events schema, ledger, artifact store, nAUC | `tests/test_core_events.py` | — |
| bwrap argv and forbidden-bind checks | argv construction only (`tests/test_core_sandbox_lab.py`) | real bwrap: §8a |
| lab MCP (stdio JSON-RPC), web blocklist proxy | `tests/test_core_sandbox_lab.py` | DDG/fetch servers: §4, §8b |
| OpenCode adapter (config, NDJSON parsing, subagents, export) | against a stub binary (`tests/test_core_opencode.py`) | real OpenCode 1.19.6: §8b |
| orchestrator + karpathy_greedy + pause/resume + crash recovery | `tests/test_core_e2e.py` | §11 |
| full local loop (real gateway, real µAR CPU training, fake model) | `python -m rsi.smoke` → `samples/events_smoke.jsonl` | — |
