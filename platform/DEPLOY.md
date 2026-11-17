# RSI-mini: cluster deployment (island mode, <node>)

For the main session only. Every block is meant to be pasted on the **cluster-A login node**. All
compute runs on <node> inside the Slurm hold (job <jobid>), through `$HOLD` =
`srun --jobid=<jobid> --overlap --ntasks=1 -N1`. Long-running processes run as windows of one tmux
session called `rsi`.

Rules every command here respects:
- Nothing heavy runs on the login node.
- Nothing is deleted permanently. Old things are moved with `mv <x> $R/_to_delete/`.
- Only `$R` and env `rsi` are touched. `$SCRATCH/otherproj*` and other projects are never touched.
- No token appears on a command line or in a log. Tokens live only in `$R/run/token` and `$R/run/agent_token`, both mode 600.
- Agents (OpenCode in bwrap) never see `$R/data/devval`, `$R/data/sealed`, `$R/run`, `$R/runs`, `$R/evald`, `~` or `~/.ssh`.

Component details (the source of each block below):
- `deploy/serving.md`: vLLM, gateway, G1b, 27B ladder (worker C).
- `deploy/evald.md`: tokenizer, evald, sandbox, noise floor (B).
- `deploy/second_order.md`: µAR, U_s, U_w (D).
- `deploy/campaign.md`: overnight scheduler (F).
- Harness, orchestrator, policies and lab MCP are in this file (A).

Status at the time of writing (from the coordinator): the repo is cloned at `$R/platform`; vLLM 9B and 27B are up; the main
gateway is up; OpenCode 1.19.6 and the MCP venv are installed (§4 done); G1b is running (§7). evald lanes: 1-2 on <node>
(GPU1/2) and 3-5 on <node> (job <jobid>, remote srun, set up by B; see `deploy/evald.md`). Steps already done only
need their check line. Nothing from A's part (§10, §12) has run on the cluster yet; it was tested on Mac CPU with fakes only (appendix).

---

## 0. Variables (paste in every new login shell)

```bash
export R=$R
export C=$R/platform                         # the git checkout (worker docs call it $P)
export P=$C
export PY=$SCRATCH/miniconda3/envs/rsi/bin/python   # serving env: vLLM, gateway, evald server, orchestrator, MCPs
export RSI_RUNS_ROOT=$R/runs
export HOLD="srun --jobid=<jobid> --overlap --ntasks=1 -N1"
```

## 1. Code: clone or update the private repo

```bash
module load git
mkdir -p $R/run $R/runs $R/out $R/tools $R/_to_delete
if [ -d $C/.git ]; then git -C $C pull --ff-only; else git clone https://github.com/<owner>/<repo>.git $C; fi
git -C $C log --oneline -1
chmod +x $C/bundle/*.sh $C/rsi/evald/start_evald.sh $C/rsi/tasks/nanochat/prepare_data.sh
```

The repo holds code only. Runs, tokens, data and envs live under `$R`. To update later, re-run the `git pull` line,
then restart only the components whose code changed.

## 2. Tokens (once; neither is ever printed)

- `token`: the main token. It is the vLLM API key, the gateway admin token, and the evald token.
- `agent_token`: the only token that enters researcher sandboxes. It is accepted only by the second gateway instance, `gateway_agent` (§6), which can only do model calls.

```bash
( umask 077; chmod 700 $R/run
  [ -s $R/run/token ]       || $PY -c "import secrets; print(secrets.token_hex(32))" > $R/run/token
  [ -s $R/run/agent_token ] || $PY -c "import secrets; print(secrets.token_hex(32))" > $R/run/agent_token )
chmod 600 $R/run/token $R/run/agent_token; ls -l $R/run/
```

## 3. tmux session

```bash
tmux has-session -t rsi 2>/dev/null || tmux new-session -d -s rsi -n shell
tmux ls
```

Each window below starts with `cd $C && $HOLD ...`, so its job runs on <node>. Watch a window with
`tmux attach -t rsi` and select it with `C-b w`; leave with `C-b d`. Each window ends with `; sleep 1d`, so its output stays
readable after the job exits.

## 4. Install OpenCode and the search MCPs (compute node, needs internet, about 2 min)

Already done on the cluster; check with `ls -l $R/tools/opencode/current/package/bin/opencode $R/envs/mcp/bin/duckduckgo-mcp-server $R/envs/mcp/bin/mcp-server-fetch` (the harness configs expect exactly these paths). To (re)install:

```bash
tmux new-window -t rsi -n install "cd $C && $HOLD bash -lc 'export R=$R PY=$PY; bash bundle/install_opencode.sh && bash bundle/install_mcps.sh' 2>&1 | tee $R/out/install.log; sleep 1d"
# after ~2 min:
grep -E "OPENCODE_BIN|tools:|MCPS_DONE|WARN|rror" $R/out/install.log
```

- **OpenCode** `opencode-linux-x64` 1.19.6 comes from the npm registry tarball (no node/npm needed) and is checked by sha1. It installs to
  `$R/tools/opencode/current/package/bin/opencode`. If running it gives "illegal instruction", rerun with
  `bash bundle/install_opencode.sh 1.19.6 "" linux-x64-baseline`.
- **MCPs**: `duckduckgo-mcp-server` and `mcp-server-fetch` go into the venv `$R/envs/mcp`. At run time each is wrapped by
  `rsi.mcp_lab.filter_proxy`, which applies the answer-repo blocklist (D4) and logs every query to the thread's lab outbox.

## 5. Tokenizer (CPU, about 2 min, once; B §1)

```bash
cd $C && $HOLD -c 8 bash rsi/tasks/nanochat/prepare_data.sh 2>&1 | tee $R/out/prepare_data.log
ls -l $R/data/tokenizer/
```

## 6. vLLM 9B on GPU0, then two gateways (C §1, §3)

```bash
tmux new-window -t rsi -n vllm9b "cd $C && $HOLD -c 6 --gres=gpu:a100:3 bash -lc 'export R=$R; bundle/serve_vllm.sh 9b 0 --lora' 2>&1 | tee -a $R/out/vllm9b.tmux.log; sleep 1d"
# wait for "Application startup complete" (1-3 min):
grep -m1 -E "startup complete|rror" $R/out/vllm-improver.log; cat $R/run/ports.json
```

vLLM and the main gateway are already up; only `gwa` (the agent-token gateway) is new. Next start two gateway instances. Both front the same vLLM and both log `llm_call` events and full trajectories into `$R/runs`.
- `gateway` (inbound = main token) is for trusted callers: toolcall_test, the lessons writer, the campaign.
- `gateway_agent` (inbound = agent token only) is the one OpenCode inside the sandbox talks to.

```bash
tmux new-window -t rsi -n gw "cd $C && $HOLD -c 2 bash -lc 'export R=$R; $PY -m rsi.gateway --runs-root $R/runs --ports-key gateway' 2>&1 | tee -a $R/out/gateway.log; sleep 1d"
tmux new-window -t rsi -n gwa "cd $C && $HOLD -c 2 bash -lc 'export R=$R; $PY -m rsi.gateway --runs-root $R/runs --ports-key gateway_agent --inbound-token-file $R/run/agent_token' 2>&1 | tee -a $R/out/gateway_agent.log; sleep 1d"
sleep 20; grep -h "listening" $R/out/gateway.log $R/out/gateway_agent.log | tail -2; cat $R/run/ports.json
```

Expected in ports.json: `vllm_improver`, `gateway` and `gateway_agent` (and `evald` later).

## 7. Gate G1b: tool-call validity of the 9B (C §4; about 10 min)

```bash
cd $C && $HOLD -c 2 bash -lc "export R=$R; $PY bundle/toolcall_test.py --model improver --out $R/out/g1b-improver.json" 2>&1 | tail -5
```

The exit code is 0 when the gate passes (`call_validity_rate ≥ 0.90`, `episode_completion_rate ≥ 0.70`). If it fails, the
fallback order is in `deploy/serving.md` §4: `--tool-choice required`, then `RSI_THINKING=off`, then the harness fallback (§10).

## 8. evald sandbox self-test (B §2; must print `"ok": true`)

```bash
cd $C && $HOLD -c 8 --gres=gpu:a100:3 bash -c "cd $C && $PY -m rsi.evald selftest --R $R --gpu 1 --smoke" 2>&1 | tee $R/out/evald_selftest.log | tail -20
```

If `--gres` is rejected for an overlapping step, drop it. If triton cannot find a C compiler, run `module load gcc; which gcc` and add
that prefix to `extra_ro_binds` (`deploy/evald.md` §6).

## 9. Noise floor S0 (gate G1a; B §3; about 1.5 h; must run BEFORE evald is started)

20 seeds of the vanilla start, 10 per lane (GPU1, GPU2). The training process has a hard wall-clock limit. The margin here is a generous
600 s so that startup time can be measured; the D5 margin is then chosen from `startup_s`.

```bash
tmux new-window -t rsi -n noise "cd $C && $HOLD -c 24 --gres=gpu:a100:3 bash bundle/noise_floor.sh 20 1,2 2>&1 | tee $R/out/noise_floor.log; sleep 1d"
# afterwards: one CSV line per seed + summary (mean/sd of dev BPB, sealed BPB, wall times, startup_s)
ls -t $R/out/noise_floor_*.csv | head -1; cat $(ls -t $R/out/noise_floor_*.summary.json | head -1)
```

Steps 10a and 10b below need only GPU0 (vLLM) and can run while the noise floor occupies GPU1 and GPU2.

## 10. Researcher sandbox self-test and one real OpenCode episode (A; gate before any campaign)

```bash
# (a) bwrap probes: devval, sealed, token, ~/.ssh, runs, other projects must all be DENIED; workspace, outbox and internet must WORK
cd $C && $HOLD -c 2 bash -lc "export R=$R; $PY -m rsi.harness.selftest sandbox --config configs/harness_cluster.json" 2>&1 | tee $R/out/sandbox_selftest.log | tail -18
#     expected last line: SANDBOX_SELFTEST OK
# (b) one real OpenCode episode: bwrap -> OpenCode -> gateway_agent -> vLLM, with lab + DDG + fetch MCPs
cd $C && $HOLD -c 2 bash -lc "export R=$R; $PY -m rsi.harness.selftest episode --config configs/harness_cluster.json --model improver" 2>&1 | tee $R/out/opencode_episode.log | tail -15
#     expected: EPISODE_SELFTEST OK, plus the events path (agent_spawn, tool_call, llm_call, message, agent_end)
```

If (a) fails:
- The "internet" probe fails because of DNS: `/etc/resolv.conf` may be a symlink into a directory that is not bound. Run
  `ls -l /etc/resolv.conf` and add that directory to `ro_paths` in `configs/harness_cluster.json` and in
  `configs/nanochat_karpathy_9b.yaml`.
- Any "must be denied" probe passes: **stop**. Do not start a campaign. That is a leak.

If (b) fails, look at `$R/out/opencode_episode.log` and the `trial-00001.stderr.txt` next to the printed trajectory.
- The provider package download failed. OpenCode fetches `@ai-sdk/openai-compatible` into the sandbox HOME on first use, which needs internet. Rerun.
- OpenCode rejects a config key. Override it through `opencode_extra` in the config, or remove the key in
  `rsi/harness/opencode.py::build_config`.
- OpenCode cannot be made to work. Switch both configs to the in-repo fallback harness:
  `--set harness.kind=mini` on the orchestrator command line. It uses the same bwrap sandbox for bash and the same lab spool;
  it has no web tools and no subagents (= the web-off arm). Record this as a deviation: the plan's fallbacks were
  pi and mini-swe-agent, but pi needs Node, which the node lacks.

## 11. Start evald with the chosen margin (B §4; lanes 3-5 on <node> per `deploy/evald.md`)

Choose the startup margin `M` (seconds) from `startup_s` in the noise-floor summary; this is the user's call (D5). Then:

```bash
export M=<startup margin in s>
tmux new-window -t rsi -n evald "cd $C && $HOLD -c 24 --gres=gpu:a100:3 bash rsi/evald/start_evald.sh 1,2 --margin-s $M 2>&1 | tee -a $R/out/evald.log; sleep 1d"
# lanes 3-5 (<node>, job <jobid>): follow the remote-lane block in deploy/evald.md (B), then check that lanes() lists 1-5
sleep 15; cd $C && $PY -c "from rsi.evald.client import EvalClient as C; c=C.from_run_dir('$R/run'); print(c.health(), c.lanes())"
```

Before the campaign, also set the same margin in the NanoChat task card. Add it to the run config's task args
(`--set task.args.startup_margin_s=$M` on the orchestrator line) so the improver is told the real limit.

## 12. First campaign: NanoChat, 9B, karpathy_greedy, 4 threads sharing 5 eval lanes (A)

Preconditions:
- §6: vLLM and both gateways are up.
- §7: G1b passed.
- §8 and §10: printed OK.
- §9: the noise floor is done and reviewed.
- §11: evald is up; `lanes()` lists lanes 1-5. If only lanes 1-2 exist, add `--set lanes=[1,2] --set policy.threads=2`.

```bash
tmux new-window -t rsi -n camp1 "cd $C && $HOLD -c 4 bash -lc 'export R=$R RSI_RUNS_ROOT=$R/runs; $PY -m rsi.orchestrator --config configs/nanochat_karpathy_9b.yaml --set task.args.startup_margin_s=$M' 2>&1 | tee -a $R/out/camp1.log; sleep 1d"
sleep 120; export RUN=$(ls -t $R/runs | grep nanochat-9b-karpathy | head -1); echo $RUN
cd $C && $PY -m rsi.orchestrator --status $R/runs/$RUN     # counts per status, pending trials, best so far
tail -n 5 $R/runs/$RUN/events.jsonl
```

What it does:
1. The vanilla start is evaluated once as the baseline.
2. Each of the 4 threads loops independently: one OpenCode episode (one edit of `train.py`/`model.py`, then `lab.submit_eval`), one
   evald run on the next free lane (shared FIFO over lanes 1-5), then keep or discard decided by the orchestrator (D5 wall-clock limit, sealed score private).
3. The run stops at `budget.trials` (90) or `wall_s` (14 h), whichever comes first.

Pause, resume, validate:

```bash
cd $C && $PY -m rsi.orchestrator --pause $R/runs/$RUN       # threads stop after their current trial; evald finishes its queue
tmux new-window -t rsi -n camp1r "cd $C && $HOLD -c 4 bash -lc 'export R=$R RSI_RUNS_ROOT=$R/runs; $PY -m rsi.orchestrator --config configs/nanochat_karpathy_9b.yaml --resume --set run=$RUN' 2>&1 | tee -a $R/out/camp1.log; sleep 1d"
cd $C && $PY -m rsi.events $R/runs/$RUN/events.jsonl        # schema check of the replay log
```

Ctrl-C in the window, or SIGTERM from Slurm, pauses rather than kills. After a hard kill, `--resume` waits again for evald jobs that were
in flight (by job id) and finishes trials that were proposed but not yet decided.

**Alternative to §12: F's overnight autopilot.** `deploy/campaign.md` runs the whole schedule: gates, µAR, U_s/U_w generations,
the other policies, endpoint confirmation, packing. Use **either** §12 **or** the autopilot, never both at once: they would
compete for the same two eval lanes. To hand over, pause §12 first (`--pause`), then:
```bash
cd $C && $HOLD bash bundle/run_campaign.sh --check                   # prints the plan, triggers and deadlines
tmux new-window -t rsi -n campaign "cd $C && $HOLD -c 4 bash bundle/run_campaign.sh 2>&1 | tee -a $R/out/campaign.tmux.log; sleep 1d"
tail -f $R/runs/campaign_status.md
```
F's scheduler assumes the ports keys `gateway`, `vllm` and `evald`, and gate results under `$R/runs/gates/` (see `NOTES_F.md` and
`deploy/campaign.md` "启动前要满足"). Check `bundle/run_campaign.sh --check` before handing over.

## 13. Bring results back for the replay (main session)

```bash
cd $R/runs && tar czf $R/out/$RUN-light.tgz $RUN/events.jsonl $RUN/ledger.jsonl $RUN/config.json $RUN/state.json $RUN/artifacts/*.diff
ls -l $R/out/$RUN-light.tgz        # small; full trajectories (LoRA data) and sealed.jsonl stay on the cluster
```

The main session copies `$R/out/$RUN-light.tgz` to the Mac over its existing connection. The visualizer replays
`events.jsonl` alone.

## 14. Optional: 27B ladder arm and the second-order kit

- 27B on <node> (job <jobid>, TP=2): `deploy/serving.md` §2. Its gateway route is `ladder-27b`. G1b for it is the same command
  as §7 with `--model ladder-27b`.
- µAR on node CPUs, U_s / U_w, LoRA on a borrowed lane: `deploy/second_order.md`.

## 15. Stop everything cleanly

```bash
cd $C && $PY -m rsi.orchestrator --pause $R/runs/$RUN      # then wait for the camp1 window to print its summary
tmux send-keys -t rsi:evald C-c; tmux send-keys -t rsi:gwa C-c; tmux send-keys -t rsi:gw C-c; tmux send-keys -t rsi:vllm9b C-c
```

---

## Appendix A: what is tested where

| piece | tested locally (Mac CPU) | first cluster check |
|---|---|---|
| events schema, ledger (torn-line safe), artifact store, nAUC | `tests/test_core_events.py` | §12 `rsi.events` check |
| bwrap argv for researcher sandbox, forbidden-bind checks | argv only (`tests/test_core_sandbox_lab.py`); no bwrap on macOS | §10a |
| lab MCP (stdio JSON-RPC), web blocklist proxy | `tests/test_core_sandbox_lab.py` | §10b |
| OpenCode adapter (config, NDJSON parsing, subagents, export) | against a stub binary (`tests/test_core_opencode.py`); the real binary was not run on the Mac | §10b |
| orchestrator + karpathy_greedy + pause/resume + crash recovery + campaign config shape | `tests/test_core_e2e.py` | §12 |
| full loop: real gateway, real µAR CPU training, fake model, mini harness | `python -m rsi.smoke` → `samples/events_smoke.jsonl` | — |
| vLLM, gateway, G1b | C: `tests/test_gateway*.py`, `toolcall_test.py --fake` | §6, §7 |
| evald, NanoChat task, noise floor | B: `tests/test_evald*.py` (sandbox=none) | §8, §9, §11 |

## Appendix B: known gaps

- The researcher sandbox and the evaluator run as the same Unix user as everything else, so bwrap is not a security boundary against kernel or bwrap bugs.
- The web log (`lab/outbox/web.jsonl`) is written from inside the sandbox, so the agent could tamper with it. The tamper-proof record is the gateway
  trajectories, which contain every tool result the model saw.
- `selftest episode` checks one episode only. The 30-episode gate is G1b (§7). It calls the model directly with OpenCode-like tool schemas,
  not through the OpenCode binary.
- The harness's OpenCode config keys were taken from the OpenCode docs and not validated against 1.19.6. §10b is the first real check.
