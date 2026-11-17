# RSI-mini event log schema (v1)

One JSON object per line in `runs/<run_id>/events.jsonl`. It is append-only, and every component writes through `platform/rsi/events.py:emit()`. The schema is shared by the orchestrator, the harness adapters, evald and the replay visualizer. Do not change field meanings; add new optional fields only.

## Common fields (every event)
| field | type | meaning |
|---|---|---|
| `t` | float | unix time (seconds, ms precision) |
| `run` | str | run id, e.g. `nanochat-9b-karpathy-20261117T2300` |
| `ev` | str | event type (see below) |
| `agent` | str | actor id: `orchestrator`, `evald`, `thread-0`, `thread-0/sub-2`, `lessons-writer`, ... |
| `parent` | str or null | parent agent id (for spawn trees / multi-agent topology) |
| `gen` | int | improver generation k (0 = I0) |
| `trial` | int or null | trial index within the run, if the event belongs to a trial |

## Event types
| `ev` | extra fields | emitted by |
|---|---|---|
| `run_start` | `policy`, `task`, `improver` {model, scaffold_hash}, `budget` {trials, wall_s}, `config_hash` | orchestrator |
| `agent_spawn` | `role` (researcher / subagent / critic / lessons-writer), `model` | orchestrator / harness adapter |
| `agent_end` | `reason` | same |
| `message` | `to` (agent id), `summary` (≤200 chars), `tokens` | harness adapter (agent↔agent or orchestrator↔agent) |
| `llm_call` | `model`, `prompt_tokens`, `completion_tokens`, `latency_s` | gateway |
| `tool_call` | `tool` (bash / edit / read / web_search / fetch / mcp:<name>), `arg_summary` (≤200 chars, no secrets), `ok` (bool), `latency_s` | harness adapter |
| `proposal` | `hypothesis` (≤300 chars), `diff_stat` {files, plus, minus}, `artifact_hash` | orchestrator |
| `eval_submit` | `artifact_hash`, `lane` (GPU lane id), `seeds` | orchestrator |
| `eval_result` | `artifact_hash`, `metric` (e.g. `val_bpb`), `value`, `seeds`, `wall_s`, `status` (ok / timeout / crash / nan / audit_fail), `audit` {level, flags[]} | evald |
| `decision` | `artifact_hash`, `verdict` (adopt / reject / rollback / explore), `best_before`, `best_after`, `reason` | orchestrator |
| `audit_flag` | `artifact_hash`, `check`, `detail` | evald |
| `lessons_update` | `scaffold_hash_before`, `scaffold_hash_after`, `n_lessons` | orchestrator (U_s) |
| `weights_update` | `adapter_path`, `n_examples`, `base_model` | orchestrator (U_w) |
| `run_end` | `best_value`, `n_trials`, `nAUC` (if computable) | orchestrator |

## Rules
- Never log secrets, tokens, or full file contents. Use summaries, hashes and stats only.
- Full trajectories (prompts and completions, which are needed as LoRA data) go to `runs/<run_id>/trajectories/` and stay separate from events.
- The visualizer must be able to replay a run from `events.jsonl` alone.
