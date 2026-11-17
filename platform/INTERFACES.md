# RSI-mini: module ownership and interfaces (v1, written by the main session)

> **Changes by worker A (integrator), v1.1** — additive only:
> - `rsi.events.emit(run, ev, agent, ...)` writes to `register_run(run, path)` if called in-process, else `$RSI_RUNS_ROOT/<run>/events.jsonl`. **Separate processes (evald) must set `RSI_RUNS_ROOT`** (or call `register_run`). Invalid events raise `rsi.events.SchemaError`. `python -m rsi.events <file>` validates a log.
> - `eval_submit` is emitted by the orchestrator (policy base), not by evald. evald emits `eval_result` and `audit_flag`.
> - `Task` also has `editable_files: tuple[str, ...] = ("train.py",)`, `dev_seeds = (0,)`, `better(a, b)`. Register tasks with `@rsi.tasks.base.register`; look up with `get_task(name)` (`"muar/<id>"` falls back to a registered `"muar"` factory called with `name=`).
> - `Harness.run_episode(...)` also takes `parent=` and its dict also has `"submitted": {hypothesis, ...} | None` (from the lab MCP outbox). Lab spool dir convention: `<workspace>/../lab/` (`card.md`, `history.jsonl`, `best.json`, `outbox/`).
> - `RunContext` also has `lanes: list[int]`, `policy_cfg: dict`, `t_start`, helpers `emit()`, `paused()`, `out_of_budget(n)`, `save_state()`, `artifacts` (content-addressed store `runs/<id>/artifacts/<hash>/`). `RunContext.create(run=, runs_root=, ...)` builds the run dir.
> - `rsi/policies/base.py` provides the shared primitives for E: `make_threads`, `ensure_baseline`, `propose`, `evaluate` (calls `EvalClient.submit(run, task, artifact_dir, seeds=, lane=, gen=, trial=[, level=])` then `result(job_id, wait=True)`), `decide`, `recover_pending`, `judge`, `write_lab_view`, `parallel`, `finish`. `rsi/ledger.py` = `Ledger`, `nauc`; `rsi/runs.py` (A) = run-dir layout and artifact store.
> - Artifact dirs passed to evald contain exactly `task.editable_files`; evald must copy any non-editable trusted files (e.g. nanochat `lib.py`) from its own copy of the task.


Several agents build this repo in parallel. **Write only inside the paths you own.** If you need something from another module, code against the interface below. If an interface is missing, add a short note to `platform/NOTES_<you>.md`; do not edit another owner's files. Only the integrator (worker A) runs `git` (add, commit); everyone else only writes files.

Runtime is island mode. Everything runs on one cluster node (<node>: 3× A100 80GB, 36 CPU, Linux, internet OK). Services talk over `127.0.0.1` on high ports recorded in `$R/run/ports.json`. Auth uses a token read from `$R/run/token` (mode 600). Never put the token on a command line or in a log.
`R=$R`. Python: serving env `$SCRATCH/miniconda3/envs/rsi/bin/python`; training env `$R/envs/rsi-train/bin/python`.
GPU layout (default): GPU0 = vLLM (improver model), GPU1 = eval lane 1, GPU2 = eval lane 2. LoRA training temporarily borrows a lane, and the scheduler pauses that lane first.

## Ownership
| worker | owns | delivers |
|---|---|---|
| **A integrator** | `rsi/events.py`, `rsi/context.py`, `rsi/ledger.py`, `rsi/orchestrator.py`, `rsi/harness/**`, `rsi/mcp_lab/**`, `rsi/policies/base.py`, `rsi/policies/karpathy_greedy.py`, `DEPLOY.md`, `tests/test_core*.py`, git | RunContext, OpenCode harness adapter (bwrap workspace sandbox, DDG plus fetch MCP, lab MCP), first policy, end-to-end smoke test, final DEPLOY.md that stitches together the other workers' `deploy/*.md` |
| **B evald** | `rsi/evald/**`, `rsi/tasks/nanochat/**`, `bundle/run_vanilla.sh`, `bundle/noise_floor.sh`, `deploy/evald.md`, `tests/test_evald*.py` | sealed evaluator service plus bwrap artifact sandbox; the NanoChat task (vanilla start, wall-clock limit per D5, sealed scoring); noise-floor job |
| **C serving** | `rsi/gateway/**`, `bundle/serve_vllm.sh`, `bundle/toolcall_test.py`, `deploy/serving.md`, `tests/test_gateway*.py`, `tests/fake_openai_server.py` | vLLM launch (Qwen3.5-9B, 27B; tool-call parser; `--enable-lora`); the logging gateway (OpenAI-compatible proxy that emits `llm_call` events and writes full trajectories); the tool-call gate test; a fake model server for everyone's tests |
| **D second-order** | `rsi/tasks/muar/**`, `rsi/update/**`, `bundle/lora_train.sh`, `deploy/second_order.md`, `tests/test_muar*.py`, `tests/test_update*.py` | the µAR CPU task family (train / held-out ID / held-out OOD, as in platform_design.md); U_s (lessons writer); U_w (SFT data builder plus PEFT LoRA training script on one A100); placebo variants |
| **E policies** | `rsi/policies/naive_trial_ledger.py`, `recursive_first_steps.py`, `apex_ai4ai.py`, `self_evolution.py`, `tests/test_policies*.py` | the four remaining paper policies per platform_build_plan.md §4 |
| **F campaign** | `rsi/campaign/**`, `campaign/*.yaml`, `bundle/run_campaign.sh`, `deploy/campaign.md`, `tests/test_campaign*.py` | the overnight scheduler: phases, triggers (time and trial count), generation swaps, lane pausing for LoRA, health checks, auto-restart, checkpoint/resume; the concrete overnight config and schedule table |
| viz (already running) | `viz/**` | replay HTML, MP4 and GIF from `events.jsonl` |

## Interfaces (Python, all under package `rsi`)

```python
# rsi/events.py  (A)  — implements EVENTS_SCHEMA.md
def emit(run: str, ev: str, agent: str, *, parent=None, gen=0, trial=None, **fields) -> None

# rsi/tasks/base.py (A writes the ABC; B and D implement)
class Task:
    name: str
    metric: str                 # e.g. "val_bpb"
    lower_is_better: bool
    def start_artifact(self) -> pathlib.Path      # dir containing the editable artifact (e.g. train.py)
    def task_card(self) -> str                    # text shown to the improver (no sealed info)
    def visible_feedback(self, result: dict) -> str   # what the improver may see from an eval_result

# rsi/evald/client.py (B)
class EvalClient:
    def __init__(self, base_url: str, token_path: str): ...
    def submit(self, run: str, task: str, artifact_dir: str, *, seeds: list[int], lane: int | None = None, gen: int = 0, trial: int | None = None) -> str   # job id
    def result(self, job_id: str, *, wait: bool = True, timeout_s: float = 3600) -> dict  # keys = eval_result event fields
    def lanes(self) -> list[dict]                 # [{lane, gpu, busy, paused}]
    def pause_lane(self, lane: int) -> None; def resume_lane(self, lane: int) -> None   # used by F around LoRA training

# rsi/gateway (C): an OpenAI-compatible HTTP proxy. Harness and policies only ever call the gateway URL from ports.json, never vLLM directly.
# Request headers X-RSI-Run / X-RSI-Agent / X-RSI-Gen / X-RSI-Trial tag the call for events and trajectories.
# Model names: "improver" (base), "improver@<adapter_name>" (LoRA, via vLLM --enable-lora), "ladder-27b".

# rsi/harness/base.py (A)
class Harness:
    def run_episode(self, *, run: str, agent: str, gen: int, trial: int, workspace: pathlib.Path, prompt: str,
                    model: str, scaffold_dir: pathlib.Path | None, budget_s: int) -> dict
    # returns {"final": str, "n_tool_calls": int, "trajectory": str(path), "ok": bool}

# rsi/context.py (A)
@dataclass
class RunContext:
    run: str; gen: int; task: Task; evald: EvalClient; harness: Harness; model: str
    scaffold_dir: pathlib.Path | None; budget: dict; run_dir: pathlib.Path; ledger: "Ledger"; rng_seed: int

# rsi/policies/base.py (A)
class Policy:
    name: str
    def __init__(self, cfg: dict): ...
    def run(self, ctx: RunContext) -> dict        # returns summary {best_value, n_trials, ...}; emits events; resumable from ctx.ledger

# rsi/update (D)
def build_sft_data(run_dirs: list[str], *, accepted_only: bool = True, placebo: bool = False, out: str) -> dict
def write_lessons(run_dirs: list[str], *, model: str, gateway_url: str, out_dir: str, generic_placebo: bool = False) -> str   # returns scaffold hash
# bundle/lora_train.sh <sft_dir> <adapter_out> <gpu>  -> PEFT LoRA on the base model; prints adapter path

# orchestrator CLI (A):  python -m rsi.orchestrator --config <run.yaml> [--resume]
# campaign CLI (F):      python -m rsi.campaign --config campaign/overnight.yaml [--dry-run] [--resume]
```

## Shared rules
- Events follow EVENTS_SCHEMA.md exactly. Full prompts and completions go only to `runs/<id>/trajectories/`.
- Every test runs on Mac CPU with `tests/fake_openai_server.py` (owned by C; until it lands, stub locally inside your own test file).
- Never write to `$R/data/devval` or `$R/data/sealed` from any improver-reachable process. The sandbox must not see them.
- No SSH, no pushing, no network installs on the Mac beyond an isolated `uv venv` under `platform/.venv` if needed (shared; create it only once, via A).
