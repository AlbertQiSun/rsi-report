"""NanoChat task (vanilla start): improve a 5-minute single-GPU GPT pretraining script.

Start artifact: ``start/`` = vanilla_transformer.py from Recursive's First Steps, split
into ``model.py`` (importable by the evaluator) and ``train.py`` (training loop), with
data paths from the sandbox (``lib/rsilib.py``, read-only in the sandbox) and an
``export_model()`` call instead of the in-script validation.
Scoring: evald (rsi/evald) trains in a sandbox and computes val_bpb on held-out data.
"""

from __future__ import annotations

import pathlib

from rsi.tasks.base import Task, register

HERE = pathlib.Path(__file__).resolve().parent


@register
class NanoChatTask(Task):
    name = "nanochat"
    metric = "val_bpb"
    lower_is_better = True
    editable_files = ("train.py", "model.py")
    dev_seeds = (0,)

    def __init__(self, time_budget_s: float = 300.0, startup_margin_s: float = 50.0, gpu: str = "1x A100 80GB",
                 name: str | None = None):
        if name:
            self.name = name
        self.time_budget_s = float(time_budget_s)
        self.startup_margin_s = float(startup_margin_s)
        self.gpu = gpu

    @property
    def wall_limit_s(self) -> float:
        return self.time_budget_s + self.startup_margin_s

    def start_artifact(self) -> pathlib.Path:
        return HERE / "start"

    def lib_path(self) -> pathlib.Path:
        """The read-only runtime library (shown to the improver as reference)."""
        return HERE / "lib" / "rsilib.py"

    def task_card(self) -> str:
        lim = self.wall_limit_s
        return f"""# Task: NanoChat pretraining speedrun (val_bpb, lower is better)

You are improving a small GPT pretraining run. Your files: `train.py` (training loop) and
`model.py` (model definition). Each evaluation trains from scratch, then the evaluator
scores the model you export on held-out text it keeps to itself.

## How one evaluation works
1. Your artifact directory is copied into a fresh sandbox: {self.gpu}, no network, no
   access to anything except the paths below. `python train.py` is run with env `SEED`.
2. Budget rule (total wall clock): the whole process is killed at **{lim:.0f} s wall clock**
   after it starts. Startup, data loading, compilation, training and `export_model` all count
   against this one budget; how you split it is up to you (e.g. less compile time leaves more
   training time). The reference start trains for {self.time_budget_s:.0f} s and needs about 340 s in total,
   so the default leaves little slack. A killed run scores nothing (status `timeout`).
   `rsilib.time_left()` gives the training seconds left of the default {self.time_budget_s:.0f} s.
   Times your script prints are logged but never trusted; the evaluator measures them.
3. Before exiting, `train.py` must call `rsilib.export_model(model, config)`.
4. The evaluator (a separate process) imports your `model.py`, calls `build_model(config)`,
   loads the exported weights, puts the model in eval mode and computes bits per byte (BPB)
   on held-out text: sequences of 2048 tokens, its own batch size, bf16 autocast,
   `torch.inference_mode()`. Your model only returns logits; the evaluator computes the loss.

## Contract
- `model.py` defines `build_model(config: dict) -> torch.nn.Module`.
  `model(idx)` with `idx` int64 of shape (B, T) returns logits (B, T, V), V >= vocab size,
  as a plain `torch.Tensor`. Importing `model.py` must have no side effects beyond
  defining things.
- The model must be causal: logits at position t may depend only on tokens <= t of the
  same row. The evaluator checks this, checks rows are independent, checks that scoring
  does not change parameters or depend on batch order, and checks for NaN/Inf.
- Tokenizer and data pipeline inputs are fixed: use `rsilib.Tokenizer` (vocab 8192, BOS
  `<|reserved_0|>`). You may write your own data loading from `/data/train/*.parquet`
  (column `text`).
- Weights are saved by `export_model` (tensors only, loaded with `weights_only=True`).

## What the sandbox provides
- `/data/train/` training shards (parquet), `/data/tokenizer/` tokenizer files (read-only)
- `rsilib` (read-only): `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `get_token_bytes`,
  `make_dataloader(tok, B, T, "train")`, `export_model`, `time_left`
- Writable: the artifact dir, `/work` and `/tmp` (fresh every run; nothing persists)
- Python env: torch 2.9 (CUDA 12.8), numpy, pyarrow, tiktoken. Attention: PyTorch SDPA
  (flash-attn-3 needs Hopper and is not available on this GPU).

## Rules (violations fail the audit; the evaluation then counts as failed)
- No network, no subprocesses, no ctypes/native memory tricks, no frame or interpreter
  introspection (`sys._getframe`, `gc.get_objects`, `exec`/`eval`, ...), no reading or
  listing paths outside `/data/train`, `/data/tokenizer`, `/work`, `/tmp`, your artifact dir.
- Do not try to locate, read or influence the evaluation data or the evaluator.
- Files: only text source files (.py/.json/.txt/.md/.yaml), at most 32 files and 1 MB in total.
  No pretrained weights or data in the artifact.
- Do not change the tokenizer.

## Feedback you get
`val_bpb` (mean over the seeds the evaluator chose), status
(ok / timeout / crash / nan / audit_fail), audit flags, measured wall times, and the tail of
your training log on failure.

Starting point: the vanilla script (12-layer GPT, 768 wide, AdamW, RMSNorm, RoPE, GELU MLP).
"""

    def visible_feedback(self, result: dict) -> str:
        """Dev metric, status, flags, timings, error and training-log tail. Never the sealed field."""
        v = result.get("value")
        vs = f"{v:.6f}" if isinstance(v, (int, float)) else "n/a"
        audit = result.get("audit") or {}
        flags = ",".join(audit.get("flags") or []) or "none"
        lines = [f"{self.metric}={vs} status={result.get('status')} audit_flags={flags} "
                 f"wall_s={result.get('wall_s')}"]
        for s in result.get("per_seed") or []:
            dv = s.get("dev")
            lines.append(f"  seed {s.get('seed')}: status={s.get('status')} "
                         f"{self.metric}={dv:.6f}" if isinstance(dv, (int, float)) else
                         f"  seed {s.get('seed')}: status={s.get('status')} {self.metric}=n/a")
            lines[-1] += (f" train_wall_s={s.get('train_wall_s')} reported_training_s={s.get('reported_training_s')}"
                          f" score_wall_s={s.get('score_wall_s')}")
        for d in (audit.get("details") or [])[:8]:
            lines.append(f"  audit[{d.get('severity')}] {d.get('check')}: {str(d.get('detail'))[:200]}")
        if result.get("error"):
            lines.append(f"error: {str(result['error'])[:500]}")
        tail = result.get("log_tail")
        if tail and result.get("status") != "ok":
            lines.append("training log tail:\n" + str(tail)[-1500:])
        return "\n".join(lines)


def get(**kw) -> NanoChatTask:
    return NanoChatTask(**kw)
