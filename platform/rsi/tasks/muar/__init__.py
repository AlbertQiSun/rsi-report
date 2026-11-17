"""µAR (micro-autoresearch) CPU task family, design doc §5.1.

``get_task("muar/train-modadd-0")`` returns a ``MuarTask`` (family factory registered as "muar").
The artifact is one ``config.json`` (13 knobs, see schema.py). Scoring is deterministic,
single-thread CPU PyTorch, ~1-3 s per seed. Splits: ``tasks.SPLITS`` (8 train / 4 held-out-ID /
4 held-out-OOD). Evaluator entry: ``rsi.tasks.muar.evaluate``. Local CPU runner, baselines and
s_ref: ``rsi.tasks.muar.runner``.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

from rsi.tasks.base import Task, register
from rsi.tasks.muar import tasks as _tasks
from rsi.tasks.muar.tasks import SPLITS, get_spec, heldout_names, start_config_and_defects, tasks_in

__all__ = ["MuarTask", "SPLITS", "tasks_in", "heldout_names", "start_config", "VISIBLE_DIAG_KEYS"]

# the only diagnostic fields an improver may see (design doc §5.1 "给改进者的反馈")
VISIBLE_DIAG_KEYS = ("steps", "params", "train_curve", "train_loss", "gap", "error")


def start_config(name: str) -> dict:
    return start_config_and_defects(get_spec(name))[0]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


@register
class MuarTask(Task):
    name = "muar"
    metric = "val_loss"
    lower_is_better = True
    editable_files = ("config.json",)
    dev_seeds = _tasks.DEV_SEEDS

    def __init__(self, name: str = "muar/train-modadd-0", **_kw):
        self.spec = get_spec(name)
        self.name = self.spec.name
        self.split = self.spec.split

    def start_artifact(self) -> pathlib.Path:
        root = os.environ.get("RSI_MUAR_START_DIR") or os.path.join(tempfile.gettempdir(), "rsi-muar-start")
        d = pathlib.Path(root) / self.spec.tid
        d.mkdir(parents=True, exist_ok=True)
        text = json.dumps(start_config(self.name), indent=1, sort_keys=True) + "\n"
        p = d / "config.json"
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
        return d

    def task_card(self) -> str:
        return _tasks.task_card(self.spec)

    def visible_feedback(self, result: dict) -> str:
        """Dev value, status, flags and whitelisted diagnostics only. Never sealed scores or s_ref."""
        if not result:
            return "no result"
        if result.get("split") == "sealed":
            return "sealed result withheld"
        v = result.get("value")
        flags = ",".join(result.get("audit", {}).get("flags", [])) or "none"
        parts = [f"val_loss={_fmt(v) if isinstance(v, (int, float)) else 'n/a'}",
                 f"status={result.get('status')}", f"flags={flags}"]
        diag = result.get("diag") or {}
        for k in VISIBLE_DIAG_KEYS:
            if diag.get(k) is not None:
                key = "train_loss@20/40/60/80/100%" if k == "train_curve" else ("train_val_gap" if k == "gap" else k)
                parts.append(f"{key}={_fmt(diag[k])}")
        return " ".join(parts)
