"""µAR action space: one JSON config with 13 knobs (design doc §5.1, "about 12 knobs").

The improver's artifact is ``config.json``. It is data, never executed. ``validate_config``
is the L0 check: unknown keys, missing keys, wrong types and out-of-range values are illegal.
``sample_config`` / ``perturb_config`` are the priors of the random-search and local-perturbation
baselines (design doc §1.5 control 7).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

ENUMS: dict[str, tuple[str, ...]] = {
    "optimizer": ("sgd", "momentum", "adam", "adamw"),
    "schedule": ("constant", "cosine", "linear", "step"),
    "activation": ("relu", "gelu", "tanh", "silu"),
    "norm": ("none", "layernorm", "rmsnorm"),
}
# name -> (lo, hi, log_scale)
FLOATS: dict[str, tuple[float, float, bool]] = {
    "lr": (1e-5, 3.0, True),
    "warmup_frac": (0.0, 0.5, False),
    "weight_decay": (0.0, 1.0, False),
    "init_scale": (0.05, 5.0, True),
    "dropout": (0.0, 0.6, False),
    "grad_clip": (0.0, 100.0, False),   # 0 = off
}
INTS: dict[str, tuple[int, int]] = {
    "batch_size": (4, 512),
    "width": (4, 1024),
    "depth": (1, 6),
}
KEYS: tuple[str, ...] = ("optimizer", "lr", "schedule", "warmup_frac", "weight_decay", "batch_size",
                         "width", "depth", "activation", "norm", "init_scale", "dropout", "grad_clip")
assert set(KEYS) == set(ENUMS) | set(FLOATS) | set(INTS)

DESCRIPTIONS = {
    "optimizer": "sgd (plain), momentum (SGD, nesterov 0.9), adam (L2-coupled weight decay), adamw (decoupled)",
    "lr": "peak learning rate",
    "schedule": "after warmup: constant | cosine (to 0) | linear (to 0) | step (x0.1 at 50% and 75%)",
    "warmup_frac": "fraction of steps with linear warmup from 0",
    "weight_decay": "weight decay coefficient",
    "batch_size": "examples per step",
    "width": "hidden units per layer",
    "depth": "number of hidden layers",
    "activation": "hidden nonlinearity",
    "norm": "normalisation after each hidden linear layer",
    "init_scale": "multiplier on the default (Kaiming-uniform) weight init",
    "dropout": "dropout probability after each hidden activation",
    "grad_clip": "max global grad norm; 0 disables clipping",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "optimizer": "adamw", "lr": 3e-3, "schedule": "cosine", "warmup_frac": 0.05, "weight_decay": 0.01,
    "batch_size": 64, "width": 128, "depth": 2, "activation": "gelu", "norm": "layernorm",
    "init_scale": 1.0, "dropout": 0.0, "grad_clip": 1.0,
}


class IllegalConfig(ValueError):
    """Raised by validate_config; ``flag`` is a short machine-readable reason."""

    def __init__(self, flag: str, msg: str):
        super().__init__(msg)
        self.flag = flag


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_config(cfg: Any) -> dict:
    """Return a normalised copy of ``cfg`` or raise IllegalConfig (L0 static check)."""
    if not isinstance(cfg, dict):
        raise IllegalConfig("schema:not_object", "config must be a JSON object")
    unknown = sorted(set(cfg) - set(KEYS))
    if unknown:
        raise IllegalConfig("schema:unknown_key", f"unknown key(s): {', '.join(unknown)}")
    missing = [k for k in KEYS if k not in cfg]
    if missing:
        raise IllegalConfig("schema:missing_key", f"missing key(s): {', '.join(missing)}")
    out: dict[str, Any] = {}
    for k, choices in ENUMS.items():
        v = cfg[k]
        if not isinstance(v, str) or v not in choices:
            raise IllegalConfig("schema:bad_enum", f"{k} must be one of {list(choices)}, got {v!r}")
        out[k] = v
    for k, (lo, hi, _) in FLOATS.items():
        v = cfg[k]
        if not _is_num(v) or not math.isfinite(float(v)):
            raise IllegalConfig("schema:bad_type", f"{k} must be a finite number, got {v!r}")
        v = float(v)
        if not (lo <= v <= hi):
            raise IllegalConfig("schema:out_of_range", f"{k}={v} outside [{lo}, {hi}]")
        out[k] = v
    for k, (lo, hi) in INTS.items():
        v = cfg[k]
        if not _is_num(v) or float(v) != int(v):
            raise IllegalConfig("schema:bad_type", f"{k} must be an integer, got {v!r}")
        v = int(v)
        if not (lo <= v <= hi):
            raise IllegalConfig("schema:out_of_range", f"{k}={v} outside [{lo}, {hi}]")
        out[k] = v
    return out


def parse_config_text(text: str) -> dict:
    try:
        cfg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise IllegalConfig("schema:bad_json", f"config.json is not valid JSON: {e}") from None
    return validate_config(cfg)


def canonical_json(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def schema_text() -> str:
    lines = []
    for k in KEYS:
        if k in ENUMS:
            rng = " | ".join(ENUMS[k])
        elif k in FLOATS:
            lo, hi, log = FLOATS[k]
            rng = f"float in [{lo:g}, {hi:g}]"
        else:
            lo, hi = INTS[k]
            rng = f"int in [{lo}, {hi}]"
        lines.append(f"  {k:<13} {rng:<38} {DESCRIPTIONS[k]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- baseline priors
def _sample_float(rng: np.random.Generator, k: str) -> float:
    lo, hi, log = FLOATS[k]
    if k == "lr":
        lo, hi = 1e-4, 0.3          # prior narrower than the legal range
    if k == "grad_clip":
        return float(rng.choice([0.0, 0.5, 1.0, 5.0]))
    if log:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _sample_int(rng: np.random.Generator, k: str) -> int:
    if k == "batch_size":
        return int(rng.choice([8, 16, 32, 64, 128, 256]))
    if k == "width":
        return int(rng.choice([16, 32, 64, 128, 256, 512]))
    lo, hi = INTS[k]
    return int(rng.integers(lo, hi + 1))


def sample_config(rng: np.random.Generator) -> dict:
    """Random-search prior (broad, log-uniform lr)."""
    cfg: dict[str, Any] = {}
    for k in KEYS:
        if k in ENUMS:
            cfg[k] = str(rng.choice(ENUMS[k]))
        elif k in FLOATS:
            cfg[k] = _sample_float(rng, k)
        else:
            cfg[k] = _sample_int(rng, k)
    cfg["dropout"] = float(rng.choice([0.0, 0.0, 0.1, 0.3]))
    cfg["weight_decay"] = float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 1e-1]))
    return validate_config(cfg)


def perturb_config(cfg: dict, rng: np.random.Generator, n_knobs: int | None = None) -> dict:
    """Local-perturbation move: change 1-3 knobs of ``cfg`` (stays legal)."""
    out = dict(cfg)
    n = n_knobs or int(rng.integers(1, 4))
    for k in rng.choice(KEYS, size=n, replace=False):
        k = str(k)
        if k in ENUMS:
            others = [c for c in ENUMS[k] if c != out[k]]
            out[k] = str(rng.choice(others))
        elif k in FLOATS:
            lo, hi, log = FLOATS[k]
            if log:
                v = out[k] * math.exp(rng.normal(0, 1.0))
            elif k == "grad_clip":
                v = float(rng.choice([0.0, 0.5, 1.0, 5.0]))
            else:
                v = out[k] + rng.normal(0, 0.15 * (hi - lo))
            out[k] = float(min(hi, max(lo, v)))
        else:
            lo, hi = INTS[k]
            if k in ("width", "batch_size"):
                v = int(round(out[k] * 2.0 ** rng.choice([-1, 1])))
            else:
                v = out[k] + int(rng.choice([-1, 1]))
            out[k] = int(min(hi, max(lo, v)))
    return validate_config(out)
