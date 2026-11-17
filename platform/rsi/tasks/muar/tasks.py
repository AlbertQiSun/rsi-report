"""µAR task registry: 16 tasks in 3 splits, randomized starting defects, task cards.

Splits (design doc §5.1, controls 5 and 6):
  train        8 = 4 types x 2 parameter sets   (modadd, parity, teacher_student, cls2d)
  heldout_id   4 = the same 4 types, new parameters, new defects
  heldout_ood  4 = 4 types never used in train   (copyrev, dyck, icl_linreg, charlm)
The design doc sketched OOD as "2 unseen types x 2"; we use 4 unseen types x 1 so all eight
requested task types are exercised with the same task count (wider OOD, n=1 per type).

Each task's start config = DEFAULT_CONFIG with 2-3 randomly drawn defects (sealed: the defect
list is never shown). The compute budget B is fixed per task as
``REF_STEPS * 64 * n_params(DEFAULT_CONFIG)`` so the default skeleton gets ~REF_STEPS steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from rsi.tasks.muar import schema
from rsi.tasks.muar.data import make_dataset
from rsi.tasks.muar.train import MAX_PARAMS, MIN_STEPS, budget_steps, n_params_for

REF_STEPS = 800
MAX_STEPS = 4000
DEV_SEEDS = (0, 1)
N_SEALED_SEEDS = 3

TYPE_BLURB = {
    "modadd": "Modular addition. Input: one-hot a and one-hot b (a, b in [0, P)). Predict (a + b) mod P. "
              "A fixed fraction of all P*P pairs is the training set; validation uses unseen pairs.",
    "parity": "Sparse parity. Input: n bits in {-1, +1}. Label: XOR of k fixed but unknown bits.",
    "copyrev": "Sequence transduction. Input: one-hot sequence of length L over V symbols. "
               "Output: the sequence copied or reversed (L softmax heads, mean cross-entropy).",
    "dyck": "Dyck-language recognition. Input: one-hot bracket string of length L with K bracket types. "
            "Label: is the string balanced? Negatives are near-misses (1-2 token edits).",
    "teacher_student": "Teacher-student regression. Input: d Gaussian features. Target: output of a fixed "
                       "random tanh MLP teacher plus noise (standardised). Loss: MSE.",
    "cls2d": "Noisy 2-D classification (points in the plane, label noise in train and validation).",
    "icl_linreg": "In-context linear regression. Input: k (x, y) pairs drawn with a fresh weight vector "
                  "per example, then a query x. Target: y at the query (standardised). Loss: MSE.",
    "charlm": "Character-level language modelling on Python source code. Input: one-hot of the previous "
              "`context` characters (top V-1 characters + <unk>). Predict the next character.",
}


@dataclass(frozen=True)
class TaskSpec:
    tid: str                 # e.g. "train-modadd-0"
    split: str               # train | heldout_id | heldout_ood
    ttype: str
    params: dict = field(hash=False)
    data_seed: int = 0
    defect_seed: int = 0

    @property
    def name(self) -> str:
        return f"muar/{self.tid}"


def _s(tid, split, ttype, data_seed, defect_seed, **params) -> TaskSpec:
    return TaskSpec(tid, split, ttype, params, data_seed, defect_seed)


SPECS: list[TaskSpec] = [
    # ---- train: 4 types x 2
    _s("train-modadd-0", "train", "modadd", 11, 101, P=23, train_frac=0.7),
    _s("train-modadd-1", "train", "modadd", 12, 102, P=37, train_frac=0.5),
    _s("train-parity-0", "train", "parity", 13, 103, n=20, k=3, n_train=3000, n_val=1000),
    _s("train-parity-1", "train", "parity", 14, 104, n=24, k=3, n_train=4000, n_val=1000),
    _s("train-teacher-0", "train", "teacher_student", 15, 105, d=16, teacher_width=32, teacher_depth=2,
       noise=0.1, n_train=1500, n_val=1000),
    _s("train-teacher-1", "train", "teacher_student", 16, 106, d=24, teacher_width=64, teacher_depth=3,
       noise=0.2, n_train=2000, n_val=1000),
    _s("train-cls2d-0", "train", "cls2d", 17, 111, kind="spirals", classes=3, jitter=0.03, label_noise=0.05,
       n_train=1500, n_val=1000),
    _s("train-cls2d-1", "train", "cls2d", 18, 115, kind="checker", jitter=0.0, label_noise=0.1,
       n_train=1500, n_val=1000),
    # ---- held-out ID: same types, new parameters and defects
    _s("id-modadd-0", "heldout_id", "modadd", 21, 202, P=31, train_frac=0.45),
    _s("id-parity-0", "heldout_id", "parity", 22, 528, n=28, k=3, n_train=4000, n_val=1000),
    _s("id-teacher-0", "heldout_id", "teacher_student", 23, 216, d=20, teacher_width=48, teacher_depth=2,
       noise=0.15, n_train=1500, n_val=1000),
    _s("id-cls2d-0", "heldout_id", "cls2d", 24, 541, kind="spirals", classes=2, jitter=0.05, label_noise=0.1,
       n_train=1500, n_val=1000),
    # ---- held-out OOD: unseen types
    _s("ood-copyrev-0", "heldout_ood", "copyrev", 31, 303, L=8, V=8, mode="reverse", n_train=3000, n_val=1000),
    _s("ood-dyck-0", "heldout_ood", "dyck", 32, 337, L=16, K=2, n_train=3000, n_val=1000),
    _s("ood-icl-0", "heldout_ood", "icl_linreg", 33, 428, d=2, k=8, noise=0.1, n_train=6000, n_val=1000),
    _s("ood-charlm-0", "heldout_ood", "charlm", 34, 363, context=12, V=40, span=14000, n_train=3000,
       n_val=1000),
]
BY_NAME = {s.name: s for s in SPECS}
SPLITS = {sp: [s.name for s in SPECS if s.split == sp] for sp in ("train", "heldout_id", "heldout_ood")}


def tasks_in(split: str) -> list[str]:
    if split in SPLITS:
        return list(SPLITS[split])
    if split == "heldout":
        return SPLITS["heldout_id"] + SPLITS["heldout_ood"]
    if split == "all":
        return [s.name for s in SPECS]
    if split in BY_NAME:
        return [split]
    raise KeyError(f"unknown split or task {split!r}")


def heldout_names() -> list[str]:
    return tasks_in("heldout")


def get_spec(name: str) -> TaskSpec:
    if not name.startswith("muar/"):
        name = "muar/" + name
    if name not in BY_NAME:
        raise KeyError(f"unknown µAR task {name!r}; known: {sorted(BY_NAME)}")
    return BY_NAME[name]


# ---------------------------------------------------------------------------- dims, budget
def dims(spec: TaskSpec) -> tuple[int, int]:
    ds = make_dataset(spec.ttype, spec.params, spec.data_seed)
    return ds.in_dim, ds.n_out


def budget_for(spec: TaskSpec) -> float:
    in_dim, n_out = dims(spec)
    return float(REF_STEPS * 64 * n_params_for(schema.DEFAULT_CONFIG, in_dim, n_out))


# ---------------------------------------------------------------------------- starting defects
# groups are mutually exclusive (at most one defect per group)
DEFECTS = {
    "lr": ["lr_high", "lr_low", "sgd_plain"],
    "capacity": ["narrow", "wide", "deep_tanh"],
    "reg": ["wd_strong", "dropout_high"],
    "batch": ["big_batch", "tiny_batch"],
    "init": ["bad_init_big", "bad_init_small"],
    "sched": ["long_warmup_const"],
}


def _apply(cfg: dict, d: str, rng: np.random.Generator) -> None:
    if d == "lr_high":
        cfg["lr"] = float(min(3.0, cfg["lr"] * rng.uniform(15, 40)))
    elif d == "lr_low":
        cfg["lr"] = float(cfg["lr"] / rng.uniform(30, 100))
    elif d == "sgd_plain":
        cfg["optimizer"] = "sgd"
    elif d == "narrow":
        cfg["width"] = int(rng.choice([4, 6, 8]))
    elif d == "wide":
        cfg["width"] = int(rng.choice([512, 768]))
    elif d == "deep_tanh":
        cfg.update(depth=int(rng.choice([5, 6])), activation="tanh", norm="none")
    elif d == "wd_strong":
        cfg["weight_decay"] = float(rng.uniform(0.3, 1.0))
        cfg["optimizer"] = "adam" if cfg["optimizer"] == "adamw" else cfg["optimizer"]
    elif d == "dropout_high":
        cfg["dropout"] = float(rng.uniform(0.4, 0.6))
    elif d == "big_batch":
        cfg["batch_size"] = 512
    elif d == "tiny_batch":
        cfg["batch_size"] = 4
    elif d == "bad_init_big":
        cfg["init_scale"] = float(rng.uniform(3.0, 5.0))
    elif d == "bad_init_small":
        cfg["init_scale"] = float(rng.uniform(0.05, 0.1))
    elif d == "long_warmup_const":
        cfg.update(schedule="constant", warmup_frac=0.5)
    else:
        raise ValueError(d)


def start_config_and_defects(spec: TaskSpec) -> tuple[dict, list[str]]:
    """Deterministic start config for a task (sealed defect list returned separately)."""
    in_dim, n_out = dims(spec)
    B = budget_for(spec)
    rng = np.random.Generator(np.random.PCG64(spec.defect_seed))
    for _ in range(100):
        cfg = dict(schema.DEFAULT_CONFIG)
        k = int(rng.integers(2, 4))
        groups = rng.choice(sorted(DEFECTS), size=k, replace=False)
        defects = [str(rng.choice(DEFECTS[str(g)])) for g in groups]
        for d in defects:
            _apply(cfg, d, rng)
        # per-task jitter so no two tasks share a start config (keeps the leak check meaningful)
        cfg["lr"] = float(min(3.0, max(1e-5, cfg["lr"] * float(np.exp(rng.uniform(-0.4, 0.4))))))
        cfg["weight_decay"] = float(min(1.0, cfg["weight_decay"] * float(np.exp(rng.uniform(-0.5, 0.5)))))
        cfg = {k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in cfg.items()}
        cfg = schema.validate_config(cfg)
        steps, n, _ = budget_steps(cfg, in_dim, n_out, B, MAX_STEPS)
        if steps >= 5 * MIN_STEPS and n <= MAX_PARAMS:
            return cfg, sorted(defects)
    raise RuntimeError(f"could not draw a legal start config for {spec.name}")


# ---------------------------------------------------------------------------- seeds (sealed)
def _h31(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def seal_salt() -> tuple[str, str]:
    """Secret salt for sealed seeds: $RSI_MUAR_SEAL_SALT, else <sealed dir>/salt, else a default.

    Returns (salt, source). The sealed dir is $RSI_MUAR_SEALED_DIR or $RSI_ROOT/data/sealed/muar; it lives
    outside every improver sandbox (INTERFACES.md shared rules).
    """
    if os.environ.get("RSI_MUAR_SEAL_SALT"):
        return os.environ["RSI_MUAR_SEAL_SALT"], "env"
    p = os.path.join(sealed_dir(), "salt")
    try:
        with open(p) as f:
            s = f.read().strip()
        if s:
            return s, "file"
    except OSError:
        pass
    return "muar-default-salt-v1", "default"


def sealed_dir() -> str:
    if os.environ.get("RSI_MUAR_SEALED_DIR"):
        return os.environ["RSI_MUAR_SEALED_DIR"]
    root = os.environ.get("RSI_ROOT", "/scratch/user/rsi")
    return os.path.join(root, "data", "sealed", "muar")


def train_seed(spec: TaskSpec, split: str, idx: int) -> int:
    """Map a seed index to the actual torch seed. Dev seeds are public indices; sealed ones are salted."""
    if split == "sealed":
        salt, _ = seal_salt()
        return _h31(salt, spec.name, "sealed", idx)
    return _h31(spec.name, "dev", idx)


# ---------------------------------------------------------------------------- task card
def task_card(spec: TaskSpec) -> str:
    in_dim, n_out = dims(spec)
    ds = make_dataset(spec.ttype, spec.params, spec.data_seed)
    B = budget_for(spec)
    pstr = ", ".join(f"{k}={v}" for k, v in sorted(spec.params.items()) if k not in ("n_train", "n_val"))
    loss = "mean squared error (standardised targets)" if ds.loss == "mse" else "cross-entropy (nats)"
    return f"""TASK {spec.name}  (µAR micro-autoresearch)
TYPE {spec.ttype}: {TYPE_BLURB[spec.ttype]}
PARAMS {pstr}
DATA train={ds.sizes()['train']} examples, input_dim={in_dim}, outputs={n_out}

Goal: minimise val_loss = {loss} on a held-out validation set, averaged over {len(DEV_SEEDS)} training seeds.
Lower is better. The model is a fixed MLP skeleton: input -> depth x [Linear(width) -> norm -> activation
-> dropout] -> Linear(outputs). You control it ONLY through config.json (13 keys, all required):

{schema.schema_text()}

Compute budget (enforced by the evaluator, not by you):
  steps = floor(B / (batch_size * n_params)), capped at {MAX_STEPS};  B = {B:.4g}
  n_params counts all weights and biases (including norm parameters).
  A bigger model or batch therefore means fewer steps. Fewer than {MIN_STEPS} steps or more than {MAX_PARAMS}
  parameters is illegal. Divergence (NaN / loss > 1e6) and timeouts count as failed attempts.
Feedback you get per attempt: val_loss (dev), status, steps, n_params, train loss at 20/40/60/80/100% of
training, and the train/val gap. Edit config.json only; invalid JSON or schema violations are rejected.
"""


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
