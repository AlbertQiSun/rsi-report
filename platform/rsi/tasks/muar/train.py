"""One µAR training run: MLP skeleton + harness-enforced compute budget, single-thread CPU.

Budget conversion (design doc §5.1): ``steps = floor(B / (batch_size * n_params))``, capped at
``max_steps``. The improver cannot buy compute by growing the model or the batch: both shrink
the step count. Fewer than ``min_steps`` steps or more than ``max_params`` parameters is illegal.
A cooperative wall-clock check aborts the run as ``timeout`` past ``hard_timeout_s``.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from rsi.tasks.muar.data import Dataset

MIN_STEPS = 10
MAX_PARAMS = 2_000_000


def _torch():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process
    torch.use_deterministic_algorithms(True)
    return torch


def n_params_for(cfg: dict, in_dim: int, n_out: int) -> int:
    w, d = cfg["width"], cfg["depth"]
    n = in_dim * w + w + (d - 1) * (w * w + w) + w * n_out + n_out
    if cfg["norm"] == "layernorm":
        n += d * 2 * w
    elif cfg["norm"] == "rmsnorm":
        n += d * w
    return n


def budget_steps(cfg: dict, in_dim: int, n_out: int, budget: float, max_steps: int) -> tuple[int, int, int]:
    """Return (steps, n_params, uncapped_steps)."""
    n = n_params_for(cfg, in_dim, n_out)
    raw = int(budget // (cfg["batch_size"] * n))
    return min(raw, max_steps), n, raw


def build_model(cfg: dict, in_dim: int, n_out: int):
    torch = _torch()
    nn = torch.nn
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
    layers, fan = [], in_dim
    for _ in range(cfg["depth"]):
        layers.append(nn.Linear(fan, cfg["width"]))
        if cfg["norm"] == "layernorm":
            layers.append(nn.LayerNorm(cfg["width"]))
        elif cfg["norm"] == "rmsnorm":
            layers.append(nn.RMSNorm(cfg["width"]))
        layers.append(acts[cfg["activation"]]())
        if cfg["dropout"] > 0:
            layers.append(nn.Dropout(cfg["dropout"]))
        fan = cfg["width"]
    layers.append(nn.Linear(fan, n_out))
    model = nn.Sequential(*layers)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(cfg["init_scale"])
    return model


def _lr_at(cfg: dict, step: int, steps: int) -> float:
    warm = int(round(cfg["warmup_frac"] * steps))
    if warm > 0 and step < warm:
        return cfg["lr"] * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    s = cfg["schedule"]
    if s == "cosine":
        f = 0.5 * (1 + math.cos(math.pi * t))
    elif s == "linear":
        f = 1.0 - t
    elif s == "step":
        f = 1.0 if t < 0.5 else (0.1 if t < 0.75 else 0.01)
    else:
        f = 1.0
    return cfg["lr"] * f


def _make_opt(torch, cfg: dict, params):
    o, lr, wd = cfg["optimizer"], cfg["lr"], cfg["weight_decay"]
    if o == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if o == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if o == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def _loss_fn(torch, ds: Dataset, out, y):
    if ds.loss == "mse":
        return torch.mean((out - y) ** 2)
    if ds.n_heads > 1:
        c = ds.n_out // ds.n_heads
        return torch.nn.functional.cross_entropy(out.reshape(-1, c), y.reshape(-1))
    return torch.nn.functional.cross_entropy(out, y)


def _eval_loss(torch, model, ds: Dataset, X: np.ndarray, Y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        tot, n = 0.0, X.shape[0]
        for i in range(0, n, 2048):
            xb = torch.from_numpy(X[i:i + 2048])
            yb = torch.from_numpy(Y[i:i + 2048])
            tot += float(_loss_fn(torch, ds, model(xb), yb)) * xb.shape[0]
    model.train()
    return tot / n


def train_and_score(cfg: dict, ds: Dataset, *, seed: int, split: str, budget: float, max_steps: int,
                    hard_timeout_s: float = 60.0) -> dict:
    """Train with ``cfg`` on ds.train, score on ds[split]. Deterministic for a given (cfg, ds, seed).

    Returns {status, value, steps, params, train_curve[5], train_loss, gap, error}.
    """
    t0 = time.perf_counter()
    steps, n_params, raw = budget_steps(cfg, ds.in_dim, ds.n_out, budget, max_steps)
    base = {"steps": steps, "params": n_params, "uncapped_steps": raw}
    if n_params > MAX_PARAMS:
        return {**base, "status": "audit_fail", "value": None, "flag": "budget:too_many_params",
                "error": f"{n_params} parameters > max {MAX_PARAMS}"}
    if steps < MIN_STEPS:
        return {**base, "status": "audit_fail", "value": None, "flag": "budget:too_few_steps",
                "error": f"budget allows only {steps} steps (< {MIN_STEPS}); shrink width/depth/batch_size"}
    torch = _torch()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = build_model(cfg, ds.in_dim, ds.n_out)
    opt = _make_opt(torch, cfg, model.parameters())
    Xtr = torch.from_numpy(ds.X["train"])
    Ytr = torch.from_numpy(ds.Y["train"])
    n = Xtr.shape[0]
    bs = min(cfg["batch_size"], n)
    perm = torch.randperm(n, generator=gen)
    pos = 0
    checkpoints = {int(round(steps * q)) - 1 for q in (0.2, 0.4, 0.6, 0.8, 1.0)}
    window, curve = [], []
    for step in range(steps):
        if pos + bs > n:
            perm = torch.randperm(n, generator=gen)
            pos = 0
        idx = perm[pos:pos + bs]
        pos += bs
        lr = _lr_at(cfg, step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = _loss_fn(torch, ds, model(Xtr[idx]), Ytr[idx])
        lv = float(loss)
        if not math.isfinite(lv) or lv > 1e6:
            return {**base, "status": "nan", "value": None, "flag": "nan", "train_curve": curve,
                    "error": f"training loss diverged at step {step}"}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        window.append(lv)
        if step in checkpoints:
            curve.append(round(float(np.mean(window[-max(1, steps // 10):])), 6))
        if (step & 63) == 0 and time.perf_counter() - t0 > hard_timeout_s:
            return {**base, "status": "timeout", "value": None, "flag": "timeout", "train_curve": curve,
                    "error": f"hard timeout {hard_timeout_s}s at step {step}"}
    val = _eval_loss(torch, model, ds, ds.X[split], ds.Y[split])
    sub = min(n, ds.X[split].shape[0])
    tr = _eval_loss(torch, model, ds, ds.X["train"][:sub], ds.Y["train"][:sub])
    if not math.isfinite(val):
        return {**base, "status": "nan", "value": None, "flag": "nan", "train_curve": curve,
                "error": "non-finite validation loss"}
    return {**base, "status": "ok", "value": round(val, 6), "train_curve": curve, "train_loss": round(tr, 6),
            "gap": round(val - tr, 6), "error": None, "seconds": round(time.perf_counter() - t0, 3)}
