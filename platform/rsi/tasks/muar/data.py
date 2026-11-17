"""Deterministic data generators for the 8 µAR task types (numpy PCG64, no downloads).

Every generator takes (params, data_seed) and returns a ``Dataset`` with three disjoint
splits: ``train`` (the model trains on it), ``dev`` (scored, visible to the improver) and
``sealed`` (scored only by the analysis script). Inputs are flat float32 vectors so one MLP
skeleton serves all types; sequence tasks use flattened one-hot encodings.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")


@dataclass
class Dataset:
    loss: str                  # "ce" (classification) or "mse" (regression, targets standardised)
    in_dim: int
    n_out: int                 # output units
    n_heads: int               # ce only: outputs are n_heads groups of n_out // n_heads classes
    X: dict                    # split -> float32 [n, in_dim]
    Y: dict                    # split -> int64 [n] / [n, n_heads] (ce) or float32 [n, n_out] (mse)

    def sizes(self) -> dict:
        return {k: int(v.shape[0]) for k, v in self.X.items()}


def _rng(data_seed: int, salt: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(f"{data_seed}:{salt}".encode()).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(h))


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(idx.shape + (n,), dtype=np.float32)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


def _split(X, Y, n_train, n_dev, n_sealed):
    a, b = n_train, n_train + n_dev
    return ({"train": X[:a], "dev": X[a:b], "sealed": X[b:b + n_sealed]},
            {"train": Y[:a], "dev": Y[a:b], "sealed": Y[b:b + n_sealed]})


# ---------------------------------------------------------------------------- task types
def modadd(p: dict, seed: int) -> Dataset:
    """(a + b) mod P from two one-hot integers; a fraction of all P^2 pairs is used for training."""
    P = int(p["P"])
    rng = _rng(seed, "modadd")
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    a, b = a.ravel(), b.ravel()
    perm = rng.permutation(P * P)
    a, b = a[perm], b[perm]
    X = np.concatenate([_onehot(a, P), _onehot(b, P)], axis=1)
    Y = ((a + b) % P).astype(np.int64)
    n_train = int(round(p["train_frac"] * P * P))
    rest = P * P - n_train
    X, Y = _split(X, Y, n_train, rest // 2, rest - rest // 2)
    return Dataset("ce", 2 * P, P, 1, X, Y)


def parity(p: dict, seed: int) -> Dataset:
    """Sparse parity: label = XOR of k secret bits out of n (inputs in {-1, +1})."""
    n, k = int(p["n"]), int(p["k"])
    rng = _rng(seed, "parity")
    support = rng.choice(n, size=k, replace=False)
    N = p["n_train"] + 2 * p["n_val"]
    bits = rng.integers(0, 2, size=(N, n))
    Y = (bits[:, support].sum(1) % 2).astype(np.int64)
    X = (2.0 * bits - 1.0).astype(np.float32)
    X, Y = _split(X, Y, p["n_train"], p["n_val"], p["n_val"])
    return Dataset("ce", n, 2, 1, X, Y)


def copyrev(p: dict, seed: int) -> Dataset:
    """Copy or reverse a length-L sequence over a V-symbol vocabulary (L output heads)."""
    L, V = int(p["L"]), int(p["V"])
    rng = _rng(seed, "copyrev")
    N = p["n_train"] + 2 * p["n_val"]
    seq = rng.integers(0, V, size=(N, L))
    tgt = seq[:, ::-1] if p["mode"] == "reverse" else seq
    X = _onehot(seq, V).reshape(N, L * V)
    X, Y = _split(X, np.ascontiguousarray(tgt).astype(np.int64), p["n_train"], p["n_val"], p["n_val"])
    return Dataset("ce", L * V, L * V, L, X, Y)


def _balanced(rng, L, K):
    """Random balanced bracket string of even length L over K bracket types (token ids 0..2K-1)."""
    out, stack = [], []
    for i in range(L):
        remaining = L - i
        can_open = len(stack) < remaining - 1 if remaining > 1 else False
        if stack and (not can_open or rng.random() < 0.5):
            out.append(2 * stack.pop() + 1)
        else:
            t = int(rng.integers(0, K))
            stack.append(t)
            out.append(2 * t)
    return out


def _is_balanced(s, K):
    stack = []
    for tok in s:
        if tok % 2 == 0:
            stack.append(tok // 2)
        elif not stack or stack.pop() != tok // 2:
            return False
    return not stack


def dyck(p: dict, seed: int) -> Dataset:
    """Dyck-K recognition: balanced strings vs near-misses (1-2 token edits), 50/50."""
    L, K = int(p["L"]), int(p["K"])
    rng = _rng(seed, "dyck")
    N = p["n_train"] + 2 * p["n_val"]
    seqs, labels = [], []
    while len(seqs) < N:
        s = _balanced(rng, L, K)
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 3))):
                j = int(rng.integers(0, L))
                if rng.random() < 0.5:   # swap two positions
                    j2 = int(rng.integers(0, L))
                    s[j], s[j2] = s[j2], s[j]
                else:                   # replace a token
                    s[j] = int(rng.integers(0, 2 * K))
        seqs.append(s)
        labels.append(int(_is_balanced(s, K)))
    seq = np.array(seqs, dtype=np.int64)
    X = _onehot(seq, 2 * K).reshape(N, L * 2 * K)
    X, Y = _split(X, np.array(labels, dtype=np.int64), p["n_train"], p["n_val"], p["n_val"])
    return Dataset("ce", L * 2 * K, 2, 1, X, Y)


def teacher_student(p: dict, seed: int) -> Dataset:
    """Regress the output of a random tanh teacher MLP plus Gaussian noise (targets standardised)."""
    d, h, depth = int(p["d"]), int(p["teacher_width"]), int(p["teacher_depth"])
    rng = _rng(seed, "teacher")
    N = p["n_train"] + 2 * p["n_val"]
    X = rng.normal(size=(N, d)).astype(np.float32)
    z, fan = X.astype(np.float64), d
    for _ in range(depth):
        W = rng.normal(size=(fan, h)) / np.sqrt(fan)
        z = np.tanh(2.0 * z @ W)
        fan = h
    y = z @ (rng.normal(size=(fan, 1)) / np.sqrt(fan))
    y = y + p["noise"] * y.std() * rng.normal(size=y.shape)
    y = ((y - y.mean()) / y.std()).astype(np.float32)
    X, Y = _split(X, y, p["n_train"], p["n_val"], p["n_val"])
    return Dataset("mse", d, 1, 1, X, Y)


def cls2d(p: dict, seed: int) -> Dataset:
    """Noisy 2-D classification (spirals / moons / circles / checker) with label noise."""
    rng = _rng(seed, "cls2d")
    N = p["n_train"] + 2 * p["n_val"]
    C = int(p.get("classes", 2))
    kind = p["kind"]
    if kind == "spirals":
        y = rng.integers(0, C, size=N)
        r = rng.uniform(0.1, 1.0, size=N)
        th = 3.5 * r * np.pi + 2 * np.pi * y / C
        X = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    elif kind == "moons":
        C = 2
        y = rng.integers(0, 2, size=N)
        th = rng.uniform(0, np.pi, size=N)
        X = np.where(y[:, None] == 0, np.stack([np.cos(th), np.sin(th)], 1),
                     np.stack([1 - np.cos(th), 0.5 - np.sin(th)], 1))
    elif kind == "circles":
        y = rng.integers(0, C, size=N)
        r = (y + 1) / C + 0.0 * y
        th = rng.uniform(0, 2 * np.pi, size=N)
        X = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    elif kind == "checker":
        C = 2
        X = rng.uniform(-1, 1, size=(N, 2))
        y = ((np.floor(X[:, 0] * 2) + np.floor(X[:, 1] * 2)) % 2).astype(np.int64)
    else:
        raise ValueError(kind)
    X = X + p["jitter"] * rng.normal(size=X.shape)
    flip = rng.random(N) < p["label_noise"]
    y = np.where(flip, rng.integers(0, C, size=N), y).astype(np.int64)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    X, Y = _split(X, y, p["n_train"], p["n_val"], p["n_val"])
    return Dataset("ce", 2, C, 1, X, Y)


def icl_linreg(p: dict, seed: int) -> Dataset:
    """In-context linear regression: k (x, y) pairs from a fresh w per example, predict y at x_q."""
    d, k = int(p["d"]), int(p["k"])
    rng = _rng(seed, "icl")
    N = p["n_train"] + 2 * p["n_val"]
    w = rng.normal(size=(N, d)) / np.sqrt(d)
    xs = rng.normal(size=(N, k + 1, d))
    ys = np.einsum("nkd,nd->nk", xs, w) + p["noise"] * rng.normal(size=(N, k + 1))
    ctx = np.concatenate([xs[:, :k], ys[:, :k, None]], axis=2).reshape(N, k * (d + 1))
    X = np.concatenate([ctx, xs[:, k]], axis=1).astype(np.float32)
    y = ys[:, k:k + 1]
    y = ((y - y.mean()) / y.std()).astype(np.float32)
    X, Y = _split(X, y, p["n_train"], p["n_val"], p["n_val"])
    return Dataset("mse", k * (d + 1) + d, 1, 1, X, Y)


def corpus_text() -> str:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return f.read()


def corpus_sha256() -> str:
    with open(CORPUS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def charlm(p: dict, seed: int) -> Dataset:
    """Next-character prediction on a frozen slice of this repo's source (corpus.txt).

    The corpus is cut into three contiguous regions (train / dev / sealed) so validation text
    never overlaps training windows. The vocabulary is the V-1 most frequent characters + <unk>.
    """
    text = corpus_text()
    ctxn, V = int(p["context"]), int(p["V"])
    rng = _rng(seed, "charlm")
    # a task-specific window of the corpus (different tasks see different code)
    span = int(p["span"])
    start = int(rng.integers(0, max(1, len(text) - span)))
    text = text[start:start + span]
    chars, counts = np.unique(np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32), return_counts=True)
    top = chars[np.argsort(-counts, kind="stable")][: V - 1]
    lut = {int(c): i for i, c in enumerate(top)}
    ids = np.array([lut.get(int(c), V - 1) for c in np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)],
                   dtype=np.int64)
    n = len(ids)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)

    def windows(lo, hi, count):
        pos = rng.integers(lo + ctxn, hi, size=count)
        ctx = np.stack([ids[q - ctxn:q] for q in pos])
        return _onehot(ctx, V).reshape(count, ctxn * V), ids[pos]

    Xtr, Ytr = windows(0, cut1, p["n_train"])
    Xdv, Ydv = windows(cut1, cut2, p["n_val"])
    Xse, Yse = windows(cut2, n, p["n_val"])
    return Dataset("ce", ctxn * V, V, 1, {"train": Xtr, "dev": Xdv, "sealed": Xse},
                   {"train": Ytr, "dev": Ydv, "sealed": Yse})


GENERATORS = {
    "modadd": modadd, "parity": parity, "copyrev": copyrev, "dyck": dyck,
    "teacher_student": teacher_student, "cls2d": cls2d, "icl_linreg": icl_linreg, "charlm": charlm,
}

_CACHE: dict = {}


def make_dataset(ttype: str, params: dict, data_seed: int) -> Dataset:
    key = (ttype, tuple(sorted(params.items())), data_seed)
    if key not in _CACHE:
        if len(_CACHE) > 32:
            _CACHE.clear()
        _CACHE[key] = GENERATORS[ttype](params, data_seed)
    return _CACHE[key]
