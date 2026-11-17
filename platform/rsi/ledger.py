"""Append-only trial ledger plus first-order summary statistics (nAUC)."""

from __future__ import annotations

import threading
import time
from typing import Any

from rsi.runs import append_jsonl, read_jsonl


# ---------------------------------------------------------------------------
# Ledger: one line per trial event; the latest line per trial is its state
# ---------------------------------------------------------------------------

class Ledger:
    """Append-only trial ledger.

    Statuses: ``pending`` (submitted, no result yet), ``keep`` / ``discard`` /
    ``crash`` (karpathy_greedy vocabulary), ``adopted`` / ``rolled_back`` /
    ``exploratory`` (naive_trial_ledger vocabulary), ``no_change``.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, **rec: Any) -> dict:
        rec.setdefault("ts", round(time.time(), 3))
        with self._lock:
            append_jsonl(self.path, rec)
        return rec

    def records(self) -> list[dict]:
        return read_jsonl(self.path)

    def trials(self) -> dict[int, dict]:
        """Latest record per trial id."""
        latest: dict[int, dict] = {}
        for r in self.records():
            if r.get("trial") is not None:
                latest.setdefault(r["trial"], {}).update(r)
        return latest

    def finished(self, thread: str | None = None) -> list[dict]:
        out = [r for r in self.trials().values() if r.get("status") != "pending"]
        if thread is not None:
            out = [r for r in out if r.get("thread") == thread]
        return sorted(out, key=lambda r: r["trial"])

    def tail_text(self, thread: str | None = None, n: int = 8, metric: str = "value") -> str:
        """results.tsv-style view (dev scores only) for agent prompts."""
        rows = self.finished(thread)[-n:]
        lines = [f"trial\tartifact\t{metric}\tstatus\tdescription"]
        for r in rows:
            v = r.get("dev")
            vs = f"{v:.6f}" if isinstance(v, (int, float)) else "-"
            lines.append(f"{r['trial']}\t{r.get('artifact_hash', '-')}\t{vs}\t{r.get('status')}\t"
                         f"{(r.get('hypothesis') or '')[:120]}")
        return "\n".join(lines)


def best_so_far_curve(values: list[float | None], lower_is_better: bool = True) -> list[float | None]:
    best = None
    out = []
    for v in values:
        if v is not None and (best is None or (v < best if lower_is_better else v > best)):
            best = v
        out.append(best)
    return out


def nauc(s0: float, values: list[float | None], s_ref: float | None) -> float | None:
    """Normalised area under the best-so-far curve (design doc §1.2).

    g(t) = (s0 - best_t) / (s0 - s_ref), averaged over t = 1..N (lower is better).
    Returns None when s_ref is unknown or degenerate.
    """
    if s_ref is None or not values or s0 == s_ref:
        return None
    curve = best_so_far_curve([s0] + list(values))[1:]
    gs = [(s0 - (b if b is not None else s0)) / (s0 - s_ref) for b in curve]
    return sum(gs) / len(gs)
