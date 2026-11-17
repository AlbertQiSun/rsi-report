"""RunContext (INTERFACES.md): everything a Policy needs for one run."""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

from rsi import events
from rsi.ledger import Ledger
from rsi.runs import ArtifactStore, RunDir


@dataclass
class RunContext:
    run: str
    gen: int
    task: Any                       # rsi.tasks.base.Task
    evald: Any                      # rsi.evald.client.EvalClient (or a test double with the same methods)
    harness: Any                    # rsi.harness.base.Harness
    model: str                      # gateway model name, e.g. "improver" or "improver@m1"
    scaffold_dir: pathlib.Path | None
    budget: dict                    # {"trials": int, "wall_s": float, "episode_s": int}
    run_dir: pathlib.Path
    ledger: Ledger
    rng_seed: int = 0
    lanes: list[int] = field(default_factory=lambda: [1])   # eval lanes the threads map onto
    policy_cfg: dict = field(default_factory=dict)
    t_start: float = field(default_factory=time.time)
    start_from: pathlib.Path | None = None      # artifact dir to start from instead of task.start_artifact()
    history_from: list = field(default_factory=list)   # prior run dirs whose (dev-only) history the improver sees

    # ---- construction -------------------------------------------------
    @classmethod
    def create(cls, *, run: str, runs_root: str | os.PathLike, **kw: Any) -> "RunContext":
        rd = RunDir(str(runs_root), run).create()
        events.register_run(run, rd.events)
        ledger = Ledger(rd.ledger_path)
        return cls(run=run, run_dir=pathlib.Path(rd.path), ledger=ledger, **kw)

    # ---- helpers ------------------------------------------------------
    @property
    def rd(self) -> RunDir:
        return RunDir(str(self.run_dir.parent), self.run)

    @property
    def artifacts(self) -> ArtifactStore:
        return ArtifactStore(str(self.run_dir / "artifacts"))

    def emit(self, ev: str, agent: str, **kw: Any) -> dict:
        kw.setdefault("gen", self.gen)
        return events.emit(self.run, ev, agent, **kw)

    def paused(self) -> bool:
        return (self.run_dir / "PAUSE").exists()

    def elapsed(self) -> float:
        return time.time() - self.t_start

    def start_dir(self) -> pathlib.Path:
        return pathlib.Path(self.start_from) if self.start_from else pathlib.Path(self.task.start_artifact())

    def n_evals(self) -> int:
        """Evaluator submissions recorded in the ledger (each distinct job id), baseline included."""
        return len({r["job_id"] for r in self.ledger.records() if r.get("job_id")})

    def history_rows(self, n: int | None = None) -> list[dict]:
        """Finished, dev-only trial rows from ``history_from`` runs (oldest first)."""
        from rsi.ledger import Ledger

        rows = []
        for d in self.history_from or []:
            led = Ledger(str(pathlib.Path(d) / "ledger.jsonl"))
            name = pathlib.Path(d).name
            for r in led.finished():
                if r.get("status") in ("no_change",):
                    continue
                rows.append({"trial": r.get("trial"), "thread": f"prior:{name}/{r.get('thread')}",
                             "artifact": r.get("artifact_hash"), "value": r.get("dev"), "status": r.get("status"),
                             "feedback": r.get("feedback", ""), "hypothesis": (r.get("hypothesis") or "")[:300]})
        return rows[-n:] if n else rows

    def out_of_budget(self, n_trials_done: int) -> bool:
        b = self.budget or {}
        if b.get("trials") is not None and n_trials_done >= int(b["trials"]):
            return True
        if b.get("evals") is not None and self.n_evals() >= int(b["evals"]):
            return True
        if b.get("wall_s") is not None and self.elapsed() >= float(b["wall_s"]):
            return True
        return False

    def save_state(self, **state: Any) -> None:
        from rsi.runs import atomic_write_json

        cur = self.load_state()
        cur.update(state)
        atomic_write_json(str(self.run_dir / "state.json"), cur)

    def load_state(self) -> dict:
        p = self.run_dir / "state.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}
