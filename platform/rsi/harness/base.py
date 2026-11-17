"""Harness ABC (INTERFACES.md): one agent episode in one workspace."""

from __future__ import annotations

import abc
import pathlib


class Harness(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def run_episode(self, *, run: str, agent: str, gen: int, trial: int, workspace: pathlib.Path,
                    prompt: str, model: str, scaffold_dir: pathlib.Path | None, budget_s: int,
                    parent: str | None = "orchestrator") -> dict:
        """Run one episode and return::

            {"final": str,               # last assistant text (summary)
             "n_tool_calls": int,
             "trajectory": str,          # path of the raw trajectory file for this episode
             "ok": bool,                 # False on harness crash / timeout
             "submitted": dict | None}   # lab submit_eval payload {hypothesis, ...} if the agent called it

        Emits agent_spawn / tool_call / message / agent_end events for ``agent`` (and its subagents).
        """
