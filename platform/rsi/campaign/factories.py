"""Factories the orchestrator can load via ``evald: {factory: "rsi.campaign.factories:muar_evald"}``.

µAR artifacts are scored on node CPUs by worker D's LocalEvalClient (evald serves only NanoChat).
Each µAR sub-run is its own orchestrator process; the campaign caps how many run at once
(``muar.cpu_workers``), so each one gets a small pool.
"""

from __future__ import annotations


def muar_evald(e: dict):
    from rsi.tasks.muar.runner import LocalEvalClient

    return LocalEvalClient(workers=int(e.get("workers", 1)))
