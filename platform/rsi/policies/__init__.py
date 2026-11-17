"""Loop policies. ``get_policy(name)`` imports ``rsi.policies.<name>`` and returns its Policy class."""

from __future__ import annotations

import importlib
import inspect

KNOWN = ["karpathy_greedy", "naive_trial_ledger", "recursive_first_steps", "apex_ai4ai", "self_evolution"]


def get_policy(name: str):
    from rsi.policies.base import Policy

    mod = importlib.import_module(f"rsi.policies.{name}")
    if hasattr(mod, "POLICY"):
        return mod.POLICY
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Policy) and obj is not Policy and getattr(obj, "name", None) == name:
            return obj
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, Policy) and obj is not Policy and obj.__module__ == mod.__name__:
            return obj
    raise ImportError(f"no Policy subclass in rsi.policies.{name}")
