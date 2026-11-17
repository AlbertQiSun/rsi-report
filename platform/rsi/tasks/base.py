"""Task ABC (INTERFACES.md). Implemented by rsi/tasks/nanochat (B) and rsi/tasks/muar (D)."""

from __future__ import annotations

import abc
import pathlib


class Task(abc.ABC):
    #: registry name, e.g. "nanochat" or "muar/modadd-1"
    name: str = ""
    #: metric reported in eval_result, e.g. "val_bpb"
    metric: str = ""
    lower_is_better: bool = True
    #: files the improver may edit, relative to start_artifact(); the orchestrator snapshots exactly these
    editable_files: tuple[str, ...] = ("train.py",)
    #: default dev seeds for a single screening eval (policies may override)
    dev_seeds: tuple[int, ...] = (0,)

    @abc.abstractmethod
    def start_artifact(self) -> pathlib.Path:
        """Directory holding the editable artifact (e.g. train.py) the run starts from."""

    @abc.abstractmethod
    def task_card(self) -> str:
        """Text shown to the improver. Must not contain sealed information."""

    def visible_feedback(self, result: dict) -> str:
        """What the improver may see from an eval_result dict (dev metric + status + flags only)."""
        v = result.get("value")
        vs = f"{v:.6f}" if isinstance(v, (int, float)) else "n/a"
        flags = ",".join(result.get("audit", {}).get("flags", [])) or "none"
        return f"{self.metric}={vs} status={result.get('status')} audit_flags={flags}"

    def better(self, a: float | None, b: float | None) -> bool:
        """True if value a is strictly better than b (None is worst)."""
        if a is None:
            return False
        if b is None:
            return True
        return a < b if self.lower_is_better else a > b


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: ``@register`` makes ``get_task(cls.name)`` work."""
    _REGISTRY[cls.name] = cls
    return cls


def get_task(name: str, **kw) -> Task:
    if name not in _REGISTRY:
        # lazy-import the known task packages so their @register runs
        import importlib

        for mod in ("rsi.tasks.nanochat", "rsi.tasks.muar"):
            try:
                importlib.import_module(mod)
            except ImportError:
                pass
    if name not in _REGISTRY:
        # family lookup: "muar/xyz" -> registered "muar" factory taking name=
        fam = name.split("/", 1)[0]
        if fam in _REGISTRY and fam != name:
            return _REGISTRY[fam](name=name, **kw)
        raise KeyError(f"unknown task {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)
