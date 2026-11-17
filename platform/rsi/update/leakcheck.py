"""Held-out leak check (design doc §5.1 "训练前用 hash 核对", risk #2).

Fingerprints of every held-out µAR task: its name, the sha256 of each task-card line that does
not also occur in a train-task card (so shared boilerplate cannot false-positive), and the
canonical start config (exact JSON and as a parsed object). ``scan_text`` reports any hit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from rsi.tasks.muar import schema
from rsi.tasks.muar.tasks import get_spec, heldout_names, start_config_and_defects, task_card, tasks_in


def _norm(line: str) -> str:
    return " ".join(line.split())


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class Fingerprints:
    names: set = field(default_factory=set)          # "muar/id-modadd-0", ...
    line_hashes: dict = field(default_factory=dict)  # sha256(normalised line) -> task name
    configs: dict = field(default_factory=dict)      # canonical json -> task name
    card_hashes: dict = field(default_factory=dict)  # sha256(full card) -> task name

    def summary(self) -> dict:
        return {"tasks": sorted(self.names), "n_line_hashes": len(self.line_hashes), "n_configs": len(self.configs),
                "digest": _h(json.dumps([sorted(self.names), sorted(self.line_hashes), sorted(self.configs)]))[:16]}


def heldout_fingerprints(heldout: list[str] | None = None) -> Fingerprints:
    heldout = list(heldout or heldout_names())
    train_lines = set()
    for t in tasks_in("train"):
        train_lines |= {_norm(x) for x in task_card(get_spec(t)).splitlines() if x.strip()}
    fp = Fingerprints()
    for t in heldout:
        spec = get_spec(t)
        card = task_card(spec)
        fp.names.add(spec.name)
        fp.card_hashes[_h(card)] = spec.name
        for line in card.splitlines():
            n = _norm(line)
            if n and n not in train_lines and len(n) >= 12:
                fp.line_hashes[_h(n)] = spec.name
        cfg = start_config_and_defects(spec)[0]
        fp.configs[schema.canonical_json(cfg)] = spec.name
    return fp


_JSON_OBJ = re.compile(r"\{[^{}]{20,2000}\}")


def scan_text(text: str, fp: Fingerprints) -> list[str]:
    """Return a list of leak descriptions found in ``text`` (empty = clean)."""
    hits = []
    for name in fp.names:
        if name in text or name.split("/", 1)[1] in text:
            hits.append(f"name:{name}")
    if _h(text) in fp.card_hashes:
        hits.append(f"card:{fp.card_hashes[_h(text)]}")
    for line in text.splitlines():
        hl = _h(_norm(line))
        if hl in fp.line_hashes:
            hits.append(f"card_line:{fp.line_hashes[hl]}")
    for m in _JSON_OBJ.finditer(text.replace("\\n", " ").replace('\\"', '"')):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and set(obj) == set(schema.KEYS):
            try:
                c = schema.canonical_json(schema.validate_config(obj))
            except schema.IllegalConfig:
                continue
            if c in fp.configs:
                hits.append(f"start_config:{fp.configs[c]}")
    return sorted(set(hits))


def example_text(ex: dict) -> str:
    """All text of a messages-format example (contents, tool-call arguments, tools)."""
    parts = []
    for m in ex.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            parts.extend(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
        elif c is not None:
            parts.append(str(c))
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            parts.append(str(fn.get("arguments", "")))
    if ex.get("tools"):
        parts.append(json.dumps(ex["tools"]))
    return "\n".join(parts)
