"""`lab` MCP server (stdio, stdlib only). Runs INSIDE the researcher sandbox.

It talks to nothing over the network and holds no token: it reads a per-thread spool written by
the orchestrator and writes the agent's submission into ``outbox/``. Layout of ``--lab DIR``::

    card.md          task card (read-only view)
    history.jsonl    this thread's finished trials: trial, artifact, value (dev only), status, feedback
    best.json        {artifact, value, metric, lower_is_better}
    outbox/submit.json   written by submit_eval (orchestrator reads it after the episode)
    outbox/calls.jsonl   every lab tool call (audit)

Tools: submit_eval(hypothesis), get_results(n), history(), task_card().
Sealed scores never exist in this directory, so they cannot leak through this server.

Usage: python -m rsi.mcp_lab.server --lab /lab
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

SERVER_INFO = {"name": "lab", "version": "0.1.0"}
DEFAULT_PROTOCOL = "2025-06-18"

TOOLS = [
    {
        "name": "submit_eval",
        "description": ("Submit the current state of your working directory for evaluation by the sealed "
                        "evaluator. Call this exactly once, after you finished editing. The result arrives "
                        "after your session ends; you will see it in history next time."),
        "inputSchema": {
            "type": "object",
            "properties": {"hypothesis": {"type": "string",
                                          "description": "One sentence: what you changed and why it should help."}},
            "required": ["hypothesis"],
        },
    },
    {
        "name": "get_results",
        "description": "Return the n most recent evaluated attempts (dev score, status, feedback) and the current best.",
        "inputSchema": {"type": "object", "properties": {"n": {"type": "integer", "description": "how many (default 5)"}}},
    },
    {
        "name": "history",
        "description": "Return every evaluated attempt in this research thread, oldest first.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "task_card",
        "description": "Return the task description, rules and the evaluation contract.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Lab:
    def __init__(self, lab_dir: str):
        self.dir = lab_dir
        self.outbox = os.path.join(lab_dir, "outbox")

    def _read(self, name: str, default: str = "") -> str:
        try:
            with open(os.path.join(self.dir, name)) as f:
                return f.read()
        except OSError:
            return default

    def rows(self) -> list[dict]:
        out = []
        for line in self._read("history.jsonl").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def best(self) -> dict:
        try:
            return json.loads(self._read("best.json", "{}") or "{}")
        except json.JSONDecodeError:
            return {}

    def log(self, tool: str, args: dict, ok: bool) -> None:
        try:
            os.makedirs(self.outbox, exist_ok=True)
            with open(os.path.join(self.outbox, "calls.jsonl"), "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "tool": tool, "args": args, "ok": ok}) + "\n")
        except OSError:
            pass

    @staticmethod
    def fmt_rows(rows: list[dict]) -> str:
        if not rows:
            return "(no evaluated attempts yet)"
        lines = []
        for r in rows:
            v = r.get("value")
            vs = f"{v:.6f}" if isinstance(v, (int, float)) else "-"
            lines.append(f"trial {r.get('trial')} [{r.get('status')}] value={vs} :: {r.get('hypothesis', '')}"
                         + (f"\n    feedback: {r['feedback']}" if r.get("feedback") else ""))
        return "\n".join(lines)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "submit_eval":
            hyp = str(args.get("hypothesis", "")).strip()
            if not hyp:
                return "error: hypothesis is required (one sentence)", True
            os.makedirs(self.outbox, exist_ok=True)
            tmp = os.path.join(self.outbox, ".submit.tmp")
            with open(tmp, "w") as f:
                json.dump({"hypothesis": hyp[:1000], "t": round(time.time(), 3)}, f)
            os.replace(tmp, os.path.join(self.outbox, "submit.json"))
            return ("Submitted. The orchestrator will evaluate your working directory after this session ends. "
                    "Stop now: reply with a one-line summary and make no further edits."), False
        if name == "get_results":
            try:
                n = max(1, min(50, int(args.get("n", 5))))
            except (TypeError, ValueError):
                n = 5
            b = self.best()
            bv = b.get("value")
            head = f"current best {b.get('metric', 'metric')}: {bv if bv is not None else 'unknown'}\n"
            return head + self.fmt_rows(self.rows()[-n:]), False
        if name == "history":
            return self.fmt_rows(self.rows()), False
        if name == "task_card":
            return self._read("card.md", "(no task card)"), False
        return f"error: unknown tool {name!r}", True


def serve(lab: Lab, stdin=sys.stdin, stdout=sys.stdout) -> None:
    def send(obj: dict) -> None:
        stdout.write(json.dumps(obj) + "\n")
        stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        mid, method = msg.get("id"), msg.get("method")
        if method is None:  # a response to us; we never send requests
            continue
        params = msg.get("params") or {}
        if mid is None:  # notification (notifications/initialized, cancelled, ...)
            continue
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": params.get("protocolVersion", DEFAULT_PROTOCOL),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            text, is_err = lab.call(name, args if isinstance(args, dict) else {})
            lab.log(name, args if isinstance(args, dict) else {}, not is_err)
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}],
                                                          "isError": is_err}})
        elif method in ("resources/list", "prompts/list"):
            key = method.split("/")[0]
            send({"jsonrpc": "2.0", "id": mid, "result": {key: []}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=os.environ.get("RSI_LAB_DIR", "/lab"))
    a = ap.parse_args(argv)
    serve(Lab(a.lab))


if __name__ == "__main__":
    main()
