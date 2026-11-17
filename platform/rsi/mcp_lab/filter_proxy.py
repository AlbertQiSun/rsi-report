"""Filtering stdio MCP proxy for web search / fetch servers (build plan §2.5 item 11, D4).

    python -m rsi.mcp_lab.filter_proxy --log /lab/outbox/web.jsonl -- <real MCP server command...>

Relays JSON-RPC lines between the harness (stdin/stdout) and the child server. Every
``tools/call`` is logged (tool, query/url, verdict). Calls whose arguments mention a blocked
answer source (public optimized NanoChat solutions) are answered by the proxy with an error and
never reach the child. The orchestrator marks trials whose web log has hits as retrieval-assisted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

# Public answer sources for the NanoChat task (case-insensitive substrings of URL or query).
DEFAULT_BLOCKLIST = [
    "first-steps-toward-automated-ai-research",
    "optimized_from_vanilla",
    "optimized_from_karpathy",
    "recursivesi",
    "karpathy/autoresearch",
    "autoresearch/blob",
    "autoresearch/tree",
    "autoresearch/raw",
    "ensue-network.ai",
    "autoresearch-at-home",
    "autoresearch@home",
    "autoresearch-mlx",
]


def blocked(args: dict, blocklist: list[str]) -> str | None:
    blob = json.dumps(args, ensure_ascii=False).lower()
    for pat in blocklist:
        if pat.lower() in blob:
            return pat
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: filter_proxy [--log FILE] [--block PAT ...] -- cmd ...", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get("RSI_WEB_LOG", "web.jsonl"))
    ap.add_argument("--block", action="append", default=[])
    ap.add_argument("--no-default-blocklist", action="store_true")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    blocklist = ([] if a.no_default_blocklist else list(DEFAULT_BLOCKLIST)) + a.block

    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    out_lock = threading.Lock()

    def write_out(line: str) -> None:
        with out_lock:
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            sys.stdout.flush()

    def log(rec: dict) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
            with open(a.log, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def pump_child() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            write_out(line)

    threading.Thread(target=pump_child, daemon=True).start()
    assert child.stdin is not None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            msg = None
        if isinstance(msg, dict) and msg.get("method") == "tools/call":
            params = msg.get("params") or {}
            args = params.get("arguments") or {}
            hit = blocked(args, blocklist)
            log({"t": round(time.time(), 3), "tool": params.get("name"),
                 "args": {k: (str(v)[:300]) for k, v in (args.items() if isinstance(args, dict) else [])},
                 "blocked": hit})
            if hit is not None and msg.get("id") is not None:
                write_out(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
                    "content": [{"type": "text", "text": "Blocked by the experiment rules: this source "
                                 "publishes solutions to the task. Search for general techniques instead."}],
                    "isError": True}}))
                continue
        try:
            child.stdin.write(line)
            child.stdin.flush()
        except BrokenPipeError:
            break
    try:
        child.stdin.close()
    except OSError:
        pass
    return child.wait()


if __name__ == "__main__":
    sys.exit(main())
