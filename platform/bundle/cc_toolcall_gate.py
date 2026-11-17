#!/usr/bin/env python3
"""Gate G1b-CC: tool-call validity and completion of the improver under the real Claude Code harness.

Runs ~10 headless Claude Code episodes (rsi.harness.claude_code, same bwrap sandbox, lab + DDG + fetch MCPs,
gateway_agent) on small scripted workspace tasks that exercise Bash, Read, Edit, Write, the lab MCP and the
Agent (subagent) tool, then checks the workspace. Prints and writes a JSON summary:

  call_validity_rate       1 - schema-invalid tool calls / tool calls          (gate: >= 0.90)
                           (invalid = tool_result is_error with InputValidationError / unknown tool / ...)
  episode_completion_rate  episodes whose workspace check passed / episodes    (gate: >= 0.70)
  tool_error_rate          all is_error tool results / tool calls (includes legitimate failures)
  harness_ok_rate          episodes without crash / timeout
  first_prompt_tokens      prompt_tokens of each episode's first llm_call (= Claude Code's system prompt + tools)
  per_episode              task, passed, ok, reason, n_tool_calls, n_invalid_calls, tools_used, n_subagents, ...

Usage (cluster, gateway_agent up, Claude Code installed):
  python bundle/cc_toolcall_gate.py --config configs/harness_claude_cluster.json --model improver \
      --out $R/out/g1b-cc-improver.json
Exit code 0 iff the gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rsi import events  # noqa: E402

SUBMIT = "Then call the lab MCP tool submit_eval with a one-sentence hypothesis, and stop."


def _r(ws: pathlib.Path, name: str) -> str:
    p = ws / name
    return p.read_text() if p.exists() else ""


def _has(ws, name, pat) -> bool:
    return re.search(pat, _r(ws, name), re.M) is not None


@dataclass
class GateTask:
    name: str
    files: dict
    prompt: str
    check: Callable[[pathlib.Path, dict], bool]
    solve: Callable[[pathlib.Path], None]          # reference solution (unit tests only)
    card: str = "Gate task card."
    needs_submit: bool = False
    extra: dict = field(default_factory=dict)


def _sub(ws, name, a, b, count=1):
    p = ws / name
    p.write_text(p.read_text().replace(a, b, count))


def _exec_ok(ws) -> bool:
    ns: dict = {}
    try:
        exec(compile(_r(ws, "train.py"), "train.py", "exec"), ns)   # noqa: S102 (gate workspace, trusted output check)
        return abs(ns["warmup"](50) - 0.5) < 1e-9 and ns["warmup"](500) == 1.0
    except Exception:
        return False


TRAIN = "LEARNING_RATE = 9e-4\nWIDTH = 64\nBATCH_SIZE = 32\nEPOCHS = 3\n"
TASKS = [
    GateTask("lr", {"train.py": TRAIN},
             "In train.py change LEARNING_RATE from 9e-4 to 2e-3. Change nothing else. " + SUBMIT,
             lambda ws, o: _has(ws, "train.py", r"^LEARNING_RATE = 2e-3$") and _has(ws, "train.py", r"^WIDTH = 64$"),
             lambda ws: _sub(ws, "train.py", "9e-4", "2e-3"), needs_submit=True),
    GateTask("width", {"train.py": TRAIN},
             "Double WIDTH in train.py (it is 64 now). Change nothing else. " + SUBMIT,
             lambda ws, o: _has(ws, "train.py", r"^WIDTH = 128$") and _has(ws, "train.py", r"^LEARNING_RATE = 9e-4$"),
             lambda ws: _sub(ws, "train.py", "WIDTH = 64", "WIDTH = 128"), needs_submit=True),
    GateTask("add_fn", {"train.py": TRAIN},
             "Append a Python function `warmup(step)` to the end of train.py that returns min(1.0, step / 100). " + SUBMIT,
             lambda ws, o: _exec_ok(ws),
             lambda ws: (ws / "train.py").write_text(TRAIN + "\n\ndef warmup(step):\n    return min(1.0, step / 100)\n"),
             needs_submit=True),
    GateTask("count_lines", {"data.txt": "".join(f"row {i}\n" for i in range(37))},
             "Use a shell command to count the lines in data.txt, then write only that number to a new file "
             "count.txt. Do not submit anything.",
             lambda ws, o: _r(ws, "count.txt").strip() == "37",
             lambda ws: (ws / "count.txt").write_text("37\n")),
    GateTask("replace_all", {"model.py": 'ACT = "relu"\nHEAD_ACT = "relu"\n# relu everywhere\n'},
             "Replace every occurrence of the word relu with gelu in model.py. " + SUBMIT,
             lambda ws, o: "relu" not in _r(ws, "model.py") and _r(ws, "model.py").count("gelu") == 3,
             lambda ws: (ws / "model.py").write_text(_r(ws, "model.py").replace("relu", "gelu")), needs_submit=True),
    GateTask("notes", {"train.py": TRAIN, "notes.md": "# notes\nThe sweep says: use BATCH_SIZE = 48 for this model.\n"},
             "Read notes.md and set BATCH_SIZE in train.py to the value it recommends. " + SUBMIT,
             lambda ws, o: _has(ws, "train.py", r"^BATCH_SIZE = 48$"),
             lambda ws: _sub(ws, "train.py", "BATCH_SIZE = 32", "BATCH_SIZE = 48"), needs_submit=True),
    GateTask("new_file", {},
             'Create a file config.json in the working directory containing the JSON object {"lr": 0.001, "steps": 500}. '
             "Do not submit anything.",
             lambda ws, o: _json_eq(ws / "config.json", {"lr": 0.001, "steps": 500}),
             lambda ws: (ws / "config.json").write_text('{"lr": 0.001, "steps": 500}\n')),
    GateTask("task_card", {"train.py": TRAIN},
             "Call the lab MCP tool task_card, do exactly what the card says, then call submit_eval with a "
             "one-sentence hypothesis, and stop.",
             lambda ws, o: _has(ws, "train.py", r"^EPOCHS = 7$"),
             lambda ws: _sub(ws, "train.py", "EPOCHS = 3", "EPOCHS = 7"),
             card="Gate task: set EPOCHS in train.py to 7 and change nothing else.", needs_submit=True),
    GateTask("run_script", {"calc.py": "print(sum(range(1000)))\n"},
             "Run calc.py with python3 and write exactly what it prints to a new file result.txt. Do not submit anything.",
             lambda ws, o: _r(ws, "result.txt").strip() == "499500",
             lambda ws: (ws / "result.txt").write_text("499500\n")),
    GateTask("subagent", {"train.py": TRAIN},
             "Use the Agent tool to start a subagent that reads train.py and reports the value of WIDTH. When it "
             "answers, set WIDTH in train.py to twice that value. " + SUBMIT,
             lambda ws, o: _has(ws, "train.py", r"^WIDTH = 128$"),
             lambda ws: _sub(ws, "train.py", "WIDTH = 64", "WIDTH = 128"), needs_submit=True),
]


def _json_eq(p: pathlib.Path, want) -> bool:
    try:
        return json.loads(p.read_text()) == want
    except (OSError, ValueError):
        return False


def setup(task: GateTask, tdir: pathlib.Path) -> pathlib.Path:
    ws = tdir / "workspace"
    ws.mkdir(parents=True)
    for n, c in task.files.items():
        (ws / n).write_text(c)
    (tdir / "lab" / "outbox").mkdir(parents=True)
    (tdir / "lab" / "card.md").write_text(task.card)
    return ws


def passed(task: GateTask, ws: pathlib.Path, out: dict) -> bool:
    good = bool(task.check(ws, out))
    if task.needs_submit:
        good = good and bool(out.get("submitted"))
    return good


def run_gate(h, *, run: str, model: str, root: pathlib.Path, tasks=TASKS, n: int = 10, parallel: int = 1,
             budget_s: int = 600) -> dict:
    chosen = [tasks[i % len(tasks)] for i in range(n)]

    def one(i: int) -> dict:
        task = chosen[i]
        tdir = root / f"gate-{i:02d}"
        ws = setup(task, tdir)
        t0 = time.time()
        try:
            out = h.run_episode(run=run, agent=f"gate-{i:02d}", gen=0, trial=i + 1, workspace=ws, prompt=task.prompt,
                                model=model, scaffold_dir=None, budget_s=budget_s)
        except Exception as e:  # a crash of one episode must not kill the gate
            out = {"ok": False, "reason": f"harness exception: {type(e).__name__}: {e}"}
        return {"i": i, "task": task.name, "passed": passed(task, ws, out), "ok": bool(out.get("ok")),
                "reason": out.get("reason"), "submitted": bool(out.get("submitted")),
                "n_tool_calls": int(out.get("n_tool_calls") or 0), "n_tool_errors": int(out.get("n_tool_errors") or 0),
                "n_invalid_calls": int(out.get("n_invalid_calls") or 0), "n_subagents": int(out.get("n_subagents") or 0),
                "tools_used": out.get("tools_used") or {}, "num_turns": (out.get("result") or {}).get("num_turns"),
                "mcp_servers": (out.get("init") or {}).get("mcp_servers"), "final": (out.get("final") or "")[:200],
                "wall_s": round(time.time() - t0, 1), "trajectory": out.get("trajectory")}

    with ThreadPoolExecutor(max(1, parallel)) as ex:
        eps = list(ex.map(one, range(n)))
    return summarize(eps)


def summarize(eps: list[dict], llm_calls: list[dict] | None = None) -> dict:
    calls = sum(e["n_tool_calls"] for e in eps)
    inval = sum(e["n_invalid_calls"] for e in eps)
    errs = sum(e["n_tool_errors"] for e in eps)
    n = len(eps) or 1
    s = {"n_episodes": len(eps), "n_tool_calls": calls,
         "call_validity_rate": round(1 - inval / calls, 4) if calls else 0.0,
         "tool_error_rate": round(errs / calls, 4) if calls else None,
         "episode_completion_rate": round(sum(e["passed"] for e in eps) / n, 4),
         "harness_ok_rate": round(sum(e["ok"] for e in eps) / n, 4),
         "episodes_without_tool_calls": sum(1 for e in eps if e["n_tool_calls"] == 0),
         "subagent_task_spawned": [e["n_subagents"] for e in eps if e["task"] == "subagent"],
         "per_episode": eps}
    s["gate_pass"] = s["call_validity_rate"] >= 0.90 and s["episode_completion_rate"] >= 0.70
    return s


def llm_stats(events_path: str) -> dict:
    first: dict[str, int] = {}
    n, pts = 0, []
    if os.path.exists(events_path):
        for e in events.read_events(events_path):
            if e.get("ev") != "llm_call":
                continue
            n += 1
            pts.append(int(e.get("prompt_tokens") or 0))
            first.setdefault(e["agent"], int(e.get("prompt_tokens") or 0))
    fp = sorted(first.values())
    return {"n_llm_calls": n, "first_prompt_tokens": {"median": statistics.median(fp) if fp else None,
                                                      "min": fp[0] if fp else None, "max": fp[-1] if fp else None},
            "max_prompt_tokens": max(pts) if pts else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="harness config JSON (kind claude_code)")
    ap.add_argument("--model", default="improver")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--budget-s", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs-root", default=None, help="default $RSI_RUNS_ROOT or $R/runs (same as the gateway's)")
    ap.add_argument("--keep", action="store_true", help="keep the gate workspaces")
    a = ap.parse_args(argv)
    from rsi.harness.mini import make_harness
    from rsi.harness.selftest import _cfg

    cfg = _cfg(a.config)
    if cfg.get("kind") != "claude_code":
        print(f"warning: config kind is {cfg.get('kind')!r}, not claude_code", file=sys.stderr)
    if not cfg.get("gateway_url"):
        from rsi.orchestrator import agent_gateway_url
        cfg["gateway_url"] = agent_gateway_url(cfg)
    R = os.environ.get("R") or os.environ.get("RSI_ROOT") or "/scratch/user/rsi"
    runs_root = pathlib.Path(a.runs_root or os.environ.get("RSI_RUNS_ROOT") or os.path.join(R, "runs"))
    run = f"ccgate-{a.model.replace('@', '_')}-{time.strftime('%Y%m%dT%H%M%S')}"
    ev_path = runs_root / run / "events.jsonl"
    events.register_run(run, str(ev_path))
    events.emit(run, "run_start", "orchestrator", policy="cc_toolcall_gate", task="gate",
                improver={"model": a.model, "scaffold_hash": None}, budget={"trials": a.n, "wall_s": None},
                config_hash="cc_toolcall_gate", harness="claude_code")
    cfg["trajectories_dir"] = str(runs_root / run / "trajectories" / "harness")
    root = pathlib.Path(tempfile.mkdtemp(prefix="ccgate-", dir=cfg.get("selftest_dir") or None))
    h = make_harness(cfg)
    t0 = time.time()
    s = run_gate(h, run=run, model=a.model, root=root, n=a.n, parallel=a.parallel, budget_s=a.budget_s)
    s.update(llm_stats(str(ev_path)))
    s.update({"run": run, "events": str(ev_path), "model": a.model, "wall_s": round(time.time() - t0, 1),
              "workspaces": str(root) if a.keep else None})
    events.emit(run, "run_end", "orchestrator", best_value=None, n_trials=a.n, nAUC=None,
                gate_pass=s["gate_pass"], call_validity_rate=s["call_validity_rate"],
                episode_completion_rate=s["episode_completion_rate"])
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
    txt = json.dumps(s, indent=1, default=str)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(txt)
    print(json.dumps({k: v for k, v in s.items() if k != "per_episode"}, indent=1, default=str))
    for e in s["per_episode"]:
        print(f"  {e['task']:12s} passed={e['passed']!s:5s} ok={e['ok']!s:5s} calls={e['n_tool_calls']:2d} "
              f"invalid={e['n_invalid_calls']} sub={e['n_subagents']} turns={e['num_turns']} {e['wall_s']}s {e['reason']}")
    print("CC_TOOLCALL_GATE", "PASS" if s["gate_pass"] else "FAIL")
    return 0 if s["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
