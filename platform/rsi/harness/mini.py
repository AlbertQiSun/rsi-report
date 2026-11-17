"""rsi-mini-agent: a minimal tool-calling harness (stdlib only).

Fallback if OpenCode cannot be made to work on the node (build plan §2.2 lists pi and
mini-swe-agent; pi needs Node, which the node lacks, so this in-repo loop is the zero-install
fallback), and the harness used by the local CPU smoke test against the fake model server.

Tools: read_file, write_file, edit_file, list_files, bash (each bash call runs inside the bwrap
sandbox with the workspace at /work), lab_submit_eval / lab_get_results / lab_history /
lab_task_card (same spool as the lab MCP). No web tools (= the web-off arm). No subagents.
Full trajectory (every request/response) -> trajectories/harness/<agent>/trial-<k>.json.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from rsi import events
from rsi.harness.base import Harness
from rsi.harness.sandbox import SandboxSpec, wrap
from rsi.mcp_lab.server import Lab

SYSTEM = ("You are a careful ML research engineer working in a sandbox. Your working directory is /work. "
          "Use the tools to inspect and edit files. Call each tool with valid JSON arguments that follow its "
          "schema exactly. When your edit is complete, call lab_submit_eval once with a one-sentence hypothesis, "
          "then reply with a one-line summary and stop.")


def _fn(name: str, desc: str, props: dict, req: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS = [
    _fn("read_file", "Read a text file in the workspace.", {"path": {"type": "string"}}, ["path"]),
    _fn("write_file", "Create or overwrite a text file in the workspace.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace one exact occurrence of old_string with new_string in a file.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"]),
    _fn("list_files", "List files in the workspace.", {}, []),
    _fn("bash", "Run a shell command in /work (no GPU, 60 s limit).", {"command": {"type": "string"}}, ["command"]),
    _fn("lab_submit_eval", "Submit the workspace for sealed evaluation. Call once when done.",
        {"hypothesis": {"type": "string"}}, ["hypothesis"]),
    _fn("lab_get_results", "Recent evaluated attempts and current best.", {"n": {"type": "integer"}}, []),
    _fn("lab_history", "All evaluated attempts in this thread.", {}, []),
    _fn("lab_task_card", "Task description and evaluation contract.", {}, []),
]

EVENT_TOOL = {"read_file": "read", "list_files": "read", "write_file": "edit", "edit_file": "edit",
              "bash": "bash"}


class MiniHarness(Harness):
    name = "mini"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.backend = self.cfg.get("sandbox", "bwrap")

    # ----------------------------------------------------------------- llm
    def chat(self, messages: list[dict], model: str, headers: dict) -> tuple[dict, dict, float]:
        url = self.cfg["gateway_url"].rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                "temperature": float(self.cfg.get("temperature", 0.7)),
                "max_tokens": int(self.cfg.get("max_output_tokens", 4096))}
        body.update(self.cfg.get("extra_body") or {})
        key = ""
        kf = self.cfg.get("api_key_file")
        if kf and os.path.exists(kf):
            key = pathlib.Path(kf).read_text().strip()
        h = {"Content-Type": "application/json", **headers}
        if key:
            h["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers=h)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=float(self.cfg.get("request_timeout_s", 600))) as r:
            resp = json.loads(r.read())
        return body, resp, time.time() - t0

    # ----------------------------------------------------------------- tools
    def _path(self, ws: pathlib.Path, p: str) -> pathlib.Path:
        p = p[len("/work/"):] if p.startswith("/work/") else p
        full = (ws / p).resolve()
        if ws.resolve() not in full.parents and full != ws.resolve():
            raise ValueError("path outside the workspace")
        return full

    def tool(self, name: str, args: dict, ws: pathlib.Path, lab: Lab, spec: SandboxSpec) -> tuple[str, bool]:
        try:
            if name == "read_file":
                return self._path(ws, args["path"]).read_text()[:60000], True
            if name == "write_file":
                fp = self._path(ws, args["path"])
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(args["content"])
                return f"wrote {len(args['content'])} chars to {args['path']}", True
            if name == "edit_file":
                fp = self._path(ws, args["path"])
                s = fp.read_text()
                n = s.count(args["old_string"])
                if n != 1:
                    return f"error: old_string found {n} times (must be exactly 1)", False
                fp.write_text(s.replace(args["old_string"], args["new_string"], 1))
                return "edited", True
            if name == "list_files":
                return "\n".join(sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file())[:500]), True
            if name == "bash":
                argv, env, cwd = wrap(spec, ["/bin/bash", "-c", args["command"]], self.backend)
                r = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
                out = (r.stdout + r.stderr)[-20000:]
                return f"exit={r.returncode}\n{out}", r.returncode == 0
            if name.startswith("lab_"):
                text, err = lab.call(name[4:], args)
                lab.log(name[4:], args, not err)
                return text, not err
            return f"error: unknown tool {name}", False
        except subprocess.TimeoutExpired:
            return "error: command timed out after 60 s", False
        except (KeyError, TypeError) as e:
            return f"error: bad arguments ({e})", False
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return f"error: {e}", False

    # ----------------------------------------------------------------- loop
    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s,
                    parent="orchestrator") -> dict:
        ws = pathlib.Path(workspace)
        lab = Lab(str(ws.parent / "lab"))
        spec = SandboxSpec(workspace=str(ws), lab=str(ws.parent / "lab"), home=None,
                           ro_paths=list(self.cfg.get("ro_paths", [])), share_net=False)
        system = SYSTEM
        if scaffold_dir and pathlib.Path(scaffold_dir).is_dir():
            notes = "\n\n".join(p.read_text() for p in sorted(pathlib.Path(scaffold_dir).glob("*.md")))
            if notes.strip():
                system += "\n\nNotes from previous generations:\n" + notes
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        traj_dir = pathlib.Path(self.cfg.get("trajectories_dir") or
                                os.path.join(os.path.dirname(events.events_path(run)), "trajectories", "harness"))
        traj_dir = traj_dir / agent.replace("/", "_")
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"trial-{trial:05d}.json"
        log: list[dict] = []
        headers = {"X-RSI-Run": run, "X-RSI-Agent": agent, "X-RSI-Gen": str(gen), "X-RSI-Trial": str(trial)}
        events.emit(run, "agent_spawn", agent, parent=parent, gen=gen, trial=trial, role="researcher",
                    model=model, harness="mini", sandbox=self.backend)
        t0 = time.time()
        n_tools, final, ok, reason = 0, "", True, "completed"
        max_turns = int(self.cfg.get("max_turns", 30))
        submitted = None
        for _turn in range(max_turns):
            if time.time() - t0 > budget_s:
                ok, reason = False, f"timeout after {budget_s}s"
                break
            try:
                body, resp, lat = self.chat(messages, model, headers)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                ok, reason = False, f"llm error: {e}"
                break
            log.append({"request": body, "response": resp, "latency_s": round(lat, 3)})
            if self.cfg.get("emit_llm_calls"):  # only when no logging gateway sits in between
                u = resp.get("usage") or {}
                events.emit(run, "llm_call", agent, gen=gen, trial=trial, model=model,
                            prompt_tokens=int(u.get("prompt_tokens") or 0),
                            completion_tokens=int(u.get("completion_tokens") or 0), latency_s=round(lat, 3))
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
            if msg.get("content"):
                final = str(msg["content"]).strip()
            if not calls:
                break
            for c in calls:
                fn = c.get("function") or {}
                name = fn.get("name", "")
                t1 = time.time()
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    text, tok = self.tool(name, args, ws, lab, spec)
                except (json.JSONDecodeError, ValueError) as e:
                    args, text, tok = {}, f"error: invalid JSON arguments ({e})", False
                n_tools += 1
                ev_tool = EVENT_TOOL.get(name, f"mcp:lab.{name[4:]}" if name.startswith("lab_") else name or "unknown")
                events.emit(run, "tool_call", agent, gen=gen, trial=trial, tool=ev_tool,
                            arg_summary=events.clip(" ".join(f"{k}={str(v)[:80]}" for k, v in args.items()), 200),
                            ok=tok, latency_s=round(time.time() - t1, 3))
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": text})
                if name == "lab_submit_eval" and tok:
                    submitted = {"hypothesis": str(args.get("hypothesis", ""))[:1000]}
        else:
            reason = f"max_turns {max_turns} reached"
        traj_path.write_text(json.dumps({"agent": agent, "trial": trial, "model": model, "messages": messages,
                                         "calls": log}, default=str))
        if final:
            events.emit(run, "message", agent, parent=parent, gen=gen, trial=trial, to=parent or "orchestrator",
                        summary=events.clip(final, 200), tokens=max(1, len(final) // 4))
        events.emit(run, "agent_end", agent, parent=parent, gen=gen, trial=trial, reason=reason,
                    n_tool_calls=n_tools, wall_s=round(time.time() - t0, 1))
        return {"final": final, "n_tool_calls": n_tools, "trajectory": str(traj_path), "ok": ok,
                "submitted": submitted, "wall_s": time.time() - t0}


def make_harness(cfg: dict) -> Harness:
    kind = cfg.get("kind", "opencode")
    if kind == "opencode":
        from rsi.harness.opencode import OpenCodeHarness

        return OpenCodeHarness(cfg)
    if kind == "mini":
        return MiniHarness(cfg)
    raise ValueError(f"unknown harness kind {kind!r}")
