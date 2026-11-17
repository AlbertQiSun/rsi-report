"""Claude Code harness adapter: one episode = one headless ``claude -p`` run in the thread's bwrap sandbox.

Model traffic: Claude Code speaks the Anthropic Messages API. ``ANTHROPIC_BASE_URL`` points at the
``gateway_agent`` instance with the tagging prefix ``/rsi/t/<run>/<agent>/<gen>/<trial>`` (Claude Code can
only set a base URL), the gateway relays ``/v1/messages`` to vLLM's native Anthropic endpoint and records
``llm_call`` events and OpenAI-shaped trajectories (rsi/gateway/anthropic.py). The agent token is passed
in the environment only (``ANTHROPIC_AUTH_TOKEN``, or ``ANTHROPIC_API_KEY`` with ``auth_env``), never argv.

Per episode a fresh read-only config dir is mounted at /rsi:

    /rsi/mcp.json            lab MCP + DuckDuckGo + fetch, both web servers wrapped by rsi.mcp_lab.filter_proxy
    /rsi/system_append.md    the improver's scaffold (scaffold_dir/*.md, lessons.md first) -> --append-system-prompt-file
    /rsi/settings.json       permissions.deny WebSearch/WebFetch, no co-author lines -> --settings

Claude Code's own state (CLAUDE_CONFIG_DIR) is ``/home/agent/.claude`` inside the sandbox HOME
(``threads/<t>/home`` on the host) and is wiped at the start of every episode, so nothing (auto memory,
transcripts) carries over between episodes except through the scaffold. Scaffold subagents
(``scaffold_dir/agents/*.md``) are copied to ``$CLAUDE_CONFIG_DIR/agents``.

Output: ``--output-format stream-json --verbose``; every stdout line is saved verbatim to
``trajectories/harness/<agent>/trial-<k>.ndjson`` and parsed live into events:
  assistant tool_use + matching user tool_result -> tool_call (Bash->bash, Edit/Write->edit, Read/Glob/Grep->read,
                                                    mcp__lab__x -> mcp:lab.x, ddg search -> web_search, fetch -> fetch)
  Agent / Task tool_use                          -> agent_spawn ``<agent>/sub-N`` (+ message), its result -> message
                                                    + agent_end; messages with that parent_tool_use_id are the
                                                    subagent's tool calls
  result                                         -> final text, num_turns, usage, is_error
After the run the session transcripts in ``$CLAUDE_CONFIG_DIR/projects`` (main + subagents) are copied to
``trial-<k>.sessions/``.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from typing import Any

from rsi import events
from rsi.harness.base import Harness
from rsi.harness.opencode import _killpg, _web_log_summary
from rsi.harness.sandbox import SandboxSpec, wrap

WORKSPACE_JUNK = ["CLAUDE.md", "CLAUDE.local.md", ".claude", ".mcp.json", "AGENTS.md", "opencode.json",
                  "opencode.jsonc", ".opencode"]
SUBAGENT_TOOLS = ("Agent", "Task")
DEFAULT_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
INVALID_PAT = re.compile(r"InputValidationError|Invalid tool parameters|No such tool available|"
                         r"tool_use_error|Error: No such tool|is not a valid|required parameter", re.I)


def tool_name(raw: str) -> str:
    """Map Claude Code tool names onto the EVENTS_SCHEMA tool vocabulary."""
    r = raw or ""
    low = r.lower()
    if low in ("bash", "bashoutput", "killshell", "killbash"):
        return "bash"
    if low in ("edit", "write", "multiedit", "notebookedit"):
        return "edit"
    if low in ("read", "glob", "grep", "ls"):
        return "read"
    if r in SUBAGENT_TOOLS:
        return "task"
    if low == "websearch":
        return "web_search"
    if low == "webfetch":
        return "fetch"
    if low.startswith("mcp__"):
        _, srv, t = (low.split("__", 2) + ["", ""])[:3]
        if srv == "lab":
            return f"mcp:lab.{t}"
        if srv in ("ddg", "duckduckgo") or "search" in t:
            return "fetch" if "fetch" in t else "web_search"
        if srv == "fetch" or "fetch" in t:
            return "fetch"
        return f"mcp:{srv}.{t}"
    return low


def summarize_args(inp: Any, n: int = 200) -> str:
    if not isinstance(inp, dict):
        return events.clip(inp, n)
    parts = []
    for k in ("command", "file_path", "path", "pattern", "query", "url", "hypothesis", "description",
              "subagent_type", "old_string"):
        if k in inp:
            v = inp[k]
            if k == "old_string":
                v = f"<{len(str(v))} chars>"
            parts.append(f"{k}={v}")
    if not parts:
        parts = [f"{k}=<{len(str(v))} chars>" if len(str(v)) > 60 else f"{k}={v}" for k, v in inp.items()]
    return events.clip(" ".join(parts), n)


def _result_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
    return "" if c is None else str(c)


class StreamParser:
    """Turns Claude Code stream-json messages into schema events for one episode."""

    def __init__(self, run: str, agent: str, gen: int, trial: int):
        self.run, self.agent, self.gen, self.trial = run, agent, gen, trial
        self.pending: dict[str, dict] = {}       # tool_use id -> {name, input, t, agent}
        self.subs: dict[str, str] = {}           # Agent tool_use id -> sub agent id
        self.open_subs: set[str] = set()
        self.n_sub = 0
        self.n_tool_calls = 0
        self.n_tool_errors = 0
        self.n_invalid = 0
        self.tools_used: dict[str, int] = {}
        self.final = ""
        self.session_id: str | None = None
        self.init: dict = {}
        self.result: dict = {}
        self.tokens = {"input": 0, "output": 0}
        self.n_retries = 0
        self.error: str | None = None
        self._usage_seen: set[str] = set()

    def emit(self, ev: str, agent: str, **kw: Any) -> None:
        kw.setdefault("gen", self.gen)
        kw.setdefault("trial", self.trial)
        events.emit(self.run, ev, agent, **kw)

    def _agent_of(self, obj: dict) -> str:
        p = obj.get("parent_tool_use_id")
        return self.subs.get(p, self.agent) if p else self.agent

    def handle(self, obj: dict) -> None:
        t = obj.get("type")
        if t == "system":
            st = obj.get("subtype")
            if st == "init":
                self.session_id = obj.get("session_id") or self.session_id
                self.init = {"model": obj.get("model"), "tools": obj.get("tools"),
                             "mcp_servers": obj.get("mcp_servers"), "mcp_server_errors": obj.get("mcp_server_errors"),
                             "permissionMode": obj.get("permissionMode"), "version": obj.get("claude_code_version")}
            elif st == "api_retry":
                self.n_retries += 1
            return
        if t == "assistant":
            agent = self._agent_of(obj)
            msg = obj.get("message") or {}
            u = msg.get("usage") or {}
            mid = msg.get("id") or ""
            if agent == self.agent and u and (not mid or mid not in self._usage_seen):   # one line per block, same id
                self._usage_seen.add(mid)
                self.tokens["input"] += int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                self.tokens["output"] += int(u.get("output_tokens") or 0)
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and agent == self.agent and str(b.get("text") or "").strip():
                    self.final = str(b["text"]).strip()
                elif b.get("type") == "tool_use":
                    tid = b.get("id") or f"anon-{len(self.pending)}"
                    self.pending[tid] = {"name": b.get("name") or "", "input": b.get("input"), "t": time.time(),
                                         "agent": agent}
                    if b.get("name") in SUBAGENT_TOOLS:
                        self._spawn(tid, b.get("input") or {}, agent)
            return
        if t == "user":
            msg = obj.get("message") or {}
            content = msg.get("content")
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    self._tool_done(b.get("tool_use_id"), bool(b.get("is_error")), _result_text(b.get("content")))
            return
        if t == "result":
            self.result = {k: obj.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "usage",
                                                   "total_cost_usd", "terminal_reason", "permission_denials")}
            self.session_id = obj.get("session_id") or self.session_id
            if isinstance(obj.get("result"), str) and obj["result"].strip():
                self.final = obj["result"].strip()
            if obj.get("is_error"):
                self.error = events.clip(str(obj.get("result") or obj.get("subtype") or "error"), 200)

    def _spawn(self, tid: str, inp: dict, parent_agent: str) -> str:
        if tid in self.subs:
            return self.subs[tid]
        sub = f"{self.agent}/sub-{self.n_sub}"
        self.n_sub += 1
        self.subs[tid] = sub
        self.open_subs.add(sub)
        self.emit("agent_spawn", sub, parent=parent_agent, role="subagent",
                  model=str(inp.get("subagent_type") or "general-purpose"), tool_use_id=tid)
        desc = inp.get("prompt") or inp.get("description") or ""
        self.emit("message", parent_agent, to=sub, summary=events.clip(desc, 200), tokens=max(1, len(str(desc)) // 4))
        return sub

    def _tool_done(self, tid: str | None, is_error: bool, text: str) -> None:
        p = self.pending.pop(tid or "", None)
        if p is None:
            return
        agent, raw = p["agent"], p["name"]
        self.n_tool_calls += 1
        self.tools_used[raw] = self.tools_used.get(raw, 0) + 1
        if is_error:
            self.n_tool_errors += 1
            if INVALID_PAT.search(text or ""):
                self.n_invalid += 1
        self.emit("tool_call", agent, parent=None if agent == self.agent else self.agent, tool=tool_name(raw),
                  arg_summary=summarize_args(p["input"]), ok=not is_error,
                  latency_s=round(time.time() - p["t"], 3), raw_tool=raw)
        if raw in SUBAGENT_TOOLS and tid in self.subs:
            sub = self.subs[tid]
            self.emit("message", sub, parent=agent, to=agent, summary=events.clip(text, 200),
                      tokens=max(1, len(text) // 4))
            self.emit("agent_end", sub, parent=agent, reason="error" if is_error else "completed")
            self.open_subs.discard(sub)

    def close(self) -> None:
        """Tool calls without a result (killed run) count as failed; open subagents end."""
        for tid in list(self.pending):
            self._tool_done(tid, True, "no tool_result (run ended)")
        for sub in sorted(self.open_subs):
            self.emit("agent_end", sub, parent=self.agent, reason="unfinished")
        self.open_subs.clear()


class ClaudeCodeHarness(Harness):
    name = "claude_code"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.backend = self.cfg.get("sandbox", "bwrap")
        self.bin = self.cfg.get("claude_bin", "claude")

    # ---------------------------------------------------------------- config files
    def mcp_config(self, lab_dir_in_sandbox: str) -> dict:
        c = self.cfg
        py = c.get("python", "python3")
        env = {"PYTHONPATH": c.get("platform_dir", "")}
        servers: dict[str, Any] = {"lab": {"type": "stdio", "command": py,
                                           "args": ["-m", "rsi.mcp_lab.server", "--lab", lab_dir_in_sandbox], "env": env}}
        proxy = ["-m", "rsi.mcp_lab.filter_proxy", "--log", f"{lab_dir_in_sandbox}/outbox/web.jsonl", "--"]
        if c.get("web", True):
            if c.get("ddg_cmd"):
                servers["ddg"] = {"type": "stdio", "command": py, "args": proxy + list(c["ddg_cmd"]), "env": env}
            if c.get("fetch_cmd"):
                servers["fetch"] = {"type": "stdio", "command": py, "args": proxy + list(c["fetch_cmd"]), "env": env}
        return {"mcpServers": servers}

    def settings(self) -> dict:
        s: dict[str, Any] = {"permissions": {"deny": list(self.disallowed())}, "includeCoAuthoredBy": False,
                             "cleanupPeriodDays": 3650}
        s.update(self.cfg.get("settings_extra") or {})
        return s

    def disallowed(self) -> list[str]:
        d = list(self.cfg.get("disallowed_tools") or ["WebSearch", "WebFetch"])
        if not self.cfg.get("subagents", True):
            d += [t for t in SUBAGENT_TOOLS if t not in d]
        return d

    def base_url(self, run: str, agent: str, gen: int, trial: int) -> str:
        g = self.cfg["gateway_url"].rstrip("/")
        g = g[:-3] if g.endswith("/v1") else g
        q = lambda s: urllib.parse.quote(str(s), safe="")  # noqa: E731
        return f"{g}/rsi/t/{q(run)}/{q(agent)}/{int(gen)}/{int(trial)}"

    def env(self, *, run, agent, gen, trial, model, home_in: str, key: str) -> dict:
        c = self.cfg
        ctx = int(c.get("context_tokens", 122880))
        out_tok = int(c.get("max_output_tokens", 8192))
        e = {
            "ANTHROPIC_BASE_URL": self.base_url(run, agent, gen, trial),
            c.get("auth_env", "ANTHROPIC_AUTH_TOKEN"): key or "none",
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,          # deprecated name, still read by older builds
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CONFIG_DIR": f"{home_in}/.claude",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(out_tok),
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1",
            "DISABLE_UPDATES": "1", "DISABLE_GROWTHBOOK": "1", "DISABLE_FEEDBACK_COMMAND": "1",
            "DISABLE_INSTALLATION_CHECKS": "1", "DISABLE_COST_WARNINGS": "1",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1", "ENABLE_TOOL_SEARCH": "false",
            "CLAUDE_CODE_FORWARD_SUBAGENT_TEXT": "1",
            "MCP_TIMEOUT": str(int(c.get("mcp_timeout_ms", 60000))),
            "MCP_CONNECTION_NONBLOCKING": "0",
            "BASH_DEFAULT_TIMEOUT_MS": str(int(c.get("bash_timeout_ms", 300000))),
            "BASH_MAX_TIMEOUT_MS": str(int(c.get("bash_max_timeout_ms", 600000))),
            "IS_SANDBOX": "1", "SHELL": "/bin/bash",
            "PATH": c.get("sandbox_path", "/usr/local/bin:/usr/bin:/bin"),
        }
        if c.get("max_thinking_tokens") is not None:
            e["MAX_THINKING_TOKENS"] = str(int(c["max_thinking_tokens"]))
        for k, v in (c.get("env_extra") or {}).items():
            e[str(k)] = str(v)
        return e

    def argv(self, *, model: str, conf_in: str, has_append: bool, prompt: str) -> list[str]:
        c = self.cfg
        a = [self.bin, "-p", "--output-format", "stream-json", "--verbose",
             "--max-turns", str(int(c.get("max_turns", 40))), "--model", model,
             "--permission-mode", c.get("permission_mode", "bypassPermissions"),
             "--mcp-config", f"{conf_in}/mcp.json", "--strict-mcp-config",
             "--settings", f"{conf_in}/settings.json", "--setting-sources", "user"]
        tools = c.get("tools", DEFAULT_TOOLS + ("," + SUBAGENT_TOOLS[0] if c.get("subagents", True) else ""))
        if tools:
            a += ["--tools", tools]
        a += ["--disallowedTools", ",".join(self.disallowed())]
        if has_append:
            a += ["--append-system-prompt-file", f"{conf_in}/system_append.md"]
        if c.get("bare"):
            a.append("--bare")
        a += list(c.get("extra_args", []))
        a += ["--", prompt]
        return a

    # ---------------------------------------------------------------- episode
    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s,
                    parent="orchestrator") -> dict:
        workspace = pathlib.Path(workspace)
        tdir = workspace.parent
        lab = tdir / "lab"
        home = tdir / "home"
        conf_dir = tdir / "episode_conf"
        home.mkdir(parents=True, exist_ok=True)
        (lab / "outbox").mkdir(parents=True, exist_ok=True)
        for junk in WORKSPACE_JUNK:   # the agent must not configure its own harness
            p = workspace / junk
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists() or p.is_symlink():
                p.unlink()
        cc_home = home / ".claude"     # fresh Claude Code state every episode (no auto memory / history carry-over)
        if cc_home.exists():
            shutil.rmtree(cc_home, ignore_errors=True)
        cc_home.mkdir(parents=True)
        for f in (home / ".claude.json",):
            if f.exists():
                f.unlink()
        if conf_dir.exists():
            shutil.rmtree(conf_dir)
        conf_dir.mkdir(parents=True)

        none_mode = self.backend == "none"
        lab_in = str(lab) if none_mode else "/lab"
        conf_in = str(conf_dir) if none_mode else "/rsi"
        home_in = str(home) if none_mode else "/home/agent"

        has_append = False
        if scaffold_dir and pathlib.Path(scaffold_dir).is_dir():
            sd = pathlib.Path(scaffold_dir)
            shutil.copytree(sd, conf_dir / "scaffold")
            mds = sorted(sd.glob("*.md"), key=lambda p: (p.name != "lessons.md", p.name))
            text = "\n\n".join(p.read_text(errors="replace").strip() for p in mds if p.read_text(errors="replace").strip())
            if text:
                (conf_dir / "system_append.md").write_text(
                    "# Notes from previous generations (scaffold)\n\n" + text + "\n")
                has_append = True
            if (sd / "agents").is_dir():
                shutil.copytree(sd / "agents", cc_home / "agents")
        (conf_dir / "mcp.json").write_text(json.dumps(self.mcp_config(lab_in), indent=1))
        (conf_dir / "settings.json").write_text(json.dumps(self.settings(), indent=1))

        traj_dir = pathlib.Path(self.cfg.get("trajectories_dir") or
                                os.path.join(os.path.dirname(events.events_path(run)), "trajectories", "harness"))
        traj_dir = traj_dir / agent.replace("/", "_")
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj = traj_dir / f"trial-{trial:05d}.ndjson"

        key = ""
        kf = self.cfg.get("api_key_file")
        if kf and os.path.exists(kf):
            key = pathlib.Path(kf).read_text().strip()
        env = self.env(run=run, agent=agent, gen=gen, trial=trial, model=model, home_in=home_in, key=key)
        if none_mode and "sandbox_path" not in self.cfg:
            env.pop("PATH")
        spec = SandboxSpec(workspace=str(workspace), lab=str(lab), home=str(home), config_dir=str(conf_dir),
                           ro_paths=list(self.cfg.get("ro_paths", [])), share_net=bool(self.cfg.get("share_net", True)),
                           env=env)
        cmd = self.argv(model=model, conf_in=conf_in, has_append=has_append, prompt=prompt)
        argv, penv, cwd = wrap(spec, cmd, self.backend)

        P = StreamParser(run, agent, gen, trial)
        events.emit(run, "agent_spawn", agent, parent=parent, gen=gen, trial=trial, role="researcher",
                    model=model, harness="claude_code", sandbox=self.backend)
        t0 = time.time()
        ok, reason = True, "completed"
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=penv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            events.emit(run, "agent_end", agent, parent=parent, gen=gen, trial=trial, reason=f"launch failed: {e}")
            return {"final": "", "n_tool_calls": 0, "trajectory": str(traj), "ok": False, "submitted": None,
                    "error": str(e)}
        killed = threading.Event()

        def _kill():
            killed.set()
            _killpg(proc)

        timer = threading.Timer(budget_s, _kill)
        timer.start()
        stderr_lines: list[str] = []
        th = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)  # type: ignore[arg-type]
        th.start()
        with open(traj, "w") as tf:
            for line in proc.stdout:  # type: ignore[union-attr]
                tf.write(line)
                tf.flush()
                s = line.strip()
                if not s.startswith("{"):
                    continue
                try:
                    P.handle(json.loads(s))
                except (json.JSONDecodeError, events.SchemaError):
                    continue
        rc = proc.wait()
        timer.cancel()
        th.join(timeout=5)
        P.close()
        wall = time.time() - t0
        subtype = (P.result or {}).get("subtype")
        if killed.is_set():
            ok, reason = False, f"timeout after {budget_s}s"
        elif subtype == "error_max_turns":
            reason = f"max_turns {self.cfg.get('max_turns', 40)} reached"
        elif rc != 0:
            ok, reason = False, f"exit code {rc}" + (f": {P.error}" if P.error else "")
        elif P.error:
            reason = f"completed with error: {P.error}"
        (traj_dir / f"trial-{trial:05d}.stderr.txt").write_text("".join(stderr_lines)[-200000:])
        sessions = self._save_sessions(cc_home, traj_dir, trial)

        submitted = None
        sp = lab / "outbox" / "submit.json"
        if sp.exists():
            try:
                submitted = json.loads(sp.read_text())
            except json.JSONDecodeError:
                submitted = None
        web_hits = _web_log_summary(lab / "outbox" / "web.jsonl")
        if P.final:
            events.emit(run, "message", agent, parent=parent, gen=gen, trial=trial, to=parent or "orchestrator",
                        summary=events.clip(P.final, 200), tokens=max(1, len(P.final) // 4))
        events.emit(run, "agent_end", agent, parent=parent, gen=gen, trial=trial, reason=reason,
                    n_tool_calls=P.n_tool_calls, wall_s=round(wall, 1), tokens=P.tokens,
                    num_turns=(P.result or {}).get("num_turns"))
        return {"final": P.final, "n_tool_calls": P.n_tool_calls, "trajectory": str(traj), "ok": ok,
                "submitted": submitted, "wall_s": wall, "tokens": P.tokens, "web": web_hits,
                "sessions": sessions, "n_subagents": P.n_sub, "n_tool_errors": P.n_tool_errors,
                "n_invalid_calls": P.n_invalid, "tools_used": P.tools_used, "result": P.result,
                "init": P.init, "n_api_retries": P.n_retries, "reason": reason}

    def _save_sessions(self, cc_home: pathlib.Path, traj_dir: pathlib.Path, trial: int) -> list[str]:
        """Copy Claude Code's session transcripts (main + subagents) next to the stream trajectory."""
        src = cc_home / "projects"
        if not src.is_dir():
            return []
        dst = traj_dir / f"trial-{trial:05d}.sessions"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        out = []
        for f in sorted(src.rglob("*.jsonl")):
            rel = f.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)
            out.append(str(rel))
        return out
