"""OpenCode harness adapter (D2 primary harness), headless, one episode = one `opencode run`.

Per episode it writes a fresh, read-only config dir mounted at /rsi inside the sandbox:

    /rsi/opencode.json   provider = the logging gateway (OpenAI-compatible), model, MCP servers
                         (lab + filtered DuckDuckGo search + filtered fetch), permissions
    /rsi/scaffold/       the improver's scaffold sigma (lessons.md, ...) -> `instructions`

and runs ``opencode run --format json --model rsi/<model> <prompt>`` in the thread's bwrap sandbox
(OPENCODE_CONFIG=/rsi/opencode.json). stdout is NDJSON; every line is saved verbatim to
``trajectories/harness/<agent>/trial-<k>.ndjson`` and parsed live into tool_call / message events.
Subagent spawns (OpenCode's ``task`` tool) become agent_spawn/agent_end for ``<agent>/sub-N``; after
the episode, ``opencode export <session>`` is saved for the main and child sessions, and the child
sessions' tool calls are emitted as tool_call events of the subagent (with their original times).

The event parser is deliberately tolerant: it walks any JSON object looking for OpenCode "parts"
(``{"type": "tool", "tool": ..., "state": {...}}``, ``{"type": "text", ...}``,
``{"type": "step-finish", "tokens": ...}``), so minor format changes between versions degrade to
fewer events instead of a crash.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Iterator

from rsi import events
from rsi.harness.base import Harness
from rsi.harness.sandbox import SandboxSpec, wrap

WORKSPACE_JUNK = ["opencode.json", "opencode.jsonc", ".opencode", "AGENTS.md", "CLAUDE.md"]


def tool_name(raw: str) -> str:
    """Map OpenCode tool ids onto EVENTS_SCHEMA tool vocabulary."""
    r = (raw or "").lower()
    if r in ("bash", "shell"):
        return "bash"
    if r in ("edit", "write", "patch", "multiedit", "apply_patch"):
        return "edit"
    if r in ("read", "list", "glob", "grep", "ls"):
        return "read"
    if r.startswith("lab_") or r.startswith("lab."):
        return f"mcp:lab.{r[4:]}"
    if "duckduckgo" in r or r.startswith("search_") or r.endswith("_search") or r == "websearch":
        return "web_search"
    if r.startswith("fetch") or r.endswith("_fetch") or r == "webfetch" or "fetch_content" in r:
        return "fetch"
    if r in ("task",):
        return "task"
    if "_" in r:  # other MCP tools are "<server>_<tool>"
        srv, _, t = r.partition("_")
        return f"mcp:{srv}.{t}"
    return r


def summarize_args(inp: Any, n: int = 200) -> str:
    if not isinstance(inp, dict):
        return events.clip(inp, n)
    parts = []
    for k in ("command", "filePath", "path", "pattern", "query", "url", "hypothesis", "description",
              "subagent_type", "oldString"):
        if k in inp:
            v = inp[k]
            if k == "oldString":
                v = f"<{len(str(v))} chars>"
            parts.append(f"{k}={v}")
    if not parts:
        parts = [f"{k}=<{len(str(v))} chars>" if len(str(v)) > 60 else f"{k}={v}" for k, v in inp.items()]
    return events.clip(" ".join(parts), n)


def iter_parts(obj: Any) -> Iterator[dict]:
    """Yield every OpenCode 'part'-like dict inside an arbitrary JSON value."""
    if isinstance(obj, dict):
        if obj.get("type") in ("tool", "text", "step-finish", "step_finish", "reasoning") and \
                ("tool" in obj or "text" in obj or "tokens" in obj):
            yield obj
            return
        for v in obj.values():
            yield from iter_parts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_parts(v)


def find_session_ids(obj: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("sessionID", "sessionId", "session_id") and isinstance(v, str):
                out.add(v)
            else:
                out |= find_session_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= find_session_ids(v)
    return out


class EpisodeParser:
    """Turns OpenCode JSON parts into schema events for one episode."""

    def __init__(self, run: str, agent: str, gen: int, trial: int):
        self.run, self.agent, self.gen, self.trial = run, agent, gen, trial
        self.seen_tools: set[str] = set()
        self.n_tool_calls = 0
        self.final = ""
        self.session_ids: set[str] = set()
        self.sub_sessions: dict[str, str] = {}   # child session id -> sub agent id
        self.n_sub = 0
        self.tokens = {"input": 0, "output": 0}
        self.error: str | None = None

    def emit(self, ev: str, agent: str, **kw: Any) -> None:
        kw.setdefault("gen", self.gen)
        kw.setdefault("trial", self.trial)
        events.emit(self.run, ev, agent, **kw)

    def handle_line(self, obj: dict, agent: str | None = None) -> None:
        agent = agent or self.agent
        if agent == self.agent:
            self.session_ids |= {s for s in find_session_ids(obj) if s not in self.sub_sessions}
        if obj.get("type") == "error" or "error" in obj and not obj.get("part"):
            err = obj.get("error")
            self.error = events.clip(json.dumps(err) if not isinstance(err, str) else err, 200)
        for part in iter_parts(obj):
            self.handle_part(part, agent)

    def handle_part(self, part: dict, agent: str, t: float | None = None) -> None:
        ptype = part.get("type")
        if ptype == "text":
            txt = part.get("text") or ""
            if txt.strip() and agent == self.agent:
                self.final = txt.strip()
            return
        if ptype in ("step-finish", "step_finish"):
            tok = part.get("tokens") or {}
            self.tokens["input"] += int(tok.get("input") or 0)
            self.tokens["output"] += int(tok.get("output") or 0)
            return
        if ptype != "tool":
            return
        state = part.get("state") or {}
        status = state.get("status")
        if status not in ("completed", "error"):
            # spawn subagents as soon as the task tool starts running
            if part.get("tool") == "task" and status == "running":
                self._spawn_sub(part, agent)
            return
        key = part.get("callID") or part.get("id") or json.dumps(part, sort_keys=True)[:200]
        if key in self.seen_tools:
            return
        self.seen_tools.add(key)
        times = state.get("time") or {}
        start, end = times.get("start"), times.get("end")
        latency = round((end - start) / 1000.0, 3) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else 0.0
        ev_t = (end / 1000.0) if t is None and isinstance(end, (int, float)) else t
        raw = part.get("tool") or ""
        ok = status == "completed"
        out = state.get("output")
        if ok and isinstance(out, str) and out.startswith("error:"):
            ok = False
        self.n_tool_calls += 1
        self.emit("tool_call", agent, parent=None if agent == self.agent else self.agent,
                  tool=tool_name(raw), arg_summary=summarize_args(state.get("input")), ok=ok,
                  latency_s=latency, raw_tool=raw, **({"t": ev_t} if ev_t else {}))
        if raw == "task":
            sub = self._spawn_sub(part, agent)
            summ = state.get("output") or state.get("title") or ""
            self.emit("message", sub, parent=agent, to=agent, summary=events.clip(summ, 200),
                      tokens=max(1, len(str(summ)) // 4))
            self.emit("agent_end", sub, parent=agent, reason="completed" if ok else "error")

    def _spawn_sub(self, part: dict, agent: str) -> str:
        state = part.get("state") or {}
        meta = state.get("metadata") or {}
        child = meta.get("sessionId") or meta.get("sessionID") or part.get("callID") or f"anon-{self.n_sub}"
        if child in self.sub_sessions:
            return self.sub_sessions[child]
        sub = f"{self.agent}/sub-{self.n_sub}"
        self.n_sub += 1
        self.sub_sessions[child] = sub
        inp = state.get("input") or {}
        self.emit("agent_spawn", sub, parent=agent, role="subagent", model=str(inp.get("subagent_type") or "general"),
                  session=child)
        desc = inp.get("prompt") or inp.get("description") or ""
        self.emit("message", agent, to=sub, summary=events.clip(desc, 200), tokens=max(1, len(str(desc)) // 4))
        return sub


class OpenCodeHarness(Harness):
    name = "opencode"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.backend = self.cfg.get("sandbox", "bwrap")
        self.bin = self.cfg.get("opencode_bin", "opencode")
        self.runs_root = self.cfg.get("runs_root")

    # ---------------------------------------------------------------- config
    def build_config(self, *, model: str, run: str, agent: str, gen: int, trial: int,
                     scaffold: bool, lab_dir_in_sandbox: str) -> dict:
        c = self.cfg
        py = c.get("python", "python3")
        lab_cmd = [py, "-m", "rsi.mcp_lab.server", "--lab", lab_dir_in_sandbox]
        mcp: dict[str, Any] = {"lab": {"type": "local", "command": lab_cmd, "enabled": True,
                                       "environment": {"PYTHONPATH": c.get("platform_dir", "")}}}
        web_log = f"{lab_dir_in_sandbox}/outbox/web.jsonl"
        proxy = [py, "-m", "rsi.mcp_lab.filter_proxy", "--log", web_log, "--"]
        if c.get("web", True):
            if c.get("ddg_cmd"):
                mcp["ddg"] = {"type": "local", "command": proxy + list(c["ddg_cmd"]), "enabled": True,
                              "environment": {"PYTHONPATH": c.get("platform_dir", "")}}
            if c.get("fetch_cmd"):
                mcp["fetch"] = {"type": "local", "command": proxy + list(c["fetch_cmd"]), "enabled": True,
                                "environment": {"PYTHONPATH": c.get("platform_dir", "")}}
        headers = {"X-RSI-Run": run, "X-RSI-Agent": agent, "X-RSI-Gen": str(gen), "X-RSI-Trial": str(trial)}
        model_entry = {"name": model, "tool_call": True, "limit": {
            "context": int(c.get("context_tokens", 65536)), "output": int(c.get("max_output_tokens", 8192))}}
        conf: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "provider": {"rsi": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "RSI gateway",
                "options": {"baseURL": c["gateway_url"].rstrip("/") + ("" if c["gateway_url"].rstrip("/").endswith("/v1") else "/v1"),
                            "apiKey": "{env:RSI_GATEWAY_KEY}", "headers": headers},
                "models": {model: model_entry}}},
            "model": f"rsi/{model}",
            "small_model": f"rsi/{model}",
            "mcp": mcp,
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
        }
        if scaffold:
            conf["instructions"] = ["/rsi/scaffold/*.md"]
        if not c.get("subagents", True):
            conf["permission"]["task"] = "deny"
        for k, v in (c.get("opencode_extra") or {}).items():
            conf[k] = v
        return conf

    # ---------------------------------------------------------------- episode
    def run_episode(self, *, run, agent, gen, trial, workspace, prompt, model, scaffold_dir, budget_s,
                    parent="orchestrator") -> dict:
        workspace = pathlib.Path(workspace)
        tdir = workspace.parent
        lab = tdir / "lab"
        home = tdir / "home"
        conf_dir = tdir / "episode_conf"
        home.mkdir(parents=True, exist_ok=True)
        for junk in WORKSPACE_JUNK:   # agent must not configure its own harness
            p = workspace / junk
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        if conf_dir.exists():
            shutil.rmtree(conf_dir)
        conf_dir.mkdir(parents=True)
        has_scaffold = bool(scaffold_dir and pathlib.Path(scaffold_dir).is_dir())
        if has_scaffold:
            shutil.copytree(scaffold_dir, conf_dir / "scaffold")
            agents_src = pathlib.Path(scaffold_dir) / "agents"
            if agents_src.is_dir():  # custom subagents defined by the scaffold
                dst = home / ".config" / "opencode" / "agent"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(agents_src, dst)
        none_mode = self.backend == "none"
        lab_in = str(lab) if none_mode else "/lab"
        conf = self.build_config(model=model, run=run, agent=agent, gen=gen, trial=trial,
                                 scaffold=has_scaffold, lab_dir_in_sandbox=lab_in)
        if none_mode and has_scaffold:
            conf["instructions"] = [str(conf_dir / "scaffold" / "*.md")]
        (conf_dir / "opencode.json").write_text(json.dumps(conf, indent=1))

        traj_dir = pathlib.Path(self.cfg.get("trajectories_dir") or
                                os.path.join(events.events_path(run).rsplit("/", 1)[0], "trajectories", "harness"))
        traj_dir = traj_dir / agent.replace("/", "_")
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj = traj_dir / f"trial-{trial:05d}.ndjson"

        key = ""
        kf = self.cfg.get("api_key_file")
        if kf and os.path.exists(kf):
            key = pathlib.Path(kf).read_text().strip()
        env = {"OPENCODE_CONFIG": str(conf_dir / "opencode.json") if none_mode else "/rsi/opencode.json",
               "RSI_GATEWAY_KEY": key or "none",
               "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
               "PATH": self.cfg.get("sandbox_path", "/usr/local/bin:/usr/bin:/bin")}
        if none_mode:
            env.pop("PATH")
        spec = SandboxSpec(workspace=str(workspace), lab=str(lab), home=str(home), config_dir=str(conf_dir),
                           ro_paths=list(self.cfg.get("ro_paths", [])), share_net=bool(self.cfg.get("share_net", True)),
                           env=env)
        cmd = [self.bin, "run", "--format", "json", "--model", f"rsi/{model}"]
        cmd += list(self.cfg.get("extra_args", []))
        cmd.append(prompt)
        argv, penv, cwd = wrap(spec, cmd, self.backend)

        P = EpisodeParser(run, agent, gen, trial)
        events.emit(run, "agent_spawn", agent, parent=parent, gen=gen, trial=trial, role="researcher",
                    model=model, harness="opencode", sandbox=self.backend)
        t0 = time.time()
        ok, reason = True, "completed"
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=penv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            events.emit(run, "agent_end", agent, parent=parent, gen=gen, trial=trial, reason=f"launch failed: {e}")
            return {"final": "", "n_tool_calls": 0, "trajectory": str(traj), "ok": False, "submitted": None,
                    "error": str(e)}
        timer = threading.Timer(budget_s, lambda: _killpg(proc))
        timer.start()
        stderr_lines: list[str] = []
        threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True).start()  # type: ignore[arg-type]
        with open(traj, "w") as tf:
            for line in proc.stdout:  # type: ignore[union-attr]
                tf.write(line)
                tf.flush()
                s = line.strip()
                if not s.startswith("{"):
                    continue
                try:
                    P.handle_line(json.loads(s))
                except (json.JSONDecodeError, events.SchemaError):
                    continue
        rc = proc.wait()
        timer.cancel()
        wall = time.time() - t0
        if wall >= budget_s - 0.5:
            ok, reason = False, f"timeout after {budget_s}s"
        elif rc != 0:
            ok, reason = False, f"exit code {rc}"
        if P.error and ok:
            reason = f"completed with error: {P.error}"
        (traj_dir / f"trial-{trial:05d}.stderr.txt").write_text("".join(stderr_lines)[-200000:])

        if self.cfg.get("export", True):
            self._export_sessions(P, spec, traj_dir, trial)

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
                    n_tool_calls=P.n_tool_calls, wall_s=round(wall, 1), tokens=P.tokens)
        return {"final": P.final, "n_tool_calls": P.n_tool_calls, "trajectory": str(traj), "ok": ok,
                "submitted": submitted, "wall_s": wall, "tokens": P.tokens, "web": web_hits,
                "sessions": sorted(P.session_ids), "n_subagents": P.n_sub}

    def _export_sessions(self, P: EpisodeParser, spec: SandboxSpec, traj_dir: pathlib.Path, trial: int) -> None:
        def export(sid: str) -> Any:
            argv, penv, cwd = wrap(spec, [self.bin, "export", sid], self.backend)
            try:
                r = subprocess.run(argv, cwd=cwd, env=penv, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired):
                return None
            txt = r.stdout
            i = txt.find("{")
            if i < 0:
                return None
            try:
                return json.loads(txt[i:])
            except json.JSONDecodeError:
                return None

        for sid in sorted(P.session_ids):
            data = export(sid)
            if data is None:
                continue
            (traj_dir / f"trial-{trial:05d}.session-{sid}.json").write_text(json.dumps(data))
            for child in find_child_sessions(data):
                if child not in P.sub_sessions:
                    P._spawn_sub({"state": {"metadata": {"sessionId": child}, "input": {}}}, P.agent)
        for child, sub in list(P.sub_sessions.items()):
            data = export(child)
            if data is None:
                continue
            (traj_dir / f"trial-{trial:05d}.sub-{child}.json").write_text(json.dumps(data))
            for part in iter_parts(data):
                try:
                    P.handle_part(part, sub)
                except events.SchemaError:
                    pass


def find_child_sessions(export: Any) -> set[str]:
    out = set()
    for part in iter_parts(export):
        if part.get("type") == "tool" and part.get("tool") == "task":
            meta = (part.get("state") or {}).get("metadata") or {}
            sid = meta.get("sessionId") or meta.get("sessionID")
            if sid:
                out.add(sid)
    return out


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(3)
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _web_log_summary(path: pathlib.Path) -> dict:
    n = blocked = 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            blocked += 1 if r.get("blocked") else 0
    return {"calls": n, "blocked": blocked}
