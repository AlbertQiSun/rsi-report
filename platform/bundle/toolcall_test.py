#!/usr/bin/env python3
"""Gate G1b: tool-call validity of the improver model through the gateway (owner: worker C).

Runs ~30 scripted multi-turn episodes with OpenCode-like tools (bash, read, edit, web_search,
lab_submit_eval = the lab MCP's submit tool as OpenCode names MCP tools: <server>_<tool>).
Tool results are canned (no real execution). For every assistant turn it checks the emitted
tool calls against the JSON schemas and prints a JSON summary:

  call_validity_rate      schema-valid tool calls / emitted tool calls        (gate: >= 0.90)
  turn_validity_rate      turns with >=1 call, all valid / turns that needed a call
  episode_completion_rate expected tools hit in order within max turns; recovering after an invalid
                          call is allowed, as in OpenCode (gate: >= 0.70)
  clean_completion_rate   completed with no schema error at all
  schema_errors           {bad_json, not_object, missing_required, wrong_type, unknown_param,
                           unknown_tool, no_tool_call, tool_call_in_content, http_error, ...}
  latency_s / ttft_s      client-side, per request ({mean, p50, p90, max})

Usage (cluster, after serve_vllm.sh 9b 0 and the gateway are up):
  python bundle/toolcall_test.py --model improver --out $R/out/g1b.json
Local self-test (CPU; fake model server + in-process gateway, a few injected errors):
  python bundle/toolcall_test.py --fake --out /tmp/g1b_fake.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
PLATFORM = HERE.parent
sys.path.insert(0, str(PLATFORM))

from rsi.gateway.client import GatewayClient, GatewayError, gateway_url  # noqa: E402
from rsi.gateway.ports import default_token_path  # noqa: E402

# ------------------------------------------------------------------ tools (OpenCode-like)
def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


TOOLS = [
    _fn("bash", "Execute a shell command in the workspace and return stdout/stderr.",
        {"command": {"type": "string", "description": "The command to run"},
         "description": {"type": "string", "description": "5-10 word description of what the command does"},
         "timeout": {"type": "number", "description": "Timeout in milliseconds"}}, ["command"]),
    _fn("read", "Read a file from the workspace. Returns numbered lines.",
        {"filePath": {"type": "string", "description": "Absolute path of the file"},
         "offset": {"type": "number", "description": "Line to start from (0-based)"},
         "limit": {"type": "number", "description": "Number of lines to read"}}, ["filePath"]),
    _fn("edit", "Replace an exact string in a file. oldString must match the file exactly.",
        {"filePath": {"type": "string", "description": "Absolute path of the file"},
         "oldString": {"type": "string", "description": "Exact text to replace"},
         "newString": {"type": "string", "description": "Replacement text"},
         "replaceAll": {"type": "boolean", "description": "Replace every occurrence"}},
        ["filePath", "oldString", "newString"]),
    _fn("web_search", "Search the web (DuckDuckGo). Returns titles, urls and snippets.",
        {"query": {"type": "string", "description": "Search query"},
         "numResults": {"type": "integer", "description": "Number of results (default 5)"}}, ["query"]),
    _fn("lab_submit_eval", "Submit the current workspace version of the editable files to the sealed "
        "evaluator. Returns a job id; the score arrives later.",
        {"hypothesis": {"type": "string", "description": "One sentence: what you changed and why it should help"},
         "seeds": {"type": "integer", "description": "Number of seeds (default 1)"}}, ["hypothesis"]),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

SYSTEM = ("You are a careful coding agent working in the directory /workspace. You can only act through the "
          "provided tools. Call exactly the tools the user asks for, with arguments that match their schemas. "
          "When the task is finished, reply with a short plain-text summary and do not call any more tools.")

TRAIN_PY = """import math
import torch

LR = 0.02
BATCH_SIZE = 32
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
N_LAYER = 8
DROPOUT = 0.0


def lr_at(step, total):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    return LR * 0.5 * (1 + math.cos(math.pi * step / total))
"""
CONFIG_YAML = "model:\n  n_head: 6\n  d_model: 384\noptim:\n  beta2: 0.95\n  grad_clip: 1.0\n"
README = "# toy task\nMetric: val_bpb (lower is better). Edit only train.py.\n"
FILES = {"/workspace/train.py": TRAIN_PY, "/workspace/config.yaml": CONFIG_YAML, "/workspace/README.md": README}


def _numbered(text: str) -> str:
    return "\n".join(f"{i + 1:5d}\t{line}" for i, line in enumerate(text.splitlines()))


def S(tool, args, output=None):
    return {"tool": tool, "args": args, "output": output}


# ------------------------------------------------------------------ episodes (30)
def build_episodes() -> list[dict]:
    eps = []
    reads = [("/workspace/train.py", "LR"), ("/workspace/train.py", "BATCH_SIZE"),
             ("/workspace/config.yaml", "n_head"), ("/workspace/config.yaml", "grad_clip"),
             ("/workspace/README.md", "the metric name"), ("/workspace/train.py", "WARMUP_STEPS")]
    for p, var in reads:
        eps.append({"kind": "read", "prompt": f"Use the read tool to open {p} and tell me the value of {var}.",
                    "steps": [S("read", {"filePath": p})]})
    cmds = ["ls -la /workspace", "python -c 'import sys; print(sys.version)'", "nvidia-smi --query-gpu=name --format=csv",
            "wc -l /workspace/train.py", "grep -n LR /workspace/train.py", "du -sh /workspace"]
    outs = ["train.py\nconfig.yaml\nREADME.md", "3.12.3", "name\nNVIDIA A100-SXM4-80GB", "16 /workspace/train.py",
            "4:LR = 0.02", "24K\t/workspace"]
    for c, o in zip(cmds, outs):
        eps.append({"kind": "bash", "prompt": f"Use the bash tool to run `{c}` and report what it prints.",
                    "steps": [S("bash", {"command": c}, o)]})
    edits = [("LR = 0.02", "LR = 0.03"), ("BATCH_SIZE = 32", "BATCH_SIZE = 64"), ("WARMUP_STEPS = 100", "WARMUP_STEPS = 200"),
             ("WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.05"), ("N_LAYER = 8", "N_LAYER = 10"), ("DROPOUT = 0.0", "DROPOUT = 0.1")]
    for old, new in edits:
        eps.append({"kind": "read_edit", "prompt": f"Read /workspace/train.py with the read tool, then use the edit tool "
                    f"to change the line `{old}` to `{new}`.",
                    "steps": [S("read", {"filePath": "/workspace/train.py"}),
                              S("edit", {"filePath": "/workspace/train.py", "oldString": old, "newString": new})]})
    qs = ["muon optimizer learning rate schedule", "nanochat val_bpb baseline", "rotary embeddings base frequency",
          "cosine vs linear warmdown language model"]
    for q in qs:
        eps.append({"kind": "search", "prompt": f"Use web_search to search for \"{q}\" and summarize the top result "
                    f"in one sentence.", "steps": [S("web_search", {"query": q})]})
    subs = [("LR = 0.02", "LR = 0.025", "a slightly higher learning rate"),
            ("WARMUP_STEPS = 100", "WARMUP_STEPS = 50", "shorter warmup"),
            ("WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.2", "stronger weight decay"),
            ("N_LAYER = 8", "N_LAYER = 9", "one more layer")]
    for old, new, why in subs:
        eps.append({"kind": "full_trial", "prompt": f"Try {why}. Step 1: read /workspace/train.py. Step 2: use edit to "
                    f"replace `{old}` with `{new}`. Step 3: run `python -m py_compile /workspace/train.py` with bash. "
                    f"Step 4: submit with lab_submit_eval and a one-sentence hypothesis.",
                    "steps": [S("read", {"filePath": "/workspace/train.py"}),
                              S("edit", {"filePath": "/workspace/train.py", "oldString": old, "newString": new}),
                              S("bash", {"command": "python -m py_compile /workspace/train.py"}, ""),
                              S("lab_submit_eval", {"hypothesis": f"{why} should lower val_bpb"})]})
    research = [("best dropout for small language models", "DROPOUT = 0.0", "DROPOUT = 0.05"),
                ("AdamW beta2 0.95 vs 0.99 small transformers", "WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.08"),
                ("learning rate warmup length small GPT", "WARMUP_STEPS = 100", "WARMUP_STEPS = 150"),
                ("depth vs width tradeoff tiny transformer", "N_LAYER = 8", "N_LAYER = 12")]
    for q, old, new in research:
        eps.append({"kind": "research_trial", "prompt": f"First call web_search for \"{q}\". Then read "
                    f"/workspace/train.py, use edit to change `{old}` to `{new}`, and finally call lab_submit_eval "
                    f"with a one-sentence hypothesis that cites what you found.",
                    "steps": [S("web_search", {"query": q}), S("read", {"filePath": "/workspace/train.py"}),
                              S("edit", {"filePath": "/workspace/train.py", "oldString": old, "newString": new}),
                              S("lab_submit_eval", {"hypothesis": f"{new} per search results"})]})
    for i, e in enumerate(eps):
        e["id"] = f"ep-{i:02d}"
        e["prompt"] = f"[{e['id']}] " + e["prompt"]
    return eps


# ------------------------------------------------------------------ validation
def validate_call(tc: dict) -> list[str]:
    """Return a list of error types for one tool call (empty = valid)."""
    fn = tc.get("function") or {}
    name = fn.get("name") or ""
    if not name:
        return ["empty_name"]
    if name not in SCHEMAS:
        return ["unknown_tool"]
    raw = fn.get("arguments")
    if raw is None or raw == "":
        raw = "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return ["bad_json"]
    if not isinstance(args, dict):
        return ["not_object"]
    sch, errs = SCHEMAS[name], []
    for r in sch["required"]:
        if r not in args:
            errs.append("missing_required")
    for k, v in args.items():
        p = sch["properties"].get(k)
        if p is None:
            errs.append("unknown_param")
            continue
        t = p["type"]
        ok = {"string": isinstance(v, str), "boolean": isinstance(v, bool),
              "integer": isinstance(v, int) and not isinstance(v, bool),
              "number": isinstance(v, (int, float)) and not isinstance(v, bool),
              "array": isinstance(v, list), "object": isinstance(v, dict)}.get(t, True)
        if not ok:
            errs.append("wrong_type")
        elif t == "string" and k in ("command", "filePath", "query", "hypothesis", "oldString") and not v.strip():
            errs.append("empty_value")
    return sorted(set(errs))


def tool_output(step: dict | None, name: str, args: dict, files: dict) -> str:
    if step is not None and step.get("output") is not None:
        return step["output"]
    if name == "read":
        p = args.get("filePath", "")
        return _numbered(files[p]) if p in files else f"Error: file not found: {p}"
    if name == "edit":
        p, old = args.get("filePath", ""), args.get("oldString", "")
        if p not in files:
            return f"Error: file not found: {p}"
        if old not in files[p]:
            return "Error: oldString not found in file"
        files[p] = files[p].replace(old, args.get("newString", ""), 1)
        return "Edit applied successfully."
    if name == "bash":
        return "(no output)"
    if name == "web_search":
        q = args.get("query", "")
        return json.dumps([{"title": f"Result about {q}", "url": "https://example.org/a",
                            "snippet": f"A short note on {q}: values in the 0.01-0.1 range work for small models."}])
    if name == "lab_submit_eval":
        return json.dumps({"job_id": "job-0001", "status": "queued"})
    return "Error: unknown tool"


# ------------------------------------------------------------------ one episode
def run_episode(ep: dict, client: GatewayClient, a, run: str, stream: bool) -> dict:
    files = dict(FILES)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ep["prompt"]}]
    steps, k = ep["steps"], 0
    max_turns = len(steps) + a.extra_turns
    calls = []        # per tool call: {name, errors}
    turns = []        # per assistant turn: {needed_call, n_calls, all_valid, latency_s, ttft_s, errors}
    nudges, final_text, complete_ok = 0, None, True
    params = {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens}
    if a.top_k is not None:
        params["top_k"] = a.top_k
    if a.thinking != "default":
        params["chat_template_kwargs"] = {"enable_thinking": a.thinking == "on"}
    if a.tool_choice != "auto":
        params["tool_choice"] = a.tool_choice
    for turn in range(max_turns):
        needed = k < len(steps)
        try:
            r = client.chat(a.model, msgs, tools=TOOLS, run=run, agent=f"toolcall/{ep['id']}", gen=0, trial=None,
                            stream=stream, **params)
        except (GatewayError, OSError) as e:
            turns.append({"needed_call": needed, "n_calls": 0, "all_valid": False, "errors": ["http_error"],
                          "latency_s": None, "ttft_s": None, "detail": str(e)[:300]})
            complete_ok = False
            break
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []
        terr = []
        if any(m in content for m in ("<tool_call>", "<function=", "</tool_call>")):
            terr.append("tool_call_in_content")
        all_valid = True
        for tc in tcs:
            errs = validate_call(tc)
            calls.append({"name": (tc.get("function") or {}).get("name"), "errors": errs})
            if errs:
                all_valid = False
                terr.extend(errs)
        if needed and not tcs:
            terr.append("no_tool_call")
        turns.append({"needed_call": needed, "n_calls": len(tcs), "all_valid": bool(tcs) and all_valid,
                      "errors": terr, "latency_s": r.get("_latency_s"), "ttft_s": r.get("_ttft_s"),
                      "usage": r.get("usage"), "finish_reason": ch.get("finish_reason")})
        if not tcs:
            if k >= len(steps):
                final_text = content
                break
            nudges += 1
            if nudges > a.max_nudges:
                break
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": "Continue the task: call the next tool now."})
            continue
        msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
            {"id": tc.get("id") or f"call_{turn}_{i}", "type": "function", "function": tc.get("function")}
            for i, tc in enumerate(tcs)]})
        for i, tc in enumerate(tcs):
            fn = tc.get("function") or {}
            name, errs = fn.get("name") or "", validate_call(tc)
            if errs:
                out = f"Error: invalid tool call ({', '.join(errs)}). Check the tool schema and try again."
            else:
                args = json.loads(fn.get("arguments") or "{}")
                hit = k < len(steps) and name == steps[k]["tool"]
                out = tool_output(steps[k] if hit else None, name, args, files)
                if hit:
                    k += 1
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{turn}_{i}", "content": out})
    hit_all = k >= len(steps)
    return {"id": ep["id"], "kind": ep["kind"], "stream": stream, "complete": bool(hit_all and complete_ok),
            "steps_hit": k, "n_steps": len(steps), "n_turns": len(turns), "nudges": nudges,
            "final_text": (final_text or "")[:200], "calls": calls, "turns": turns,
            "edit_applied": files["/workspace/train.py"] != FILES["/workspace/train.py"]}


# ------------------------------------------------------------------ summary
def _dist(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 3), "p50": round(q(0.5), 3),
            "p90": round(q(0.9), 3), "max": round(xs[-1], 3)}


def summarize(results: list[dict], a, run: str, url: str, wall: float) -> dict:
    calls = [c for r in results for c in r["calls"]]
    turns = [t for r in results for t in r["turns"]]
    need = [t for t in turns if t["needed_call"]]
    errs: dict[str, int] = {}
    for t in turns:
        for e in t["errors"]:
            errs[e] = errs.get(e, 0) + 1
    by_tool: dict[str, dict] = {}
    for c in calls:
        d = by_tool.setdefault(c["name"] or "<empty>", {"calls": 0, "valid": 0})
        d["calls"] += 1
        d["valid"] += not c["errors"]
    by_mode = {}
    for mode in (False, True):
        cs = [c for r in results if r["stream"] == mode for c in r["calls"]]
        es = [r for r in results if r["stream"] == mode]
        if es:
            by_mode["stream" if mode else "nonstream"] = {
                "episodes": len(es), "calls": len(cs),
                "call_validity_rate": round(sum(not c["errors"] for c in cs) / len(cs), 4) if cs else None,
                "completion_rate": round(sum(r["complete"] for r in es) / len(es), 4)}
    by_kind = {}
    for r in results:
        d = by_kind.setdefault(r["kind"], {"episodes": 0, "complete": 0})
        d["episodes"] += 1
        d["complete"] += r["complete"]
    n_valid = sum(not c["errors"] for c in calls)
    cvr = round(n_valid / len(calls), 4) if calls else 0.0
    tvr = round(sum(t["all_valid"] for t in need) / len(need), 4) if need else 0.0
    comp = round(sum(r["complete"] for r in results) / len(results), 4) if results else 0.0
    usage = [t.get("usage") or {} for t in turns]
    return {
        "gate": "G1b", "pass": cvr >= a.min_validity and comp >= a.min_completion,
        "thresholds": {"call_validity_rate": a.min_validity, "episode_completion_rate": a.min_completion},
        "call_validity_rate": cvr, "turn_validity_rate": tvr, "episode_completion_rate": comp,
        "clean_completion_rate": round(sum(r["complete"] and not any(t["errors"] for t in r["turns"])
                                           for r in results) / len(results), 4) if results else 0.0,
        "n_episodes": len(results), "n_requests": len(turns), "n_tool_calls": len(calls), "n_valid_tool_calls": n_valid,
        "schema_errors": dict(sorted(errs.items(), key=lambda kv: -kv[1])),
        "by_tool": by_tool, "by_mode": by_mode, "by_kind": by_kind,
        "latency_s": _dist([t["latency_s"] for t in turns]), "ttft_s": _dist([t["ttft_s"] for t in turns]),
        "tokens": {"prompt": sum(int(u.get("prompt_tokens") or 0) for u in usage),
                   "completion": sum(int(u.get("completion_tokens") or 0) for u in usage)},
        "wall_s": round(wall, 1), "run": run, "gateway": url, "model": a.model,
        "settings": {"stream": a.stream, "tool_choice": a.tool_choice, "thinking": a.thinking,
                     "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "max_tokens": a.max_tokens,
                     "concurrency": a.concurrency, "extra_turns": a.extra_turns, "max_nudges": a.max_nudges},
        "episodes": [{k: v for k, v in r.items() if k not in ("calls", "turns")} |
                     {"errors": [e for t in r["turns"] for e in t["errors"]]} for r in results],
    }


# ------------------------------------------------------------------ fake mode (CPU self-test)
INJECT = {3: "bad_json", 8: "missing_required", 13: "no_tool_call", 19: "unknown_tool", 24: "wrong_type",
          27: "content_leak"}


def fake_script(eps: list[dict]) -> dict:
    """Rules that make the fake model follow each episode, with a few injected faults."""
    rules = []
    for i, ep in enumerate(eps):
        tag = f"[{ep['id']}]"
        seq = [{"tool_calls": [{"name": s["tool"], "arguments": s["args"]}]} for s in ep["steps"]]
        fault = INJECT.get(i)
        if fault == "bad_json":
            seq.insert(0, {"tool_calls": [{"name": ep["steps"][0]["tool"], "arguments": '{"filePath": "/workspace/'}]})
        elif fault == "missing_required":
            seq.insert(0, {"tool_calls": [{"name": ep["steps"][0]["tool"], "arguments": {}}]})
        elif fault == "unknown_tool":
            seq.insert(0, {"tool_calls": [{"name": "grep", "arguments": {"pattern": "LR"}}]})
        elif fault == "wrong_type":
            seq.insert(0, {"tool_calls": [{"name": "read", "arguments": {"filePath": "/workspace/train.py",
                                                                          "limit": "all"}}]})
        elif fault == "content_leak":  # parser failure: raw tool-call markup left in content, never completes
            seq = [{"content": '<tool_call>\n<function=read>\n<parameter=filePath>/workspace/train.py'
                               '</parameter>\n</function>\n</tool_call>'}] * 3
        if fault == "no_tool_call":
            rules.append({"match": {"any_contains": tag, "n_tool_msgs": 0}, "times": 1,
                          "respond": {"content": "I will now read the file."}})
        for j, resp in enumerate(seq):
            rules.append({"match": {"any_contains": tag, "n_tool_msgs": j}, "respond": resp})
        rules.append({"match": {"any_contains": tag, "min_tool_msgs": len(seq)},
                      "respond": {"content": f"Done with {ep['id']}."}})
    return {"models": ["improver"], "rules": rules, "default": {"content": "unscripted"}}


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1b tool-call gate")
    ap.add_argument("--gateway", default=None, help="gateway base URL (default: ports.json 'gateway')")
    ap.add_argument("--ports-file", default=None)
    ap.add_argument("--token-file", default=None, help="default $R/run/token")
    ap.add_argument("--model", default="improver")
    ap.add_argument("--run", default=None, help="run id for trajectories (default toolcall-g1b-<ts>)")
    ap.add_argument("--episodes", type=int, default=30, help="use the first N of the 30 episodes")
    ap.add_argument("--stream", choices=["mix", "on", "off"], default="mix")
    ap.add_argument("--tool-choice", default="auto", help="auto | required | none")
    ap.add_argument("--thinking", choices=["default", "on", "off"], default="default",
                    help="send chat_template_kwargs.enable_thinking (default: leave to server)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--extra-turns", type=int, default=3)
    ap.add_argument("--max-nudges", type=int, default=1)
    ap.add_argument("--min-validity", type=float, default=0.90)
    ap.add_argument("--min-completion", type=float, default=0.70)
    ap.add_argument("--out", default=None, help="write the JSON summary here too")
    ap.add_argument("--details", action="store_true", help="include per-turn details in the output")
    ap.add_argument("--fake", action="store_true", help="CPU self-test: fake model server + in-process gateway")
    ap.add_argument("--runs-root", default=None, help="--fake only: where the gateway writes runs")
    a = ap.parse_args(argv)

    eps = build_episodes()[: a.episodes]
    run = a.run or time.strftime("toolcall-g1b-%Y%m%dT%H%M%S")
    stack = []
    if a.fake:
        sys.path.insert(0, str(PLATFORM / "tests"))
        from fake_openai_server import FakeServer
        from rsi.gateway.server import GatewayConfig, GatewayThread
        fk = FakeServer(fake_script(eps)).start()
        rr = a.runs_root or tempfile.mkdtemp(prefix="g1b-fake-")
        gw = GatewayThread(GatewayConfig(runs_root=rr, upstreams={"improver": fk.base}, quiet=True)).__enter__()
        stack = [gw.__exit__, fk.stop]
        url, client = gw.url, GatewayClient(gw.url, token="")
        print(f"[fake] gateway {url} runs_root {rr}", file=sys.stderr)
    else:
        url = a.gateway or gateway_url(a.ports_file)
        client = GatewayClient(url, token_path=a.token_file or default_token_path())
    t0 = time.time()
    lock, done = threading.Lock(), [0]

    def one(i_ep):
        i, ep = i_ep
        stream = {"on": True, "off": False}.get(a.stream, i % 2 == 1)
        res = run_episode(ep, client, a, run, stream)
        with lock:
            done[0] += 1
            print(f"[{done[0]:2d}/{len(eps)}] {ep['id']} {ep['kind']:<14} stream={int(stream)} "
                  f"complete={int(res['complete'])} steps={res['steps_hit']}/{res['n_steps']} "
                  f"errors={[e for t in res['turns'] for e in t['errors']]}", file=sys.stderr)
        return res

    try:
        with ThreadPoolExecutor(max(1, a.concurrency)) as ex:
            results = list(ex.map(one, enumerate(eps)))
    finally:
        for f in stack:
            f()
    summary = summarize(results, a, run, url, time.time() - t0)
    if a.details:
        summary["details"] = results
    text = json.dumps(summary, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
