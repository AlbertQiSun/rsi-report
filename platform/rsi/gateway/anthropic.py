"""Anthropic Messages API support for the gateway (``POST /v1/messages``, used by Claude Code).

vLLM (0.34+) serves ``/v1/messages`` natively, so the gateway only relays bytes. For recording it
converts each call to the OpenAI chat shape the rest of the platform already reads
(``rsi/update/trajectories.py``, ``sft.py``, the lessons writer):

    request   {"model", "messages": [system?, user, assistant(tool_calls, reasoning_content), tool, ...],
               "tools": [{"type": "function", "function": {name, description, parameters}}], ...}
    response  {"message": {"role": "assistant", "content", "reasoning_content", "tool_calls"},
               "finish_reason": stop | tool_calls | length, "id"}

and keeps the raw Anthropic response content in ``response_anthropic`` (small). The raw request is
not kept (it is the converted ``request`` plus cache-control markers and beta fields only).

Conversion rules (Anthropic -> OpenAI)
  system (str | text blocks)          -> one leading {"role": "system"} message (texts joined by a blank line)
  user text / tool_result blocks       -> {"role": "tool", "tool_call_id", "content"} per tool_result (in order),
                                          then one {"role": "user"} with the remaining text (system reminders etc.)
                                          tool_result.is_error -> "is_error": true on the tool message
  assistant text / thinking / tool_use -> {"role": "assistant", "content", "reasoning_content",
                                           "tool_calls": [{id, type: function, function: {name, arguments: json}}]}
  image / document blocks              -> "[image]" / "[document]" placeholders
  stop_reason end_turn|stop_sequence -> stop, tool_use -> tool_calls, max_tokens -> length
  usage  input_tokens (+ cache_read_input_tokens + cache_creation_input_tokens) -> prompt_tokens,
         output_tokens -> completion_tokens (original keys kept)
"""

from __future__ import annotations

import json
from typing import Any

STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
            "pause_turn": "stop", "refusal": "stop"}


def _block_text(b: Any) -> str:
    if isinstance(b, str):
        return b
    if not isinstance(b, dict):
        return str(b)
    t = b.get("type")
    if t == "text":
        return str(b.get("text") or "")
    if t == "image":
        return "[image]"
    if t == "document":
        return "[document]"
    if "text" in b:
        return str(b["text"])
    return ""


def _content_text(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x for x in (_block_text(b) for b in c) if x)
    return str(c)


def system_text(system: Any) -> str:
    return _content_text(system)


def assistant_from_blocks(blocks: Any) -> dict:
    msg: dict[str, Any] = {"role": "assistant"}
    if isinstance(blocks, str):
        msg["content"] = blocks
        return msg
    texts, thinking, calls = [], [], []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            texts.append(str(b.get("text") or ""))
        elif t in ("thinking", "redacted_thinking"):
            if b.get("thinking"):
                thinking.append(str(b["thinking"]))
        elif t in ("tool_use", "server_tool_use"):
            inp = b.get("input")
            calls.append({"id": b.get("id"), "type": "function",
                          "function": {"name": b.get("name") or "",
                                       "arguments": inp if isinstance(inp, str) else json.dumps(inp if inp is not None else {},
                                                                                                ensure_ascii=False)}})
    txt = "".join(texts)
    msg["content"] = txt if txt else None
    if thinking:
        msg["reasoning_content"] = "\n".join(thinking)
    if calls:
        msg["tool_calls"] = calls
    return msg


def messages_to_openai(body: dict) -> list[dict]:
    out: list[dict] = []
    sys_t = system_text(body.get("system"))
    if sys_t:
        out.append({"role": "system", "content": sys_t})
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role, c = m.get("role"), m.get("content")
        if role == "assistant":
            out.append(assistant_from_blocks(c))
            continue
        if isinstance(c, list):
            rest: list[str] = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tm = {"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": _content_text(b.get("content"))}
                    if b.get("is_error"):
                        tm["is_error"] = True
                    out.append(tm)
                else:
                    t = _block_text(b)
                    if t:
                        rest.append(t)
            if rest:
                out.append({"role": "user", "content": "\n".join(rest)})
        else:
            out.append({"role": role or "user", "content": _content_text(c)})
    return out


def tools_to_openai(tools: Any) -> list[dict] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        if "input_schema" not in t:      # server tools (web_search_2025...) have no schema
            out.append({"type": "function", "function": {"name": t["name"], "description": t.get("type", ""),
                                                          "parameters": {"type": "object", "properties": {}}}})
            continue
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description") or "",
                                                      "parameters": t.get("input_schema") or {}}})
    return out or None


def request_to_openai(body: dict) -> dict:
    """The converted request that goes into the trajectory record."""
    req: dict[str, Any] = {"model": body.get("model"), "messages": messages_to_openai(body)}
    tools = tools_to_openai(body.get("tools"))
    if tools:
        req["tools"] = tools
    for k in ("max_tokens", "temperature", "top_p", "top_k", "stream", "stop_sequences", "tool_choice", "thinking"):
        if k in body:
            req[k] = body[k]
    req["api"] = "anthropic"
    return req


DEFAULT_DROP = ("output_config", "context_management")


def sanitize_body(body: dict, drop=DEFAULT_DROP, thinking: str = "enabled") -> tuple[dict, list[str]]:
    """Make a Claude Code request digestible for vLLM's /v1/messages; return (body, changes).

    drop      top-level keys removed (Anthropic-only features with no vLLM meaning)
    thinking  what to do with ``thinking: {"type": "adaptive"}`` (sent by recent Claude Code for unknown
              models; not part of the older schema vLLM implements):
              enabled -> {"type": "enabled", "budget_tokens": max_tokens - 1};  drop -> remove;  pass -> keep
    """
    out, changes = dict(body), []
    for k in drop or ():
        if k in out:
            out.pop(k)
            changes.append(f"drop:{k}")
    th = out.get("thinking")
    if isinstance(th, dict) and th.get("type") not in ("enabled", "disabled") and thinking != "pass":
        if thinking == "drop":
            out.pop("thinking")
            changes.append("drop:thinking")
        else:
            mt = int(out.get("max_tokens") or 8192)
            out["thinking"] = {"type": "enabled", "budget_tokens": max(1, mt - 1)}
            changes.append(f"thinking:{th.get('type')}->enabled")
    return out, changes


def normalize_usage(u: dict | None) -> dict:
    u = dict(u or {})
    inp = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + \
        int(u.get("cache_creation_input_tokens") or 0)
    u.setdefault("prompt_tokens", inp)
    u.setdefault("completion_tokens", int(u.get("output_tokens") or 0))
    return u


def final_from_message(obj: dict) -> dict:
    """Non-stream /v1/messages response -> recorded final {message, finish_reason, id}."""
    return {"message": assistant_from_blocks(obj.get("content")),
            "finish_reason": STOP_MAP.get(obj.get("stop_reason"), obj.get("stop_reason")), "id": obj.get("id")}


class AnthropicStreamAssembler:
    """Rebuild the final message from Anthropic SSE events (message_start, content_block_*, message_delta)."""

    def __init__(self):
        self.id = None
        self.model = None
        self.blocks: dict[int, dict] = {}
        self.partial_json: dict[int, list[str]] = {}
        self.stop_reason = None
        self.usage: dict = {}
        self.error = None
        self.n_events = 0

    def feed(self, ev: dict) -> None:
        self.n_events += 1
        t = ev.get("type")
        if t == "message_start":
            m = ev.get("message") or {}
            self.id, self.model = m.get("id"), m.get("model")
            self.usage.update({k: v for k, v in (m.get("usage") or {}).items() if v is not None})
        elif t == "content_block_start":
            i = int(ev.get("index", 0))
            self.blocks[i] = dict(ev.get("content_block") or {})
        elif t == "content_block_delta":
            i = int(ev.get("index", 0))
            d = ev.get("delta") or {}
            b = self.blocks.setdefault(i, {"type": "text", "text": ""})
            dt = d.get("type")
            if dt == "text_delta":
                b["text"] = (b.get("text") or "") + str(d.get("text") or "")
            elif dt == "thinking_delta":
                b["thinking"] = (b.get("thinking") or "") + str(d.get("thinking") or "")
            elif dt == "signature_delta":
                b["signature"] = (b.get("signature") or "") + str(d.get("signature") or "")
            elif dt == "input_json_delta":
                self.partial_json.setdefault(i, []).append(str(d.get("partial_json") or ""))
        elif t == "content_block_stop":
            i = int(ev.get("index", 0))
            if i in self.partial_json:
                raw = "".join(self.partial_json.pop(i))
                try:
                    self.blocks.setdefault(i, {})["input"] = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    self.blocks.setdefault(i, {})["input"] = raw  # invalid JSON from the model: keep the string
        elif t == "message_delta":
            d = ev.get("delta") or {}
            if d.get("stop_reason"):
                self.stop_reason = d["stop_reason"]
            self.usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
        elif t == "error":
            self.error = ev.get("error") or ev

    def content(self) -> list[dict]:
        for i, parts in list(self.partial_json.items()):   # stream cut before content_block_stop
            self.blocks.setdefault(i, {})["input"] = "".join(parts)
        return [self.blocks[i] for i in sorted(self.blocks)]

    def result(self) -> dict:
        return {"message": assistant_from_blocks(self.content()),
                "finish_reason": STOP_MAP.get(self.stop_reason, self.stop_reason), "id": self.id}
