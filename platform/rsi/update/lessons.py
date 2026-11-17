"""U_s: the improver writes <=300-word lessons for its next generation (design doc architecture B).

    write_lessons(run_dirs, model=..., gateway_url=..., out_dir=..., generic_placebo=False) -> scaffold hash

Real arm: the model reads a compact summary of D_k (per run: task type, baseline, adopted changes with
their config diffs and gains, rejected and illegal attempts with feedback) and writes lessons.
Placebo arm (generic_placebo=True): the same model writes generic tuning advice of the same length
WITHOUT seeing any trajectory (design doc §1.5 control 3). Both arms produce an identically titled
``lessons.md`` so the next-generation improver cannot tell them apart (the manifest that says which
arm it is goes NEXT TO the scaffold dir, ``<out_dir>.manifest.json``, never inside it).

The call goes through the logging gateway (agent "lessons-writer"), so it lands in the trajectories
and emits llm_call events like every other model call. The token is read from a file and never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request

from rsi.runs import artifact_hash
from rsi.update import leakcheck
from rsi.update.trajectories import label, run_task, trials

TITLE = "# Notes for the researcher\n\n"
MAX_SUMMARY_CHARS = 12000

SYSTEM = ("You are an ML research agent. You are writing a short memo that will be given to the next "
          "generation of yourself before it starts new hyperparameter-search tasks.")

REAL_PROMPT = """Below is a summary of your own previous research runs: small MLP training tasks where each
attempt edits a JSON config (optimizer, lr, schedule, warmup_frac, weight_decay, batch_size, width,
depth, activation, norm, init_scale, dropout, grad_clip) under a fixed compute budget
(steps = B / (batch_size * n_params)), scored by validation loss (lower is better).

{summary}

Write lessons for your future self: what kinds of changes helped, which hurt or were illegal, how to
read the feedback (train curve, train/val gap, steps), and a good order in which to try things.
Rules: at most {words} words; bullet points; lessons must transfer to NEW tasks, so do not name
specific tasks and do not copy whole configs. Output only the memo."""

PLACEBO_PROMPT = """Write a memo of general advice on tuning the training configuration of small MLPs:
a JSON config with optimizer, lr, schedule, warmup_frac, weight_decay, batch_size, width, depth,
activation, norm, init_scale, dropout, grad_clip, trained under a fixed compute budget
(steps = B / (batch_size * n_params)) and scored by validation loss.
Rules: about {words} words (never more than {max_words}); bullet points; general advice only.
Output only the memo."""


# ---------------------------------------------------------------------------- summary of D_k
def _diff_lines(run_dir: str, ahash: str | None, n: int = 8) -> str:
    if not ahash:
        return ""
    p = os.path.join(run_dir, "artifacts", f"{ahash}.diff")
    try:
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    except OSError:
        return ""
    return "; ".join(lines[:n])


def summarize_runs(run_dirs: list[str], heldout: list[str] | None = None) -> str:
    fp = leakcheck.heldout_fingerprints(heldout)
    blocks = []
    for rd in run_dirs:
        task = run_task(rd) or "?"
        if task in fp.names:
            raise ValueError(f"refusing to write lessons from a held-out task run: {rd} ({task})")
        ttype = task.split("/", 1)[-1].rsplit("-", 1)[0].split("-", 1)[-1] if task.startswith("muar/") else task
        tr = sorted(trials(rd).values(), key=lambda r: r["trial"])
        base = next((r.get("dev") for r in tr if r.get("status") == "baseline"), None)
        acc = [r for r in tr if label(r) == "accepted"]
        rej = [r for r in tr if label(r) == "rejected"]
        ill = [r for r in tr if label(r) == "illegal"]
        best = min([r["dev"] for r in acc if isinstance(r.get("dev"), (int, float))] or [base or float("nan")])
        lines = [f"## run on a '{ttype}' task: baseline {base}, best {best}, "
                 f"{len(acc)} adopted / {len(rej)} rejected / {len(ill)} failed attempts"]
        prev = base
        for r in acc:
            gain = (prev - r["dev"]) if isinstance(prev, (int, float)) and isinstance(r.get("dev"), (int, float)) else None
            lines.append(f"+ ADOPTED {r.get('dev')} (gain {gain:.4g}) : {(r.get('hypothesis') or '')[:160]}"
                         f" | {_diff_lines(rd, r.get('artifact_hash'))}" if gain is not None else
                         f"+ ADOPTED {r.get('dev')} : {(r.get('hypothesis') or '')[:160]}")
            prev = r.get("dev", prev)
        for r in rej[:6]:
            lines.append(f"- rejected {r.get('dev')} : {(r.get('hypothesis') or '')[:140]}"
                         f" | {_diff_lines(rd, r.get('artifact_hash'), 4)}")
        for r in ill[:4]:
            lines.append(f"x failed ({r.get('eval_status') or r.get('status')}): {(r.get('hypothesis') or '')[:100]}"
                         f" | feedback: {(r.get('feedback') or r.get('reason') or '')[:160]}")
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n[... truncated]"
    return text


# ---------------------------------------------------------------------------- gateway call
def _token(token_path: str | None) -> str | None:
    p = token_path or os.environ.get("RSI_TOKEN_PATH")
    if not p:
        try:
            from rsi.gateway.ports import default_token_path   # worker C: $RSI_TOKEN_FILE or $R/run/token
            p = default_token_path()
        except ImportError:
            p = os.path.join(os.environ.get("R") or os.environ.get("RSI_ROOT", ""), "run", "token")
    try:
        with open(p) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _endpoint(gateway_url: str) -> str:
    u = gateway_url.rstrip("/")
    return u + "/chat/completions" if u.endswith("/v1") else u + "/v1/chat/completions"


def chat(gateway_url: str, model: str, messages: list, *, run: str, gen: int, token_path: str | None = None,
         max_tokens: int = 1024, temperature: float = 0.3, timeout_s: float = 300) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json", "X-RSI-Run": run, "X-RSI-Agent": "lessons-writer",
               "X-RSI-Gen": str(gen)}
    tok = _token(token_path)
    if tok:
        headers["Authorization"] = "Bearer " + tok
    req = urllib.request.Request(_endpoint(gateway_url), data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def clip_words(text: str, max_words: int) -> str:
    out, n = [], 0
    for line in text.splitlines():
        w = line.split()
        if n + len(w) > max_words:
            if max_words - n > 0:
                out.append(" ".join(w[: max_words - n]))
            break
        out.append(line)
        n += len(w)
    return "\n".join(out).strip()


def scaffold_hash(d: str) -> str:
    files = {}
    for base, _, fns in os.walk(d):
        for fn in fns:
            fp = os.path.join(base, fn)
            with open(fp, "rb") as f:
                files[os.path.relpath(fp, d)] = f.read()
    return artifact_hash(files)


def write_lessons(run_dirs: list[str], *, model: str, gateway_url: str, out_dir: str, generic_placebo: bool = False,
                  max_words: int = 300, target_words: int | None = None, base_scaffold: str | None = None,
                  run: str | None = None, gen: int = 0, token_path: str | None = None,
                  heldout: list[str] | None = None) -> str:
    """Write ``out_dir/lessons.md`` (plus copies of ``base_scaffold``) and return the scaffold hash."""
    run = run or "lessons-" + "-".join(os.path.basename(os.path.normpath(r)) for r in run_dirs[:1])
    if generic_placebo:
        words = min(max_words, target_words or max_words)
        prompt = PLACEBO_PROMPT.format(words=words, max_words=max_words)
        summary_sha = None
    else:
        summary = summarize_runs(run_dirs, heldout)
        prompt = REAL_PROMPT.format(summary=summary, words=max_words)
        summary_sha = artifact_hash({"summary": summary.encode()})
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = chat(gateway_url, model, msgs, run=run, gen=gen, token_path=token_path)
    if len(text.split()) > max_words:   # one polite retry, then hard clip
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Too long. Rewrite it in at most {max_words} words."}]
        text = chat(gateway_url, model, msgs, run=run, gen=gen, token_path=token_path)
    text = clip_words(text, max_words)
    hits = leakcheck.scan_text(text, leakcheck.heldout_fingerprints(heldout))
    if hits:
        raise RuntimeError(f"lessons mention held-out material: {hits}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    if base_scaffold and os.path.isdir(base_scaffold):
        shutil.copytree(base_scaffold, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lessons.md"), "w") as f:
        f.write(TITLE + text + "\n")
    h = scaffold_hash(out_dir)
    manifest = {"scaffold_hash": h, "kind": "generic_placebo" if generic_placebo else "lessons",
                "source_runs": [] if generic_placebo else [os.path.basename(os.path.normpath(r)) for r in run_dirs],
                "model": model, "words": len(text.split()), "summary_sha": summary_sha,
                "base_scaffold": base_scaffold}
    with open(os.path.normpath(out_dir) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--model", default="improver")
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generic-placebo", action="store_true")
    ap.add_argument("--target-words", type=int)
    ap.add_argument("--base-scaffold")
    ap.add_argument("--gen", type=int, default=0)
    a = ap.parse_args(argv)
    print(write_lessons(a.runs, model=a.model, gateway_url=a.gateway_url, out_dir=a.out_dir,
                        generic_placebo=a.generic_placebo, target_words=a.target_words,
                        base_scaffold=a.base_scaffold, gen=a.gen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
