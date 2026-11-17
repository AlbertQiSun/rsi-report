"""U_sd: the improver designs its own weight update (SEAL-style self-edit of the *update recipe*).

Round k (k = 1, 2): the base improver (M0, via the logging gateway) reads a summary of its own trial pool D_0
(and, from round 2 on, its previous recipes plus how the resulting models did on the TRAIN tasks, paired
against M0 and M1 on the same seeds) and writes a recipe:

    {"include": ["accepted", "hindsight", "rejected"],   # which trials' calls become SFT targets
     "min_rel_gain": 0.0,          # accepted trials only: keep if (parent_dev - dev) / parent_dev >= this
     "task_families": ["modadd", "parity", "teacher", "cls2d"],
     "calls_per_trial": "all" | "last" | 1..8,           # last k calls of each kept trial
     "per_task_cap": 20..200,
     "lora": {"rank": 8|16|32, "lr": 2e-5..3e-4, "epochs": 1..4},
     "rationale": "..."}

The platform validates/clips it, counts the examples it selects (must be 40..3000, else the model is told
and retries), builds the SFT set (``build_recipe_sft``) and trains with the recipe's LoRA args. The control
(``random_placebo``) is the same number of calls drawn at random from the legal pool, trained with the same
LoRA args: it separates "the model's choice of data" from "more / different SFT data".
Held-out tasks never enter the prompt or the feedback (feedback uses train tasks only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import urllib.request
from collections import Counter, defaultdict

from rsi.update import leakcheck
from rsi.update.lessons import _endpoint, _token
from rsi.update.sft import _collect, _hindsight_trials, _split, _write
from rsi.update.trajectories import run_task, trials

LABELS = ("accepted", "hindsight", "rejected")
FAMILIES = ("modadd", "parity", "teacher", "cls2d")
DEFAULT = {"include": ["accepted"], "min_rel_gain": 0.0, "task_families": list(FAMILIES), "calls_per_trial": "all",
           "per_task_cap": 60, "lora": {"rank": 16, "lr": 1e-4, "epochs": 2}}
MIN_N, MAX_N = 40, 3000


def family(task: str | None) -> str:
    m = re.match(r"muar/(?:train|id|ood)-([a-z0-9]+)-\d+", task or "")
    return m.group(1) if m else (task or "")


# ---------------------------------------------------------------- pool
def pool_rows(run_dirs: list[str]) -> list[dict]:
    rows = []
    for rd in run_dirs:
        task = run_task(rd)
        tr = trials(rd)
        dev_of = {r["artifact_hash"]: r.get("dev") for r in tr.values()
                  if r.get("artifact_hash") and r.get("eval_status") == "ok" and r.get("dev") is not None}
        hs = _hindsight_trials(tr)
        for t, r in sorted(tr.items()):
            st = r.get("status")
            if st in ("baseline", "pending"):
                continue
            lab = ("accepted" if st in ("keep", "adopted") else
                   ("hindsight" if t in hs else "rejected") if st in ("discard", "rolled_back", "rejected")
                   and r.get("eval_status", "ok") == "ok" else "no_change" if st == "no_change" else "illegal")
            pdev = dev_of.get(r.get("parent_hash"))
            dev = r.get("dev") if r.get("eval_status") == "ok" else None
            gain = (pdev - dev) / pdev if (pdev and dev is not None and pdev > 0) else None
            rows.append({"run": os.path.basename(os.path.normpath(rd)), "task": task, "family": family(task),
                         "trial": t, "label": lab, "dev": dev, "parent_dev": pdev, "rel_gain": gain,
                         "hypothesis": (r.get("hypothesis") or "")[:160]})
    return rows


def _q(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    xs = sorted(xs)
    pick = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]  # noqa: E731
    return f"n={len(xs)} min={xs[0]:+.3f} p25={pick(.25):+.3f} median={pick(.5):+.3f} p75={pick(.75):+.3f} max={xs[-1]:+.3f}"


def summarize_pool(rows: list[dict], n_calls: dict) -> str:
    out = ["Trials per task family and outcome (a trial = one proposed edit, evaluated once):"]
    fam_lab = Counter((r["family"], r["label"]) for r in rows)
    out.append("family   | accepted | hindsight | rejected | no_change | illegal | gen. calls in accepted/hindsight/rejected")
    for f in FAMILIES:
        c = [fam_lab[(f, l)] for l in ("accepted", "hindsight", "rejected", "no_change", "illegal")]
        calls = "/".join(str(n_calls.get((f, l), 0)) for l in LABELS)
        out.append(f"{f:8s} | {c[0]:8d} | {c[1]:9d} | {c[2]:8d} | {c[3]:9d} | {c[4]:7d} | {calls}")
    out.append("")
    out.append("Relative improvement over the parent state ((parent_loss - loss) / parent_loss; >0 = better):")
    for lab in LABELS:
        out.append(f"  {lab:9s}: " + _q([r["rel_gain"] for r in rows if r["label"] == lab and r["rel_gain"] is not None]))
    out.append("")
    acc = sorted([r for r in rows if r["label"] == "accepted" and r["rel_gain"] is not None],
                 key=lambda r: -r["rel_gain"])
    out.append("Examples of accepted trials (largest gains first):")
    for r in acc[:10]:
        out.append(f"  [{r['family']}, gain {r['rel_gain']:+.3f}] {r['hypothesis']}")
    out.append("Examples of accepted trials with the smallest gains:")
    for r in acc[-5:]:
        out.append(f"  [{r['family']}, gain {r['rel_gain']:+.3f}] {r['hypothesis']}")
    rej = [r for r in rows if r["label"] == "rejected" and r["rel_gain"] is not None]
    rej = sorted(rej, key=lambda r: hashlib.sha256(f"{r['run']}{r['trial']}".encode()).hexdigest())[:6]
    out.append("Examples of rejected trials:")
    for r in rej:
        out.append(f"  [{r['family']}, gain {r['rel_gain']:+.3f}] {r['hypothesis']}")
    return "\n".join(out)


# ---------------------------------------------------------------- recipe
def validate_recipe(obj) -> tuple[dict, list[str]]:
    notes = []
    if not isinstance(obj, dict):
        raise ValueError("recipe must be a JSON object")
    r = {}
    inc = obj.get("include") or DEFAULT["include"]
    inc = [x for x in (inc if isinstance(inc, list) else [inc]) if x in LABELS]
    if not inc:
        raise ValueError(f"include must be a non-empty subset of {list(LABELS)}")
    r["include"] = sorted(set(inc), key=LABELS.index)
    try:
        g = obj.get("min_rel_gain")
        r["min_rel_gain"] = None if g is None else max(-1.0, min(1.0, float(g)))
    except (TypeError, ValueError):
        r["min_rel_gain"] = None
        notes.append("min_rel_gain unreadable -> none")
    fams = obj.get("task_families") or list(FAMILIES)
    fams = [f for f in (fams if isinstance(fams, list) else [fams]) if f in FAMILIES]
    if not fams:
        raise ValueError(f"task_families must be a non-empty subset of {list(FAMILIES)}")
    r["task_families"] = fams
    c = obj.get("calls_per_trial", "all")
    if c not in ("all", "last"):
        try:
            c = max(1, min(8, int(c)))
        except (TypeError, ValueError):
            c = "all"
            notes.append("calls_per_trial unreadable -> all")
    r["calls_per_trial"] = c
    try:
        r["per_task_cap"] = max(20, min(200, int(obj.get("per_task_cap", 60))))
    except (TypeError, ValueError):
        r["per_task_cap"] = 60
    lo = obj.get("lora") or {}
    rank = lo.get("rank", 16)
    r_rank = min((8, 16, 32), key=lambda x: abs(x - float(rank))) if isinstance(rank, (int, float)) else 16
    try:
        lr = max(2e-5, min(3e-4, float(lo.get("lr", 1e-4))))
    except (TypeError, ValueError):
        lr = 1e-4
    try:
        ep = max(1.0, min(4.0, float(lo.get("epochs", 2))))
    except (TypeError, ValueError):
        ep = 2.0
    r["lora"] = {"rank": r_rank, "alpha": 2 * r_rank, "lr": lr, "epochs": ep}
    r["rationale"] = str(obj.get("rationale", ""))[:2000]
    for k, v in (("rank", rank), ("lr", lo.get("lr")), ("epochs", lo.get("epochs"))):
        if v is not None and str(v) != str(r["lora"][k]):
            notes.append(f"lora.{k} {v} -> {r['lora'][k]}")
    return r, notes


def lora_args(recipe: dict) -> str:
    lo = recipe["lora"]
    return f"--rank {lo['rank']} --alpha {lo['alpha']} --lr {lo['lr']:.6g} --epochs {lo['epochs']:g}"


# ---------------------------------------------------------------- SFT
def _legal_examples(run_dirs: list[str], fp, report: dict) -> list[dict]:
    exs = _collect(run_dirs, "legal", fp, hindsight_ok=False, min_accepted=0, max_chars=60000,
                   max_calls_per_trial=None, report=report)
    gains, hs = {}, {}
    for r in pool_rows(run_dirs):
        gains[(r["run"], r["trial"])] = r["rel_gain"]
        hs[(r["run"], r["trial"])] = r["label"] == "hindsight"
    for ex in exs:
        k = (ex["meta"]["run"], ex["meta"]["trial"])
        if ex["meta"]["label"] == "rejected" and hs.get(k):
            ex["meta"]["label"] = "hindsight"
        ex["meta"]["rel_gain"] = gains.get(k)
        ex["meta"]["family"] = family(ex["meta"]["task"])
    return exs


def select(exs: list[dict], recipe: dict) -> list[dict]:
    keep = []
    inc = set(recipe["include"])
    for ex in exs:
        m = ex["meta"]
        lab = m["label"]
        if not (lab in inc or (lab == "hindsight" and "rejected" in inc)):
            continue
        if m["family"] not in recipe["task_families"]:
            continue
        if lab == "accepted" and recipe.get("min_rel_gain") is not None:
            if m.get("rel_gain") is None or m["rel_gain"] < recipe["min_rel_gain"]:
                continue
        keep.append(ex)
    c = recipe["calls_per_trial"]
    if c != "all":
        k = 1 if c == "last" else int(c)
        by: dict = defaultdict(list)
        for ex in keep:      # _collect preserves call order within a trial
            by[(ex["meta"]["run"], ex["meta"]["trial"])].append(ex)
        keep = [e for lst in by.values() for e in lst[-k:]]
    by_task: dict = defaultdict(list)
    for ex in keep:
        by_task[ex["meta"]["task"]].append(ex)
    out = []
    for t, lst in sorted(by_task.items()):
        out.extend(sorted(lst, key=lambda e: e["meta"]["hash"])[:recipe["per_task_cap"]])
    return out


def count_selected(run_dirs: list[str], recipe: dict) -> int:
    rep = {"excluded_runs": [], "n_calls_unmatched": 0, "n_dupes": 0, "n_leak_dropped": 0, "leak_hits": [],
           "n_too_long": 0, "capped": {}}
    return len(select(_legal_examples(run_dirs, leakcheck.heldout_fingerprints(None), rep), recipe))


def build_recipe_sft(run_dirs: list[str], recipe: dict, out: str, *, random_placebo: bool = False,
                     match_n: int | None = None, salt: str = "sdr") -> dict:
    fp = leakcheck.heldout_fingerprints(None)
    report = {"excluded_runs": [], "n_calls_unmatched": 0, "n_dupes": 0, "n_leak_dropped": 0, "leak_hits": [],
              "n_too_long": 0, "capped": {}}
    pool = _legal_examples(run_dirs, fp, report)
    if random_placebo:
        n = int(match_n or 0)
        exs = sorted(pool, key=lambda e: hashlib.sha256((salt + e["meta"]["hash"]).encode()).hexdigest())[:n]
        report["placebo_shortfall"] = max(0, n - len(exs))
    else:
        exs = select(pool, recipe)
    train, valid = _split(exs, 0.1)
    os.makedirs(out, exist_ok=True)
    sha_t = _write(os.path.join(out, "train.jsonl"), train)
    sha_v = _write(os.path.join(out, "valid.jsonl"), valid)
    rescan = 0
    for fn in ("train.jsonl", "valid.jsonl"):
        with open(os.path.join(out, fn)) as f:
            for line in f:
                if leakcheck.scan_text(leakcheck.example_text(json.loads(line)), fp):
                    rescan += 1
    if rescan:
        raise RuntimeError(f"held-out leak check failed on written SFT files ({rescan} examples)")
    by_label, by_task = Counter(), Counter()
    for ex in train + valid:
        by_label[ex["meta"]["label"]] += 1
        by_task[ex["meta"]["task"]] += 1
    man = {"mode": "random_placebo" if random_placebo else "recipe", "recipe": recipe, "n_train": len(train),
           "n_valid": len(valid), "n_pool": len(pool), "by_label": dict(by_label), "by_task": dict(sorted(by_task.items())),
           "runs": [os.path.basename(os.path.normpath(r)) for r in run_dirs],
           "sha256": {"train.jsonl": sha_t, "valid.jsonl": sha_v},
           "leak_check": {"ok": True, "rescan_hits": rescan, "n_dropped": report["n_leak_dropped"]},
           "report": {k: v for k, v in report.items() if k != "leak_hits"}, "ok": len(train) > 0}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    return man


# ---------------------------------------------------------------- design call
SYSTEM = ("You are an ML research agent. You are about to be fine-tuned (LoRA) on transcripts of your own past "
          "research attempts. You decide how that fine-tuning is done.")

TASK = """## What you are deciding
You are the improver model (Qwen3.5-9B) of a small automated-research loop. In each run you get a tiny
training task (a µAR task: a small transformer trained under a fixed step budget, metric = validation loss,
lower is better) and 12 trials; in each trial you edit the training config through tools and submit it; an
edit is ACCEPTED if it beats the current best, otherwise REJECTED.

Your past runs on 8 training tasks (4 families x 2 variants, 5 seeds each, 40 runs) are listed below. Each
trial left transcripts of your model calls. We will fine-tune YOU on a subset of those calls (input = the full
prompt at that moment, target = your response) with LoRA, and then use the fine-tuned model as the improver
on NEW µAR tasks you have not seen (some of the same families, some different kinds of tasks).
The goal: the fine-tuned model should improve new tasks FASTER (reach lower loss in fewer trials) than you do now.

Choose the recipe. Options:
- include: subset of ["accepted", "hindsight", "rejected"].
  accepted = trials that became the new best. hindsight = rejected trials that were the best proposal among
  those tried from the same parent state. rejected = all other legal-but-not-adopted trials.
- min_rel_gain: number or null. For accepted trials only: keep a trial only if its relative gain >= this.
- task_families: subset of ["modadd", "parity", "teacher", "cls2d"].
- calls_per_trial: "all", "last", or an integer 1..8 (the last k model calls of each kept trial).
- per_task_cap: integer 20..200, max calls kept per training task.
- lora: {"rank": 8|16|32, "lr": 2e-5..3e-4, "epochs": 1..4}. The training run is capped at 55 minutes.
The selected set must contain between 40 and 3000 calls.
For reference, the platform's fixed default recipe (already tried, gave a model NO faster than you) is:
include ["accepted"], min_rel_gain null, all families, calls "all", per_task_cap 60, rank 16, lr 1e-4, epochs 2
(438 calls).

## Your past trials
{pool}
{history}
## Answer
Think about what in these transcripts would actually make a better researcher on new tasks, then answer
with ONE JSON object and nothing after it:
{"include": [...], "min_rel_gain": ..., "task_families": [...], "calls_per_trial": ..., "per_task_cap": ...,
 "lora": {"rank": ..., "lr": ..., "epochs": ...}, "rationale": "<= 120 words"}"""


def chat(gateway_url: str, model: str, messages: list, *, run: str, gen: int, token_path: str | None,
         max_tokens: int = 8192, temperature: float = 0.6, timeout_s: float = 900) -> tuple[str, str]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": True}}
    headers = {"Content-Type": "application/json", "X-RSI-Run": run, "X-RSI-Agent": "update-designer",
               "X-RSI-Gen": str(gen)}
    tok = _token(token_path)
    if tok:
        headers["Authorization"] = "Bearer " + tok
    req = urllib.request.Request(_endpoint(gateway_url), data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read())
    ch = data["choices"][0]
    content = ch["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip(), ch.get("finish_reason") or ""


def parse_json(text: str):
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):        # last top-level object that parses
        depth = 0
        for i in range(s, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and ("include" in obj or "lora" in obj):
                    return obj
                break
    return None


def history_text(prev: list[dict]) -> str:
    if not prev:
        return ""
    out = ["", "## Your previous recipes and what they produced"]
    for i, p in enumerate(prev, 1):
        r = dict(p["recipe"])
        out.append(f"Round {i} recipe: {json.dumps({k: v for k, v in r.items() if k != 'rationale'})}")
        out.append(f"Round {i} rationale: {r.get('rationale', '')}")
        out.append(f"Selected {p.get('n_examples')} calls. Training: {p.get('train_note') or 'n/a'}")
        fb = p.get("feedback")
        if fb:
            out.append("Then the fine-tuned model ran as the improver on the 8 TRAINING tasks with 5 fresh seeds each "
                       "(same tasks/seeds also run by you without fine-tuning = M0, and by the default-recipe model = M1).")
            out.append("Score = normalized area under the improvement curve over 12 trials (0 = never improves, "
                       "1 = reaches the reference loss immediately; higher is better):")
            out.append("task | this model | M0 (you, no fine-tune) | M1 (default recipe)")
            f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
            for t in sorted(fb["per_task"]):
                row = fb["per_task"][t]
                out.append(f"{t} | {f3(row.get('this'))} | {f3(row.get('m0'))} | {f3(row.get('m1'))}")
            m = fb["mean"]
            out.append(f"mean | {f3(m.get('this'))} | {f3(m.get('m0'))} | {f3(m.get('m1'))}")
    out.append("Use this evidence to write a better recipe (you may keep, change or reverse any choice).")
    return "\n".join(out) + "\n"


def train_feedback(spec: dict) -> dict | None:
    """Per-task nAUC of the previous round's model vs M0 and M1 on the train tasks (same seeds)."""
    fbs = spec.get("feedback")
    if not fbs:
        return None
    from rsi.update.analysis import arm_summary, curves_from_runs

    raw = json.load(open(spec["sref_path"]))
    s_ref = {t: (v.get("s_ref") if isinstance(v, dict) else v) for t, v in raw.items()}
    s0 = {t: v["s0"] for t, v in raw.items() if isinstance(v, dict) and v.get("s0") is not None}
    res = {"per_task": defaultdict(dict), "mean": {}}
    for key in ("this", "m0", "m1"):
        dirs = fbs.get(key) or []
        if not dirs:
            continue
        c, s = curves_from_runs(dirs, 12)
        s0x = dict(s0)
        for t, v in s.items():
            s0x.setdefault(t, v)
        summ = arm_summary(c, s0x, s_ref, 12)
        for t, v in summ["per_task"].items():
            res["per_task"][t][key] = v
        res["mean"][key] = summ["mean"]
    res["per_task"] = dict(res["per_task"])
    return res


def design(spec: dict) -> dict:
    run_dirs = spec["run_dirs"]
    out_dir = spec["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    rows = pool_rows(run_dirs)
    rep = {"excluded_runs": [], "n_calls_unmatched": 0, "n_dupes": 0, "n_leak_dropped": 0, "leak_hits": [],
           "n_too_long": 0, "capped": {}}
    exs = _legal_examples(run_dirs, leakcheck.heldout_fingerprints(None), rep)
    n_calls = Counter((e["meta"]["family"], e["meta"]["label"]) for e in exs)
    prev = []
    for p in spec.get("previous") or []:
        rec = json.load(open(p["recipe_path"]))
        man = json.load(open(p["manifest_path"])) if p.get("manifest_path") and os.path.exists(p["manifest_path"]) else {}
        prev.append({"recipe": rec, "n_examples": man.get("n_train", 0) + man.get("n_valid", 0) if man else None,
                     "train_note": p.get("train_note"), "feedback": train_feedback({**spec, "feedback": p.get("feedback")})})
    prompt = TASK.replace("{pool}", summarize_pool(rows, n_calls)).replace("{history}", history_text(prev))
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    attempts = []
    recipe, n = None, None
    for attempt in range(4):
        text, finish = chat(spec["gateway_url"], spec.get("model", "improver"), msgs, run=spec["run"],
                            gen=int(spec.get("gen", 1)), token_path=spec.get("token_path"))
        att = {"attempt": attempt, "finish_reason": finish, "response": text}
        attempts.append(att)
        obj = parse_json(text)
        err = None
        if obj is None:
            err = "I could not find a valid JSON recipe object in your answer."
        else:
            try:
                recipe, notes = validate_recipe(obj)
                att["clipped"] = notes
                n = len(select(exs, recipe))
                att["n_selected"] = n
                if not (MIN_N <= n <= MAX_N):
                    err = f"That recipe selects {n} calls; it must select between {MIN_N} and {MAX_N}."
            except ValueError as e:
                err = f"Invalid recipe: {e}."
        if err is None:
            break
        att["error"] = err
        recipe = None
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": err + " Answer again with one corrected JSON object."}]
    json.dump({"messages": msgs, "attempts": attempts, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()},
              open(os.path.join(out_dir, "design_log.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "prompt.md"), "w") as f:
        f.write(prompt)
    if recipe is None:
        raise RuntimeError("no valid recipe after 4 attempts")
    recipe["n_selected"] = n
    recipe["round"] = int(spec.get("round", 1))
    json.dump(recipe, open(os.path.join(out_dir, "recipe.json"), "w"), indent=1, ensure_ascii=False)
    return {"recipe_path": os.path.join(out_dir, "recipe.json"), "n_selected": n, "attempts": len(attempts),
            "lora_args": lora_args(recipe), "include": recipe["include"], "out_dir": out_dir}
