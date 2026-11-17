"""U_w data: build messages-format SFT JSONL from gateway trajectories + the ledger.

    build_sft_data(run_dirs, accepted_only=True, placebo=False, out=DIR) -> manifest dict

One example per gateway call that belongs to a selected trial:
``{"messages": request.messages + [response message], "tools": [...], "meta": {...}}``.
Training (rsi.update.lora_train) puts loss on the final assistant message only, i.e.
"input = the full prompt at that moment, output = the model's proposal" (design doc §5.1).

Selection
  accepted (default)   trials whose ledger status is keep / adopted
  placebo              legal-but-rejected trials (eval ok, status discard / rolled_back), subsampled to
                       the size of the accepted set built from the same runs (design doc §1.5 control 3)
  accepted_only=False  accepted + legal-rejected
  hindsight fallback   if fewer than ``min_accepted`` accepted trials: for every (run, thread, parent state)
                       the best-scoring legal proposal is relabelled as a positive (design doc §5.1 退路)
Then: exact dedup, held-out leak scan (drop + report; written files are re-scanned and must be clean),
length cap, per-task cap, ~10% valid split by trial (calls of one trial never straddle the split).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

from rsi.update import leakcheck
from rsi.update.trajectories import label, load_calls, run_task, trials


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _hindsight_trials(tr: dict[int, dict]) -> set[int]:
    """Top-1 legal proposal per (thread, parent state)."""
    groups: dict = defaultdict(list)
    for t, r in tr.items():
        if r.get("eval_status") == "ok" and isinstance(r.get("dev"), (int, float)) and r.get("status") not in (
                "baseline", "pending", "no_change"):
            groups[(r.get("thread"), r.get("parent_hash"))].append((r["dev"], t))
    return {min(v)[1] for v in groups.values() if v}


def _collect(run_dirs, mode: str, fp, *, hindsight_ok: bool, min_accepted: int, max_chars: int,
             max_calls_per_trial: int | None, report: dict) -> list[dict]:
    """mode: accepted | rejected | legal."""
    exs: list[dict] = []
    n_acc_trials = 0
    per_run = []
    for rd in run_dirs:
        task = run_task(rd)
        run_id = os.path.basename(os.path.normpath(rd))
        if task in fp.names:
            report["excluded_runs"].append({"run": run_id, "reason": f"held-out task {task}"})
            continue
        tr = trials(rd)
        n_acc_trials += sum(1 for r in tr.values() if label(r) == "accepted")
        per_run.append((rd, run_id, task, tr))
    use_hindsight = (mode == "accepted" and hindsight_ok and n_acc_trials < min_accepted)
    report["n_accepted_trials"] = n_acc_trials
    report["hindsight"] = use_hindsight
    for rd, run_id, task, tr in per_run:
        hs = _hindsight_trials(tr) if use_hindsight else set()
        per_trial: dict = defaultdict(int)
        for c in load_calls(rd):
            if c.trial is None or c.trial not in tr:
                report["n_calls_unmatched"] += 1
                continue
            lab = label(tr[c.trial])
            if lab == "rejected" and c.trial in hs:
                lab = "hindsight"
            want = {"accepted": ("accepted", "hindsight"), "rejected": ("rejected",),
                    "legal": ("accepted", "rejected")}[mode]
            if lab not in want:
                continue
            if max_calls_per_trial and per_trial[c.trial] >= max_calls_per_trial:
                continue
            per_trial[c.trial] += 1
            ex = {"messages": list(c.messages) + [c.response]}
            if c.tools:
                ex["tools"] = c.tools
            ex["meta"] = {"run": run_id, "task": task or "", "trial": c.trial, "agent": c.agent, "gen": c.gen,
                          "label": lab, "source": c.source}
            exs.append(ex)
    # dedup (exact, on what the model is trained on)
    seen, out = set(), []
    for ex in exs:
        h = _hash({"m": ex["messages"], "t": ex.get("tools")})
        if h in seen:
            report["n_dupes"] += 1
            continue
        seen.add(h)
        ex["meta"]["hash"] = h[:16]
        text = leakcheck.example_text(ex)
        hits = leakcheck.scan_text(text, fp)
        if hits:
            report["n_leak_dropped"] += 1
            report["leak_hits"].extend(hits[:3])
            continue
        if len(text) > max_chars:
            report["n_too_long"] += 1
            continue
        out.append(ex)
    return out


def _cap(exs: list[dict], per_task_cap: int | None, report: dict) -> list[dict]:
    if not per_task_cap:
        return exs
    by_task: dict = defaultdict(list)
    for ex in exs:
        by_task[ex["meta"]["task"]].append(ex)
    out = []
    for t, lst in sorted(by_task.items()):
        lst = sorted(lst, key=lambda e: e["meta"]["hash"])
        if len(lst) > per_task_cap:
            report["capped"][t] = len(lst) - per_task_cap
        out.extend(lst[:per_task_cap])
    return out


def _split(exs: list[dict], valid_frac: float) -> tuple[list[dict], list[dict]]:
    groups: dict = defaultdict(list)
    for ex in exs:
        groups[(ex["meta"]["run"], ex["meta"]["trial"])].append(ex)
    keys = sorted(groups, key=lambda k: hashlib.sha256(repr(k).encode()).hexdigest())
    target = int(round(valid_frac * len(exs)))
    if len(exs) >= 2 and len(groups) >= 2:
        target = max(1, target)
    valid, train = [], []
    for k in keys:
        (valid if len(valid) < target else train).extend(groups[k])
    if not train and valid:   # never leave train empty
        train, valid = valid, []
    return train, valid


def _write(path: str, exs: list[dict]) -> str:
    h = hashlib.sha256()
    with open(path, "w") as f:
        for ex in exs:
            line = json.dumps(ex, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            h.update(line.encode())
    return h.hexdigest()


def build_sft_data(run_dirs: list[str], *, accepted_only: bool = True, placebo: bool = False, out: str,
                   per_task_cap: int | None = 60, valid_frac: float = 0.1, heldout: list[str] | None = None,
                   min_accepted: int = 40, hindsight: bool = True, max_chars: int = 60000,
                   max_calls_per_trial: int | None = None, match_n: int | None = None) -> dict:
    """Build ``out/train.jsonl``, ``out/valid.jsonl`` and ``out/manifest.json``; return the manifest."""
    fp = leakcheck.heldout_fingerprints(heldout)
    mode = "rejected" if placebo else ("accepted" if accepted_only else "legal")
    report = {"excluded_runs": [], "n_calls_unmatched": 0, "n_dupes": 0, "n_leak_dropped": 0, "leak_hits": [],
              "n_too_long": 0, "capped": {}}
    exs = _collect(run_dirs, mode, fp, hindsight_ok=hindsight, min_accepted=min_accepted, max_chars=max_chars,
                   max_calls_per_trial=max_calls_per_trial, report=report)
    exs = _cap(exs, per_task_cap, report)
    target_n = match_n
    if placebo and target_n is None:   # size-match the accepted set from the same runs
        acc_rep = {"excluded_runs": [], "n_calls_unmatched": 0, "n_dupes": 0, "n_leak_dropped": 0,
                   "leak_hits": [], "n_too_long": 0, "capped": {}}
        acc = _cap(_collect(run_dirs, "accepted", fp, hindsight_ok=hindsight, min_accepted=min_accepted,
                            max_chars=max_chars, max_calls_per_trial=max_calls_per_trial, report=acc_rep),
                   per_task_cap, acc_rep)
        target_n = len(acc)
    if target_n is not None:
        report["match_n"] = target_n
        report["placebo_shortfall"] = max(0, target_n - len(exs))
        exs = sorted(exs, key=lambda e: e["meta"]["hash"])[:target_n]
    train, valid = _split(exs, valid_frac)
    os.makedirs(out, exist_ok=True)
    sha_train = _write(os.path.join(out, "train.jsonl"), train)
    sha_valid = _write(os.path.join(out, "valid.jsonl"), valid)
    # re-scan what was actually written (belt and braces)
    rescan = 0
    for fn in ("train.jsonl", "valid.jsonl"):
        with open(os.path.join(out, fn)) as f:
            for line in f:
                if leakcheck.scan_text(leakcheck.example_text(json.loads(line)), fp):
                    rescan += 1
    if rescan:
        raise RuntimeError(f"held-out leak check failed on written SFT files ({rescan} examples)")
    by_label: dict = defaultdict(int)
    by_task: dict = defaultdict(int)
    for ex in train + valid:
        by_label[ex["meta"]["label"]] += 1
        by_task[ex["meta"]["task"]] += 1
    report["leak_hits"] = sorted(set(report["leak_hits"]))[:20]
    manifest = {
        "mode": "placebo_rejected" if placebo else mode, "n_train": len(train), "n_valid": len(valid),
        "by_label": dict(by_label), "by_task": dict(sorted(by_task.items())),
        "runs": [os.path.basename(os.path.normpath(r)) for r in run_dirs],
        "sha256": {"train.jsonl": sha_train, "valid.jsonl": sha_valid},
        "leak_check": {"ok": True, "heldout": fp.summary(), "n_dropped": report["n_leak_dropped"],
                       "rescan_hits": rescan},
        "params": {"per_task_cap": per_task_cap, "valid_frac": valid_frac, "min_accepted": min_accepted,
                   "hindsight": hindsight, "max_chars": max_chars, "max_calls_per_trial": max_calls_per_trial},
        "report": report, "ok": len(train) > 0,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (runs/<id>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--all-legal", action="store_true", help="accepted + legal-rejected")
    ap.add_argument("--per-task-cap", type=int, default=60)
    ap.add_argument("--min-accepted", type=int, default=40)
    ap.add_argument("--no-hindsight", action="store_true")
    ap.add_argument("--match-n", type=int)
    a = ap.parse_args(argv)
    m = build_sft_data(a.runs, accepted_only=not a.all_legal, placebo=a.placebo, out=a.out,
                       per_task_cap=a.per_task_cap, min_accepted=a.min_accepted, hindsight=not a.no_hindsight,
                       match_n=a.match_n)
    print(json.dumps({k: m[k] for k in ("mode", "n_train", "n_valid", "by_label", "leak_check", "ok")}, indent=1))
    return 0 if m["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
