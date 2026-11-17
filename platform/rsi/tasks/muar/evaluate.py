"""µAR evaluator entry point (called by evald, the local CPU runner, baselines and analysis).

Python:
    evaluate_artifact(task, artifact_dir, seeds=[0, 1], split="dev") -> result dict
    evaluate_config(task, cfg_dict_or_json_text, seeds, split)      -> result dict
CLI (one JSON object on stdout, or --out FILE):
    python -m rsi.tasks.muar.evaluate --task muar/train-modadd-0 --artifact DIR [--seeds 0,1]
        [--split dev|sealed] [--out result.json]

The result carries the eval_result event fields (metric, value, seeds, wall_s, status, audit) plus
``diag`` (visible diagnostics) and ``split``. ``value`` is the mean val loss over seeds when every seed
is ok, else None. Sealed results (split="sealed") must only go to the analysis script / sealed.jsonl;
``MuarTask.visible_feedback`` refuses to render them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rsi.runs import artifact_hash
from rsi.tasks.muar import schema
from rsi.tasks.muar.data import make_dataset
from rsi.tasks.muar.tasks import MAX_STEPS, budget_for, get_spec, train_seed
from rsi.tasks.muar.train import train_and_score

METRIC = "val_loss"
CONFIG_FILE = "config.json"
MAX_CONFIG_BYTES = 4096
HARD_TIMEOUT_S = float(os.environ.get("RSI_MUAR_TIMEOUT_S", "60"))


def run_seed(task: str, cfg: dict, seed_idx: int, split: str = "dev") -> dict:
    """One training run (already-validated cfg). Picklable entry for process pools."""
    spec = get_spec(task)
    ds = make_dataset(spec.ttype, spec.params, spec.data_seed)
    r = train_and_score(cfg, ds, seed=train_seed(spec, split, seed_idx), split=split,
                        budget=budget_for(spec), max_steps=MAX_STEPS, hard_timeout_s=HARD_TIMEOUT_S)
    r["seed"] = seed_idx
    return r


def _illegal(task, seeds, split, flag, msg, ahash, t0) -> dict:
    return {"task": task, "artifact_hash": ahash, "metric": METRIC, "value": None, "seeds": list(seeds),
            "wall_s": round(time.time() - t0, 3), "status": "audit_fail", "split": split,
            "audit": {"level": "L0", "flags": [flag]}, "diag": {"error": msg}}


def combine(task: str, seeds: list[int], split: str, per_seed: list[dict], ahash: str | None, t0: float) -> dict:
    """Aggregate per-seed results into one eval_result-shaped dict."""
    statuses = [r["status"] for r in per_seed]
    flags = sorted({r["flag"] for r in per_seed if r.get("flag")})
    if all(s == "ok" for s in statuses):
        status = "ok"
        value = round(sum(r["value"] for r in per_seed) / len(per_seed), 6)
    else:
        status = next(s for s in ("audit_fail", "nan", "timeout", "crash") if s in statuses)
        value = None
    oks = [r for r in per_seed if r["status"] == "ok"]
    first = per_seed[0]
    curve = None
    if oks and all(len(r.get("train_curve") or []) == 5 for r in oks):
        curve = [round(sum(r["train_curve"][i] for r in oks) / len(oks), 6) for i in range(5)]
    diag = {
        "steps": first.get("steps"), "params": first.get("params"), "uncapped_steps": first.get("uncapped_steps"),
        "train_curve": curve,
        "train_loss": round(sum(r["train_loss"] for r in oks) / len(oks), 6) if oks else None,
        "gap": round(sum(r["gap"] for r in oks) / len(oks), 6) if oks else None,
        "per_seed": [r.get("value") for r in per_seed],
        "error": next((r.get("error") for r in per_seed if r.get("error")), None),
    }
    return {"task": task, "artifact_hash": ahash, "metric": METRIC, "value": value, "seeds": list(seeds),
            "wall_s": round(time.time() - t0, 3), "status": status, "split": split,
            "audit": {"level": "L0", "flags": flags}, "diag": diag}


def evaluate_config(task: str, cfg, seeds=(0, 1), split: str = "dev", *, ahash: str | None = None,
                    pool=None) -> dict:
    """Validate and score one config. ``cfg`` may be a dict or JSON text. ``pool`` = optional executor."""
    t0 = time.time()
    seeds = [int(s) for s in seeds]
    if split not in ("dev", "sealed"):
        raise ValueError("split must be dev or sealed")
    get_spec(task)
    try:
        if isinstance(cfg, (str, bytes)):
            if len(cfg) > MAX_CONFIG_BYTES:
                raise schema.IllegalConfig("schema:too_large", f"config.json larger than {MAX_CONFIG_BYTES} bytes")
            cfg = schema.parse_config_text(cfg if isinstance(cfg, str) else cfg.decode())
        else:
            cfg = schema.validate_config(cfg)
    except schema.IllegalConfig as e:
        return _illegal(task, seeds, split, e.flag, str(e), ahash, t0)
    if pool is not None:
        futs = [pool.submit(run_seed, task, cfg, s, split) for s in seeds]
        per_seed = [f.result() for f in futs]
    else:
        per_seed = [run_seed(task, cfg, s, split) for s in seeds]
    return combine(task, seeds, split, per_seed, ahash, t0)


def read_artifact(artifact_dir: str) -> tuple[bytes | None, str]:
    p = os.path.join(artifact_dir, CONFIG_FILE)
    try:
        with open(p, "rb") as f:
            data = f.read(MAX_CONFIG_BYTES + 1)
    except OSError:
        return None, artifact_hash({})
    return data, artifact_hash({CONFIG_FILE: data})


def evaluate_artifact(task: str, artifact_dir: str, seeds=(0, 1), split: str = "dev", *, pool=None) -> dict:
    t0 = time.time()
    data, ahash = read_artifact(artifact_dir)
    if data is None:
        return _illegal(task, list(seeds), split, "schema:missing_file", f"{CONFIG_FILE} not found", ahash, t0)
    if len(data) > MAX_CONFIG_BYTES:
        return _illegal(task, list(seeds), split, "schema:too_large",
                        f"{CONFIG_FILE} larger than {MAX_CONFIG_BYTES} bytes", ahash, t0)
    return evaluate_config(task, data.decode("utf-8", errors="replace"), seeds, split, ahash=ahash, pool=pool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--artifact", required=True, help="directory containing config.json")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--split", default="dev", choices=["dev", "sealed"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = evaluate_artifact(a.task, a.artifact, [int(s) for s in a.seeds.split(",") if s != ""], a.split)
    text = json.dumps(res, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
