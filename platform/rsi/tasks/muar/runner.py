"""Local CPU runner for µAR: process pool, an EvalClient stand-in, baselines and s_ref.

Every worker is a separate process pinned to one thread (OMP/MKL/torch = 1), so N workers use N
cores and results stay bit-identical to a serial run. On the cluster node use ~34 of 36 CPUs.

CLI:
  python -m rsi.tasks.muar.runner list
  python -m rsi.tasks.muar.runner eval --task muar/id-modadd-0 --config cfg.json [--seeds 0,1] [--split dev]
  python -m rsi.tasks.muar.runner bench [--split all] [--workers 8]          # Gate 1 timing smoke test
  python -m rsi.tasks.muar.runner baseline --method random,local --split heldout --rng-seeds 0,1,2 \
        --n 12 --workers 34 --out baselines.json
  python -m rsi.tasks.muar.runner sref --split all --n-random 300 --n-local 100 --workers 34 [--out PATH]

Python:
  LocalEvalClient(workers=8)   # same methods as rsi.evald.client.EvalClient: submit/result/lanes/pause/resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import numpy as np

from rsi.runs import atomic_write_json, read_json
from rsi.tasks.muar import schema
from rsi.tasks.muar.evaluate import evaluate_artifact, evaluate_config, read_artifact
from rsi.tasks.muar.tasks import (DEV_SEEDS, N_SEALED_SEEDS, SPLITS, get_spec, sealed_dir,
                                  start_config_and_defects, tasks_in)


# ---------------------------------------------------------------------------- pool
def _init_worker() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    import torch

    torch.set_num_threads(1)


def default_workers() -> int:
    if os.environ.get("RSI_MUAR_WORKERS"):
        return int(os.environ["RSI_MUAR_WORKERS"])
    return max(1, (os.cpu_count() or 2) - 2)


def make_pool(workers: int | None = None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers or default_workers(), mp_context=mp.get_context("spawn"),
                               initializer=_init_worker)


def _eval_job(task: str, cfg: dict, seeds: list[int], split: str = "dev") -> dict:
    return evaluate_config(task, cfg, seeds, split)


# ---------------------------------------------------------------------------- baselines
def baseline_run(task: str, method: str, rng_seed: int, n: int, seeds=DEV_SEEDS) -> dict:
    """One non-LLM search run (design doc §1.5 control 7), greedy keep/discard like the LLM arms.

    random: each proposal is a fresh sample from the prior.  local: perturb 1-3 knobs of the current best.
    Returns values (dev, None for failed proposals), aligned with the LLM arms' trial curves.
    """
    rng = np.random.Generator(np.random.PCG64([rng_seed, int(hashlib.sha256(task.encode()).hexdigest()[:8], 16),
                                               {"random": 1, "local": 2}[method]]))
    start, _ = start_config_and_defects(get_spec(task))
    r0 = evaluate_config(task, start, seeds)
    best_cfg, best = start, r0["value"]
    values, statuses, configs = [], [], []
    for _ in range(n):
        cfg = schema.sample_config(rng) if method == "random" else schema.perturb_config(best_cfg, rng)
        r = evaluate_config(task, cfg, seeds)
        v = r["value"] if r["status"] == "ok" else None
        values.append(v)
        statuses.append(r["status"])
        configs.append(cfg)
        if v is not None and (best is None or v < best):
            best, best_cfg = v, cfg
    return {"task": task, "method": method, "rng_seed": rng_seed, "s0": r0["value"], "values": values,
            "statuses": statuses, "best": best, "best_config": best_cfg, "configs": configs}


def run_baselines(tasks: list[str], methods: list[str], rng_seeds: list[int], n: int, *, workers=None,
                  seeds=DEV_SEEDS, progress=False) -> list[dict]:
    jobs = [(t, m, s) for t in tasks for m in methods for s in rng_seeds]
    out = []
    with make_pool(workers) as pool:
        futs = {pool.submit(baseline_run, t, m, s, n, list(seeds)): (t, m, s) for t, m, s in jobs}
        for i, f in enumerate(as_completed(futs)):
            out.append(f.result())
            if progress:
                print(f"[{i + 1}/{len(jobs)}] {futs[f]} best={out[-1]['best']}", file=sys.stderr, flush=True)
    out.sort(key=lambda r: (r["task"], r["method"], r["rng_seed"]))
    return out


def baseline_curves(results: list[dict]) -> dict:
    """{arm: {task: {rng_seed: values}}} in the format rsi.update.analysis expects."""
    arms: dict = {}
    for r in results:
        arms.setdefault(f"R-{r['method']}", {}).setdefault(r["task"], {})[str(r["rng_seed"])] = r["values"]
    return arms


# ---------------------------------------------------------------------------- s_ref
def sref_path() -> str:
    return os.path.join(sealed_dir(), "sref.json")


def compute_sref(tasks: list[str], *, n_random: int = 300, n_local: int = 100, workers=None,
                 seeds=DEV_SEEDS, rng_seed: int = 12345, progress: bool = False) -> dict:
    """s_ref per task = best dev score of a large random search + local search from its top configs.

    Only used to normalise nAUC; never shown to the improver. Fold arm bests in later with fold_sref().
    """
    out: dict = {}
    with make_pool(workers) as pool:
        futs = {}
        for t in tasks:
            rng = np.random.Generator(np.random.PCG64([rng_seed, int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)]))
            start, _ = start_config_and_defects(get_spec(t))
            futs[pool.submit(_eval_job, t, start, list(seeds))] = (t, "start", start)
            for _ in range(n_random):
                cfg = schema.sample_config(rng)
                futs[pool.submit(_eval_job, t, cfg, list(seeds))] = (t, "random", cfg)
        res: dict = {t: [] for t in tasks}
        s0: dict = {}
        for i, f in enumerate(as_completed(futs)):
            t, kind, cfg = futs[f]
            r = f.result()
            if kind == "start":
                s0[t] = r["value"]
            if r["status"] == "ok":
                res[t].append((r["value"], cfg))
            if progress and (i + 1) % 200 == 0:
                print(f"sref random: {i + 1}/{len(futs)}", file=sys.stderr, flush=True)
        # local search: 4 independent chains per task from its top-4 random configs
        n_chains = 4
        lfuts = {}
        for t in tasks:
            top = sorted(res[t], key=lambda x: x[0])[:n_chains]
            for j, (_, cfg) in enumerate(top):
                lfuts[pool.submit(_local_chain, t, cfg, max(1, n_local // n_chains), rng_seed + j, list(seeds))] = t
        for f in as_completed(lfuts):
            t = lfuts[f]
            res[t].extend(f.result())
    for t in tasks:
        best_v, best_cfg = min(res[t], key=lambda x: x[0]) if res[t] else (None, None)
        if s0.get(t) is not None and best_v is not None:
            best_v = min(best_v, s0[t])
        out[t] = {"s_ref": best_v, "s0": s0.get(t), "best_config": best_cfg, "n_evals": len(res[t]) + 1,
                  "source": "offline random+local search"}
    return out


def _local_chain(task: str, cfg: dict, n: int, rng_seed: int, seeds: list[int]) -> list:
    rng = np.random.Generator(np.random.PCG64([rng_seed, 7]))
    best_cfg = cfg
    r = evaluate_config(task, cfg, seeds)
    best = r["value"]
    found = []
    for _ in range(n):
        c = schema.perturb_config(best_cfg, rng)
        r = evaluate_config(task, c, seeds)
        if r["status"] == "ok":
            found.append((r["value"], c))
            if best is None or r["value"] < best:
                best, best_cfg = r["value"], c
    return found


def save_sref(path: str, table: dict) -> None:
    cur = read_json(path, {}) or {}
    for t, row in table.items():
        old = cur.get(t, {})
        if old.get("s_ref") is not None and row.get("s_ref") is not None and old["s_ref"] < row["s_ref"]:
            row = {**row, "s_ref": old["s_ref"], "best_config": old.get("best_config"), "source": old.get("source")}
        cur[t] = {**old, **row}
    atomic_write_json(path, cur)


def fold_sref(path: str, task: str, value: float | None, source: str) -> None:
    """Lower s_ref of ``task`` if an arm found a better dev value (design doc §1.1: all-arm best)."""
    if value is None:
        return
    cur = read_json(path, {}) or {}
    row = cur.setdefault(task, {})
    if row.get("s_ref") is None or value < row["s_ref"]:
        row.update(s_ref=value, source=source)
        atomic_write_json(path, cur)


def load_sref(path: str | None = None) -> dict:
    return read_json(path or sref_path(), {}) or {}


# ---------------------------------------------------------------------------- EvalClient stand-in
class LocalEvalClient:
    """Drop-in for rsi.evald.client.EvalClient that scores µAR artifacts on local CPUs.

    submit() snapshots config.json at submission time; job ids are deterministic per
    (run, trial, artifact hash, seeds) so a resumed policy gets the same job. Emits eval_result
    events (agent "evald") when ``emit_events`` and the run is registered with rsi.events.
    """

    def __init__(self, workers: int | None = None, *, emit_events: bool = True, sealed_log: bool = False):
        self._pool = make_pool(workers)
        self._jobs: dict[str, Future] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._paused: set[int] = set()
        self.emit_events = emit_events
        self.sealed_log = sealed_log

    def submit(self, run: str, task: str, artifact_dir: str, *, seeds: list[int], lane: int | None = None,
               gen: int = 0, trial: int | None = None, **_kw) -> str:
        data, ahash = read_artifact(artifact_dir)
        jid = "muar-" + hashlib.sha256(json.dumps([run, trial, ahash, list(seeds), task]).encode()).hexdigest()[:16]
        with self._lock:
            if jid in self._jobs:
                return jid
            text = (data or b"").decode("utf-8", errors="replace") if data is not None else None
            if text is None:
                fut = self._pool.submit(evaluate_artifact, task, artifact_dir, list(seeds), "dev")
            else:
                fut = self._pool.submit(_eval_text, task, text, list(seeds), ahash)
            self._jobs[jid] = fut
            self._meta[jid] = {"run": run, "gen": gen, "trial": trial, "lane": lane, "task": task, "emitted": False}
        return jid

    def result(self, job_id: str, *, wait: bool = True, timeout_s: float = 3600) -> dict:
        fut = self._jobs[job_id]
        if not wait and not fut.done():
            return {"status": "pending"}
        try:
            res = fut.result(timeout=timeout_s)
        except Exception as e:  # noqa: BLE001 - worker crash is a crash result, not an orchestrator error
            res = {"metric": "val_loss", "value": None, "seeds": [], "wall_s": 0.0, "status": "crash",
                   "audit": {"level": "L0", "flags": ["worker_crash"]}, "diag": {"error": type(e).__name__}}
        meta = self._meta[job_id]
        if self.emit_events and not meta["emitted"]:
            meta["emitted"] = True
            try:
                from rsi import events

                events.emit(meta["run"], "eval_result", "evald", parent="orchestrator", gen=meta["gen"],
                            trial=meta["trial"], artifact_hash=res.get("artifact_hash"), metric=res["metric"],
                            value=res["value"], seeds=res["seeds"], wall_s=res["wall_s"], status=res["status"],
                            audit=res["audit"])
            except Exception:  # noqa: BLE001 - events are best-effort for local runs
                pass
        return dict(res)

    def lanes(self) -> list[dict]:
        busy = any(not f.done() for f in self._jobs.values())
        return [{"lane": 0, "gpu": None, "busy": busy, "paused": 0 in self._paused}]

    def pause_lane(self, lane: int) -> None:
        self._paused.add(lane)

    def resume_lane(self, lane: int) -> None:
        self._paused.discard(lane)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _eval_text(task: str, text: str, seeds: list[int], ahash: str) -> dict:
    return evaluate_config(task, text, seeds, "dev", ahash=ahash)


def score_sealed(task: str, artifact_dir: str, n_seeds: int = N_SEALED_SEEDS, pool=None) -> dict:
    """Sealed re-score (sealed val split + salted seeds). For the analysis script only."""
    return evaluate_artifact(task, artifact_dir, list(range(n_seeds)), "sealed", pool=pool)


# ---------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("eval")
    e.add_argument("--task", required=True)
    e.add_argument("--config", required=True)
    e.add_argument("--seeds", default="0,1")
    e.add_argument("--split", default="dev", choices=["dev", "sealed"])
    b = sub.add_parser("bench")
    b.add_argument("--split", default="all")
    b.add_argument("--workers", type=int)
    bl = sub.add_parser("baseline")
    bl.add_argument("--method", default="random,local")
    bl.add_argument("--split", default="heldout")
    bl.add_argument("--rng-seeds", default="0,1,2")
    bl.add_argument("--n", type=int, default=12)
    bl.add_argument("--workers", type=int)
    bl.add_argument("--out", required=True)
    s = sub.add_parser("sref")
    s.add_argument("--split", default="all")
    s.add_argument("--n-random", type=int, default=300)
    s.add_argument("--n-local", type=int, default=100)
    s.add_argument("--workers", type=int)
    s.add_argument("--out", default=None, help="default: <sealed dir>/sref.json")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for sp, names in SPLITS.items():
            print(sp, " ".join(names))
        return 0
    if a.cmd == "eval":
        with open(a.config) as f:
            text = f.read()
        print(json.dumps(evaluate_config(a.task, text, [int(x) for x in a.seeds.split(",")], a.split), sort_keys=True))
        return 0
    if a.cmd == "bench":
        names = tasks_in(a.split)
        t0 = time.time()
        with make_pool(a.workers) as pool:
            futs = {pool.submit(_eval_job, t, start_config_and_defects(get_spec(t))[0], list(DEV_SEEDS)): t
                    for t in names}
            rows = {futs[f]: f.result() for f in as_completed(futs)}
        for t in names:
            r = rows[t]
            print(f"{t:24s} status={r['status']:10s} value={r['value']} wall_s={r['wall_s']} steps={r['diag'].get('steps')}")
        print(f"total {time.time() - t0:.1f}s for {len(names)} evals x {len(DEV_SEEDS)} seeds")
        return 0
    if a.cmd == "baseline":
        res = run_baselines(tasks_in(a.split), a.method.split(","), [int(x) for x in a.rng_seeds.split(",")],
                            a.n, workers=a.workers, progress=True)
        atomic_write_json(a.out, {"runs": res, "arms": baseline_curves(res)})
        print(a.out)
        return 0
    if a.cmd == "sref":
        table = compute_sref(tasks_in(a.split), n_random=a.n_random, n_local=a.n_local, workers=a.workers,
                             progress=True)
        path = a.out or sref_path()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_sref(path, table)
        for t, row in table.items():
            print(f"{t:24s} s0={row['s0']} s_ref={row['s_ref']}")
        print(path)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
