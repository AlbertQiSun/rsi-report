"""EvalService: per-lane job queues, sandboxed training, sealed scoring, audits, events.

Lanes are local (a GPU of this node) or remote (a GPU in another Slurm job, reached with
``srun --jobid=... --overlap`` from a SLURM_*-free env; the seed then runs on that node via
``python -m rsi.evald.seed_runner``). Per seed of a job (rsi/evald/seed.py):
  1. copy the (L0-checked) artifact into a fresh dir on node-local scratch
  2. train in sandbox B (bwrap: no network, read-only train shards + tokenizer + env,
     writable /work and /tmp only, one GPU); wall-clock hard limit enforced from here,
     whole process group SIGKILLed at time_budget_s + startup_margin_s
  3. L1 host checks: guard channel, honeypots, reported vs measured time, export, ckpt size
  4. score dev in sandbox C (read-only: pristine artifact code, exported weights, packed
     dev rows; no network; no other writable place than tmpfs)
  5. score sealed in a second sandbox C (sealed rows only). Sealed numbers go to the
     result's ``sealed`` field and to sealed.jsonl files; never to events or feedback.
Host side is stdlib only (no torch).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from statistics import mean

from rsi import events
from rsi.evald import audit
from rsi.evald.channel import FifoReader
from rsi.evald.config import GUARD_DIR, PLATFORM_ROOT, SCORER_DIR, EvaldConfig, LaneCfg
from rsi.evald.sandbox import (Mount, SandboxSpec, clean_slurm_env, gpu_info, make_launcher, resolve_slurm_gpu,
                               run_limited, srun_argv)
from rsi.evald.seed import SeedRunner, empty_seed_result
from rsi.runs import append_jsonl, artifact_hash, atomic_write_json, read_json

STATUS_ORDER = ["audit_fail", "nan", "crash", "timeout", "ok"]
METRIC = "val_bpb"
AGENT = "evald"


def _now() -> float:
    return round(time.time(), 3)


class LaneState:
    def __init__(self, cfg: LaneCfg):
        self.cfg = cfg
        self.lane = cfg.lane
        self.gpu = cfg.gpu
        self.remote = cfg.remote
        self.queue: collections.deque[str] = collections.deque()
        self.paused = False
        self.busy: str | None = None
        self.thread: threading.Thread | None = None


class EvalService:
    def __init__(self, cfg: EvaldConfig, launcher=None):
        self.cfg = cfg
        self.launcher = launcher or make_launcher(cfg.sandbox, bwrap=cfg.bwrap, extra_ro=cfg.extra_ro_binds)
        for d in (cfg.state_dir, os.path.join(cfg.state_dir, "jobs"), os.path.join(cfg.state_dir, "cache"),
                  cfg.scratch_dir, os.path.join(cfg.scratch_dir, "jobs")):
            os.makedirs(d, exist_ok=True)
        os.chmod(cfg.state_dir, 0o700)
        os.chmod(cfg.scratch_dir, 0o700)
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.jobs: dict[str, dict] = {}
        self.lanes: dict[int, LaneState] = {l.lane: LaneState(l) for l in cfg.lanes}
        self.runner = SeedRunner(cfg, self.launcher)
        self._gpu: dict[int, object] = {}
        self._prep_lock = threading.Lock()
        self._stop = threading.Event()
        self._load_state()

    # ------------------------------------------------------------------ state
    def _jdir(self, job_id: str) -> str:
        return os.path.join(self.cfg.state_dir, "jobs", job_id)

    def _save_job(self, job: dict) -> None:
        atomic_write_json(os.path.join(self._jdir(job["id"]), "job.json"), job)

    def _save_lanes(self) -> None:
        atomic_write_json(os.path.join(self.cfg.state_dir, "lanes.json"),
                          {str(k): {"paused": v.paused} for k, v in self.lanes.items()})

    def _load_state(self) -> None:
        lanes = read_json(os.path.join(self.cfg.state_dir, "lanes.json"), {}) or {}
        for k, v in lanes.items():
            if int(k) in self.lanes:
                self.lanes[int(k)].paused = bool(v.get("paused"))
        jroot = os.path.join(self.cfg.state_dir, "jobs")
        pending = []
        for jid in os.listdir(jroot):
            job = read_json(os.path.join(jroot, jid, "job.json"))
            if not job:
                continue
            res = read_json(os.path.join(jroot, jid, "result.json"))
            if res is not None:
                job["state"] = "done"
            else:
                job["state"] = "queued"
                pending.append(job)
            self.jobs[jid] = job
        for job in sorted(pending, key=lambda j: j.get("t_submit", 0)):
            lane = job.get("lane")
            if lane not in self.lanes:
                lane = self._pick_lane()
                job["lane"] = lane
            self.lanes[lane].queue.append(job["id"])

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "EvalService":
        if self.cfg.sandbox == "bwrap":
            for ls in self.lanes.values():
                if ls.remote or ls.gpu is None:
                    continue   # remote lanes resolve their GPU inside the Slurm step
                self._gpu[ls.lane] = (resolve_slurm_gpu(self.cfg.nvidia_smi)[0] if ls.gpu == "slurm"
                                      else gpu_info(int(ls.gpu), self.cfg.nvidia_smi))
        self.runner.prepare_hf_template()
        for ls in self.lanes.values():
            ls.thread = threading.Thread(target=self._worker, args=(ls,), name=f"evald-lane{ls.lane}", daemon=True)
            ls.thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self.cond:
            self.cond.notify_all()
        for ls in self.lanes.values():
            if ls.thread is not None:
                ls.thread.join(timeout=timeout)

    def _pick_lane(self, exclude: int | None = None) -> int:
        cands = [l for l in self.lanes.values() if not l.paused and l.lane != exclude] or \
                [l for l in self.lanes.values() if l.lane != exclude] or list(self.lanes.values())
        return min(cands, key=lambda l: (len(l.queue) + (1 if l.busy else 0), l.lane)).lane

    def submit(self, run: str, task: str, artifact_dir: str, *, seeds: list[int], lane: int | None = None,
               gen: int = 0, trial: int | None = None, events_path: str | None = None,
               emit_submit: bool = True) -> str:
        if not isinstance(run, str) or not run or "/" in run or run.startswith("."):
            raise ValueError("bad run id")
        if not str(task).startswith("nanochat"):
            raise ValueError(f"evald only serves the nanochat task family, not {task!r}")
        seeds = [int(s) for s in seeds]
        if not 1 <= len(seeds) <= 32:
            raise ValueError("need 1..32 seeds")
        if lane is not None and int(lane) not in self.lanes:
            raise ValueError(f"unknown lane {lane}; lanes: {sorted(self.lanes)}")
        job_id = time.strftime("j%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        jdir = self._jdir(job_id)
        os.makedirs(jdir, exist_ok=True)
        cfg = self.cfg
        files, flags = audit.collect_artifact(artifact_dir, max_files=cfg.max_files,
                                              max_file_bytes=cfg.max_file_bytes,
                                              max_total_bytes=cfg.max_artifact_bytes)
        flags += audit.static_check(files)
        adir = os.path.join(jdir, "artifact")
        for name, data in files.items():
            fp = os.path.join(adir, name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as f:
                f.write(data)
        os.makedirs(adir, exist_ok=True)
        ahash = artifact_hash(files) if files else "empty"
        with self.lock:
            chosen = int(lane) if lane is not None else self._pick_lane()
            job = {"id": job_id, "run": run, "task": task, "gen": int(gen), "trial": trial, "seeds": seeds,
                   "lane": chosen, "lane_requested": lane is not None, "artifact_hash": ahash,
                   "artifact_src": os.path.abspath(artifact_dir), "events_path": events_path,
                   "t_submit": _now(), "state": "queued", "l0_flags": flags,
                   "audit_rules": audit.AUDIT_RULES_VERSION}
            self.jobs[job_id] = job
            self._save_job(job)
        if emit_submit:
            self._emit(job, "eval_submit", artifact_hash=ahash, lane=chosen, seeds=seeds, job_id=job_id)
        if audit.has_hard(flags):
            self._finish(job, self._l0_fail_result(job, flags))
        else:
            with self.cond:
                self.lanes[chosen].queue.append(job_id)
                self.cond.notify_all()
        return job_id

    def result(self, job_id: str, wait: bool = True, timeout_s: float = 3600.0) -> dict:
        deadline = time.time() + timeout_s
        with self.cond:
            while True:
                job = self.jobs.get(job_id)
                if job is None:
                    raise KeyError(job_id)
                if job["state"] == "done":
                    break
                left = deadline - time.time()
                if not wait or left <= 0:
                    return {"job_id": job_id, "done": False, "state": job["state"], "lane": job.get("lane")}
                self.cond.wait(timeout=min(left, 5.0))
        res = read_json(os.path.join(self._jdir(job_id), "result.json"))
        res["done"] = True
        return res

    def lanes_info(self) -> list[dict]:
        with self.lock:
            return [{"lane": l.lane, "gpu": l.gpu, "busy": l.busy is not None, "job": l.busy,
                     "paused": l.paused, "queued": len(l.queue), "remote": l.remote,
                     "jobid": l.cfg.jobid, "node": l.cfg.node}
                    for l in sorted(self.lanes.values(), key=lambda x: x.lane)]

    def pause_lane(self, lane: int) -> None:
        with self.cond:
            ls = self.lanes[int(lane)]
            ls.paused = True
            # move auto-assigned queued jobs to unpaused lanes
            if any(not l.paused for l in self.lanes.values()):
                keep = collections.deque()
                for jid in ls.queue:
                    job = self.jobs[jid]
                    if job.get("lane_requested"):
                        keep.append(jid)
                    else:
                        new = self._pick_lane(exclude=ls.lane)
                        job["lane"] = new
                        self._save_job(job)
                        self.lanes[new].queue.append(jid)
                ls.queue = keep
            self._save_lanes()
            self.cond.notify_all()

    def resume_lane(self, lane: int) -> None:
        with self.cond:
            self.lanes[int(lane)].paused = False
            self._save_lanes()
            self.cond.notify_all()

    def list_jobs(self, run: str | None = None) -> list[dict]:
        with self.lock:
            return [{k: j.get(k) for k in ("id", "run", "gen", "trial", "lane", "state", "artifact_hash", "seeds", "t_submit")}
                    for j in self.jobs.values() if run is None or j.get("run") == run]

    # ---------------------------------------------------------------- workers
    def _next_job(self, ls: LaneState) -> str | None:
        if ls.queue:
            return ls.queue.popleft()
        # steal an auto-assigned job from another lane's queue
        for other in sorted(self.lanes.values(), key=lambda l: -len(l.queue)):
            if other is ls:
                continue
            for jid in list(other.queue):
                job = self.jobs[jid]
                if not job.get("lane_requested") and (other.paused or other.busy is not None):
                    other.queue.remove(jid)
                    job["lane"] = ls.lane
                    return jid
        return None

    def _worker(self, ls: LaneState) -> None:
        while not self._stop.is_set():
            with self.cond:
                jid = None
                while not self._stop.is_set():
                    if not ls.paused:
                        jid = self._next_job(ls)
                        if jid:
                            break
                    self.cond.wait(timeout=1.0)
                if self._stop.is_set() or jid is None:
                    return
                job = self.jobs[jid]
                job["state"] = "running"
                job["lane"] = ls.lane
                job["t_start"] = _now()
                ls.busy = jid
                self._save_job(job)
                self.cond.notify_all()
            try:
                res = self._run_job(job, ls)
            except Exception as e:  # infrastructure failure: record, keep serving
                traceback.print_exc()
                res = self._base_result(job, ls.lane)
                res.update(status="crash", error=f"evaluator infrastructure error: {type(e).__name__}",
                           infra_error=f"{type(e).__name__}: {e}"[:500])
            with self.cond:
                ls.busy = None
            self._finish(job, res)

    # ----------------------------------------------------------------- events
    def _emit(self, job: dict, ev: str, **fields) -> None:
        try:
            if job.get("events_path"):
                events.register_run(job["run"], job["events_path"])
            elif self.cfg.runs_root:
                events.register_run(job["run"], os.path.join(self.cfg.runs_root, job["run"], "events.jsonl"))
            events.emit(job["run"], ev, AGENT, gen=int(job.get("gen") or 0), trial=job.get("trial"), **fields)
        except Exception as e:  # never let logging kill an evaluation
            print(f"[evald] event emit failed ({ev}): {type(e).__name__}: {e}")

    def _sealed_paths(self, job: dict) -> list[str]:
        out = [os.path.join(self.cfg.state_dir, "sealed.jsonl")]
        if job.get("events_path"):
            out.append(os.path.join(os.path.dirname(job["events_path"]), "sealed.jsonl"))
        elif self.cfg.runs_root:
            out.append(os.path.join(self.cfg.runs_root, job["run"], "sealed.jsonl"))
        return out

    # ---------------------------------------------------------------- results
    def _base_result(self, job: dict, lane: int | None) -> dict:
        return {"job_id": job["id"], "run": job["run"], "task": job["task"], "gen": job["gen"], "trial": job["trial"],
                "lane": lane, "artifact_hash": job["artifact_hash"], "metric": METRIC, "value": None,
                "seeds": job["seeds"], "wall_s": 0.0, "status": "crash",
                "audit": {"level": "L1", "flags": [], "details": [], "rules": audit.AUDIT_RULES_VERSION},
                "per_seed": [], "sealed": {"value": None, "per_seed": [], "flags": []},
                "error": None, "log_tail": "", "sandbox": self.cfg.sandbox,
                "wall_limit_s": self.cfg.wall_limit_s, "t_submit": job.get("t_submit"), "t_start": job.get("t_start")}

    def _l0_fail_result(self, job: dict, flags: list[dict]) -> dict:
        res = self._base_result(job, job.get("lane"))
        res["status"] = "audit_fail"
        res["audit"].update(level="L0", flags=sorted({f["check"] for f in flags}), details=flags)
        hard = [f for f in flags if f["severity"] == "hard"]
        res["error"] = "L0 audit failed: " + "; ".join(f["detail"] for f in hard[:5])
        return res

    def _finish(self, job: dict, res: dict) -> None:
        res["t_end"] = _now()
        atomic_write_json(os.path.join(self._jdir(job["id"]), "result.json"), res)
        # sealed numbers: private files only
        # "value" = sealed val_bpb (mean over seeds; null unless status ok): read by orchestrator-side
        # policies only (e.g. recursive_first_steps). "sealed" kept as an alias.
        rec = {"t": res["t_end"], "job_id": job["id"], "run": job["run"], "gen": job["gen"], "trial": job["trial"],
               "artifact_hash": job["artifact_hash"], "seeds": job["seeds"], "status": res["status"],
               "value": res["sealed"]["value"], "dev": res["value"], "sealed": res["sealed"]["value"],
               "per_seed": res["sealed"]["per_seed"], "private_flags": res["sealed"]["flags"]}
        for p in self._sealed_paths(job):
            try:
                append_jsonl(p, rec)
                os.chmod(p, 0o600)
            except OSError as e:
                print(f"[evald] cannot write sealed log {p}: {e}")
        ev_fields = {k: res[k] for k in ("artifact_hash", "metric", "value", "seeds", "wall_s", "status")}
        ev_fields["audit"] = {"level": res["audit"]["level"], "flags": res["audit"]["flags"]}
        ev_fields["job_id"] = job["id"]
        ev_fields["lane"] = res.get("lane")
        ev_fields["per_seed"] = [{k: s.get(k) for k in ("seed", "status", "dev", "train_wall_s")} for s in res["per_seed"]]
        if res.get("error"):
            ev_fields["error"] = str(res["error"])[:200]
        self._emit(job, "eval_result", **ev_fields)
        for d in res["audit"]["details"]:
            self._emit(job, "audit_flag", artifact_hash=job["artifact_hash"], check=d["check"],
                       detail=events.clip(d["detail"], 200), severity=d.get("severity"), seed=d.get("seed"))
        for d in res["sealed"]["flags"]:
            self._emit(job, "audit_flag", artifact_hash=job["artifact_hash"], check="hidden:" + d["check"],
                       detail="details in the evaluator's private log", severity=d.get("severity"), seed=d.get("seed"))
        with self.cond:
            job["state"] = "done"
            job["t_end"] = res["t_end"]
            self._save_job(job)
            self.cond.notify_all()

    # ------------------------------------------------------------------ jobs
    def _run_job(self, job: dict, ls: LaneState) -> dict:
        t0 = time.time()
        res = self._base_result(job, ls.lane)
        res["audit"]["details"] = [dict(f, seed=None) for f in job.get("l0_flags", [])]
        seeds_res = []
        for seed in job["seeds"]:
            if self._stop.is_set():
                raise RuntimeError("evald stopping")
            seeds_res.append(self._run_seed(job, ls, seed))
        details = list(res["audit"]["details"])
        for s in seeds_res:
            details += [dict(f, seed=s["seed"]) for f in s.pop("flags")]
        status = min((s["status"] for s in seeds_res), key=STATUS_ORDER.index)
        devs = [s["dev"] for s in seeds_res if s["status"] == "ok" and s["dev"] is not None]
        if len(devs) >= 2 and max(devs) - min(devs) < 1e-9:
            details.append({"check": "seed_ignored", "detail": "identical dev scores across different seeds",
                            "severity": "soft", "seed": None})
        sealed_flags = []
        for s in seeds_res:
            sealed_flags += [dict(f, seed=s["seed"]) for f in s.pop("private_flags")]
        sealed_ps = [s.pop("private") for s in seeds_res]
        sv = [p["value"] for p in sealed_ps if p.get("value") is not None]
        res.update(
            status=status,
            value=mean(devs) if status == "ok" and devs else None,
            wall_s=round(time.time() - t0, 2),
            per_seed=seeds_res,
            sealed={"value": mean(sv) if status == "ok" and len(sv) == len(seeds_res) else None,
                    "per_seed": sealed_ps, "flags": sealed_flags},
        )
        res["audit"]["details"] = details
        res["audit"]["flags"] = sorted({d["check"] for d in details})
        bad = next((s for s in seeds_res if s["status"] != "ok"), None)
        res["error"] = bad.get("error") if bad else None
        res["log_tail"] = (bad or seeds_res[-1]).pop("log_tail", "")
        for s in seeds_res:
            s.pop("log_tail", None)
        infra = [n for s in seeds_res for n in s.pop("infra", [])]
        if infra:
            res["infra_notes"] = infra
        return res

    # ------------------------------------------------------------ seeds
    def _run_seed(self, job: dict, ls: LaneState, seed: int) -> dict:
        packed = {}
        for which, split in self.cfg.splits.items():
            try:
                packed[which] = self._ensure_packed(split)
            except Exception as e:  # noqa: BLE001 - scorer reports the missing split as an infra crash
                print(f"[evald] packing {split} failed: {e}")
        if ls.remote:
            return self._run_seed_remote(job, ls, seed, packed)
        return self.runner.run_seed(job, ls.lane, self._gpu.get(ls.lane), seed, packed)

    def _run_seed_remote(self, job: dict, ls: LaneState, seed: int, packed: dict) -> dict:
        """Run one seed on a GPU of another Slurm job, as a job step on that node."""
        cfg = self.cfg
        jdir = self._jdir(job["id"])
        spec_p = os.path.join(jdir, f"remote_s{seed}.json")
        out_p = os.path.join(jdir, f"seed_s{seed}.json")
        if os.path.exists(out_p):
            os.unlink(out_p)
        atomic_write_json(spec_p, {"cfg": cfg.to_json(), "job": job, "lane": ls.lane, "gpu": ls.gpu,
                                   "seed": seed, "packed": packed})
        argv = srun_argv(ls.cfg, cfg) + [cfg.host_python, "-m", "rsi.evald.seed_runner", spec_p, out_p]
        env = clean_slurm_env()
        env["PYTHONPATH"] = cfg.platform_root
        limit = cfg.wall_limit_s + 2 * cfg.score_timeout_s + cfg.gpu_idle_wait_s + cfg.remote_slack_s
        logp = os.path.join(jdir, f"remote_s{seed}.log")
        with open(logp, "wb") as lf:
            proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True, cwd=cfg.platform_root)
            r = run_limited(proc, limit)
        res = read_json(out_p)
        if isinstance(res, dict) and res.get("seed") == seed and res.get("status"):
            return res
        out = empty_seed_result(seed, "crash", "evaluator infrastructure error (remote lane)")
        out["infra"].append(f"remote lane {ls.lane} (job {ls.cfg.jobid}) step failed: rc={r['rc']} "
                            f"killed={r['killed_reason']}; see {logp}")
        return out

    # ------------------------------------------------------------ eval data
    def _fingerprint(self, split: str) -> str:
        h = hashlib.sha256()
        d = self.cfg.split_dir(split)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".parquet"):
                st = os.stat(os.path.join(d, fn))
                h.update(f"{fn}:{st.st_size}:{int(st.st_mtime)}".encode())
        st = os.stat(os.path.join(self.cfg.split_dir("tokenizer"), "tokenizer.pkl"))
        h.update(f"tok:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:10]

    def _ensure_packed(self, split: str) -> str:
        """Tokenize + pack an eval split once (trusted code only), cached on node-local scratch."""
        cfg = self.cfg
        with self._prep_lock:
            path = os.path.join(cfg.state_dir, "cache",
                                f"{split}_T{cfg.seq_len}_R{cfg.eval_rows}_{self._fingerprint(split)}.pt")
            if os.path.isfile(path):
                return path
            pd = os.path.join(cfg.scratch_dir, f"prep_{split}_{uuid.uuid4().hex[:6]}")
            os.makedirs(os.path.join(pd, "out"), exist_ok=True)
            fifo = FifoReader(os.path.join(pd, "audit.fifo")).start()
            mounts = self.runner._env_mount() + [
                Mount(SCORER_DIR, "/opt/rsi/scorer", "ro"),
                Mount(GUARD_DIR, "/opt/rsi/guard", "ro"),
                Mount(cfg.split_dir(split), "/in/data", "ro"),
                Mount(cfg.split_dir("tokenizer"), "/in/tokenizer", "ro"),
                Mount(os.path.join(pd, "out"), "/out", "rw"),
                Mount(fifo.path, "/opt/rsi/chan/audit.fifo", "rw"),
            ]
            if cfg.sandbox == "bwrap":
                mounts.append(Mount(None, "/tmp", "tmpfs"))
            tmp_in = "/tmp" if cfg.sandbox == "bwrap" else os.path.join(pd, "out")
            env = self.runner._env_common(None, tmp_in)
            env.update({"HOME": tmp_in, "PYTHONPATH": "/opt/rsi/guard", "RSI_AUDIT_PIPE": "/opt/rsi/chan/audit.fifo",
                        "RSI_GUARD_ROLE": "prepare", "RSI_GUARD_DENY": ""})
            argv = [cfg.train_python, "-u", "/opt/rsi/scorer/scorer.py", "prepare", "--data", "/in/data",
                    "--tokenizer", "/in/tokenizer", "--seq-len", str(cfg.seq_len), "--rows", str(cfg.eval_rows),
                    "--out", "/out/rows.pt"]
            spec = SandboxSpec(argv=argv, env=env, cwd="/out", mounts=mounts, gpu=None)
            logp = os.path.join(cfg.state_dir, f"prepare_{split}.log")
            with open(logp, "wb") as logf:
                proc = self.launcher.popen(spec, stdout=logf, stderr=subprocess.STDOUT)
                r = run_limited(proc, cfg.prepare_timeout_s)
            msgs = fifo.stop()
            res = [m for m in msgs if m.get("k") == "result"]
            outp = os.path.join(pd, "out", "rows.pt")
            if r["rc"] != 0 or not res or not res[-1].get("ok") or not os.path.isfile(outp):
                shutil.rmtree(pd, ignore_errors=True)
                raise RuntimeError(f"packing {split} failed (rc={r['rc']}); see {logp}")
            os.replace(outp, path)
            os.chmod(path, 0o600)
            shutil.rmtree(pd, ignore_errors=True)
            info = dict(res[-1])
            info.pop("k", None)
            atomic_write_json(path + ".json", info)
            return path
