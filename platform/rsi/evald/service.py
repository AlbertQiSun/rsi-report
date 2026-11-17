"""EvalService: per-lane job queues, sandboxed training, sealed scoring, audits, events.

Per seed of a job:
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
from rsi.evald.config import GUARD_DIR, NANOCHAT_LIB_DIR, SCORER_DIR, EvaldConfig
from rsi.evald.honeypot import Honeypots
from rsi.evald.sandbox import Mount, SandboxSpec, gpu_compute_pids, gpu_info, make_launcher, run_limited
from rsi.runs import append_jsonl, artifact_hash, atomic_write_json, read_json

STATUS_ORDER = ["audit_fail", "nan", "crash", "timeout", "ok"]
METRIC = "val_bpb"
AGENT = "evald"


def _now() -> float:
    return round(time.time(), 3)


def _tail_bytes(path: str, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _visible_tail(text: str, n_lines: int = 40, n_chars: int = 3000) -> str:
    lines = [l for l in text.replace("\r", "\n").split("\n") if l.strip()]
    out = "\n".join(lines[-n_lines:])
    return out[-n_chars:]


class LaneState:
    def __init__(self, lane: int, gpu: int | None):
        self.lane = lane
        self.gpu = gpu
        self.queue: collections.deque[str] = collections.deque()
        self.paused = False
        self.busy: str | None = None
        self.thread: threading.Thread | None = None


class EvalService:
    def __init__(self, cfg: EvaldConfig, launcher=None):
        self.cfg = cfg
        self.launcher = launcher or make_launcher(cfg.sandbox, bwrap=cfg.bwrap, extra_ro=cfg.extra_ro_binds)
        for d in (cfg.state_dir, os.path.join(cfg.state_dir, "jobs"), cfg.scratch_dir,
                  os.path.join(cfg.scratch_dir, "cache"), os.path.join(cfg.scratch_dir, "jobs")):
            os.makedirs(d, exist_ok=True)
        os.chmod(cfg.state_dir, 0o700)
        os.chmod(cfg.scratch_dir, 0o700)
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.jobs: dict[str, dict] = {}
        self.lanes: dict[int, LaneState] = {l.lane: LaneState(l.lane, l.gpu) for l in cfg.lanes}
        self._gpu: dict[int, object] = {}
        self._prep_lock = threading.Lock()
        self._stop = threading.Event()
        self._hf_template: str | None = None
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
                if ls.gpu is not None:
                    self._gpu[ls.lane] = gpu_info(ls.gpu, self.cfg.nvidia_smi)
        self._prepare_hf_template()
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

    def _prepare_hf_template(self) -> None:
        if not self.cfg.hf_cache_dirs:
            return
        tpl = os.path.join(self.cfg.scratch_dir, "hf_template", "hub")
        os.makedirs(tpl, exist_ok=True)
        for d in self.cfg.hf_cache_dirs:
            dst = os.path.join(tpl, os.path.basename(d.rstrip("/")))
            if os.path.isdir(d) and not os.path.isdir(dst):
                shutil.copytree(d, dst + ".tmp", symlinks=True, dirs_exist_ok=True)
                os.replace(dst + ".tmp", dst)
        self._hf_template = tpl

    def _copy_hf(self, dest_hf_home: str) -> None:
        hub = os.path.join(dest_hf_home, "hub")
        if self._hf_template and os.path.isdir(self._hf_template):
            shutil.copytree(self._hf_template, hub, symlinks=True, dirs_exist_ok=True)
        else:
            os.makedirs(hub, exist_ok=True)

    # ------------------------------------------------------------------ API
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
                     "paused": l.paused, "queued": len(l.queue)} for l in sorted(self.lanes.values(), key=lambda x: x.lane)]

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
                "error": None, "log_tail": "", "sandbox": self.launcher.name,
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
        rec = {"t": res["t_end"], "job_id": job["id"], "run": job["run"], "gen": job["gen"], "trial": job["trial"],
               "artifact_hash": job["artifact_hash"], "seeds": job["seeds"], "status": res["status"],
               "dev": res["value"], "sealed": res["sealed"]["value"], "per_seed": res["sealed"]["per_seed"],
               "private_flags": res["sealed"]["flags"]}
        for p in self._sealed_paths(job):
            try:
                append_jsonl(p, rec)
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

    def _deny_list(self, role: str = "train") -> list[str]:
        """Path prefixes the guard reports. In the scorer, its own inputs are added by
        scorer.py itself after it has read them (before importing submitted code)."""
        cfg = self.cfg
        d = list(cfg.deny_paths)
        for split in cfg.splits.values():
            d.append(cfg.split_dir(split))
        # (never a sys.path entry such as /opt/rsi/guard: the import system lists those)
        d += [os.path.join(cfg.scratch_dir, "hf_template"), "/opt/rsi/chan"]
        if role == "train":
            d += [cfg.state_dir, os.path.join(cfg.scratch_dir, "cache")]
            if cfg.sandbox == "none":
                d += [SCORER_DIR]
        return d

    def _wait_gpu_idle(self, gpu) -> list[int]:
        deadline = time.time() + self.cfg.gpu_idle_wait_s
        while True:
            pids = gpu_compute_pids(gpu.uuid, self.cfg.nvidia_smi)
            if not pids or time.time() >= deadline:
                return pids
            time.sleep(5)

    def _env_common(self, gpu, tmp_inside: str) -> dict:
        py_bin = os.path.dirname(self.cfg.train_python)
        return {
            "PATH": f"{py_bin}:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "TMPDIR": tmp_inside,
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": gpu.uuid if gpu is not None else "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }

    def _env_mount(self) -> list[Mount]:
        prefix = os.path.dirname(os.path.dirname(os.path.abspath(self.cfg.train_python)))
        return [Mount(prefix, prefix, "ro")] if self.cfg.sandbox == "bwrap" else []

    def _run_seed(self, job: dict, ls: LaneState, seed: int) -> dict:
        cfg = self.cfg
        gpu = self._gpu.get(ls.lane)
        jdir = self._jdir(job["id"])
        sd = os.path.join(cfg.scratch_dir, "jobs", job["id"], f"s{seed}")
        shutil.rmtree(sd, ignore_errors=True)
        work = os.path.join(sd, "work")
        for sub in ("artifact", "export", "home", "cache", "hf"):
            os.makedirs(os.path.join(work, sub), exist_ok=True)
        tmp = os.path.join(sd, "tmp")
        os.makedirs(tmp, exist_ok=True)
        shutil.copytree(os.path.join(jdir, "artifact"), os.path.join(work, "artifact"), dirs_exist_ok=True)
        self._copy_hf(os.path.join(work, "hf"))
        out = {"seed": seed, "status": None, "dev": None, "train_wall_s": None, "score_wall_s": None,
               "reported_training_s": None, "reported_total_s": None, "attn_backend": None, "error": None,
               "flags": [], "private_flags": [], "private": {"seed": seed, "value": None, "status": None},
               "infra": [], "log_tail": ""}
        flags = out["flags"]
        if gpu is not None:
            busy = self._wait_gpu_idle(gpu)
            if busy:
                out["infra"].append(f"GPU {gpu.index} had compute processes {busy[:5]} at start (seed {seed})")

        # ---------------- training sandbox ----------------
        fifo = FifoReader(os.path.join(sd, "audit.fifo")).start()
        mounts = self._env_mount() + [
            Mount(cfg.split_dir("train"), "/data/train", "ro"),
            Mount(cfg.split_dir("tokenizer"), "/data/tokenizer", "ro"),
            Mount(NANOCHAT_LIB_DIR, "/opt/rsi/lib", "ro"),
            Mount(GUARD_DIR, "/opt/rsi/guard", "ro"),
            Mount(work, "/work", "rw"),
            Mount(fifo.path, "/opt/rsi/chan/audit.fifo", "rw"),
        ]
        if cfg.sandbox == "bwrap":
            mounts.append(Mount(tmp, "/tmp", "rw"))
        hp = None
        if cfg.sandbox == "bwrap" and cfg.honeypots and cfg.honeypot_targets:
            hp = Honeypots(os.path.join(sd, "hp"), cfg.honeypot_targets)
            for host, inside in hp.create():
                mounts.append(Mount(host, inside, "ro"))
            hp.start()
        t0 = time.time()
        env = self._env_common(gpu, "/tmp" if cfg.sandbox == "bwrap" else tmp)
        env.update({
            "HOME": "/work/home", "XDG_CACHE_HOME": "/work/cache",
            "PYTHONPATH": "/opt/rsi/lib:/opt/rsi/guard",
            "SEED": str(seed),
            "RSI_DATA_DIR": "/data/train", "RSI_TOKENIZER_DIR": "/data/tokenizer", "RSI_EXPORT_DIR": "/work/export",
            "RSI_SEQ_LEN": str(cfg.seq_len),
            "RSI_WALL_LIMIT_S": f"{cfg.wall_limit_s:.0f}", "RSI_DEADLINE_UNIX": f"{t0 + cfg.wall_limit_s:.3f}",
            "RSI_AUDIT_PIPE": "/opt/rsi/chan/audit.fifo", "RSI_GUARD_ROLE": "train",
            "RSI_GUARD_DENY": ":".join(self._deny_list()),
            "HF_HOME": "/work/hf", "HF_HUB_CACHE": "/work/hf/hub",
            "TORCHINDUCTOR_CACHE_DIR": "/work/cache/inductor", "TRITON_CACHE_DIR": "/work/cache/triton",
        })
        env.update({k: str(v) for k, v in cfg.train_env_extra.items()})
        spec = SandboxSpec(argv=[cfg.train_python, "-u", "train.py"], env=env, cwd="/work/artifact",
                           mounts=mounts, gpu=gpu)
        logp = os.path.join(jdir, f"train_s{seed}.log")

        def tick():
            if hp is not None:
                hp.poll()
            try:
                if os.path.getsize(logp) > cfg.max_log_bytes:
                    return "log_flood"
            except OSError:
                pass
            return None

        with open(logp, "wb") as logf:
            proc = self.launcher.popen(spec, stdout=logf, stderr=subprocess.STDOUT)
            r = run_limited(proc, cfg.wall_limit_s, t0=t0, tick=tick)
        msgs = fifo.stop()
        hits = hp.stop() if hp is not None else []
        out["train_wall_s"] = round(r["wall_s"], 2)

        log_text = _tail_bytes(logp, 256 * 1024)
        parsed = audit.parse_train_log(log_text)
        out["attn_backend"] = parsed["attn_backend"]
        out["reported_training_s"] = parsed["summary"].get("training_seconds")
        out["reported_total_s"] = parsed["summary"].get("total_seconds")
        out["log_tail"] = _visible_tail(log_text)
        gf, gstats = audit.guard_flags(msgs, where="train")
        flags += gf
        for h in hits:
            flags.append({"check": "forbidden_path_access", "detail": f"opened decoy {h['path']}", "severity": "hard"})
        if gstats["hello"] == 0:
            flags.append({"check": "guard_missing", "detail": "no guard report from the training process",
                          "severity": "soft"})
        flags += audit.time_check(parsed["summary"], r["wall_s"])

        status = None
        export = os.path.join(work, "export")
        if r["killed_reason"] == "timeout":
            status, out["error"] = "timeout", f"killed at the {cfg.wall_limit_s:.0f}s wall-clock limit"
        elif r["killed_reason"] == "log_flood":
            status, out["error"] = "crash", f"killed: log output exceeded {cfg.max_log_bytes} bytes"
            flags.append({"check": "log_flood", "detail": out["error"], "severity": "soft"})
        elif r["rc"] != 0:
            status = "crash"
            out["error"] = f"train.py exited with code {r['rc']}" + (
                " (FAIL: loss NaN or exploding)" if parsed["fail_marker"] else "")
        else:
            sdp = os.path.join(export, "state_dict.pt")
            if not (os.path.isfile(sdp) and os.path.isfile(os.path.join(export, "config.json"))) \
                    or os.path.islink(sdp):
                status, out["error"] = "crash", "train.py finished without export_model() (no state_dict.pt/config.json)"
            elif os.path.getsize(sdp) > cfg.max_ckpt_bytes:
                flags.append({"check": "ckpt_size", "severity": "hard",
                              "detail": f"state_dict.pt is {os.path.getsize(sdp)} bytes > {cfg.max_ckpt_bytes}"})
        if audit.has_hard(flags):
            status = "audit_fail"
            out["error"] = out["error"] or "runtime audit failed: " + "; ".join(
                f["detail"] for f in flags if f["severity"] == "hard")[:300]

        # ---------------- scoring sandboxes ----------------
        if status is None:
            dev = self._score(job, ls, seed, sd, "dev", gpu)
            out["score_wall_s"] = dev.get("wall_s")
            flags += dev["flags"]
            if dev.get("nonparam_bytes") is not None and dev.get("sd_bytes"):
                frac = dev["nonparam_bytes"] / max(1, dev["sd_bytes"])
                if frac > cfg.max_nonparam_frac:
                    flags.append({"check": "ckpt_nonparam", "severity": "soft",
                                  "detail": f"{frac:.0%} of the checkpoint is non-parameter buffers"})
            if dev.get("attn_backend"):
                out["attn_backend_score"] = dev["attn_backend"]
            if audit.has_hard(flags):
                status = "audit_fail"
                out["error"] = "runtime audit failed: " + "; ".join(
                    f["detail"] for f in flags if f["severity"] == "hard")[:300]
                out["private"]["dev_raw"] = dev.get("bpb")
            elif dev["status"] != "ok":
                status = dev["status"]
                out["error"] = dev.get("error_visible") or f"scoring failed ({dev['status']})"
            else:
                status = "ok"
                out["dev"] = dev["bpb"]
                out["n_params"] = dev.get("n_params")
                sealed = self._score(job, ls, seed, sd, "sealed", gpu)
                out["private"].update(value=sealed.get("bpb"), status=sealed["status"],
                                      wall_s=sealed.get("wall_s"), error=sealed.get("error"))
                out["private_flags"] += sealed["flags"]
                if audit.has_hard(sealed["flags"]):
                    status = "audit_fail"
                    out["dev"] = None
                    out["private"]["dev_raw"] = dev["bpb"]
                    flags.append({"check": "hidden_check_failed", "severity": "hard",
                                  "detail": "a runtime check failed during held-out scoring"})
                    out["error"] = "runtime audit failed during held-out scoring"
        out["status"] = status
        if not cfg.keep_scratch:
            shutil.rmtree(sd, ignore_errors=True)
        return out

    def _score(self, job: dict, ls: LaneState, seed: int, sd: str, which: str, gpu) -> dict:
        """Run the trusted scorer on one split in a fresh sandbox. Returns a dict with
        status/bpb/flags (+ error, error_visible, wall_s, n_params, ...)."""
        cfg = self.cfg
        split = cfg.splits[which]
        try:
            packed = self._ensure_packed(split)
        except Exception as e:
            return {"status": "crash", "bpb": None, "flags": [], "error": f"prepare {split}: {e}"[:300],
                    "error_visible": "evaluator infrastructure error (eval data)"}
        ssd = os.path.join(sd, f"score_{which}")
        tmp = os.path.join(ssd, "tmp")
        os.makedirs(tmp, exist_ok=True)
        hf = os.path.join(ssd, "hf")
        self._copy_hf(hf)
        fifo = FifoReader(os.path.join(ssd, "audit.fifo")).start()
        mounts = self._env_mount() + [
            Mount(SCORER_DIR, "/opt/rsi/scorer", "ro"),
            Mount(GUARD_DIR, "/opt/rsi/guard", "ro"),
            Mount(os.path.join(self._jdir(job["id"]), "artifact"), "/in/code", "ro"),
            Mount(os.path.join(sd, "work", "export"), "/in/export", "ro"),
            Mount(packed, "/in/eval/rows.pt", "ro"),
            Mount(os.path.join(cfg.split_dir("tokenizer"), "token_bytes.pt"), "/in/eval/token_bytes.pt", "ro"),
            Mount(hf, "/work/hf", "rw"),
            Mount(fifo.path, "/opt/rsi/chan/audit.fifo", "rw"),
        ]
        if cfg.sandbox == "bwrap":
            mounts.append(Mount(None, "/tmp", "tmpfs"))
        tmp_in = "/tmp" if cfg.sandbox == "bwrap" else tmp
        env = self._env_common(gpu, tmp_in)
        env.update({
            "HOME": tmp_in, "PYTHONPATH": "/opt/rsi/guard",
            "RSI_AUDIT_PIPE": "/opt/rsi/chan/audit.fifo", "RSI_GUARD_ROLE": f"score-{which}",
            "RSI_GUARD_DENY": ":".join(self._deny_list("score")),
            "HF_HOME": "/work/hf", "HF_HUB_CACHE": "/work/hf/hub",
            "TORCHINDUCTOR_CACHE_DIR": f"{tmp_in}/inductor", "TRITON_CACHE_DIR": f"{tmp_in}/triton",
        })
        argv = [cfg.train_python, "-u", "/opt/rsi/scorer/scorer.py", "score",
                "--rows-file", "/in/eval/rows.pt", "--token-bytes", "/in/eval/token_bytes.pt",
                "--code", "/in/code", "--export", "/in/export", "--batch", str(cfg.score_batch),
                "--device", cfg.score_device, "--seed", str(seed), "--probe-tol", str(cfg.probe_tol)]
        if cfg.score_no_autocast:
            argv.append("--no-autocast")
        spec = SandboxSpec(argv=argv, env=env, cwd="/in/code", mounts=mounts, gpu=gpu)
        logp = os.path.join(self._jdir(job["id"]), f"score_{which}_s{seed}.log")
        t0 = time.time()
        with open(logp, "wb") as logf:
            proc = self.launcher.popen(spec, stdout=logf, stderr=subprocess.STDOUT)
            r = run_limited(proc, cfg.score_timeout_s, t0=t0)
        msgs = fifo.stop()
        gf, _ = audit.guard_flags(msgs, where="score")
        results = [m for m in msgs if m.get("k") == "result"]
        out = {"wall_s": round(r["wall_s"], 2), "flags": list(gf)}
        if len(results) > 1:
            out["flags"].append({"check": "result_channel_tamper", "severity": "hard",
                                 "detail": f"{len(results)} result messages on the scorer channel"})
        if r["timed_out"]:
            out.update(status="crash", bpb=None, error="scoring timed out",
                       error_visible=f"scoring timed out after {cfg.score_timeout_s:.0f}s")
            return out
        if not results:
            out.update(status="crash", bpb=None, error=f"scorer produced no result (rc={r['rc']})",
                       error_visible="scoring crashed (no result)")
            return out
        m = results[-1]
        for f in m.get("flags") or []:
            if isinstance(f, dict) and "check" in f:
                out["flags"].append({"check": str(f["check"])[:40], "detail": str(f.get("detail", ""))[:300],
                                     "severity": "hard" if f.get("severity") == "hard" else "soft"})
        status = m.get("status") if m.get("status") in ("ok", "nan", "crash") else "crash"
        bpb = m.get("bpb")
        if status == "ok" and not isinstance(bpb, (int, float)):
            status = "crash"
        err = m.get("error")
        vis = None
        if err:
            vis = str(err)[:300] if str(err).startswith("contract:") else "scoring raised " + str(err).split(":", 1)[0][:60]
        elif status == "nan":
            vis = "non-finite loss on held-out data"
        out.update(status=status, bpb=float(bpb) if status == "ok" else None, error=err, error_visible=vis,
                   n_params=m.get("n_params"), nonparam_bytes=m.get("nonparam_bytes"), sd_bytes=m.get("sd_bytes"),
                   attn_backend=m.get("attn_backend"), probes=m.get("probes"))
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
            path = os.path.join(cfg.scratch_dir, "cache",
                                f"{split}_T{cfg.seq_len}_R{cfg.eval_rows}_{self._fingerprint(split)}.pt")
            if os.path.isfile(path):
                return path
            pd = os.path.join(cfg.scratch_dir, "cache", f"prep_{split}_{uuid.uuid4().hex[:6]}")
            os.makedirs(os.path.join(pd, "out"), exist_ok=True)
            fifo = FifoReader(os.path.join(pd, "audit.fifo")).start()
            mounts = self._env_mount() + [
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
            env = self._env_common(None, tmp_in)
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
