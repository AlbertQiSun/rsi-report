"""Overnight campaign scheduler (worker F).

One process, one loop. Every tick it:
  1. applies the human override file and checks STOP / PAUSE files;
  2. runs health checks when due (services, GPU memory, disk) and handles outages;
  3. advances every running step (state machines below) and starts every pending step
     whose trigger fired: ``after`` / ``needs`` steps finished, flags, ``not_before`` clock,
     resources (``holds``) free; ``latest_start`` passed -> the step is skipped;
  4. writes the hourly status line and checkpoints ``state.json`` atomically.

Long work runs as runner-wrapped subprocesses in their own session, so a crash of this
process loses nothing: ``--resume`` re-adopts live processes or restarts dead ones with
``--resume``. The step kinds are documented in campaign/overnight.yaml.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

from rsi import events
from rsi.campaign import analysis
from rsi.campaign.config import CODE_ROOT, Clock, deep_merge, expand, fmt_time, run_checks
from rsi.campaign.services import (GatewayLoraLoader, HttpProbe, Ports, clean_slurm_env, disk_free_gb, gpu_memory,
                                   make_eval_client)
from rsi.runs import atomic_write_json, read_json

TERMINAL = {"done", "partial", "failed", "skipped"}
OK_STATES = {"done", "partial"}

# Dry-run replacements for every external command (logical name -> template).
FAKE_COMMANDS = {
    "orchestrator": "{py} -m rsi.campaign.fakes orchestrator --config {config}",
    "lora_train": "{py} -m rsi.campaign.fakes lora {sft_dir} {adapter_out} {gpu} --duration_s {est_s}",
    "gate": "{py} -m rsi.campaign.fakes gate --name {step} --out {out} --checks {checks_json} --duration_s {est_s}",
    "serve": "{py} -m rsi.campaign.fakes serve --name {probe} --load_s {load_s}",
    "restart": "{py} -m rsi.campaign.fakes restart --name {service}",
    "action": "{py} -m rsi.campaign.fakes action {what} --spec {spec} --duration_s {est_s}",
    "final": "{py} -m rsi.campaign.fakes action pack --duration_s 60",
}


class StopCampaign(Exception):
    pass


# ===========================================================================
# process management
# ===========================================================================

class Procs:
    def __init__(self, camp: "Campaign"):
        self.c = camp
        self.popen: dict[int, subprocess.Popen] = {}

    def launch(self, argv: list[str], *, log: str, result: str, env: dict | None = None) -> dict:
        os.makedirs(os.path.dirname(result), exist_ok=True)
        if os.path.exists(result):
            os.replace(result, result + f".old{int(time.time())}")
        full = [sys.executable, "-m", "rsi.campaign.runner", "--result", result, "--log", log, "--"] + argv
        p = subprocess.Popen(full, cwd=CODE_ROOT, env=env or self.c.child_env(), start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.popen[p.pid] = p
        return {"pid": p.pid, "result": result, "log": log, "t_launch": self.c.now(), "argv": argv}

    def alive(self, rec: dict) -> bool:
        pid = rec.get("pid")
        if not pid:
            return False
        p = self.popen.get(pid)
        if p is not None:
            return p.poll() is None
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        try:  # make sure the pid is still our runner (pid reuse after a reboot / long gap)
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=5).stdout
            return rec["result"] in cmd
        except (OSError, subprocess.SubprocessError):
            return True

    def poll(self, rec: dict) -> tuple[str, dict | None]:
        """-> ("running", None) | ("done", result) | ("lost", None)"""
        if self.alive(rec):
            return "running", None
        res = read_json(rec["result"])
        if res is not None:
            return "done", res
        return "lost", None

    def kill(self, rec: dict | None, sig: int = signal.SIGTERM) -> None:
        if not rec or not rec.get("pid"):
            return
        try:
            os.killpg(rec["pid"], sig)
        except OSError:
            try:
                os.kill(rec["pid"], sig)
            except OSError:
                pass


# ===========================================================================
# steps
# ===========================================================================

class Step:
    def __init__(self, camp: "Campaign", cfg: dict):
        self.c = camp
        self.cfg = cfg
        self.id = cfg["id"]

    # ---- state ----------------------------------------------------------
    @property
    def st(self) -> dict:
        return self.c.state["steps"].setdefault(self.id, {"status": "pending", "restarts": 0})

    @property
    def status(self) -> str:
        return self.st["status"]

    def set(self, status: str, reason: str | None = None) -> None:
        st = self.st
        prev = st["status"]
        st["status"] = status
        if reason:
            st["reason"] = reason
        if status == "running" and prev != "running":
            st.setdefault("t_start", self.c.now())
        if status in TERMINAL:
            st["t_end"] = self.c.now()
        self.c.dirty = True
        if prev != status:
            self.c.log(f"step {self.id}: {prev} -> {status}" + (f" ({reason})" if reason else ""))
            self.c.msg(self.id, f"{status}" + (f": {reason}" if reason else ""), campaign_ev=f"step_{status}",
                       phase=self.cfg.get("phase"))
            if status == "failed":
                self.c.error(self.id, reason or "failed")
            if status in TERMINAL:
                self.c.apply_step_flags(self)

    @property
    def est_s(self) -> float:
        return float(self.cfg.get("est_s") or 600)

    def run_dirs(self) -> list[str]:
        return []

    def outputs(self) -> dict:
        return self.st.get("out") or {}

    # ---- lifecycle (overridden) ----------------------------------------
    def start(self) -> None:
        raise NotImplementedError

    def poll(self) -> None:
        raise NotImplementedError

    def request_stop(self, why: str, now_kill: bool = False) -> None:
        self.st["stop_requested"] = self.c.now()
        self.st["stop_reason"] = why
        for rec in self.procs():
            self.c.procs.kill(rec, signal.SIGKILL if now_kill else signal.SIGTERM)

    def procs(self) -> list[dict]:
        p = self.st.get("proc")
        return [p] if p else []

    def recover(self) -> None:
        """Called once after --resume for a step that was running/interrupted."""
        self.st["status"] = "running"

    def progress(self) -> str:
        return ""

    def deadline_hit(self) -> bool:
        d = self.cfg.get("_deadline")
        return d is not None and self.c.now() >= d

    # ---- helpers --------------------------------------------------------
    def step_dir(self, *parts: str) -> str:
        return os.path.join(self.c.dir, "steps", self.id, *parts)

    def cmd(self, key_or_template: str, **extra: Any) -> list[str]:
        return self.c.build_cmd(key_or_template, step=self.id, est_s=self.est_s, **extra)


# ---------------------------------------------------------------------------
class ActionStep(Step):
    """cmd / lessons / sft / endpoint: one subprocess; ``checks`` turn it into a gate."""

    def argv(self) -> list[str]:
        k = self.cfg["kind"]
        out = self.out_path()
        if k == "cmd":
            checks = json.dumps(self.cfg.get("checks") or {})
            if self.c.dry:
                return self.c.build_cmd("gate", step=self.id, out=out, checks_json=checks, est_s=self.est_s)
            return self.c.build_cmd(self.cfg["cmd"], step=self.id, out=out, est_s=self.est_s,
                                    model=self.cfg.get("model", "improver"))
        spec = self.spec()
        sp = self.step_dir("spec.json")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        atomic_write_json(sp, spec)
        if self.c.dry:
            return self.c.build_cmd("action", what=k, spec=sp, est_s=self.est_s)
        return [self.c.ctx["py"], "-m", "rsi.campaign.actions", k, "--spec", sp]

    def out_path(self) -> str:
        if self.cfg.get("cache"):
            return self.c.expand(self.cfg["cache"])
        return self.step_dir("out.json")

    def spec(self) -> dict:
        k = self.cfg["kind"]
        srcs = []
        for sid in self.cfg.get("sources", []):
            src = self.c.steps[sid]
            srcs += src.done_dirs() if isinstance(src, MuarArmStep) else src.run_dirs()
        if k == "lessons":
            match = self.cfg.get("match_words_of")
            return {"run_dirs": srcs, "model": self.cfg.get("model", "improver"),
                    "gateway_url": self.c.expand(self.c.cfg.get("gateway_url", "")),
                    "out_dir": self.c.expand(self.cfg.get("out_dir") or os.path.join(self.c.dir, "scaffolds", self.id)),
                    "generic_placebo": bool(self.cfg.get("generic_placebo")),
                    "match_words_of": self.c.steps[match].scaffold_dir() if match else None,
                    "run": self.c.camp_run, "gen": int(self.cfg.get("gen", 1)), "token_path": self.c.ctx.get("token")}
        if k == "sft":
            return {"run_dirs": srcs, "accepted_only": bool(self.cfg.get("accepted_only", True)),
                    "placebo": bool(self.cfg.get("placebo")),
                    "out": self.c.expand(self.cfg.get("out") or os.path.join(self.c.dir, "sft", self.id))}
        if k == "selfdesign":
            def dirs(sid):
                st = self.c.steps[sid]
                return st.done_dirs() if isinstance(st, MuarArmStep) else st.run_dirs()
            prev = []
            for p in self.cfg.get("previous") or []:
                d_out = self.c.steps[p["design"]].outputs() or {}
                s_out = self.c.steps[p["sft"]].outputs() or {}
                prev.append({"recipe_path": d_out.get("recipe_path"),
                             "manifest_path": os.path.join(s_out.get("out") or "", "manifest.json"),
                             "feedback": {"this": dirs(p["fb_this"]), "m0": dirs(p["fb_m0"]), "m1": dirs(p["fb_m1"])}})
            return {"run_dirs": srcs, "model": self.cfg.get("model", "improver"),
                    "gateway_url": self.c.expand(self.c.cfg.get("gateway_url", "")),
                    "token_path": self.c.ctx.get("token"), "run": self.c.camp_run, "gen": int(self.cfg.get("gen", 1)),
                    "round": int(self.cfg.get("round", 1)), "previous": prev,
                    "sref_path": self.c.expand(self.c.cfg["muar"].get("sref", "")),
                    "out_dir": self.c.expand(self.cfg.get("out_dir") or os.path.join(self.c.dir, "selfdesign", self.id))}
        if k == "sft_recipe":
            d_out = self.c.steps[self.cfg["recipe_from"]].outputs() or {}
            match = self.cfg.get("match_n_of")
            return {"run_dirs": srcs, "recipe_path": d_out.get("recipe_path"),
                    "random_placebo": bool(self.cfg.get("random_placebo")), "salt": self.cfg.get("salt", self.id),
                    "match_manifest": os.path.join((self.c.steps[match].outputs() or {}).get("out") or "",
                                                   "manifest.json") if match else None,
                    "out": self.c.expand(self.cfg.get("out") or os.path.join(self.c.dir, "sft", self.id))}
        if k == "endpoint":
            return {"run_dirs": srcs, "top_k": int(self.cfg.get("top_k", 3)), "seeds": self.cfg.get("seeds", []),
                    "include_start": self.c.expand(self.cfg.get("include_start") or ""),
                    "run": self.c.run_id(self.id), "runs_root": self.c.ctx["runs_root"],
                    "evald_url": self.c.expand(self.c.cfg.get("evald_url", "")),
                    "token_path": self.c.ctx.get("token"), "task": self.cfg.get("task", "nanochat")}
        return {}

    def start(self) -> None:
        if self.cfg["kind"] == "cmd" and self.cfg.get("cache"):
            cached = read_json(self.out_path())
            if isinstance(cached, dict):
                ok, failed = run_checks(cached, self.cfg.get("checks") or {})
                if ok:
                    self.st["out"] = cached
                    self.set("done", "cached gate result passes")
                    return
        self.st["proc"] = self.c.procs.launch(self.argv(), log=self.step_dir("log.txt"),
                                              result=self.step_dir("result.json"))
        self.set("running")

    def recover(self) -> None:
        super().recover()
        if not self.st.get("proc") or self.c.procs.poll(self.st["proc"])[0] == "lost":
            self.st["proc"] = None
            self.start()

    def poll(self) -> None:
        if self.status == "waiting_human":
            self.c.poll_human_gate(self)
            return
        rec = self.st.get("proc")
        if rec is None:
            return
        state, res = self.c.procs.poll(rec)
        if state != "running" and self.st.get("stop_requested") is not None:
            self.st["proc"] = None
            self.set("failed", self.st.get("stop_reason"))
            return
        if state == "running":
            if self.deadline_hit() or self.c.now() - rec["t_launch"] > float(self.cfg.get("timeout_s", 4 * self.est_s + 1800)):
                self.c.procs.kill(rec, signal.SIGKILL)
                self.finish(False, {}, ["timeout"])
            return
        if state == "lost":
            res = {"rc": -1, "out": None}
        out = (res or {}).get("out") or {}
        if self.cfg["kind"] == "cmd" and not out:
            out = read_json(self.out_path()) or {}
        self.st["out"] = out
        if res.get("rc") != 0:
            tries = self.st.get("tries", 0)
            if tries < int(self.cfg.get("retries", 1)):
                self.st["tries"] = tries + 1
                self.c.log(f"{self.id}: rc={res.get('rc')}, retry {tries + 1}")
                self.st["proc"] = self.c.procs.launch(self.argv(), log=self.step_dir("log.txt"),
                                                      result=self.step_dir("result.json"))
                return
            self.finish(False, out, [f"rc={res.get('rc')}"])
            return
        checks = dict(self.cfg.get("checks") or {})
        if self.cfg["kind"] in ("sft", "sft_recipe") and self.cfg.get("min_examples"):
            checks.setdefault("n_examples", f">= {self.cfg['min_examples']}")
            checks.setdefault("heldout_overlap", "<= 0")
            out.setdefault("heldout_overlap", 0)
        ok, failed = run_checks(out, checks)
        self.finish(ok, out, failed)

    def finish(self, ok: bool, out: dict, failed: list[str]) -> None:
        self.st["proc"] = None
        self.st["out"] = out
        if ok:
            if self.cfg["kind"] == "lessons":
                self.c.ev("lessons_update", agent="lessons-writer", gen=int(self.cfg.get("gen", 1)),
                          scaffold_hash_before=self.cfg.get("base_scaffold", "sigma0"),
                          scaffold_hash_after=out.get("scaffold_hash"), n_lessons=out.get("n_lessons"),
                          step=self.id, placebo=bool(self.cfg.get("generic_placebo")))
            self.set("done")
        else:
            self.c.gate_failed(self, out, failed)

    def scaffold_dir(self) -> str | None:
        return (self.outputs() or {}).get("out_dir") or self.spec().get("out_dir")

    def progress(self) -> str:
        return ""


# ---------------------------------------------------------------------------
class RunStep(Step):
    """One orchestrator run (``python -m rsi.orchestrator --config ...``)."""

    @property
    def run(self) -> str:
        return self.c.run_id(self.id)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.c.ctx["runs_root"], self.run)

    def run_dirs(self) -> list[str]:
        return [self.run_dir]

    def stats(self) -> dict:
        return analysis.run_stats(os.path.join(self.run_dir, "events.jsonl"))

    def build_config(self) -> str:
        r = deep_merge(self.c.cfg.get("run_defaults") or {}, self.cfg.get("run") or {})
        for ov in self.cfg.get("overlays") or []:
            if self.c.flag(ov["when"]):
                r = deep_merge(r, ov.get("set") or {})
        r = self.c.resolve_refs(r, self)
        r["run"] = self.run
        r["runs_root"] = self.c.ctx["runs_root"]
        r.setdefault("gen", int(self.cfg.get("gen", 0)))
        if self.c.dry:
            r["_fake"] = deep_merge({"trial_s": float(self.cfg.get("est_trial_s", 570))}, self.cfg.get("fake") or {})
        path = os.path.join(self.c.dir, "configs", f"{self.run}.yaml")
        self.c.write_config(path, r)
        self.st["config"] = path
        return path

    def launch(self, resume: bool) -> None:
        path = self.st.get("config") if resume and self.st.get("config") else self.build_config()
        argv = self.c.build_cmd("orchestrator", config=path)
        if resume or os.path.exists(os.path.join(self.run_dir, "events.jsonl")):
            argv.append("--resume")
        self.st["proc"] = self.c.procs.launch(argv, log=self.step_dir("log.txt"), result=self.step_dir("result.json"))
        self.st["last_progress"] = self.c.now()

    def start(self) -> None:
        self.launch(resume=False)
        self.set("running")

    def recover(self) -> None:
        super().recover()
        if self.st.get("proc") and self.c.procs.poll(self.st["proc"])[0] == "lost":
            self.st["proc"] = None
            self.st["relaunch_at"] = self.c.now()

    def pause_file(self) -> str:
        return os.path.join(self.run_dir, "PAUSE")

    def write_pause(self, owner: str) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.pause_file(), "w") as f:
            f.write(f"campaign:{owner}\n")
        self.st["pause_owner"] = owner

    def clear_pause(self, owner: str) -> None:
        if self.st.get("pause_owner") == owner:
            try:
                os.remove(self.pause_file())
            except OSError:
                pass
            self.st.pop("pause_owner", None)

    def human_paused(self) -> bool:
        """A PAUSE file written by a person (or ``orchestrator --pause``), not by us or by the
        orchestrator's own SIGTERM handler."""
        try:
            text = open(self.pause_file()).read()
        except OSError:
            return False
        return not (text.startswith("campaign:") or text.startswith("signal"))

    def request_stop(self, why: str, now_kill: bool = False) -> None:
        if self.st.get("stop_requested") is None:
            self.st["stop_requested"] = self.c.now()
            self.st["stop_reason"] = why
            self.write_pause("stop")
            self.c.log(f"{self.id}: stop requested ({why})")
        if now_kill:
            self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)

    def poll(self) -> None:
        s = self.stats()
        rec = self.st.get("proc")
        now = self.c.now()
        if s.get("mtime"):
            self.st["last_progress"] = max(self.st.get("last_progress", 0), self.c.real_to_virtual(s["mtime"]))
        stopping = self.st.get("stop_requested") is not None
        # trial-count trigger and deadline
        if not stopping:
            limit = self.cfg.get("stop_when_trials")
            if limit and s["n_trials"] >= int(limit):
                self.request_stop(f"trial budget reached ({s['n_trials']}/{limit})")
            elif self.deadline_hit():
                self.request_stop("deadline")
            else:
                ab = self.cfg.get("abort_if")
                if ab and s["n_trials"] >= int(ab.get("min_trials", 10)) and (s.get("ok_rate") or 0) < float(ab["ok_rate_lt"]):
                    self.c.error(self.id, f"abort: ok_rate {s.get('ok_rate'):.2f} after {s['n_trials']} trials")
                    self.st["aborted"] = True
                    self.request_stop("abort_if")
        if rec is None:
            # waiting to relaunch (after crash / health pause / human pause)
            if stopping:
                self.set("partial" if s["n_trials"] else "failed", self.st.get("stop_reason"))
                return
            if self.c.outage or self.human_paused() or self.c.paused:
                return
            if now >= self.st.get("relaunch_at", 0):
                self.launch(resume=True)
            return
        state, res = self.c.procs.poll(rec)
        if state == "running":
            if stopping:
                grace = float(self.c.cfg.get("stop_grace_s", 900))
                waited = now - self.st["stop_requested"]
                if waited > 2 * grace:
                    self.c.procs.kill(rec, signal.SIGKILL)
                elif waited > grace and not self.st.get("sigterm_sent"):
                    self.st["sigterm_sent"] = True
                    self.c.procs.kill(rec, signal.SIGTERM)
            elif not self.c.outage and not self.c.paused and not self.human_paused():
                stall = float(self.cfg.get("stall_s", 2400))
                if now - self.st.get("last_progress", now) > stall:
                    self.c.error(self.id, f"stalled {int(now - self.st['last_progress'])}s without events; restarting")
                    self.c.procs.kill(rec, signal.SIGTERM)
                    self.st["stall_kill"] = True
            return
        # process ended
        self.st["proc"] = None
        rc = (res or {}).get("rc", -1)
        s = analysis.run_stats(os.path.join(self.run_dir, "events.jsonl"))
        if s.get("run_end") is not None and rc == 0:
            self.set("done", f"run_end, {s['n_trials']} trials")
            return
        if stopping:
            self.set("partial" if s["n_trials"] else "failed", self.st.get("stop_reason"))
            return
        if rc == 0 and (self.c.outage or self.c.paused or self.human_paused()):
            return  # it paused at a trial boundary because we (or a human) asked; relaunch later
        # crash, lost or stall
        n = self.st.get("restarts", 0)
        if n >= int(self.cfg.get("max_restarts", self.c.cfg.get("max_restarts", 3))):
            self.set("failed" if not s["n_trials"] else "partial", f"gave up after {n} restarts (rc={rc})")
            return
        self.st["restarts"] = n + 1
        backoff = float(self.c.cfg.get("restart_backoff_s", 120)) * (n + 1)
        self.st["relaunch_at"] = now + backoff
        self.st.pop("stall_kill", None)
        self.c.error(self.id, f"run exited rc={rc} without run_end; restart {n + 1} with --resume in {int(backoff)}s")
        self.c.msg(self.id, f"restart {n + 1}", campaign_ev="restart")

    def health_pause(self, on: bool) -> None:
        if on:
            if not self.st.get("pause_owner"):
                self.write_pause("health")
        else:
            self.clear_pause("health")

    def progress(self) -> str:
        s = self.stats()
        lim = self.cfg.get("stop_when_trials") or (self.cfg.get("run") or {}).get("budget", {}).get("trials")
        best = f" best={s['best']:.4f}" if isinstance(s.get("best"), (int, float)) else ""
        return f"{s['n_trials']}/{lim or '?'}{best}"


# ---------------------------------------------------------------------------
class MuarArmStep(Step):
    """One µAR arm = one orchestrator run per (task, seed), on the shared CPU worker pool."""

    def subs(self) -> dict[str, dict]:
        st = self.st
        if "subs" not in st:
            mu = self.c.cfg["muar"]
            splits = self.cfg.get("splits") or ["train"]
            if isinstance(splits, str):
                splits = [splits]
            names = []
            for sp in splits:
                for s2 in (mu["heldout"] if sp == "heldout" else [sp]):
                    split_tasks = list(mu["tasks"][s2])
                    idx = self.cfg.get("task_index") or list(range(len(split_tasks)))
                    if self.c.dry and (self.c.cfg.get("dry_run") or {}).get("muar_tasks_per_split"):
                        idx = idx[: int(self.c.cfg["dry_run"]["muar_tasks_per_split"])]
                    names += [split_tasks[i] for i in idx if i < len(split_tasks)]
            seeds = self.cfg["seeds"]
            if isinstance(seeds, str):
                seeds = mu["seeds"][seeds]
            dr = self.c.cfg.get("dry_run") or {}
            if self.c.dry and dr.get("muar_seeds"):  # dry-run only: smaller fan-out, same arms and triggers
                seeds = list(seeds)[: int(dr["muar_seeds"])]
            st["subs"] = {}
            for t in names:
                for sd in seeds:
                    slug = t.replace("muar/", "").replace("/", "-").replace("_", "")
                    run = f"{self.c.run_id(self.id)}-{slug}-s{sd}"
                    st["subs"][run] = {"run": run, "task": t, "seed": int(sd), "status": "pending", "tries": 0}
        return st["subs"]

    def run_dirs(self) -> list[str]:
        return [os.path.join(self.c.ctx["runs_root"], r) for r in self.subs()]

    def done_dirs(self) -> list[str]:
        return [os.path.join(self.c.ctx["runs_root"], r) for r, s in self.subs().items() if s["status"] == "done"]

    def sub_config(self, sub: dict) -> str:
        mu = self.c.cfg["muar"]
        N = int(self.cfg.get("N", mu["N"]))
        r = {"task": sub["task"], "policy": self.cfg.get("policy", mu.get("policy", "karpathy_greedy")),
             "policy_cfg": self.cfg.get("policy_cfg") or {}, "model": self.cfg.get("model", "improver"),
             "scaffold_dir": self.cfg.get("scaffold"), "budget": {"trials": N, "wall_s": N * float(mu["est_step_s"]) * 20},
             "rng_seed": sub["seed"], "task_seed": sub["seed"], "seeds": [sub["seed"]], "lanes": [],
             "gen": int(self.cfg.get("gen", 0)), "run": sub["run"], "runs_root": self.c.ctx["runs_root"]}
        r = deep_merge(mu.get("run_extra") or {}, r)   # e.g. evald factory = LocalEvalClient (CPU scoring)
        r = self.c.resolve_refs(r, self)
        if self.c.dry:
            r["_fake"] = deep_merge({"trial_s": float(mu["est_step_s"])}, self.cfg.get("fake") or {})
        path = os.path.join(self.c.dir, "configs", f"{sub['run']}.yaml")
        self.c.write_config(path, r)
        return path

    def start(self) -> None:
        self.subs()
        self.set("running")

    def procs(self) -> list[dict]:
        return [s["proc"] for s in self.subs().values() if s.get("proc")]

    def recover(self) -> None:
        super().recover()
        for sub in self.subs().values():
            if sub["status"] == "running" and sub.get("proc"):
                if self.c.procs.poll(sub["proc"])[0] == "lost":
                    sub["status"], sub["proc"] = "pending", None

    def request_stop(self, why: str, now_kill: bool = False) -> None:
        self.st["stop_requested"] = self.c.now()
        self.st["stop_reason"] = why
        for sub in self.subs().values():
            if sub["status"] == "running":
                self.c.procs.kill(sub.get("proc"), signal.SIGTERM)
            if sub["status"] == "pending":
                sub["status"] = "cancelled"

    def running_count(self) -> int:
        return sum(1 for s in self.subs().values() if s["status"] == "running")

    def poll(self) -> None:
        if self.status == "waiting_human":
            self.c.poll_human_gate(self)
            return
        subs = self.subs()
        if self.deadline_hit() and self.st.get("stop_requested") is None:
            self.request_stop("deadline")
        for sub in subs.values():
            if sub["status"] != "running":
                continue
            state, res = self.c.procs.poll(sub["proc"])
            if state == "running":
                if self.c.now() - sub["proc"]["t_launch"] > float(self.cfg.get("sub_timeout_s", 3600)):
                    self.c.procs.kill(sub["proc"], signal.SIGKILL)
                continue
            s = analysis.run_stats(os.path.join(self.c.ctx["runs_root"], sub["run"], "events.jsonl"))
            sub["proc"] = None
            if (res or {}).get("rc") == 0 and s.get("run_end") is not None:
                sub["status"] = "done"
            elif self.st.get("stop_requested") is not None:
                sub["status"] = "cancelled"
            elif (res or {}).get("rc") == 0 and (self.c.outage or self.c.paused):
                sub["status"] = "pending"
            elif sub["tries"] < int(self.cfg.get("sub_retries", 1)):
                sub["tries"] += 1
                sub["status"] = "pending"
            else:
                sub["status"] = "failed"
                self.c.error(self.id, f"sub-run {sub['run']} failed (rc={(res or {}).get('rc')})")
            self.c.dirty = True
        # launch within the shared CPU pool
        if self.st.get("stop_requested") is None and not self.c.outage and not self.c.paused:
            for sub in subs.values():
                if self.c.muar_free() <= 0 or self.c.muar_blocked_by_earlier(self):
                    break
                if sub["status"] == "pending":
                    path = self.sub_config(sub)
                    argv = self.c.build_cmd(self.c.cfg["muar"].get("command", "orchestrator"), config=path)
                    if os.path.exists(os.path.join(self.c.ctx["runs_root"], sub["run"], "events.jsonl")):
                        argv.append("--resume")
                    sub["proc"] = self.c.procs.launch(argv, log=self.step_dir("logs", sub["run"] + ".txt"),
                                                      result=self.step_dir("results", sub["run"] + ".json"))
                    sub["status"] = "running"
                    self.c.dirty = True
        if any(s["status"] in ("pending", "running") for s in subs.values()):
            return
        n = len(subs)
        n_done = sum(1 for s in subs.values() if s["status"] == "done")
        frac = n_done / max(1, n)
        if self.cfg.get("checks"):
            agg = self.aggregate()
            ok, failed = run_checks(agg, self.cfg["checks"])
            self.st["out"] = agg
            if ok:
                self.set("done", f"gate pass {agg}")
            else:
                self.c.gate_failed(self, agg, failed)
            return
        if frac >= float(self.cfg.get("min_ok_frac", 0.9)):
            self.set("done" if n_done == n else "partial", f"{n_done}/{n} sub-runs")
        elif n_done:
            self.set("partial", f"only {n_done}/{n} sub-runs" + (f" ({self.st['stop_reason']})" if self.st.get("stop_reason") else ""))
        else:
            self.set("failed", f"0/{n} sub-runs")

    def aggregate(self) -> dict:
        n_tr = n_ok = 0
        walls = []
        for d in self.run_dirs():
            s = analysis.run_stats(os.path.join(d, "events.jsonl"))
            n_tr += s["n_trials"]
            n_ok += s["n_ok"]
            if s.get("mean_wall_s") is not None:
                walls.append(s["mean_wall_s"])
        return {"n_trials": n_tr, "ok_rate": (n_ok / n_tr) if n_tr else 0.0,
                "mean_wall_s": (sum(walls) / len(walls)) if walls else None}

    def table(self) -> dict:
        return analysis.arm_table(self.subs(), self.c.ctx["runs_root"])

    def progress(self) -> str:
        subs = self.subs() if "subs" in self.st else {}
        n_done = sum(1 for s in subs.values() if s["status"] == "done")
        vals = [v for t in self.table().values() for v in t.values()] if n_done else []
        m = f" nAUC={sum(vals) / len(vals):.3f}" if vals else ""
        return f"{n_done}/{len(subs)} runs{m}"


# ---------------------------------------------------------------------------
class LoraStep(Step):
    """Pause an eval lane, train adapters on its GPU, load them into vLLM, resume the lane."""

    def lane(self) -> int | None:
        """None (``lane: null``) = training on a separate node; no eval lane is paused."""
        if "lane" in self.cfg and self.cfg["lane"] is None:
            return None
        return int(self.cfg.get("lane", self.c.cfg.get("lora_lane", 2)))

    def gpu(self) -> int | str:
        if self.lane() is None:
            return self.cfg.get("gpu", "slurm")
        try:
            for l in self.c.evald.lanes():
                if int(l.get("lane")) == self.lane() and l.get("gpu") is not None:
                    return int(l["gpu"])
        except Exception:  # noqa: BLE001
            pass
        return int((self.c.cfg.get("lane_gpus") or {}).get(self.lane(), self.lane()))

    def start(self) -> None:
        self.st.update({"stage": "pause" if self.lane() is not None else "train", "job": 0, "adapters": {},
                        "tries": {}})
        self.set("running")
        self.poll()

    def recover(self) -> None:
        super().recover()
        if self.st.get("proc") and self.c.procs.poll(self.st["proc"])[0] == "lost":
            self.st["proc"] = None  # the train stage relaunches the current job
        if self.lane() is None:
            if self.st.get("stage") == "stopped":
                self.st["stage"] = "train"
            return
        if self.st.get("stage") == "stopped" or (
                self.st.get("stage") in ("wait_idle", "train") and str(self.lane()) not in self.c.state["paused_lanes"]):
            self.st["stage"] = "pause"  # trained adapters are kept; training continues with the next job

    def request_stop(self, why: str, now_kill: bool = False) -> None:
        self.st["stop_requested"] = self.c.now()
        self.st["stop_reason"] = why
        self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)

    def release_lane(self) -> None:
        lane = self.lane()
        if lane is None:
            return
        if self.c.state["paused_lanes"].get(str(lane)) == self.id:
            try:
                self.c.evald.resume_lane(lane)
                self.c.state["paused_lanes"].pop(str(lane), None)
                self.c.msg(self.id, f"resume lane {lane}", campaign_ev="lane_resume", lane=lane)
            except Exception as e:  # noqa: BLE001
                self.c.error(self.id, f"resume_lane({lane}) failed: {e}")

    def fail(self, why: str) -> None:
        self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)
        self.st["proc"] = None
        self.release_lane()
        self.set("failed", why)

    def poll(self) -> None:
        now = self.c.now()
        lane = self.lane()
        jobs = self.cfg["jobs"]
        if self.st.get("stop_requested") is not None and self.st["stage"] != "resume":
            self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)
            self.st["proc"] = None
            self.release_lane()
            self.st["stage"] = "stopped"
            self.set("partial" if self.st["adapters"] else "failed", self.st.get("stop_reason"))
            return
        stage = self.st["stage"]
        if stage == "pause":
            try:
                self.c.evald.pause_lane(lane)
            except Exception as e:  # noqa: BLE001
                self.set("failed", f"pause_lane({lane}) failed: {e}")
                return
            self.c.state["paused_lanes"][str(lane)] = self.id
            self.c.msg(self.id, f"pause lane {lane} for LoRA", campaign_ev="lane_pause", lane=lane)
            self.st.update({"stage": "wait_idle", "t_pause": now})
            stage = "wait_idle"
        if stage == "wait_idle":
            try:
                busy = any(l.get("busy") for l in self.c.evald.lanes() if int(l.get("lane")) == lane)
            except Exception as e:  # noqa: BLE001
                busy = True
                self.c.log(f"{self.id}: lanes() failed: {e}")
            if busy:
                if now - self.st["t_pause"] > float(self.cfg.get("idle_timeout_s", 1800)):
                    self.fail(f"lane {lane} still busy after pause")
                return
            self.st.update({"stage": "train", "t_idle": now})
            stage = "train"
        if stage == "train":
            if self.st["job"] >= len(jobs):
                self.st["stage"] = "load"
                stage = "load"
            else:
                job = jobs[self.st["job"]]
                name = job["name"]
                rec = self.st.get("proc")
                if rec is None:
                    sft = self.c.steps[job["sft"]]
                    sft_dir = (sft.outputs() or {}).get("out") or sft.spec()["out"]
                    out = os.path.join(self.c.ctx["adapters"], name)
                    key = "lora_train" if self.c.dry else self.cfg.get("train_cmd", "lora_train")
                    argv = self.c.build_cmd(key, sft_dir=sft_dir, adapter_out=out, gpu=self.gpu(),
                                            est_s=float(self.cfg.get("est_train_s", 2700)))
                    argv, env = self.c.remote_wrap(self.cfg, argv)
                    if job.get("recipe_from") and not self.c.dry:   # U_sd: LoRA args chosen by the model
                        from rsi.update.selfdesign import lora_args
                        rp = (self.c.steps[job["recipe_from"]].outputs() or {}).get("recipe_path")
                        env = dict(env or self.c.child_env())
                        env["LORA_ARGS"] = lora_args(json.load(open(rp)))
                        self.c.log(f"{self.id}: {name} LORA_ARGS={env['LORA_ARGS']}")
                    self.st["proc"] = self.c.procs.launch(argv, log=self.step_dir(f"train-{name}.txt"),
                                                          result=self.step_dir(f"train-{name}.json"), env=env)
                    self.st["proc"]["adapter_out"] = out
                    self.c.msg(self.id, f"train {name} on gpu {self.gpu()}", campaign_ev="lora_train_start",
                               lane=lane, adapter=name)
                    return
                state, res = self.c.procs.poll(rec)
                if state == "running":
                    if now - rec["t_launch"] > float(self.cfg.get("train_timeout_s", 4500)):
                        self.c.error(self.id, f"training {name} exceeded train_timeout_s")
                        self.c.procs.kill(rec, signal.SIGKILL)
                    return
                self.st["proc"] = None
                path = ((res or {}).get("out") or {}).get("adapter_path") or rec["adapter_out"]
                if (res or {}).get("rc") == 0 and os.path.isdir(path):
                    self.st["adapters"][name] = path
                    self.st["job"] += 1
                    self.c.msg(self.id, f"trained {name}", campaign_ev="lora_train_end", adapter=name)
                else:
                    t = self.st["tries"].get(name, 0)
                    if t < int(self.cfg.get("retries", 1)):
                        self.st["tries"][name] = t + 1
                        self.c.error(self.id, f"training {name} failed (rc={(res or {}).get('rc')}); retry")
                    else:
                        self.fail(f"training {name} failed")
                return
        if stage == "load" and not self.cfg.get("load", True):
            # adapters are served later by a `service` step that preloads them (LORA_MODULES)
            for job in jobs:
                name = job["name"]
                self.c.state.setdefault("remote_adapters", {})[name] = self.st["adapters"][name]
                sft = self.c.steps[job["sft"]]
                self.c.ev("weights_update", agent="campaign", gen=int(self.cfg.get("gen", 1)),
                          adapter_path=self.st["adapters"][name], n_examples=(sft.outputs() or {}).get("n_examples"),
                          base_model=self.cfg.get("base_model", self.c.cfg.get("base_model", "improver")),
                          adapter=name, step=self.id)
            self.st["stage"] = "resume"
            stage = "resume"
        if stage == "load":
            for job in jobs:
                name = job["name"]
                if name in self.c.state["loaded_adapters"]:
                    continue
                path = self.st["adapters"][name]
                ok, why = self.c.loader.load(name, path)
                if not ok or not self.c.loader.verify(name):
                    t = self.st.setdefault("load_tries", 0)
                    if t >= 3:
                        self.fail(f"loading {name} into vLLM failed: {why}")
                    self.st["load_tries"] = t + 1
                    return
                self.c.state["loaded_adapters"][name] = path
                sft = self.c.steps[job["sft"]]
                self.c.ev("weights_update", agent="campaign", gen=int(self.cfg.get("gen", 1)), adapter_path=path,
                          n_examples=(sft.outputs() or {}).get("n_examples"),
                          base_model=self.c.cfg.get("base_model", "improver"), adapter=name, step=self.id)
            self.st["stage"] = "resume"
            stage = "resume"
        if stage == "resume":
            self.release_lane()
            self.set("done", f"adapters {sorted(self.st['adapters'])}")

    def progress(self) -> str:
        return f"{self.st.get('stage')} job {self.st.get('job', 0) + 1}/{len(self.cfg['jobs'])}"


# ---------------------------------------------------------------------------
class ServiceStep(Step):
    """Host a model server on a lane's GPU for a while (fallback for the 27B ladder)."""

    lane = LoraStep.lane
    gpu = LoraStep.gpu
    release_lane = LoraStep.release_lane

    def start(self) -> None:
        self.st.update({"stage": "pause" if self.lane() is not None else "launch", "restarts_svc": 0})
        self.set("running")
        self.poll()

    def lora_modules(self) -> dict[str, str]:
        """{adapter name: path} from the lora steps in ``adapters_from`` (preloaded at vLLM start)."""
        out: dict[str, str] = {}
        for sid in self.cfg.get("adapters_from") or []:
            out.update(self.c.steps[sid].st.get("adapters") or {})
        return out

    def adapters_ready(self) -> bool:
        mods = self.lora_modules()
        if not mods:
            return True
        if self.c.dry:  # the fake loader stands in for "vLLM listed the preloaded adapter"
            return all(self.c.loader.load(n, p)[0] for n, p in mods.items())
        la = self.c.cfg["lora_admin"]
        ld = GatewayLoraLoader(self.c.expand(la["load_url"]), self.c.expand(la["list_url"]), self.c.ctx.get("token"),
                               base=self.cfg.get("served_name", "improver"))
        return all(ld.verify(n) for n in mods)

    def recover(self) -> None:
        super().recover()
        if self.st.get("proc") and self.c.procs.poll(self.st["proc"])[0] == "lost":
            self.st["proc"] = None
            if self.st["stage"] in ("wait_healthy", "serving"):
                self.st["stage"] = "launch"
        if self.lane() is None:
            if self.st.get("stage") in ("stopping", "resume") and self.st.get("stop_reason"):
                self.st.update({"stage": "launch", "proc": None, "fail": None})
            return
        if str(self.lane()) not in self.c.state["paused_lanes"] and self.st.get("stage") != "pause":
            self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)
            self.st.update({"stage": "pause", "proc": None, "fail": None})

    def request_stop(self, why: str, now_kill: bool = False) -> None:
        self.st["stop_requested"] = self.c.now()
        self.st["stop_reason"] = why
        if self.st.get("stage") not in ("stopping", "resume"):
            self.st["stage"] = "stopping"
            self.st["t_stop"] = self.c.now()
            self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)

    def up(self, value: bool) -> None:
        f = self.cfg.get("up_flag")
        if f:
            self.c.state["flags"][f] = value

    def poll(self) -> None:
        now = self.c.now()
        lane = self.lane()
        stage = self.st["stage"]
        if stage == "pause" and lane is None:
            self.st["stage"] = stage = "launch"
        if stage == "pause":
            self.c.evald.pause_lane(lane)
            self.c.state["paused_lanes"][str(lane)] = self.id
            self.c.msg(self.id, f"pause lane {lane} for service", campaign_ev="lane_pause", lane=lane)
            self.st.update({"stage": "wait_idle", "t_pause": now})
            return
        if stage == "wait_idle":
            if any(l.get("busy") for l in self.c.evald.lanes() if int(l.get("lane")) == lane):
                if now - self.st["t_pause"] > float(self.cfg.get("idle_timeout_s", 1800)):
                    self.st["stage"] = "resume"
                    self.st["fail"] = "lane busy"
                return
            self.st["stage"] = "launch"
            stage = "launch"
        if stage == "launch":
            mods = ",".join(f"{n}={os.path.abspath(p)}" for n, p in sorted(self.lora_modules().items()))
            if self.c.dry:
                argv = self.c.build_cmd("serve", probe=self.cfg["probe"], load_s=self.cfg.get("load_s", 600))
            else:
                argv = self.c.build_cmd(self.cfg["cmd"], gpu=self.gpu(), lora_modules=mods or None)
            argv, env = self.c.remote_wrap(self.cfg, argv)
            self.st["proc"] = self.c.procs.launch(argv, log=self.step_dir("serve.txt"), result=self.step_dir("serve.json"),
                                                  env=env)
            self.st.update({"stage": "wait_healthy", "t_launch": now})
            return
        if stage == "wait_healthy":
            ok, _ = self.c.probe.check(self.cfg["probe"], self.c.expand(self.cfg.get("url", "")))
            ok = ok and self.adapters_ready()
            if ok:
                self.up(True)
                self.st["stage"] = "serving"
                self.c.msg(self.id, "service up", campaign_ev="service_up")
            elif now - self.st["t_launch"] > float(self.cfg.get("start_timeout_s", 1800)) or \
                    self.c.procs.poll(self.st["proc"])[0] != "running":
                self.c.procs.kill(self.st.get("proc"), signal.SIGTERM)
                self.st["fail"] = "service did not come up"
                self.st["stage"] = "stopping"
                self.st["t_stop"] = now
            return
        if stage == "serving":
            until = [self.c.steps[s] for s in self.cfg.get("until", [])]
            if (until and all(s.status in TERMINAL for s in until)) or self.deadline_hit():
                self.request_stop("until/deadline")
                return
            if self.c.procs.poll(self.st["proc"])[0] != "running":
                self.up(False)
                if self.st["restarts_svc"] < int(self.cfg.get("max_restarts", 1)):
                    self.st["restarts_svc"] += 1
                    self.c.error(self.id, "service died; restarting")
                    self.st["stage"] = "launch"
                else:
                    self.st["fail"] = "service died"
                    self.st["stage"] = "resume"
            return
        if stage == "stopping":
            self.up(False)
            rec = self.st.get("proc")
            if rec and self.c.procs.poll(rec)[0] == "running":
                if now - self.st["t_stop"] > float(self.cfg.get("stop_grace_s", 300)):
                    self.c.procs.kill(rec, signal.SIGKILL)
                return
            self.st["proc"] = None
            self.st["stage"] = "resume"
            stage = "resume"
        if stage == "resume":
            self.up(False)
            self.release_lane()
            if self.st.get("fail"):
                for f in (self.cfg.get("fail_flags") or {}).items():
                    self.c.state["flags"][f[0]] = f[1]
                self.set("failed", self.st["fail"])
            else:
                self.set("done", self.st.get("stop_reason"))

    def progress(self) -> str:
        return str(self.st.get("stage"))


# ---------------------------------------------------------------------------
class DecisionStep(Step):
    """Apply the pre-registered Δ₂ rule; record it, emit a ``decision`` event, set flags.

    Real mode: worker D's ``rsi.update.analysis`` (s_ref from the sealed sref.json, curves from the
    run ledgers) in a subprocess. Dry-run: the in-process implementation in rsi.campaign.analysis
    on the fake runs' ``run_end.nAUC``. Arm references are step ids, or ``<cmd step>:<arm>`` for
    an ``arms`` block in that step's output file (the µAR runner's non-LLM baselines)."""

    def refs(self) -> list[str]:
        cfg = self.cfg
        out = [cfg["arm"], cfg["base"]] + ([cfg["placebo"]] if cfg.get("placebo") else []) + list(cfg.get("aa") or [])
        for ex in cfg.get("also") or []:
            out += [ex["arm"], ex["base"]]
        return list(dict.fromkeys(out))

    def start(self) -> None:
        self.set("running")
        if not self.c.dry:
            spec = self.analysis_spec()
            sp = self.step_dir("spec.json")
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            atomic_write_json(sp, spec)
            argv = [self.c.ctx["py"], "-m", "rsi.campaign.actions", "decision", "--spec", sp]
            self.st["proc"] = self.c.procs.launch(argv, log=self.step_dir("log.txt"), result=self.step_dir("result.json"))
            return
        self.poll()

    def recover(self) -> None:
        super().recover()
        if not self.c.dry and (not self.st.get("proc") or self.c.procs.poll(self.st["proc"])[0] == "lost"):
            self.st["proc"] = None
            self.st["status"] = "pending"

    def analysis_spec(self) -> dict:
        cfg = self.cfg
        runs, files = {}, {}
        for ref in self.refs():
            if ":" in ref:
                sid, key = ref.split(":", 1)
                files[ref] = [self.c.steps[sid].out_path(), key]
            else:
                s = self.c.steps[ref]
                if isinstance(s, MuarArmStep) and s.status in OK_STATES:
                    runs[ref] = s.done_dirs()
        compare = [[cfg["arm"], cfg["base"]] + ([cfg["placebo"]] if cfg.get("placebo") else [])]
        compare += [[ex["arm"], ex["base"]] for ex in cfg.get("also") or []]
        labels = {cfg["arm"]: cfg["arm"]}
        return {"runs": runs, "arms_files": files, "aa": cfg.get("aa"), "compare": compare,
                "names": [ex["name"] for ex in cfg.get("also") or []], "labels": labels,
                "sref_path": self.c.expand(self.c.cfg["muar"].get("sref", "")),
                "n_trials": int(self.c.cfg["muar"]["N"]),
                "n_boot": int(self.c.cfg.get("analysis", {}).get("n_boot", 10000)),
                "floor": self.c.cfg.get("analysis", {}).get("floor", "per_task")}

    def table(self, ref: str | None) -> dict | None:
        if not ref or ":" in ref:
            return None
        s = self.c.steps[ref]
        if s.status not in OK_STATES or not isinstance(s, MuarArmStep):
            return None
        return s.table()

    def compute_inprocess(self) -> dict:
        cfg = self.cfg
        n_boot = int(self.c.cfg.get("analysis", {}).get("n_boot", 2000))
        if self.c.dry:
            n_boot = min(n_boot, int((self.c.cfg.get("dry_run") or {}).get("n_boot", 300)))
        arm, base = self.table(cfg["arm"]), self.table(cfg["base"])
        aa = None
        if cfg.get("aa"):
            a0, a1 = self.table(cfg["aa"][0]), self.table(cfg["aa"][1])
            if a0 and a1:
                aa = (a0, a1)
        res: dict[str, Any] = {}
        if not arm or not base:
            res.update({"pass": False, "harmful": False, "reason": "missing arm data", "nodata": True})
        else:
            res.update(analysis.delta2_decision(arm, base, aa=aa, placebo=self.table(cfg.get("placebo")), n_boot=n_boot))
        extra = []
        for ex in cfg.get("also") or []:
            a, b = self.table(ex["arm"]), self.table(ex["base"])
            if a and b:
                extra.append({"name": ex["name"], **analysis.bootstrap(a, b, n_boot=n_boot, seed=4)})
        res["also"] = extra
        return res

    def poll(self) -> None:
        if self.status == "waiting_human":
            self.c.poll_human_gate(self)
            return
        cfg = self.cfg
        if self.c.dry:
            res = self.compute_inprocess()
        else:
            rec = self.st.get("proc")
            if rec is None:
                return
            state, out = self.c.procs.poll(rec)
            if state == "running":
                return
            self.st["proc"] = None
            res = (out or {}).get("out") or {}
            if (out or {}).get("rc") != 0 or not res:
                res = {"pass": False, "harmful": False, "nodata": True,
                       "reason": f"analysis failed (rc={(out or {}).get('rc')}); see {self.step_dir('log.txt')}"}
        res["label"] = cfg.get("label", self.id)
        if cfg.get("label_if_flag") and self.c.flag(cfg["label_if_flag"][0]):
            res["label"] += " " + cfg["label_if_flag"][1]
        self.st["out"] = res
        self.c.state["decisions"][self.id] = res
        atomic_write_json(os.path.join(self.c.dir, "decisions.json"), self.c.state["decisions"])
        self.c.update_pairs()
        m = res.get("main") or {}
        self.c.ev("decision", agent="campaign", gen=int(cfg.get("gen", 1)), trial=None,
                  artifact_hash=str(cfg.get("artifact", cfg["arm"])),
                  verdict="adopt" if res.get("pass") else "reject", best_before=m.get("mean_b"),
                  best_after=m.get("mean_a"), reason=f"{res['label']}: {res.get('reason')}"[:300],
                  delta2=m.get("delta"), ci=m.get("ci"), p=m.get("p_signflip"), step=self.id)
        for flag, key in (cfg.get("set_flags") or {}).items():
            self.c.state["flags"][flag] = bool(res.get(key))
        self.c.status_line(event=f"DECISION {res['label']}: {analysis.fmt_delta(res)} — {str(res.get('reason'))[:300]}")
        if cfg.get("pair_with") or any(s.cfg.get("pair_with") == self.id for s in self.c.steps.values()):
            pt = self.c.pairs_text()
            if pt:
                self.c.status_line(event=f"Δ₂ 9B vs 27B — {pt}")
        req = cfg.get("require")
        if req and not res.get("nodata"):
            ok, failed = run_checks({"delta": m.get("delta")}, {"delta": req})
            if not ok:
                self.c.gate_failed(self, res, failed)
                return
        self.set("done", analysis.fmt_delta(res))


# ===========================================================================
# the campaign
# ===========================================================================

KINDS = {"cmd": ActionStep, "lessons": ActionStep, "sft": ActionStep, "endpoint": ActionStep, "run": RunStep,
         "selfdesign": ActionStep, "sft_recipe": ActionStep,
         "muar_arm": MuarArmStep, "lora": LoraStep, "service": ServiceStep, "decision": DecisionStep}


class Campaign:
    def __init__(self, cfg: dict, *, dry_run: bool = False, time_scale: float | None = None,
                 until: float | None = None, tick_s: float | None = None, quiet: bool = False):
        self.cfg = cfg
        self.ctx = dict(cfg["_ctx"])
        self.dry = dry_run
        self.quiet = quiet
        self.dir = self.ctx["campaign_dir"]
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.ctx["runs_root"], exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.tz = cfg["_tz"]
        self.until = until if until is not None else cfg["_gpu_stop"]
        self.tick_s = tick_s if tick_s is not None else (0.05 if dry_run else float(cfg.get("tick_s", 10)))
        self.scale = float(time_scale or (cfg.get("dry_run", {}).get("time_scale", 2000) if dry_run else 1.0))
        self.state: dict = {}
        self.dirty = False
        self.outage = False
        self.paused = False
        self.procs = Procs(self)
        self.steps: dict[str, Step] = {s["id"]: KINDS[s["kind"]](self, s) for s in cfg["steps"]}
        self.ports = Ports(self.ctx.get("ports", ""))
        self.fake_dir = os.path.join(self.dir, "fake")
        self.camp_run = f"campaign-{cfg['campaign']}"
        self.clock: Clock | None = None
        self._lock_fd = None
        self._override_mtime = None

    # ---- backends -------------------------------------------------------
    def _backends(self) -> None:
        if self.dry:
            from rsi.campaign import fakes

            self.evald = fakes.FakeEvalClient(self.fake_dir, self.clock, lanes=self.cfg.get("lanes", [1, 2]))
            self.loader = fakes.FakeLoader(self.fake_dir, self.clock)
            self.probe = fakes.FakeProbe(self.fake_dir)
        else:
            token = self.ctx.get("token")
            self.evald = _LazyEvald(self.expand(self.cfg.get("evald_url", "")), token)
            la = self.cfg["lora_admin"]
            self.loader = GatewayLoraLoader(self.expand(la["load_url"]), self.expand(la["list_url"]), token,
                                            base=la.get("base", "improver"))
            self.probe = HttpProbe(token)

    def child_env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = CODE_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["RSI_RUNS_ROOT"] = self.ctx["runs_root"]
        env["R"] = self.ctx["R"]
        env["RSI_CAMPAIGN"] = self.cfg["campaign"]
        if self.dry:
            env.update({"RSI_CAMPAIGN_TIME_SCALE": str(self.scale), "RSI_CAMPAIGN_FAKE_DIR": self.fake_dir,
                        "RSI_CAMPAIGN_REAL0": str(self.clock.real0), "RSI_CAMPAIGN_VIRT0": str(self.clock.virt0)})
        return env

    # ---- small helpers --------------------------------------------------
    def now(self) -> float:
        return self.clock.now()

    def real_to_virtual(self, t_real: float) -> float:
        return self.clock.virt0 + (t_real - self.clock.real0) * self.clock.scale

    def fmt(self, t: float | None) -> str:
        return fmt_time(t, self.tz)

    def expand(self, s: Any) -> Any:
        return expand(s, self.ctx, self.ports.resolve)

    def run_id(self, step_id: str) -> str:
        return f"{self.cfg['campaign']}-{step_id}".replace("_", "-")

    def build_cmd(self, key_or_template: str, **extra: Any) -> list[str]:
        cmds = self.cfg.get("commands") or {}
        if self.dry and key_or_template in FAKE_COMMANDS:
            tmpl = FAKE_COMMANDS[key_or_template]
        elif key_or_template in cmds:
            tmpl = cmds[key_or_template]
        else:
            tmpl = key_or_template
        if self.dry and tmpl not in FAKE_COMMANDS.values():
            raise RuntimeError(f"dry-run would run a real command: {tmpl}")
        ctx = dict(self.ctx)
        ctx.update({k: v for k, v in extra.items() if v is not None})
        if self.dry:
            ctx["py"] = sys.executable
        return [expand(tok, ctx, self.ports.resolve) for tok in shlex.split(tmpl)]

    def remote_wrap(self, cfg: dict, argv: list[str]) -> tuple[list[str], dict | None]:
        """Steps with ``remote: {jobid, srun_args}`` run inside another Slurm job (e.g. the 27B
        node node-d): ``srun --jobid=J --overlap --ntasks=1 -N1 <srun_args> --cpu-bind=none``,
        with SLURM_* cleared except SLURM_CONF. Dry-runs ignore it."""
        rem = cfg.get("remote")
        if not rem or self.dry:
            return argv, None
        pre = ["srun", f"--jobid={rem['jobid']}", "--overlap", "--ntasks=1", "-N1"]
        pre += [str(a) for a in rem.get("srun_args") or []] + ["--cpu-bind=none"]
        return pre + argv, clean_slurm_env(self.child_env())

    def update_pairs(self) -> None:
        """Δ₂(9B) next to Δ₂(27B): every decision with ``pair_with`` forms one row."""
        def brief(r: dict | None) -> dict | None:
            if not r:
                return None
            m = r.get("main") or {}
            return {"delta": m.get("delta"), "ci": m.get("ci"), "pass": bool(r.get("pass")),
                    "harmful": bool(r.get("harmful")), "text": analysis.fmt_delta(r)}
        rows = []
        for s in self.steps.values():
            pw = s.cfg.get("pair_with")
            if s.cfg["kind"] != "decision" or not pw:
                continue
            rows.append({"comparison": s.cfg.get("pair_label", s.id), "steps": {"9B": pw, "27B": s.id},
                         "9B": brief(self.state["decisions"].get(pw)), "27B": brief(self.state["decisions"].get(s.id))})
        self.state["pairs"] = rows
        atomic_write_json(os.path.join(self.dir, "decisions.json"), {**self.state["decisions"], "_9b_vs_27b": rows})

    def pairs_text(self) -> str:
        rows = [r for r in self.state.get("pairs") or [] if r.get("9B") or r.get("27B")]
        return "; ".join(f"{r['comparison']}: 9B {(r['9B'] or {}).get('text', 'n/a')} | 27B {(r['27B'] or {}).get('text', 'n/a')}"
                         for r in rows)

    def write_config(self, path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            import yaml  # type: ignore

            text = yaml.safe_dump(obj, sort_keys=False)
        except ImportError:  # JSON is valid YAML
            text = json.dumps(obj, indent=1)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    def resolve_refs(self, r: dict, step: Step) -> dict:
        """``frontier:<step>`` / ``runs:<step>`` / ``lessons:<step>`` references -> paths."""
        def res(v: Any) -> Any:
            if isinstance(v, str) and ":" in v and v.split(":", 1)[0] in ("frontier", "runs", "lessons"):
                kind, sid = v.split(":", 1)
                src = self.steps[sid]
                if kind == "frontier":
                    fr = analysis.frontier(src.run_dirs())
                    if fr is None:
                        self.error(step.id, f"no frontier artifact in {sid}; starting from the default start")
                        return None
                    self.log(f"{step.id}: frontier from {sid} = {fr['artifact_hash']} ({fr['value']:.4f})")
                    return fr["path"]
                if kind == "runs":
                    return src.run_dirs()
                if kind == "lessons":
                    return src.scaffold_dir()
            if isinstance(v, dict):
                return {k: res(x) for k, x in v.items()}
            if isinstance(v, list):
                return [res(x) for x in v]
            return self.expand(v) if isinstance(v, str) else v
        return res(r)

    # ---- flags ----------------------------------------------------------
    def flag(self, name: str) -> bool:
        neg = name.startswith("!")
        v = bool(self.state["flags"].get(name.lstrip("!"), False))
        return (not v) if neg else v

    def apply_step_flags(self, step: Step) -> None:
        key = "on_done_flags" if step.status in OK_STATES else "on_fail_flags"
        for k, v in (step.cfg.get(key) or {}).items():
            self.state["flags"][k] = v

    # ---- logging, events, errors ---------------------------------------
    def log(self, text: str) -> None:
        line = f"[{self.fmt(self.now()) if self.clock else '-'}] {text}"
        with open(os.path.join(self.dir, "campaign.log"), "a") as f:
            f.write(line + "\n")
        if not self.quiet:
            print(line, flush=True)

    def ev(self, ev: str, *, agent: str = "campaign", gen: int = 0, trial: int | None = None, **fields: Any) -> None:
        try:
            events.emit(self.camp_run, ev, agent, gen=gen, trial=trial, **fields)
        except Exception as e:  # noqa: BLE001 — never let logging kill the campaign
            self.log(f"event emit failed: {e}")

    def msg(self, to: str, summary: str, **fields: Any) -> None:
        self.ev("message", to=to, summary=summary, tokens=0, vt=round(self.now(), 1), **fields)

    def error(self, where: str, text: str) -> None:
        self.state.setdefault("errors", []).append({"vt": self.now(), "where": where, "msg": text[:300]})
        self.state["errors"] = self.state["errors"][-200:]
        self.dirty = True
        self.log(f"ERROR {where}: {text}")

    # ---- gates and the human --------------------------------------------
    def gate_failed(self, step: Step, metrics: dict, failed: list[str]) -> None:
        on_fail = step.cfg.get("on_fail") or {}
        step.st["gate_failed"] = failed
        self.state["gates"][step.id] = {"pass": False, "metrics": metrics, "failed": failed}
        wait = float(on_fail.get("wait_human_s", 0))
        self.status_line(event=f"ALERT gate {step.id} FAILED: {', '.join(failed)}; "
                               f"{'waiting ' + str(int(wait / 60)) + ' min for override.yaml, then ' if wait else ''}"
                               f"fallback: {self.describe_fallback(step, metrics)}")
        if wait > 0:
            step.st["human_deadline"] = self.now() + wait
            step.set("waiting_human", "gate failed: " + ", ".join(failed))
        else:
            self.apply_fallback(step, metrics)

    def describe_fallback(self, step: Step, metrics: dict) -> str:
        fb = self.pick_fallback(step, metrics)
        if fb is None:
            return "none"
        return fb.get("note") or json.dumps(fb.get("set_flags", {}))

    def pick_fallback(self, step: Step, metrics: dict) -> dict | None:
        for fb in (step.cfg.get("on_fail") or {}).get("fallbacks", []):
            cond = fb.get("if")
            if cond is None or run_checks(metrics, cond)[0]:
                return fb
        return None

    def apply_fallback(self, step: Step, metrics: dict) -> None:
        fb = self.pick_fallback(step, metrics) or {}
        for k, v in (fb.get("set_flags") or {}).items():
            self.state["flags"][k] = v
        self.msg(step.id, f"fallback applied: {fb.get('note') or fb.get('set_flags')}", campaign_ev="fallback")
        step.set(fb.get("status", "failed"), f"gate failed; fallback {fb.get('note') or fb.get('set_flags')}")

    def poll_human_gate(self, step: Step) -> None:
        if step.id in self.state["override_applied"].get("force_pass", []):
            self.state["gates"][step.id]["pass"] = True
            step.set("done", "forced pass by human override")
        elif self.now() >= step.st.get("human_deadline", 0):
            self.apply_fallback(step, self.state["gates"].get(step.id, {}).get("metrics", {}))

    def read_override(self) -> None:
        path = os.path.join(self.dir, "override.yaml")
        try:
            m = os.stat(path).st_mtime
        except OSError:
            return
        if m == self._override_mtime:
            return
        self._override_mtime = m
        try:
            from rsi.campaign.config import load_raw

            ov = load_raw(path) or {}
        except Exception as e:  # noqa: BLE001
            self.error("override", f"cannot parse override.yaml: {e}")
            return
        applied = self.state["override_applied"]
        for sid in ov.get("force_pass") or []:
            applied.setdefault("force_pass", [])
            if sid not in applied["force_pass"]:
                applied["force_pass"].append(sid)
        for k, v in (ov.get("flags") or {}).items():
            if applied.setdefault("flags", {}).get(k) != v:
                applied["flags"][k] = v
                self.state["flags"][k] = v
                self.log(f"override: flag {k} = {v}")
        for sid in ov.get("skip") or []:
            s = self.steps.get(sid)
            if s and s.status == "pending":
                s.set("skipped", "human override")
        for sid in ov.get("stop") or []:
            s = self.steps.get(sid)
            if s and s.status == "running" and s.st.get("stop_requested") is None:
                s.request_stop("human override")
        self.dirty = True

    # ---- triggers -------------------------------------------------------
    def try_start(self, step: Step) -> None:
        cfg = step.cfg
        deps_after = [self.steps[d] for d in cfg["after"]]
        deps_need = [self.steps[d] for d in cfg["needs"]]
        if any(d.status not in TERMINAL for d in deps_after + deps_need):
            return
        bad = [d.id for d in deps_need if d.status not in OK_STATES]
        if bad:
            step.set("skipped", f"needs {','.join(bad)} which did not succeed")
            return
        for f in cfg["requires_flags"]:
            if not self.flag(f):
                step.set("skipped", f"flag {f} is false")
                return
        for f in cfg["skip_if_flags"]:
            if self.flag(f):
                step.set("skipped", f"flag {f} is true")
                return
        now = self.now()
        ls = cfg.get("_latest_start")
        if ls is not None and now > ls:
            step.set("skipped", f"latest_start {self.fmt(ls)} passed")
            return
        dl = cfg.get("_deadline")
        if dl is not None and now >= dl:
            step.set("skipped", f"deadline {self.fmt(dl)} passed before start")
            return
        if any(not self.flag(f) for f in cfg["wait_for_flags"]):
            return
        nb = cfg.get("_not_before")
        if nb is not None and now < nb:
            return
        if self.paused or (self.outage and cfg["kind"] in ("run", "muar_arm", "lessons")):
            return
        for h in cfg["holds"]:
            holder = self.state["holds"].get(h)
            if holder and holder != step.id and self.steps[holder].status not in TERMINAL:
                return
        for h in cfg["holds"]:
            self.state["holds"][h] = step.id
        if cfg.get("phase") is not None and cfg["phase"] not in self.state["phases_started"]:
            self.state["phases_started"].append(cfg["phase"])
            self.msg("campaign", f"phase {cfg['phase']} begins with {step.id}", campaign_ev="phase_start",
                     phase=cfg["phase"])
            self.status_line(event=f"PHASE {cfg['phase']} starts ({step.id})")
        try:
            step.start()
        except Exception as e:  # noqa: BLE001
            self.error(step.id, f"start failed: {e}\n{traceback.format_exc()[-800:]}")
            step.set("failed", f"start failed: {e}")

    def muar_blocked_by_earlier(self, step) -> bool:
        """Pool priority = yaml order: a µAR arm may not take a free slot while an earlier arm still has pending
        sub-runs (otherwise every arm refills its own finished slots and later arms keep the pool forever)."""
        for s in self.steps.values():
            if s is step:
                return False
            if isinstance(s, MuarArmStep) and s.status == "running" and s.st.get("stop_requested") is None \
                    and any(x["status"] == "pending" for x in s.subs().values()):
                return True
        return False

    def muar_free(self) -> int:
        cap = int(self.cfg["muar"]["cpu_workers"])
        used = sum(s.running_count() for s in self.steps.values() if isinstance(s, MuarArmStep) and s.status in ("running", "waiting_human"))
        return cap - used

    # ---- health ---------------------------------------------------------
    def health(self) -> None:
        h = self.cfg.get("health") or {}
        hs = self.state["health"]
        now = self.now()
        if now - hs.get("last", 0) < float(h.get("interval_s", 180)):
            return
        hs["last"] = now
        report = {}
        outage = False
        for svc in h.get("services") or []:
            name = svc["name"]
            ok, detail = self.probe.check(name, self.expand(svc.get("url", "")))
            fails = 0 if ok else hs.setdefault("fails", {}).get(name, 0) + 1
            hs.setdefault("fails", {})[name] = fails
            report[name] = "ok" if ok else f"DOWN({fails}) {detail}"
            if fails >= int(h.get("fail_threshold", 2)) and svc.get("critical", True):
                outage = True
                self.try_restart_service(svc, fails)
            if ok and hs.get("was_down", {}).get(name):
                hs["was_down"][name] = False
                self.log(f"health: {name} recovered")
                if svc.get("reload_adapters"):
                    self.reload_adapters()
            if not ok and fails >= int(h.get("fail_threshold", 2)):
                hs.setdefault("was_down", {})[name] = True
        gm = None if self.dry else gpu_memory()
        if gm:
            report["gpu_mem_gb"] = {g["gpu"]: round(g["used_mb"] / 1024, 1) for g in gm}
            g0 = [g for g in gm if g["gpu"] == int(h.get("serving_gpu", 0))]
            if g0 and (g0[0]["total_mb"] - g0[0]["used_mb"]) / 1024 < float(h.get("gpu_min_free_gb", 1.0)):
                self.error("health", f"GPU{g0[0]['gpu']} nearly full ({g0[0]['used_mb'] / 1024:.1f} GB used)")
        free = disk_free_gb(self.ctx["runs_root"])
        if free is not None:
            report["disk_free_gb"] = round(free, 1)
            if free < float(h.get("disk_hard_min_gb", 10)):
                outage = True
                self.error("health", f"disk nearly full: {free:.1f} GB free; pausing runs")
            elif free < float(h.get("disk_min_free_gb", 50)):
                self.error("health", f"disk low: {free:.1f} GB free")
        for lane, owner in list(self.state["paused_lanes"].items()):
            if self.steps[owner].status in TERMINAL:  # orphaned pause (crash between steps): release it
                try:
                    self.evald.resume_lane(int(lane))
                    self.state["paused_lanes"].pop(lane, None)
                    self.error("health", f"lane {lane} was left paused by {owner}; resumed")
                except Exception as e:  # noqa: BLE001
                    self.error("health", f"cannot resume orphan lane {lane}: {e}")
        hs["report"] = report
        self.set_outage(outage)
        self.dirty = True

    def set_outage(self, on: bool) -> None:
        if on == self.outage:
            return
        self.outage = on
        self.state["outage"] = on
        self.status_line(event=("ALERT outage: pausing agent runs" if on else "outage over: resuming runs"))
        self.msg("campaign", "outage" if on else "recovered", campaign_ev="health")
        for s in self.steps.values():
            if isinstance(s, RunStep) and s.status == "running":
                s.health_pause(on)

    def try_restart_service(self, svc: dict, fails: int) -> None:
        hs = self.state["health"]
        name = svc["name"]
        if not svc.get("restart_cmd"):
            return
        r = hs.setdefault("restarts", {}).setdefault(name, {"n": 0, "last": 0, "proc": None})
        if self.now() - r["last"] < float(svc.get("restart_wait_s", 900)):
            return  # give the previous restart time to come up
        if r["proc"] and self.procs.poll(r["proc"])[0] == "running":
            if not svc.get("long_running"):
                return
            self.procs.kill(r["proc"], signal.SIGTERM)  # a server we restarted earlier that is still unhealthy
        if r["n"] >= int((self.cfg.get("health") or {}).get("max_service_restarts", 2)):
            if not r.get("gave_up"):
                r["gave_up"] = True
                self.status_line(event=f"ALERT {name} still down after {r['n']} restarts; needs a human")
            return
        r["n"] += 1
        r["last"] = self.now()
        argv = self.build_cmd("restart", service=name) if self.dry else self.build_cmd(svc["restart_cmd"])
        r["proc"] = self.procs.launch(argv, log=os.path.join(self.dir, "services", f"{name}.txt"),
                                      result=os.path.join(self.dir, "services", f"{name}-{r['n']}.json"))
        self.error("health", f"{name} down {fails}x: restart {r['n']} launched")
        self.msg(name, f"restart {r['n']}", campaign_ev="service_restart")

    def reload_adapters(self) -> None:
        for name, path in self.state["loaded_adapters"].items():
            ok, why = self.loader.load(name, path)
            self.log(f"reload adapter {name}: {why}")

    # ---- status ---------------------------------------------------------
    def status_line(self, event: str | None = None) -> None:
        path = self.ctx["status_md"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write(f"# Campaign status ({self.cfg['campaign']})\n\n"
                        "Times are Beijing time" + (" (dry-run, virtual clock)" if self.dry else "") + ".\n\n"
                        "| time | phase | running (trials/budget, best) | finished | Δ₂ so far | lanes | health | errors 1h |\n"
                        "|---|---|---|---|---|---|---|---|\n")
            if event:
                f.write(f"| {self.fmt(self.now())} | **{event}** |||||||\n")
                return
            running = [s for s in self.steps.values() if s.status in ("running", "waiting_human")]
            started = self.state.get("phases_started") or [0]
            phase = max([s.cfg.get("phase", 0) for s in running], default=started[-1])
            run_txt = "; ".join(f"{s.id} {s.progress()}" for s in running)[:600] or "-"
            fin = sum(1 for s in self.steps.values() if s.status in OK_STATES)
            fails = sum(1 for s in self.steps.values() if s.status == "failed")
            skipped = sum(1 for s in self.steps.values() if s.status == "skipped")
            d2 = "; ".join(f"{v.get('label', k)} {analysis.fmt_delta(v)}" for k, v in self.state["decisions"].items()) or "-"
            if self.pairs_text():
                d2 += " ‖ 9B vs 27B: " + self.pairs_text()
            lanes = ", ".join(f"L{l} paused by {o}" for l, o in self.state["paused_lanes"].items()) or "all eval lanes up"
            hr = self.state["health"].get("report", {})
            health = ", ".join(f"{k}:{v}" for k, v in hr.items() if not isinstance(v, dict))[:200] or "-"
            errs = [e for e in self.state.get("errors", []) if self.now() - e["vt"] < 3600]
            err_txt = f"{len(errs)}" + (f" (last: {errs[-1]['where']}: {errs[-1]['msg'][:80]})" if errs else "")
            f.write(f"| {self.fmt(self.now())} | P{phase} | {run_txt} | {fin} ok, {fails} failed, {skipped} skipped "
                    f"| {d2} | {lanes} | {health} | {err_txt} |\n")
        atomic_write_json(os.path.join(self.dir, "status.json"), {
            "vt": self.now(), "steps": {k: {"status": s.status, "progress": s.progress(), "reason": s.st.get("reason")}
                                        for k, s in self.steps.items()},
            "flags": self.state["flags"], "decisions": self.state["decisions"], "paused_lanes": self.state["paused_lanes"]})

    # ---- persistence ----------------------------------------------------
    def save(self) -> None:
        self.state["clock"] = self.clock.state()
        atomic_write_json(self.state_path, self.state)
        self.dirty = False

    def acquire_lock(self) -> bool:
        self._lock_fd = os.open(os.path.join(self.dir, "campaign.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        os.ftruncate(self._lock_fd, 0)
        os.write(self._lock_fd, str(os.getpid()).encode())
        return True

    def init_state(self, resume: bool) -> str:
        """-> 'fresh' | 'resumed'. Raises StopCampaign on refusal."""
        old = read_json(self.state_path)
        if old and not resume:
            raise StopCampaign(f"{self.state_path} exists; use --resume (or move the directory away)")
        if old and old.get("ended"):
            raise StopCampaign("campaign already ended (state.ended); nothing to resume")
        if old:
            self.state = old
            c = old["clock"]
            self.clock = Clock(c["scale"], c["real0"], c["virt0"])
            self.outage = bool(old.get("outage"))
            mode = "resumed"
        else:
            virt0 = self.cfg.get("_planned_start") if self.dry else None
            self.clock = Clock(self.scale, None, virt0)
            flags = dict(self.cfg.get("flags") or {})
            self.state = {"campaign": self.cfg["campaign"], "config_hash": self.cfg["_hash"], "dry_run": self.dry,
                          "steps": {}, "flags": flags, "gates": {}, "decisions": {}, "holds": {},
                          "paused_lanes": {}, "loaded_adapters": {}, "health": {}, "errors": [],
                          "override_applied": {}, "phases_started": [], "started_vt": None}
            mode = "fresh"
        for k in ("gates", "decisions", "holds", "paused_lanes", "loaded_adapters", "health", "override_applied"):
            self.state.setdefault(k, {})
        self.state.setdefault("phases_started", [])
        self._backends()
        events.register_run(self.camp_run, os.path.join(self.dir, "events.jsonl"))
        return mode

    # ---- main loop ------------------------------------------------------
    def run(self, resume: bool = False) -> int:
        if os.path.exists(os.path.join(self.dir, "STOP")):
            print(f"STOP file present in {self.dir}; remove it to start or resume", file=sys.stderr)
            return 2
        if not self.acquire_lock():
            print(f"another campaign process holds {self.dir}/campaign.lock", file=sys.stderr)
            return 3
        try:
            mode = self.init_state(resume)
        except StopCampaign as e:
            print(str(e), file=sys.stderr)
            return 2
        if mode == "fresh":
            self.state["started_vt"] = self.now()
            planned = sum(int(s.get("stop_when_trials") or 0) for s in self.cfg["steps"] if s["kind"] == "run")
            self.ev("run_start", policy="campaign", task=self.cfg.get("task", "overnight"),
                    improver={"model": "improver", "scaffold_hash": None},
                    budget={"trials": planned, "wall_s": self.until - self.now()}, config_hash=self.cfg["_hash"],
                    dry_run=self.dry, n_steps=len(self.steps))
            self.log(f"campaign {self.cfg['campaign']} started ({'DRY-RUN x' + str(self.scale) if self.dry else 'real'}),"
                     f" runs until {self.fmt(self.until)}")
            self.status_line(event=f"campaign start ({'dry-run' if self.dry else 'real'})")
        else:
            self.state["resumes"] = self.state.get("resumes", 0) + 1
            self.msg("campaign", f"resume #{self.state['resumes']}", campaign_ev="resume")
            self.status_line(event=f"campaign resumed (#{self.state['resumes']})")
            self.log("resuming")
            for s in self.steps.values():
                if s.status in ("running", "interrupted"):
                    if s.status == "interrupted":
                        s.st["status"] = "running"
                        s.st.pop("stop_requested", None)
                        s.st.pop("sigterm_sent", None)
                        if isinstance(s, RunStep):
                            s.clear_pause("stop")
                        if isinstance(s, MuarArmStep):
                            for sub in s.subs().values():
                                if sub["status"] == "cancelled":
                                    sub["status"] = "pending"
                    s.recover()
        self.save()
        try:
            return self.loop()
        except KeyboardInterrupt:
            self.log("KeyboardInterrupt: stopping gracefully")
            return self.shutdown("interrupt", final=False, now_kill=True)
        finally:
            self.save()

    def loop(self) -> int:
        last_status = self.state.get("last_status_vt") or self.now()
        while True:
            self.read_override()
            stop = os.path.join(self.dir, "STOP")
            if os.path.exists(stop):
                now_kill = "now" in open(stop).read().lower()
                return self.shutdown("STOP file", final=False, now_kill=now_kill)
            was = self.paused
            self.paused = os.path.exists(os.path.join(self.dir, "PAUSE"))
            if was != self.paused:
                self.status_line(event="PAUSE file: no new steps; runs pause at trial boundary" if self.paused
                                 else "PAUSE removed: continuing")
                for s in self.steps.values():
                    if isinstance(s, RunStep) and s.status == "running":
                        s.write_pause("campaign-pause") if self.paused else s.clear_pause("campaign-pause")
            if self.now() >= self.until:
                return self.shutdown(f"reached {self.fmt(self.until)} (gpu_stop / --until)", final=True)
            self.health()
            for s in self.steps.values():
                if s.status in ("running", "waiting_human"):
                    try:
                        s.poll()
                    except Exception as e:  # noqa: BLE001
                        self.error(s.id, f"poll error: {e}\n{traceback.format_exc()[-600:]}")
            for s in self.steps.values():
                if s.status == "pending":
                    self.try_start(s)
            if self.now() - last_status >= float(self.cfg.get("status_interval_s", 3600)):
                last_status = self.now()
                self.state["last_status_vt"] = last_status
                self.status_line()
            self.save()
            if all(s.status in TERMINAL for s in self.steps.values()):
                return self.shutdown("all steps finished", final=True)
            time.sleep(self.tick_s)

    def shutdown(self, reason: str, *, final: bool, now_kill: bool = False) -> int:
        self.log(f"shutdown: {reason}")
        running = [s for s in self.steps.values() if s.status in ("running", "waiting_human")]
        for s in running:
            if s.status == "waiting_human":
                s.set("failed" if final else "pending", reason)
                continue
            s.request_stop(reason, now_kill=now_kill)
        grace = float(self.cfg.get("stop_grace_s", 900)) * (0.2 if now_kill else 2.2)
        t0 = self.now()
        while any(s.status == "running" for s in running) and self.now() - t0 < grace:
            for s in running:
                if s.status == "running":
                    try:
                        s.poll()
                    except Exception as e:  # noqa: BLE001
                        self.error(s.id, f"poll during shutdown: {e}")
            self.save()
            time.sleep(self.tick_s)
        for s in running:
            if s.status == "running":
                for rec in s.procs():
                    self.procs.kill(rec, signal.SIGKILL)
                if isinstance(s, (LoraStep, ServiceStep)):
                    s.release_lane()
                s.st["status"] = "partial" if final else "interrupted"
                s.st["proc"] = None
            elif not final and s.st.get("stop_reason") == reason and (
                    s.status in ("partial", "failed") or (s.status == "done" and isinstance(s, ServiceStep))):
                # stopped only because of STOP: resume it later rather than count it as finished
                s.st["status"] = "interrupted"
                s.st.pop("t_end", None)
        for lane, owner in list(self.state["paused_lanes"].items()):
            try:
                self.evald.resume_lane(int(lane))
                self.state["paused_lanes"].pop(lane, None)
                self.log(f"resumed lane {lane} (was paused by {owner})")
            except Exception as e:  # noqa: BLE001
                self.error("shutdown", f"cannot resume lane {lane}: {e}")
        for name, r in (self.state["health"].get("restarts") or {}).items():
            if r.get("proc"):
                self.procs.kill(r["proc"], signal.SIGTERM)
        if final:
            for c in self.cfg.get("final_cmds") or []:
                argv = self.build_cmd("final") if self.dry else self.build_cmd(c)
                rec = self.procs.launch(argv, log=os.path.join(self.dir, "final.txt"),
                                        result=os.path.join(self.dir, "final.json"))
                t1 = time.time()
                while self.procs.poll(rec)[0] == "running" and time.time() - t1 < 1800:
                    time.sleep(min(1.0, self.tick_s * 10))
            self.state["ended"] = True
            best = None
            n_tr = 0
            for s in self.steps.values():
                if isinstance(s, RunStep):
                    st = s.stats()
                    n_tr += st["n_trials"]
                    if st.get("best") is not None and (best is None or st["best"] < best):
                        best = st["best"]
            self.status_line(event=f"campaign END: {reason}")
            self.status_line()
            self.ev("run_end", best_value=best, n_trials=n_tr, nAUC=None, reason=reason,
                    decisions={k: analysis.fmt_delta(v) for k, v in self.state["decisions"].items()})
        else:
            self.status_line(event=f"campaign STOPPED: {reason} (resume with --resume)")
            self.msg("campaign", f"stopped: {reason}", campaign_ev="stop")
        self.save()
        return 0


class _LazyEvald:
    """Import worker B's EvalClient only when first used (so --help works without it)."""

    def __init__(self, url: str, token: str | None):
        self.url, self.token, self._c = url, token, None

    def __getattr__(self, name: str):
        if self._c is None:
            self._c = make_eval_client(self.url, self.token)
        return getattr(self._c, name)
