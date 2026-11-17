"""evald configuration (stdlib only)."""

from __future__ import annotations

import dataclasses
import getpass
import sys
import glob
import json
import os
from dataclasses import dataclass, field

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
GUARD_DIR = os.path.join(PKG_DIR, "guard")
SCORER_DIR = os.path.join(PKG_DIR, "scorer")
NANOCHAT_DIR = os.path.join(os.path.dirname(PKG_DIR), "tasks", "nanochat")
NANOCHAT_LIB_DIR = os.path.join(NANOCHAT_DIR, "lib")
NANOCHAT_START_DIR = os.path.join(NANOCHAT_DIR, "start")
PLATFORM_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))

EVAL_TOKENS = 40 * 524288  # same as Recursive / Karpathy lib.py


DEFAULT_REMOTE_SRUN_ARGS = ["--cpus-per-task=6", "--mem=60G", "--gres=gpu:a100:1", "--cpu-bind=none"]


@dataclass
class LaneCfg:
    lane: int
    gpu: int | str | None   # nvidia-smi index on the evald node; "slurm" = the GPU Slurm gives the step; None = CPU
    jobid: str | None = None          # set => remote lane: run each seed as a step of this Slurm job
    node: str | None = None           # optional --nodelist for the step
    srun_args: list | None = None     # resource args for the step (default DEFAULT_REMOTE_SRUN_ARGS)
    srun: str = "srun"

    @property
    def remote(self) -> bool:
        return self.jobid is not None


@dataclass
class EvaldConfig:
    state_dir: str                      # private, persistent: jobs/<id>/{job,result}.json, logs, sealed.jsonl
    scratch_dir: str                    # node-local, per-seed sandboxes: /tmp/rsi-evald-$USER
    data_root: str                      # holds train/ devval/ sealed/ tokenizer/
    train_python: str                   # python of the training env (used inside both sandboxes)
    lanes: list[LaneCfg] = field(default_factory=lambda: [LaneCfg(1, 1), LaneCfg(2, 2)])
    sandbox: str = "bwrap"              # "bwrap" | "none" (Mac tests; prints a warning)
    time_budget_s: float = 300.0        # D5: hard wall-clock limit = time_budget_s + startup_margin_s
    startup_margin_s: float = 50.0      # total wall rule: 300+50 = 350 s (S0: vanilla max 344 s; margin test 2026-11-17)
    score_timeout_s: float = 1200.0
    prepare_timeout_s: float = 3600.0
    seq_len: int = 2048
    eval_tokens: int = EVAL_TOKENS
    score_batch: int = 32
    score_device: str = "auto"
    score_no_autocast: bool = False
    probe_tol: float = 0.02
    splits: dict = field(default_factory=lambda: {"dev": "devval", "sealed": "sealed"})
    hf_cache_dirs: list[str] = field(default_factory=list)   # HF hub dirs (FA3 kernels) copied fresh into each sandbox
    deny_paths: list[str] = field(default_factory=list)      # extra paths the guard reports (prefixes; "x*" = glob prefix)
    honeypots: bool = True              # decoy files at forbidden paths + inotify (Linux, bwrap only)
    honeypot_targets: list[str] = field(default_factory=list)
    max_artifact_bytes: int = 1_000_000
    max_file_bytes: int = 300_000
    max_files: int = 32
    max_ckpt_bytes: int = 4 << 30
    max_nonparam_frac: float = 0.25
    max_log_bytes: int = 256 << 20
    gpu_idle_wait_s: float = 600.0
    host_python: str = sys.executable   # python for evald's own code (remote seed runner); stdlib only
    platform_root: str = PLATFORM_ROOT  # must be on shared storage for remote lanes
    remote_slack_s: float = 300.0       # extra outer timeout for a remote step (srun startup etc.)
    keep_scratch: bool = False
    extra_ro_binds: list[str] = field(default_factory=list)  # e.g. a module-provided gcc prefix
    runs_root: str | None = None        # default: $RSI_RUNS_ROOT or ./runs (rsi.events)
    train_env_extra: dict = field(default_factory=dict)      # extra env for the training sandbox (smoke tests)
    bwrap: str = "bwrap"
    nvidia_smi: str = "nvidia-smi"

    @property
    def wall_limit_s(self) -> float:
        return float(self.time_budget_s) + float(self.startup_margin_s)

    @property
    def eval_rows(self) -> int:
        return int(self.eval_tokens) // int(self.seq_len)

    def split_dir(self, name: str) -> str:
        return os.path.join(self.data_root, name)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "EvaldConfig":
        d = dict(d)
        d["lanes"] = [LaneCfg(**x) if isinstance(x, dict) else x for x in d.get("lanes", [])]
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def for_cluster(cls, R: str, lanes: list[LaneCfg], **kw) -> "EvaldConfig":
        """Defaults for island mode on node-a (R=/scratch/user/rsi)."""
        user = getpass.getuser()
        home = os.path.expanduser("~")
        hf = sorted(glob.glob(os.path.join(R, "hf", "hub", "models--kernels-community--flash-attn3"))
                    + glob.glob(os.path.join(R, "hf", "hub", "models--varunneal--flash-attention-3")))
        scratch_user = os.path.dirname(R.rstrip("/"))   # /scratch/user
        # never deny a prefix of the training env itself
        home_deny = home if not os.path.abspath(R).startswith(home.rstrip("/") + "/") else os.path.join(home, ".ssh")
        cfg = dict(
            state_dir=os.path.join(R, "evald"),
            scratch_dir=f"/tmp/rsi-evald-{user}",
            data_root=os.path.join(R, "data"),
            train_python=os.path.join(R, "envs", "rsi-train", "bin", "python"),
            lanes=lanes,
            hf_cache_dirs=hf,
            runs_root=os.path.join(R, "runs"),
            deny_paths=[
                os.path.join(R, "data", "devval"), os.path.join(R, "data", "sealed"),
                "/data/devval", "/data/sealed",
                os.path.join(R, "run"), os.path.join(R, "evald"), os.path.join(R, "runs"),
                os.path.join(R, "platform"), os.path.join(R, "bundle"), os.path.join(R, "hf"),
                home_deny, os.path.join(scratch_user, "otherproj") + "*",
            ],
            honeypot_targets=[
                os.path.join(R, "data", "devval"), os.path.join(R, "data", "sealed"),
                "/data/devval", "/data/sealed",
                os.path.join(home, ".ssh"), os.path.join(scratch_user, "otherproj"),
            ],
        )
        cfg.update(kw)
        return cls(**cfg)


def load_config(path: str) -> EvaldConfig:
    with open(path) as f:
        return EvaldConfig.from_json(json.load(f))


def parse_lanes(spec: str) -> list[LaneCfg]:
    """Comma list of lane specs:
      "1"                      lane 1 on local GPU 1
      "1:0"                    lane 1 on local GPU 0;  "1:cpu" CPU lane (tests)
      "3@10000002"             remote lane 3: steps of Slurm job 10000002, GPU chosen by Slurm
      "3-5@10000002:node-b" lanes 3,4,5 remote, pinned to node node-b
    """
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        remote = None
        if "@" in part:
            part, remote = part.split("@", 1)
        gpu_s = None
        if ":" in part:
            part, gpu_s = part.split(":", 1)
        if "-" in part:
            a, b = part.split("-", 1)
            ids = list(range(int(a), int(b) + 1))
        else:
            ids = [int(part)]
        for lane in ids:
            if remote is not None:
                jobid, _, node = remote.partition(":")
                if len(ids) > 1:
                    # A range of remote lanes on one node: pin lane k to that node's GPU (k - first).
                    # With --overlap, Slurm does NOT give concurrent steps distinct GPUs (all three lanes
                    # landed on GPU0 of node-b and OOMed on 2026-11-17), so the step sees all GPUs
                    # of the hold and the seed runner binds only the pinned one into the sandbox.
                    srun_args = [a for a in DEFAULT_REMOTE_SRUN_ARGS if not a.startswith("--gres=")]
                    srun_args.append(f"--gres=gpu:a100:{len(ids)}")
                    out.append(LaneCfg(lane, lane - ids[0], jobid=jobid, node=node or None, srun_args=srun_args))
                else:
                    out.append(LaneCfg(lane, "slurm", jobid=jobid, node=node or None))
            elif gpu_s is None:
                out.append(LaneCfg(lane, lane))
            elif gpu_s.lower() in ("cpu", "none", ""):
                out.append(LaneCfg(lane, None))
            elif gpu_s.lower() == "slurm":
                out.append(LaneCfg(lane, "slurm"))
            else:
                out.append(LaneCfg(lane, int(gpu_s)))
    return out
