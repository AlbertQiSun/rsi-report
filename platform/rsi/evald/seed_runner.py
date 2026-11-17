"""Remote seed runner: ``python -m rsi.evald.seed_runner <spec.json> <out.json>``.

Launched by evald as a Slurm job step on a remote lane's node (see service._run_seed_remote).
Runs the same per-seed pipeline as a local lane (rsi/evald/seed.py) with node-local scratch,
FIFO guard channel and honeypots on *this* node, then writes the seed result JSON to the
shared private state dir. Stdlib only.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import traceback

from rsi.evald.config import EvaldConfig
from rsi.evald.sandbox import gpu_info, make_launcher, resolve_slurm_gpu
from rsi.evald.seed import SeedRunner, empty_seed_result
from rsi.runs import atomic_write_json


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec_p, out_p = argv[0], argv[1]
    with open(spec_p) as f:
        spec = json.load(f)
    seed = int(spec["seed"])
    note = f"node={socket.gethostname()}"
    try:
        cfg = EvaldConfig.from_json(spec["cfg"])
        launcher = make_launcher(cfg.sandbox, bwrap=cfg.bwrap, extra_ro=cfg.extra_ro_binds)
        gpu = None
        if cfg.sandbox == "bwrap" and spec.get("gpu") is not None:
            if spec["gpu"] == "slurm":
                gpu, how = resolve_slurm_gpu(cfg.nvidia_smi)
                note += f" gpu={gpu.index} ({gpu.uuid}, via {how})"
            else:
                gpu = gpu_info(int(spec["gpu"]), cfg.nvidia_smi)
                note += f" gpu={gpu.index}"
        runner = SeedRunner(cfg, launcher)
        runner.prepare_hf_template()
        print(f"[seed_runner] {note} job={spec['job']['id']} seed={seed} start {time.strftime('%H:%M:%S')}",
              flush=True)
        res = runner.run_seed(spec["job"], int(spec["lane"]), gpu, seed, spec.get("packed") or {})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res = empty_seed_result(seed, "crash", "evaluator infrastructure error (remote lane)")
        res["infra"].append(f"seed_runner: {type(e).__name__}: {str(e)[:300]}")
    res.setdefault("infra", []).append(note)
    res["node"] = socket.gethostname()
    atomic_write_json(out_p, res)
    print(f"[seed_runner] done status={res.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
