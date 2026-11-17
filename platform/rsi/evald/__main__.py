"""CLI: python -m rsi.evald {serve, vanilla, selftest}

serve     start the HTTP service (writes its port to $R/run/ports.json as "evald")
vanilla   run the vanilla start artifact for N seeds across lanes, one CSV line per seed;
          through a running evald if one answers (mode auto/evald), else in-process (direct)
selftest  probe the bwrap sandbox on one GPU and pre-pack the eval splits
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time

from rsi.evald.config import NANOCHAT_START_DIR, EvaldConfig, load_config, parse_lanes

CSV_COLS = ["seed", "lane", "status", "dev_bpb", "sealed_bpb", "wall_s", "train_wall_s", "score_wall_s",
            "reported_training_s", "reported_total_s", "startup_s", "attn_backend", "job_id"]


def _cfg_from_args(a) -> EvaldConfig:
    if a.config:
        cfg = load_config(a.config)
    else:
        cfg = EvaldConfig.for_cluster(a.R, parse_lanes(a.lanes))
    over = {}
    if a.sandbox:
        over["sandbox"] = a.sandbox
    if a.margin_s is not None:
        over["startup_margin_s"] = a.margin_s
    if a.time_budget_s is not None:
        over["time_budget_s"] = a.time_budget_s
    if getattr(a, "no_honeypots", False):
        over["honeypots"] = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _common(p):
    p.add_argument("--R", default=os.environ.get("R", "/scratch/user/rsi"))
    p.add_argument("--lanes", default="1,2", help='"1,2" = lane i on GPU i; "1:0,2:1" explicit')
    p.add_argument("--config", help="JSON EvaldConfig (overrides the cluster defaults)")
    p.add_argument("--sandbox", choices=["bwrap", "none"])
    p.add_argument("--margin-s", type=float, dest="margin_s")
    p.add_argument("--time-budget-s", type=float, dest="time_budget_s")
    p.add_argument("--no-honeypots", action="store_true")


def cmd_serve(a) -> int:
    from rsi.evald.server import EvaldServer, read_token, write_port
    from rsi.evald.service import EvalService

    cfg = _cfg_from_args(a)
    run_dir = os.path.join(a.R, "run")
    token_file = a.token_file or os.path.join(run_dir, "token")
    token = read_token(token_file)
    svc = EvalService(cfg).start()
    srv = EvaldServer(svc, token, "127.0.0.1", a.port)
    write_port(a.ports_file or os.path.join(run_dir, "ports.json"), "evald", srv.port)
    with open(os.path.join(cfg.state_dir, "config.json"), "w") as f:
        json.dump(cfg.to_json(), f, indent=1)
    print(f"[evald] serving on {srv.url}  lanes={svc.lanes_info()}  sandbox={cfg.sandbox} "
          f"wall_limit={cfg.wall_limit_s:.0f}s state={cfg.state_dir}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    srv.start()
    while not stop.is_set():
        time.sleep(1)
    srv.shutdown()
    svc.stop()
    return 0


def _row(res: dict, seed_row: dict | None, sealed_row: dict | None) -> dict:
    s = seed_row or {}
    tr, rt = s.get("train_wall_s"), s.get("reported_training_s")
    return {"seed": s.get("seed"), "lane": res.get("lane"), "status": s.get("status") or res.get("status"),
            "dev_bpb": s.get("dev"), "sealed_bpb": (sealed_row or {}).get("value"), "wall_s": res.get("wall_s"),
            "train_wall_s": tr, "score_wall_s": s.get("score_wall_s"), "reported_training_s": rt,
            "reported_total_s": s.get("reported_total_s"),
            "startup_s": round(tr - rt, 1) if isinstance(tr, (int, float)) and isinstance(rt, (int, float)) else None,
            "attn_backend": s.get("attn_backend"), "job_id": res.get("job_id")}


def run_seeds(target, *, artifact: str, seeds: list[int], lanes: list[int], run: str, out_csv: str | None,
              task: str = "nanochat") -> list[dict]:
    """Submit one job per seed (round-robin over lanes); print one CSV line per seed as results arrive."""
    jobs = []
    for i, s in enumerate(seeds):
        jid = target.submit(run, task, artifact, seeds=[s], lane=lanes[i % len(lanes)] if lanes else None)
        jobs.append((s, jid))
    w = csv.DictWriter(sys.stdout, fieldnames=CSV_COLS)
    w.writeheader()
    fh = open(out_csv, "w", newline="") if out_csv else None
    fw = csv.DictWriter(fh, fieldnames=CSV_COLS) if fh else None
    if fw:
        fw.writeheader()
    rows = []
    for s, jid in jobs:
        res = target.result(jid, wait=True, timeout_s=24 * 3600)
        ps = (res.get("per_seed") or [{}])[0] or {"seed": s}
        ps.setdefault("seed", s)
        sp = ((res.get("sealed") or {}).get("per_seed") or [{}])[0]
        row = _row(res, ps, sp)
        rows.append(row)
        w.writerow(row)
        sys.stdout.flush()
        if fw:
            fw.writerow(row)
            fh.flush()
    if fh:
        fh.close()
    return rows


def _parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def cmd_vanilla(a) -> int:
    seeds = _parse_seeds(a.seeds)
    lanes = [l.lane for l in parse_lanes(a.lanes)]
    run = a.run or time.strftime("vanilla-%Y%m%dT%H%M%S")
    artifact = a.artifact or NANOCHAT_START_DIR
    target = None
    if a.mode in ("auto", "evald"):
        from rsi.evald.client import EvalClient

        try:
            c = EvalClient.from_run_dir(os.path.join(a.R, "run"))
            if c.health():
                target = c
        except Exception as e:  # noqa: BLE001
            if a.mode == "evald":
                print(f"evald not reachable: {e}", file=sys.stderr)
                return 2
        if target is None and a.mode == "evald":
            print("evald not reachable", file=sys.stderr)
            return 2
    svc = None
    if target is None:
        from rsi.evald.service import EvalService

        print("[vanilla] evald not up: running in-process (direct mode)", file=sys.stderr)
        svc = EvalService(_cfg_from_args(a)).start()
        target = svc
    print(f"[vanilla] run={run} via {'evald' if svc is None else 'direct'} seeds={seeds} lanes={lanes}",
          file=sys.stderr)
    try:
        run_seeds(target, artifact=artifact, seeds=seeds, lanes=lanes, run=run, out_csv=a.csv)
    finally:
        if svc is not None:
            svc.stop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rsi.evald")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    _common(s)
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--token-file")
    s.add_argument("--ports-file")
    v = sub.add_parser("vanilla")
    _common(v)
    v.add_argument("--seeds", default="0-19")
    v.add_argument("--mode", choices=["auto", "evald", "direct"], default="auto")
    v.add_argument("--run")
    v.add_argument("--artifact")
    v.add_argument("--csv")
    t = sub.add_parser("selftest")
    _common(t)
    t.add_argument("--gpu", type=int, default=1)
    t.add_argument("--skip-prepare", action="store_true")
    t.add_argument("--smoke", action="store_true", help="also train+score the vanilla start for ~30 s")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return cmd_serve(a)
    if a.cmd == "vanilla":
        return cmd_vanilla(a)
    from rsi.evald.selftest import run_selftest

    return run_selftest(_cfg_from_args(a), gpu=a.gpu, prepare=not a.skip_prepare, smoke=a.smoke)


if __name__ == "__main__":
    sys.exit(main())
