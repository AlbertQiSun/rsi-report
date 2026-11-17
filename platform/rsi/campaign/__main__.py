"""CLI: python -m rsi.campaign --config campaign/overnight.yaml [--resume] [--dry-run] [--check]

Exit codes: 0 finished or stopped by STOP; 2 refused (config error, STOP present, state exists
without --resume); 3 another campaign process holds the lock; anything else = crash
(bundle/run_campaign.sh restarts the campaign with --resume, bounded).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from rsi.campaign.config import ConfigError, fmt_time, load_config, parse_time


def plan_table(cfg: dict) -> str:
    rows = ["phase | step | kind | planned | not_before | latest_start | deadline | after/needs"]
    for s in cfg["steps"]:
        deps = ",".join(s["after"] + s["needs"])
        rows.append(f"P{s.get('phase', '?')} | {s['id']} | {s['kind']} | {s.get('plan', '')} | "
                    f"{fmt_time(s['_not_before'], cfg['_tz'])} | {fmt_time(s['_latest_start'], cfg['_tz'])} | "
                    f"{fmt_time(s['_deadline'], cfg['_tz'])} | {deps}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rsi.campaign")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from state.json (fresh start if none)")
    ap.add_argument("--dry-run", action="store_true", help="fake components + compressed clock, CPU only")
    ap.add_argument("--root", help="override $R (dry-run default: a new temp dir)")
    ap.add_argument("--time-scale", type=float, help="dry-run: virtual seconds per real second (default 2000)")
    ap.add_argument("--until", help="stop (final) at this clock time instead of gpu_stop, e.g. 2026-11-18T09:00")
    ap.add_argument("--tick", type=float, help="seconds between scheduler ticks")
    ap.add_argument("--check", action="store_true", help="validate the config and print the plan, then exit")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    root = a.root
    if a.dry_run and not root:
        root = tempfile.mkdtemp(prefix="rsi-campaign-dry-")
    try:
        cfg = load_config(a.config, root_override=root)
    except (ConfigError, OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if a.check:
        print(plan_table(cfg))
        print(f"\nR={cfg['_ctx']['R']}  campaign_dir={cfg['_ctx']['campaign_dir']}  gpu_stop={fmt_time(cfg['_gpu_stop'], cfg['_tz'])}")
        return 0
    from rsi.campaign.scheduler import Campaign

    until = parse_time(a.until, cfg["_tz"]) if a.until else None
    camp = Campaign(cfg, dry_run=a.dry_run, time_scale=a.time_scale, until=until, tick_s=a.tick, quiet=a.quiet)
    if a.dry_run:
        print(f"dry-run root: {root}", flush=True)
    return camp.run(resume=a.resume)


if __name__ == "__main__":
    sys.exit(main())
