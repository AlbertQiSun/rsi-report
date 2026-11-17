#!/usr/bin/env python3
"""Download only selected tensors from HF safetensors repos via HTTP Range (stdlib only).

usage: python3 hfslice.py <repo> <out.safetensors> --include REGEX [--include REGEX ...] [--exclude REGEX] [--dry-run]
Writes one valid safetensors file containing exactly the matched tensors (dtype/shape/bytes unchanged).
Uses HF_TOKEN from the environment if set (never pass it on the command line).
"""
import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.request

API = "https://huggingface.co"


def req(url, rng=None):
    h = {"User-Agent": "hfslice/1"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--include", action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    inc = [re.compile(x) for x in a.include]
    exc = [re.compile(x) for x in a.exclude]
    tree = json.loads(req(f"{API}/api/models/{a.repo}/tree/{a.revision}"))
    shards = sorted(f["path"] for f in tree if f["path"].endswith(".safetensors"))
    plan = []  # (shard_url, name, dtype, shape, abs_start, abs_end)
    for s in shards:
        url = f"{API}/{a.repo}/resolve/{a.revision}/{s}"
        n = struct.unpack("<Q", req(url, (0, 7)))[0]
        hdr = json.loads(req(url, (8, 7 + n)))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if any(p.search(k) for p in inc) and not any(p.search(k) for p in exc):
                b0, b1 = v["data_offsets"]
                plan.append((url, k, v["dtype"], v["shape"], base + b0, base + b1))
    total = sum(p[5] - p[4] for p in plan)
    print(f"{a.repo}: {len(plan)} tensors, {total / 1e9:.2f} GB from {len(shards)} shards", flush=True)
    if a.dry_run:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plan.sort(key=lambda p: p[1])
    header, off = {}, 0
    for _, k, dt, shp, s0, s1 in plan:
        header[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + (s1 - s0)]}
        off += s1 - s0
    header["__metadata__"] = {"source": a.repo, "revision": a.revision, "tool": "hfslice"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = a.out + ".partial"
    done = 0
    t0 = time.time()
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for i, (url, k, dt, shp, s0, s1) in enumerate(plan):
            CH = 256 << 20
            for c0 in range(s0, s1, CH):
                f.write(req(url, (c0, min(s1, c0 + CH) - 1)))
            done += s1 - s0
            if i % 20 == 0 or i == len(plan) - 1:
                el = time.time() - t0
                print(f"  [{i + 1}/{len(plan)}] {done / 1e9:.2f}/{total / 1e9:.2f} GB  {done / 1e6 / max(el, 1e-3):.0f} MB/s", flush=True)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
