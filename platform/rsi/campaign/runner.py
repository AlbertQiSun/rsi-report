"""Tiny process wrapper: ``python -m rsi.campaign.runner --result R --log L -- cmd ...``

Runs ``cmd`` with stdout/stderr appended to ``L``. When it exits, writes ``R``
atomically: ``{"rc", "t0", "t1", "out"}`` where ``out`` is the last JSON object
line the command printed (or null). The campaign tracks the runner's pid and
reads ``R``; because the runner lives in its own session, it (and its child)
survive a crash of the campaign process and can be re-adopted on ``--resume``.
SIGTERM / SIGINT are forwarded to the child.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time


def last_json_line(path: str, offset: int) -> dict | None:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()[-262144:]
    except OSError:
        return None
    for line in reversed(data.decode(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def write_result(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-res-")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: runner --result R --log L -- cmd ...", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, cmd = argv[:i], argv[i + 1:]
    kv = dict(zip(opts[::2], opts[1::2]))
    result, log = kv["--result"], kv["--log"]
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    t0 = time.time()
    lf = open(log, "ab", buffering=0)
    offset = lf.tell()
    lf.write(f"\n=== runner start {time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()} ===\n".encode())
    offset = lf.tell()
    try:
        child = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError as e:
        lf.write(f"runner: cannot start command: {e}\n".encode())
        write_result(result, {"rc": 127, "t0": t0, "t1": time.time(), "out": None, "error": str(e)})
        return 127

    def forward(sig, _frame):
        try:
            child.send_signal(sig)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = child.wait()
    lf.write(f"=== runner end rc={rc} ===\n".encode())
    lf.close()
    write_result(result, {"rc": rc, "t0": t0, "t1": time.time(), "out": last_json_line(log, offset)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
