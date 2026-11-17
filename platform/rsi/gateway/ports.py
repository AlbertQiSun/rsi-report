"""ports.json / token helpers shared by the gateway, serve_vllm.sh and clients. Stdlib only.

``$R/run/ports.json`` is a flat JSON object ``{service_key: port}``. Writers merge under an
exclusive flock so concurrent services never clobber each other's entries.

Values are ints (a port on 127.0.0.1 of the node that wrote it) or URL strings.
Keys written by worker C:
    vllm_<name>  int   vLLM on this node, served-model-name <name> (bundle/serve_vllm.sh, local mode)
    <name>       str   "http://host:port" base URL that serves model <name>; written by serve_vllm.sh in
                       both modes (local: 127.0.0.1; --expose: the node's cluster IP, i.e. the edge gateway).
                       The gateway prefers this over vllm_<name>. Names: improver, ladder-27b.
    gateway      int   the logging gateway on the orchestrator node (python -m rsi.gateway)
"""

from __future__ import annotations

import fcntl
import json
import os
import socket


def default_rsi_root() -> str:
    return os.environ.get("R") or os.environ.get("RSI_ROOT") or "/scratch/user/rsi"


def default_ports_path() -> str:
    return os.environ.get("RSI_PORTS") or os.path.join(default_rsi_root(), "run", "ports.json")


def default_token_path() -> str:
    return os.environ.get("RSI_TOKEN_FILE") or os.path.join(default_rsi_root(), "run", "token")


def read_token(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path) as f:
            tok = f.read().strip()
        return tok or None
    except FileNotFoundError:
        return None


def read_ports(path: str | None = None) -> dict:
    path = path or default_ports_path()
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def merge_ports(updates: dict, path: str | None = None, remove: list[str] | None = None) -> dict:
    """Merge ``updates`` into ports.json atomically (flock); never drops other keys."""
    path = path or default_ports_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = b""
        while chunk := os.read(fd, 65536):
            raw += chunk
        try:
            data = json.loads(raw.decode() or "{}")
            if not isinstance(data, dict):
                data = {}
        except ValueError:
            data = {}
        data.update(updates)
        for k in remove or []:
            data.pop(k, None)
        out = (json.dumps(data, indent=1, sort_keys=True) + "\n").encode()
        os.lseek(fd, 0, 0)
        os.ftruncate(fd, 0)
        os.write(fd, out)
        return data
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def cluster_ip() -> str:
    """This node's cluster-network IPv4 (env BIND_IP wins). Never 127.x."""
    ip = os.environ.get("BIND_IP")
    if ip:
        return ip
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if not sa[0].startswith("127."):
                return sa[0]
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # no packet is sent
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    if ip.startswith("127."):
        raise RuntimeError("could not determine cluster IP; set BIND_IP")
    return ip


def free_port(host: str = "127.0.0.1", lo: int = 20000, hi: int = 60999) -> int:
    """A currently free TCP port in [lo, hi] (random high port; tiny race until bound)."""
    import random
    for _ in range(200):
        p = random.randint(lo, hi)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


if __name__ == "__main__":  # tiny CLI used by bundle/serve_vllm.sh
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("free")
    f.add_argument("--host", default="127.0.0.1")
    sub.add_parser("ip")
    rm = sub.add_parser("remove")
    rm.add_argument("--path", default=None)
    rm.add_argument("keys", nargs="+")
    m = sub.add_parser("merge")
    m.add_argument("--path", default=None)
    m.add_argument("kv", nargs="+", help="key=port")
    g = sub.add_parser("get")
    g.add_argument("--path", default=None)
    g.add_argument("key")
    a = ap.parse_args()
    if a.cmd == "free":
        print(free_port(a.host))
    elif a.cmd == "ip":
        print(cluster_ip())
    elif a.cmd == "remove":
        merge_ports({}, a.path, remove=a.keys)
    elif a.cmd == "merge":
        upd = {}
        for kv in a.kv:
            k, v = kv.split("=", 1)
            upd[k] = int(v) if v.isdigit() else v
        merge_ports(upd, a.path)
    else:
        v = read_ports(a.path).get(a.key)
        if v is None:
            raise SystemExit(1)
        print(v)
