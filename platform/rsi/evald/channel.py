"""Host side of the rsi_guard report channel: a FIFO evald reads while a sandbox runs."""

from __future__ import annotations

import json
import os
import select
import threading
import time


class FifoReader:
    """Creates ``path`` as a FIFO and collects JSON-line messages in a thread.

    Opened O_RDONLY|O_NONBLOCK before the sandbox starts, so writers (opened
    O_NONBLOCK in rsi_guard) never get ENXIO; EOF just means "no writer right now".
    """

    def __init__(self, path: str, max_msgs: int = 5000):
        self.path = path
        self.max_msgs = max_msgs
        self.msgs: list[dict] = []
        self.n_total = 0
        self.n_bad = 0
        if os.path.exists(path):
            os.unlink(path)
        os.mkfifo(path, 0o600)
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self._buf = b""
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "FifoReader":
        self._th.start()
        return self

    def _consume(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if not line.strip():
                continue
            self.n_total += 1
            try:
                m = json.loads(line)
                if not isinstance(m, dict):
                    raise ValueError
            except Exception:
                self.n_bad += 1
                continue
            if len(self.msgs) < self.max_msgs or m.get("k") == "result":
                self.msgs.append(m)
        if len(self._buf) > 1 << 20:  # a writer that never ends lines
            self._buf = b""
            self.n_bad += 1

    def _read_some(self, timeout: float) -> bool:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return False
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return False
        if not data:
            time.sleep(min(timeout, 0.05))  # EOF: no writer at the moment
            return False
        self._consume(data)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read_some(0.2)

    def stop(self) -> list[dict]:
        self._stop.set()
        self._th.join(timeout=5)
        for _ in range(1000):  # drain
            if not self._read_some(0.0):
                break
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
        return self.msgs
