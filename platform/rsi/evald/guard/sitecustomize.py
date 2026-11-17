# Loaded automatically by every Python process in an evald sandbox (PYTHONPATH).
# Installs the rsi_guard audit hook when $RSI_AUDIT_PIPE is set.
import os as _os

if _os.environ.get("RSI_AUDIT_PIPE"):
    try:
        import rsi_guard as _g

        _g.install()
    except Exception:  # never break the interpreter
        pass
