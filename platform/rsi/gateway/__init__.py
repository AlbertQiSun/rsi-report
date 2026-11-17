"""RSI logging gateway (owner: worker C). See server.py for the full contract.

    python -m rsi.gateway [--port 0] [--upstream improver=http://127.0.0.1:P] ...
    from rsi.gateway import GatewayClient, gateway_url

Names are imported lazily so ``python -m rsi.gateway.ports`` runs without warnings.
"""

_LAZY = {"GatewayClient": "client", "gateway_url": "client", "GatewayConfig": "server",
         "GatewayThread": "server", "make_server": "server"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
