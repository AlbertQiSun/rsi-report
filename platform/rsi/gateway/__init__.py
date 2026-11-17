"""RSI logging gateway (owner: worker C). See server.py for the full contract.

    python -m rsi.gateway [--port 0] [--upstream improver=http://127.0.0.1:P] ...
    from rsi.gateway import GatewayClient, gateway_url
"""

from .client import GatewayClient, gateway_url  # noqa: F401
from .server import GatewayConfig, GatewayThread, make_server  # noqa: F401
