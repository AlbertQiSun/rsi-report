#!/bin/bash
# Python MCP servers for web search + fetch (the node has no node/npm), in their own venv $R/envs/mcp.
#   duckduckgo-mcp-server  (no API key; tools: search, fetch_content)
#   mcp-server-fetch       (official reference fetch server; tool: fetch)
# Both are wrapped at run time by rsi.mcp_lab.filter_proxy (blocklist + query log).
# Usage: bash bundle/install_mcps.sh   (on the compute node; needs internet)
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
V=$R/envs/mcp
[ -x "$V/bin/python" ] || "$PY" -m venv "$V"
"$V/bin/pip" install --upgrade pip >/dev/null
"$V/bin/pip" install "duckduckgo-mcp-server" "mcp-server-fetch"
"$V/bin/pip" freeze | grep -i -E "duckduckgo|mcp" | tee "$V/VERSIONS.txt"
# stdio handshake check: initialize + tools/list on each server
for s in duckduckgo-mcp-server mcp-server-fetch; do
  printf '%s\n%s\n%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | timeout 60 "$V/bin/$s" 2>/dev/null | "$V/bin/python" -c '
import json,sys
for l in sys.stdin:
    m=json.loads(l)
    if m.get("id")==2: print(sys.argv[1], "tools:", [t["name"] for t in m["result"]["tools"]])' "$s" || echo "WARN: $s handshake failed"
done
echo MCPS_DONE
