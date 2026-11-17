#!/bin/bash
# Install the native Claude Code Linux binary into $R/tools/claude/<ver>/claude (no node/npm, nothing in $HOME).
#
# Source (looked up 2026-11-18): Anthropic's official installer https://claude.ai/install.sh redirects to
#   https://downloads.claude.ai/claude-code-releases/bootstrap.sh, which downloads
#     $BASE/<version>/manifest.json                    (per-platform sha256 "checksum" and "size")
#     $BASE/<version>/<platform>/claude                (the native binary; platform linux-x64 | linux-x64-musl)
#   with BASE=https://downloads.claude.ai/claude-code-releases ($BASE/stable and $BASE/latest hold version strings;
#   on 2026-11-18 stable=2.1.311, latest=2.1.326). The official script then runs `claude install`, which writes
#   to ~/.local and ~/.claude; this script deliberately does NOT (no `claude install`, no $HOME writes).
# The sha256 below is pinned from the 2.1.311 manifest (fetched 2026-11-18) and re-checked against the
# manifest at install time; the binary must match both.
#
# Usage: bash bundle/install_claude_code.sh [version] [sha256] [platform]
#   platform: linux-x64 (default, glibc) | linux-x64-musl
# Result: $R/tools/claude/<ver>/claude (glibc) or $R/tools/claude/<ver>-musl/claude, symlink $R/tools/claude/current.
set -euo pipefail
R=${R:-/scratch/user/rsi}
PY=${PY:-/scratch/user/miniconda3/envs/rsi/bin/python}
BASE=https://downloads.claude.ai/claude-code-releases
VER=${1:-2.1.311}
PLATFORM=${3:-linux-x64}
case "$VER/$PLATFORM" in
  2.1.311/linux-x64)      DEF_SHA=86dc1fcc69498159eaf15f446613e847010f9036bfbbdffd4c49350be8489ada ;;   # size 293776752
  2.1.311/linux-x64-musl) DEF_SHA=cd40dd327145ee47aa0deacd571801c1d782388f709d71242eee6b91de8ed11f ;;   # size 287442392
  *) DEF_SHA="" ;;
esac
SHA=${2:-$DEF_SHA}
case "$PLATFORM" in
  linux-x64) D=$R/tools/claude/$VER ;;
  linux-x64-musl) D=$R/tools/claude/$VER-musl ;;
  *) echo "unsupported platform $PLATFORM" >&2; exit 2 ;;
esac
mkdir -p "$D"
CURL=(curl -fsSL --proto =https --tlsv1.2 --retry 3)
if [ ! -x "$D/claude" ]; then
  "${CURL[@]}" -o "$D/manifest.json" "$BASE/$VER/manifest.json"
  read -r M_SHA M_SIZE < <("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1]))["platforms"][sys.argv[2]]; print(p["checksum"], p["size"])' "$D/manifest.json" "$PLATFORM")
  if [ -z "$SHA" ]; then echo "WARNING: no sha256 pinned for $VER/$PLATFORM; using the manifest's"; SHA=$M_SHA; fi
  if [ "$SHA" != "$M_SHA" ]; then echo "ERROR: pinned sha256 $SHA != manifest $M_SHA" >&2; exit 3; fi
  "${CURL[@]}" -o "$D/claude.part" "$BASE/$VER/$PLATFORM/claude"
  echo "$SHA  $D/claude.part" | sha256sum -c -
  [ "$(stat -c %s "$D/claude.part")" = "$M_SIZE" ] || { echo "ERROR: size mismatch" >&2; exit 3; }
  chmod 755 "$D/claude.part"
  mv "$D/claude.part" "$D/claude"
fi
ln -sfn "$(basename "$D")" "$R/tools/claude/current"
# version probe with a throwaway HOME/config dir (never the real ~ or ~/.claude)
PROBE=$R/tools/claude/.probe-home
mkdir -p "$PROBE"
env -i HOME="$PROBE" CLAUDE_CONFIG_DIR="$PROBE/.claude" PATH=/usr/bin:/bin \
  DISABLE_AUTOUPDATER=1 DISABLE_UPDATES=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \
  "$R/tools/claude/current/claude" --version
echo "CLAUDE_BIN=$R/tools/claude/current/claude"
