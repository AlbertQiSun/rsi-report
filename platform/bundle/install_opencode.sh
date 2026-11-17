#!/bin/bash
# Install the OpenCode Linux x64 binary into $R/tools/opencode/<ver>/ (no node/npm needed).
# Source: the npm registry tarball of the platform package (same bytes `npm i opencode-ai` installs),
# checked against the registry's sha1 (dist.shasum, looked up 2026-11-17 for 1.19.6).
# Usage: bash bundle/install_opencode.sh [version] [sha1] [variant]
#   variant: linux-x64 (default) | linux-x64-baseline (CPUs without AVX2)
set -euo pipefail
R=${R:-/scratch/user/rsi}
VER=${1:-1.19.6}
VARIANT=${3:-linux-x64}
case "$VER/$VARIANT" in
  1.19.6/linux-x64)          DEF_SHA=167f49072a7af90684417ac1b7aab69d417e5cdc ;;
  1.19.6/linux-x64-baseline) DEF_SHA=7e8dbec59fdffcf07498d0af18bb3ca648cbeb2d ;;
  *) DEF_SHA="" ;;
esac
SHA=${2:-$DEF_SHA}
PKG=opencode-$VARIANT
D=$R/tools/opencode/$VER-$VARIANT
mkdir -p "$D"
if [ ! -x "$D/package/bin/opencode" ]; then
  curl -L --fail --retry 3 -o "$D/pkg.tgz" "https://registry.npmjs.org/$PKG/-/$PKG-$VER.tgz"
  if [ -n "$SHA" ]; then echo "$SHA  $D/pkg.tgz" | sha1sum -c -; else echo "WARNING: no sha1 pinned for $VER/$VARIANT"; sha1sum "$D/pkg.tgz"; fi
  tar -xzf "$D/pkg.tgz" -C "$D"
fi
ln -sfn "$VER-$VARIANT" "$R/tools/opencode/current"
"$R/tools/opencode/current/package/bin/opencode" --version
echo "OPENCODE_BIN=$R/tools/opencode/current/package/bin/opencode"
