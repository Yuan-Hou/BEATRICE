#!/usr/bin/env bash
# Restore all xz-compressed data files in place (removes the .xz suffix).
set -euo pipefail
cd "$(dirname "$0")/.."
find data -name '*.xz' -print -exec xz -d {} \;
echo "done"
