#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/update-sqlite-vector.sh [TAG]
# Falls back to latest GitHub release if TAG is not provided.
if [[ -n "${1:-}" ]]; then
  VERSION="$1"
else
  VERSION=$(curl -sfL "https://api.github.com/repos/sqliteai/sqlite-vector/releases/latest" \
    | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
  if [[ -z "$VERSION" ]]; then
    echo "Error: failed to fetch latest release. Please specify a tag." >&2
    echo "Usage: $0 <tag>  (e.g., $0 v0.9.90)" >&2
    exit 1
  fi
fi

REPO="https://raw.githubusercontent.com/sqliteai/sqlite-vector/${VERSION}"
DEST="vendor/sqlite-vector"

mkdir -p "${DEST}/src" "${DEST}/libs/fp16"

for f in sqlite-vector.{c,h} distance-{cpu,sse2,avx2,avx512,neon}.{c,h}; do
  curl -sfL "${REPO}/src/${f}" -o "${DEST}/src/${f}"
done
for f in fp16.h bitcasts.h macros.h; do
  curl -sfL "${REPO}/libs/fp16/${f}" -o "${DEST}/libs/fp16/${f}"
done
curl -sfL "${REPO}/LICENSE.md" -o "${DEST}/LICENSE.md"
echo "${VERSION}" > "${DEST}/VERSION"
echo "sqlite-vector vendored at ${VERSION}"
