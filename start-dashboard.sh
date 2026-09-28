#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"
# Prefer an installed modern Node; Codex's bundled runtime is the local fallback.
dashboard_node="$(command -v node || true)"
if [ -z "$dashboard_node" ] || ! "$dashboard_node" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' >/dev/null 2>&1; then
  dashboard_node="${HOME}/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
fi
if [ ! -x "$dashboard_node" ] || ! "$dashboard_node" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' >/dev/null 2>&1; then
  echo 'Node.js 18 이상이 필요합니다. 설치 후 다시 실행해 주세요.' >&2
  exit 1
fi
exec "$dashboard_node" dashboard/server.mjs
