#!/usr/bin/env bash
set -euo pipefail

cd /experiment

if [ ! -d .venv ]; then
    python -m venv --system-site-packages .venv
fi

if [ $# -eq 0 ]; then
    set -- python -m main
fi

uv sync --no-dev
exec "$@"
