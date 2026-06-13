#!/bin/bash
set -eo pipefail

if command -v uv >/dev/null 2>&1; then
  uv sync
else
  python -m venv .venv
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  fi
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi
