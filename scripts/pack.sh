#!/bin/bash
set -eo pipefail

python -m compileall src
echo "[pack] source check completed"
