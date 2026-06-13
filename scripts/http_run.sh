#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-5000}"

while getopts "p:h" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    h) echo "用法: $0 [-p 端口]"; exit 0 ;;
    \?) exit 1 ;;
  esac
done

python "$WORK_DIR/src/main.py" -m http -p "$PORT"
