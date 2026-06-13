#!/bin/bash
set -e

mode=""
node=""
input=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
  echo "用法: $0 -m <http|flow|node> [-n <节点ID>] [-i <输入JSON或图片路径>]"
}

while getopts "m:n:i:h" opt; do
  case "$opt" in
    m) mode="$OPTARG" ;;
    n) node="$OPTARG" ;;
    i) input="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage; exit 1 ;;
  esac
done

if [ -z "$mode" ]; then
  usage
  exit 1
fi

cmd=(python "$WORK_DIR/src/main.py" -m "$mode")
if [ -n "$node" ]; then
  cmd+=(-n "$node")
fi
if [ -n "$input" ]; then
  cmd+=(-i "$input")
fi

"${cmd[@]}"
