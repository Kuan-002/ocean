#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {ac|grpo} /external/path/to/run_dir [evaluator options...]" >&2
  exit 2
fi

method=$1
run_dir=$2
shift 2

case "$method" in
  ac|grpo) ;;
  *)
    echo "Error: method must be 'ac' or 'grpo'." >&2
    exit 2
    ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}

cd "$repo_dir"
exec "$python_bin" scripts/eval_selector_v1_test.py \
  --method "$method" \
  --run_dir "$run_dir" \
  "$@"
