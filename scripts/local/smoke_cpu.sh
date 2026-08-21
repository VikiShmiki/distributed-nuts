#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNROOT="${RUNROOT:-${SCRATCH:-$HOME}/abnuts_runs/${SLURM_JOB_ID:-local}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RUNROOT"

echo "Project root: $PROJECT_ROOT"
echo "Run root: $RUNROOT"
echo "Python: $PYTHON_BIN"

PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m abnuts.experiments.smoke \
        --config "$PROJECT_ROOT/configs/smoke.yaml" \
        --backend cpu \
        --out "$RUNROOT/smoke_cpu"
