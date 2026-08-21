#!/usr/bin/env bash
#SBATCH --job-name=abnuts-profile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=viktor.najdovski@students.finki.ukim.mk
#SBATCH --mail-type=FAIL

set -euo pipefail

# Edit these defaults for the local cluster before submitting.
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/AGENTS.md" ]]; then
        PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    else
        PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
# shellcheck source=scripts/hpc/common.sh
source "$PROJECT_ROOT/scripts/hpc/common.sh"
IMAGE="${IMAGE:-$PROJECT_ROOT/images/abnuts.sif}"
RUNROOT="${RUNROOT:-${SCRATCH:-$HOME}/abnuts_runs/${SLURM_JOB_ID:-local}}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
ABNUTS_BACKEND="${ABNUTS_BACKEND:-gpu}"
NSYS_BIN="${NSYS_BIN:-nsys}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/configs/funnel_ablation.yaml}"
PROFILE_NAME="${PROFILE_NAME:-full}"
RAW_OUT="${RAW_OUT:-/runs/raw/profiling/full}"
LATEX_OUT="${LATEX_OUT:-/runs/latex/profiling/full}"
TRACE_BASENAME="${TRACE_BASENAME:-abnuts_profile_${SLURM_JOB_ID:-local}}"
EXTRA_BENCHMARK_ARGS="${EXTRA_BENCHMARK_ARGS:-}"

if [[ -n "$MODULES_TO_LOAD" ]]; then
    if command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        module load $MODULES_TO_LOAD
    else
        echo "Requested MODULES_TO_LOAD='$MODULES_TO_LOAD', but no module command is available." >&2
        exit 2
    fi
fi

mkdir -p "$RUNROOT/nsight" "$PROJECT_ROOT/logs"

echo "SLURM job id: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Run root: $RUNROOT"
echo "Image: $IMAGE"
echo "Profile: $PROFILE_NAME"
echo "Backend: $ABNUTS_BACKEND"
echo "Raw output directory: $RUNROOT/raw/profiling/$PROFILE_NAME"
echo "LaTeX output directory: $RUNROOT/latex/profiling/$PROFILE_NAME"
echo "Nsight trace basename: $RUNROOT/nsight/$TRACE_BASENAME"
echo "SLURM stdout: $PROJECT_ROOT/logs/abnuts-profile-${SLURM_JOB_ID:-local}.out"
echo "SLURM stderr: $PROJECT_ROOT/logs/abnuts-profile-${SLURM_JOB_ID:-local}.err"
abnuts_print_slurm_time_limit

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found on the host; continuing so CPU syntax checks remain valid."
fi

export NSYS_BIN CONFIG_PATH PROFILE_NAME RAW_OUT LATEX_OUT TRACE_BASENAME ABNUTS_BACKEND EXTRA_BENCHMARK_ARGS

CONTAINER_COMMAND='
set -euo pipefail
cd /workspace
# The image pip-installs a snapshot of src/ at build time. Without this the
# bound workspace is shadowed by that snapshot and the job silently runs stale
# code: verified on 2026-08-09 against an image built 2026-05-26, which predates
# the U-turn fix, the control-flow transition and the per-bucket executor.
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
python - <<'GUARD'
import inspect, pathlib, sys
import abnuts
resolved = pathlib.Path(inspect.getfile(abnuts)).resolve()
if not str(resolved).startswith("/workspace/src/"):
    sys.exit(
        f"abnuts resolved to {resolved}, not the bound workspace. "
        "The job would run a stale snapshot; refusing to produce results."
    )
print(f"abnuts source verified: {resolved}")
GUARD
mkdir -p "$(dirname "$RAW_OUT")" "$(dirname "$LATEX_OUT")" /runs/nsight
BENCHMARK_COMMAND=(
    python -m abnuts.experiments.run_benchmark
    --config "$CONFIG_PATH"
    --backend "$ABNUTS_BACKEND"
    --profile "$PROFILE_NAME"
    --enable-timing-breakdown
    --enable-profiler-markers
    --out "$RAW_OUT"
)
if [[ -n "$EXTRA_BENCHMARK_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=($EXTRA_BENCHMARK_ARGS)
    BENCHMARK_COMMAND+=("${EXTRA_ARGS[@]}")
fi
if command -v "$NSYS_BIN" >/dev/null 2>&1; then
    "$NSYS_BIN" profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --output "/runs/nsight/$TRACE_BASENAME" \
        "${BENCHMARK_COMMAND[@]}"
else
    echo "Nsight Systems command $NSYS_BIN not found inside the container; running timing breakdown only." >&2
    "${BENCHMARK_COMMAND[@]}"
fi
python -m abnuts.analysis.report --input "$RAW_OUT" --out "$LATEX_OUT"
'

abnuts_run_with_timeout "$SINGULARITY_BIN" exec --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    --bind "$RUNROOT:/runs" \
    "$IMAGE" \
    bash -lc "$CONTAINER_COMMAND"
