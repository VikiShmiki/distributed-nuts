#!/usr/bin/env bash
#SBATCH --job-name=abnuts-correctness
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-0
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
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUNROOT="${RUNROOT:-${SCRATCH:-$HOME}/abnuts_runs/$RUN_ID}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
ABNUTS_BACKEND="${ABNUTS_BACKEND:-gpu}"
ABNUTS_CORRECTNESS_PROFILE="${ABNUTS_CORRECTNESS_PROFILE:-paper}"
ABNUTS_ARRAY_TASK_ID="${ABNUTS_ARRAY_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
ABNUTS_DRY_RUN="${ABNUTS_DRY_RUN:-0}"
ABNUTS_LIST_TASKS="${ABNUTS_LIST_TASKS:-0}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_CORRECTNESS_ARGS="${EXTRA_CORRECTNESS_ARGS:-}"

TASK_SPECS=(
    "long_trace_funnel|configs/long_trace.yaml|${ABNUTS_CORRECTNESS_PROFILE}"
)

if [[ "$ABNUTS_LIST_TASKS" == "1" ]]; then
    for index in "${!TASK_SPECS[@]}"; do
        echo "$index ${TASK_SPECS[$index]}"
    done
    exit 0
fi

if (( ABNUTS_ARRAY_TASK_ID < 0 || ABNUTS_ARRAY_TASK_ID >= ${#TASK_SPECS[@]} )); then
    echo "Array task ${ABNUTS_ARRAY_TASK_ID} is out of range 0-$(( ${#TASK_SPECS[@]} - 1 ))." >&2
    exit 2
fi

IFS="|" read -r TASK_NAME CONFIG_REL PROFILE_NAME <<< "${TASK_SPECS[$ABNUTS_ARRAY_TASK_ID]}"
OUTPUT_REL="raw/long_trace/${PROFILE_NAME}_${TASK_NAME}"
HOST_OUT="$RUNROOT/$OUTPUT_REL"
CONTAINER_OUT="/runs/$OUTPUT_REL"
CONTAINER_CONFIG="/workspace/$CONFIG_REL"
LOG_STEM="logs/abnuts-correctness-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${ABNUTS_ARRAY_TASK_ID}"

if [[ -n "$MODULES_TO_LOAD" ]]; then
    if command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        module load $MODULES_TO_LOAD
    else
        echo "Requested MODULES_TO_LOAD='$MODULES_TO_LOAD', but no module command is available." >&2
        exit 2
    fi
fi

mkdir -p "$RUNROOT" "$PROJECT_ROOT/logs"

echo "SLURM job id: ${SLURM_JOB_ID:-local}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID:-$ABNUTS_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Run root: $RUNROOT"
echo "Image: $IMAGE"
echo "Task: $TASK_NAME"
echo "Config: $PROJECT_ROOT/$CONFIG_REL"
echo "Profile: $PROFILE_NAME"
echo "Backend: $ABNUTS_BACKEND"
echo "Output directory: $HOST_OUT"
echo "Log files: $PROJECT_ROOT/${LOG_STEM}.out and $PROJECT_ROOT/${LOG_STEM}.err"
abnuts_print_slurm_time_limit

if [[ -f "$HOST_OUT/manifest.json" && "$OVERWRITE" != "1" ]]; then
    if [[ ! -f "$HOST_OUT/config.yaml" ]]; then
        cp "$PROJECT_ROOT/$CONFIG_REL" "$HOST_OUT/config.yaml"
    fi
    echo "Skipping completed correctness task because manifest exists: $HOST_OUT/manifest.json"
    exit 0
fi

if [[ "$ABNUTS_DRY_RUN" == "1" ]]; then
    echo "Dry run: would execute correctness task $TASK_NAME and copy config.yaml after success."
    exit 0
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "Expected Singularity image is missing: $IMAGE" >&2
    echo "Build it with scripts/hpc/build_image.sh or set IMAGE to an existing .sif file." >&2
    exit 2
fi

if ! command -v "$SINGULARITY_BIN" >/dev/null 2>&1; then
    echo "Could not find Singularity command '$SINGULARITY_BIN'." >&2
    echo "Set SINGULARITY_BIN or load the cluster's Singularity/Apptainer module." >&2
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found on the host; continuing because this script may run CPU-only configs."
fi

export CONTAINER_CONFIG CONTAINER_OUT PROFILE_NAME OVERWRITE ABNUTS_BACKEND EXTRA_CORRECTNESS_ARGS

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
mkdir -p "$(dirname "$CONTAINER_OUT")"
RUN_ARGS=(
    python -m abnuts.experiments.run_long_trace
    --config "$CONTAINER_CONFIG"
    --backend "$ABNUTS_BACKEND"
    --profile "$PROFILE_NAME"
    --out "$CONTAINER_OUT"
)
if [[ "$OVERWRITE" == "1" ]]; then
    RUN_ARGS+=(--overwrite)
fi
if [[ -n "$EXTRA_CORRECTNESS_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=($EXTRA_CORRECTNESS_ARGS)
    RUN_ARGS+=("${EXTRA_ARGS[@]}")
fi
"${RUN_ARGS[@]}"
cp "$CONTAINER_CONFIG" "$CONTAINER_OUT/config.yaml"
'

abnuts_run_with_timeout "$SINGULARITY_BIN" exec --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    --bind "$RUNROOT:/runs" \
    "$IMAGE" \
    bash -lc "$CONTAINER_COMMAND"
