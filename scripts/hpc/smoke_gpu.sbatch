#!/usr/bin/env bash
#SBATCH --job-name=abnuts-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
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
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Run root: $RUNROOT"
echo "Image: $IMAGE"
echo "Validating JAX GPU visibility, then running the smoke manifest command."
abnuts_print_slurm_time_limit

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found on the host; continuing because this scaffold may be syntax-checked off-GPU."
fi

abnuts_run_with_timeout "$SINGULARITY_BIN" exec --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    --bind "$RUNROOT:/runs" \
    "$IMAGE" \
    python - <<'PY'
import jax

jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp

devices = jax.devices("gpu")
if not devices:
    raise SystemExit("No JAX GPU devices are visible inside the container.")

value = jnp.arange(8, dtype=jnp.float32).sum()
value.block_until_ready()
print(f"JAX GPU validation ok: backend={jax.default_backend()}, devices={devices}")
PY

# PYTHONPATH makes the bound workspace shadow the snapshot of src/ that the
# image pip-installs at build time. Without it the job silently runs stale code.
abnuts_run_with_timeout "$SINGULARITY_BIN" exec --nv \
    --env PYTHONPATH=/workspace/src \
    --bind "$PROJECT_ROOT:/workspace" \
    --bind "$RUNROOT:/runs" \
    "$IMAGE" \
    python -m abnuts.experiments.smoke \
        --config /workspace/configs/smoke.yaml \
        --backend cpu \
        --out /runs/smoke_gpu
