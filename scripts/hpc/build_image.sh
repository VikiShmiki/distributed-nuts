#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults for the local cluster. Module names vary across HPC sites.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEF_FILE="${DEF_FILE:-$PROJECT_ROOT/singularity/abnuts.def}"
IMAGE_DIR="${IMAGE_DIR:-$PROJECT_ROOT/images}"
IMAGE="${IMAGE:-$IMAGE_DIR/abnuts.sif}"
SINGULARITY_BIN="${SINGULARITY_BIN:-singularity}"
SINGULARITY_BUILD_FLAGS="${SINGULARITY_BUILD_FLAGS:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"

if [[ -n "$MODULES_TO_LOAD" ]]; then
    if command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        module load $MODULES_TO_LOAD
    else
        echo "Requested MODULES_TO_LOAD='$MODULES_TO_LOAD', but no module command is available." >&2
        exit 2
    fi
fi

if ! command -v "$SINGULARITY_BIN" >/dev/null 2>&1; then
    echo "Could not find Singularity command '$SINGULARITY_BIN'." >&2
    echo "Set SINGULARITY_BIN or load the cluster's Singularity/Apptainer module." >&2
    exit 2
fi

mkdir -p "$IMAGE_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Definition:   $DEF_FILE"
echo "Image:        $IMAGE"
echo "Builder:      $SINGULARITY_BIN"

# shellcheck disable=SC2086
"$SINGULARITY_BIN" build $SINGULARITY_BUILD_FLAGS "$IMAGE" "$DEF_FILE"
