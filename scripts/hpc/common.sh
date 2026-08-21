#!/usr/bin/env bash

# Shared helpers for ABNUTS SLURM wrappers.

abnuts_parse_slurm_duration_seconds() {
    local value="${1:-}"
    value="${value//[[:space:]]/}"
    if [[ -z "$value" || "$value" == "N/A" || "$value" == "UNLIMITED" || "$value" == "NOT_SET" ]]; then
        return 1
    fi

    local days=0
    local clock="$value"
    if [[ "$clock" == *-* ]]; then
        days="${clock%%-*}"
        clock="${clock#*-}"
    fi

    local parts=()
    IFS=":" read -r -a parts <<< "$clock"

    local hours=0
    local minutes=0
    local seconds=0
    case "${#parts[@]}" in
        3)
            hours="${parts[0]}"
            minutes="${parts[1]}"
            seconds="${parts[2]}"
            ;;
        2)
            minutes="${parts[0]}"
            seconds="${parts[1]}"
            ;;
        1)
            seconds="${parts[0]}"
            ;;
        *)
            return 1
            ;;
    esac

    if ! [[ "$days" =~ ^[0-9]+$ && "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    echo $((10#$seconds + 60 * 10#$minutes + 3600 * 10#$hours + 86400 * 10#$days))
}

abnuts_slurm_field() {
    local format="$1"
    if ! command -v squeue >/dev/null 2>&1; then
        return 1
    fi

    local candidates=()
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        candidates+=("${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}")
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        candidates+=("$SLURM_JOB_ID")
    fi
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
        candidates+=("$SLURM_ARRAY_JOB_ID")
    fi
    if (( ${#candidates[@]} == 0 )); then
        return 1
    fi

    local candidate=""
    local value=""
    for candidate in "${candidates[@]}"; do
        value="$(squeue -h -j "$candidate" -o "$format" 2>/dev/null | awk 'NF {print $1; exit}')"
        if [[ -n "$value" ]]; then
            echo "$value"
            return 0
        fi
    done
    return 1
}

abnuts_print_slurm_time_limit() {
    local limit=""
    local remaining=""
    limit="$(abnuts_slurm_field "%l" || true)"
    remaining="$(abnuts_slurm_field "%L" || true)"
    if [[ -n "$limit" || -n "$remaining" ]]; then
        echo "SLURM time limit: ${limit:-unknown}; remaining at launch: ${remaining:-unknown}"
    fi
    echo "Walltime guard cushion: ${ABNUTS_TIMEOUT_CUSHION_SECONDS:-600}s"
    if [[ -n "${ABNUTS_JOB_TIMEOUT_SECONDS:-}" ]]; then
        echo "Walltime guard override: ${ABNUTS_JOB_TIMEOUT_SECONDS}s"
    else
        echo "Walltime guard mode: auto from SLURM remaining time"
    fi
}

abnuts_resolve_timeout_seconds() {
    if [[ -n "${ABNUTS_JOB_TIMEOUT_SECONDS:-}" ]]; then
        echo "$ABNUTS_JOB_TIMEOUT_SECONDS"
        return 0
    fi

    local remaining_text=""
    local remaining_seconds=""
    local cushion="${ABNUTS_TIMEOUT_CUSHION_SECONDS:-600}"
    remaining_text="$(abnuts_slurm_field "%L" || true)"
    if [[ -n "$remaining_text" ]] && remaining_seconds="$(abnuts_parse_slurm_duration_seconds "$remaining_text")"; then
        if (( remaining_seconds > cushion )); then
            echo $((remaining_seconds - cushion))
            return 0
        fi
        echo 1
        return 0
    fi

    echo "${ABNUTS_FALLBACK_JOB_TIMEOUT_SECONDS:-42300}"
}

abnuts_run_with_timeout() {
    local timeout_seconds=""
    timeout_seconds="$(abnuts_resolve_timeout_seconds)"
    if ! [[ "$timeout_seconds" =~ ^[0-9]+$ ]]; then
        echo "Invalid timeout seconds: $timeout_seconds" >&2
        return 2
    fi

    if (( timeout_seconds == 0 )); then
        echo "Walltime guard disabled by ABNUTS_JOB_TIMEOUT_SECONDS=0."
        "$@"
        return $?
    fi

    if (( timeout_seconds <= 60 )); then
        echo "Refusing to start command with only ${timeout_seconds}s before the walltime guard." >&2
        return 124
    fi

    if ! command -v timeout >/dev/null 2>&1; then
        echo "The 'timeout' command is unavailable; running without a local walltime guard." >&2
        "$@"
        return $?
    fi

    local kill_after="${ABNUTS_TIMEOUT_KILL_AFTER_SECONDS:-120}"
    echo "Running with local walltime guard: timeout=${timeout_seconds}s kill_after=${kill_after}s"

    local status=0
    if timeout --signal=TERM --kill-after="${kill_after}s" "${timeout_seconds}s" "$@"; then
        status=0
    else
        status=$?
    fi

    if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
        echo "Command stopped by ABNUTS walltime guard before the SLURM hard limit." >&2
    fi
    return "$status"
}
