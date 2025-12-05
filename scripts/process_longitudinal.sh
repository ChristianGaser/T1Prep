#!/usr/bin/env bash
# Batch longitudinal processing helper
# Groups time points by subject, runs inverse-consistent realignment, then per-time-point tool invocations

set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REALIGN_SCRIPT="$SCRIPT_DIR/realign_longitudinal.sh"

OUT_DIR=""
TEMP_TOOL_CMD="temp_tool"
SUBJECT_IDS_RAW=""
DRY_RUN=0

declare -a TIMEPOINT_ORDER=()
declare -A TIMEPOINT_DATA=()
declare -a REALIGN_ARGS=()
declare -a TEMP_TOOL_ARGS=()

die() {
    echo "ERROR: $*" >&2
    exit 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

add_timepoint() {
    local definition="$1"
    if [[ "$definition" != *:* ]]; then
        die "Time point definition must be LABEL:path1,path2,..."
    fi
    local label="${definition%%:*}"
    local data="${definition#*:}"
    label="$(trim "$label")"
    [[ -n "$label" ]] || die "Time point label cannot be empty"
    [[ -z "${TIMEPOINT_DATA[$label]:-}" ]] || die "Duplicate definition for time point '$label'"

    IFS=',' read -r -a raw_paths <<< "$data"
    local -a cleaned_paths=()
    for path in "${raw_paths[@]}"; do
        local cleaned="$(trim "$path")"
        [[ -n "$cleaned" ]] || die "Empty path found in time point '$label'"
        cleaned_paths+=("$cleaned")
    done

    [[ ${#cleaned_paths[@]} -gt 0 ]] || die "No paths provided for time point '$label'"

    TIMEPOINT_ORDER+=("$label")
    TIMEPOINT_DATA["$label"]="$(printf '%s\n' "${cleaned_paths[@]}")"
}

print_usage() {
    cat <<'USAGE'
Batch longitudinal processing.

Usage:
    scripts/process_longitudinal.sh \
        --timepoint T1:/sub-01_tp1.nii.gz,/sub-02_tp1.nii.gz \
        --timepoint T2:/sub-01_tp2.nii.gz,/sub-02_tp2.nii.gz \
        --out-dir /path/to/derivatives \
        [--subject-ids sub-01,sub-02] \
        [--temp-tool my_tool] [--temp-tool-arg "--flag"] \
        [--realign-arg "--some-option"] [--dry-run]

Notes:
    - All subjects must share the same number of time points.
    - realign_longitudinal.sh must succeed before temp_tool steps run.
    - Additional arguments can be supplied via repeated --*-arg flags.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --timepoint)
                shift || die "--timepoint requires LABEL:paths"
                add_timepoint "$1"
                ;;
            --out-dir)
                shift || die "--out-dir requires a path"
                OUT_DIR="$1"
                ;;
            --temp-tool)
                shift || die "--temp-tool requires a command"
                TEMP_TOOL_CMD="$1"
                ;;
            --realign-arg)
                shift || die "--realign-arg needs a value"
                REALIGN_ARGS+=("$1")
                ;;
            --temp-tool-arg)
                shift || die "--temp-tool-arg needs a value"
                TEMP_TOOL_ARGS+=("$1")
                ;;
            --subject-ids)
                shift || die "--subject-ids requires comma-separated IDs"
                SUBJECT_IDS_RAW="$1"
                ;;
            --dry-run)
                DRY_RUN=1
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                print_usage
                die "Unknown option: $1"
                ;;
        esac
        shift || true
    done
}

validate_inputs() {
    [[ ${#TIMEPOINT_ORDER[@]} -gt 0 ]] || die "At least one --timepoint is required"
    [[ -n "$OUT_DIR" ]] || die "--out-dir is required"
    [[ -x "$REALIGN_SCRIPT" ]] || die "Realignment script not executable: $REALIGN_SCRIPT"
    command -v "$TEMP_TOOL_CMD" >/dev/null 2>&1 || die "temp tool command not found: $TEMP_TOOL_CMD"

    local first_tp="${TIMEPOINT_ORDER[0]}"
    mapfile -t FIRST_LIST <<< "${TIMEPOINT_DATA[$first_tp]}"
    SUBJECT_COUNT=${#FIRST_LIST[@]}
    [[ $SUBJECT_COUNT -gt 0 ]] || die "No subjects detected in first time point"

    for tp in "${TIMEPOINT_ORDER[@]}"; do
        mapfile -t paths <<< "${TIMEPOINT_DATA[$tp]}"
        [[ ${#paths[@]} -eq $SUBJECT_COUNT ]] || die "Time point '$tp' has ${#paths[@]} entries but expected $SUBJECT_COUNT"
    done

    declare -ag SUBJECT_IDS=()
    if [[ -n "$SUBJECT_IDS_RAW" ]]; then
        IFS=',' read -r -a SUBJECT_IDS <<< "$SUBJECT_IDS_RAW"
        [[ ${#SUBJECT_IDS[@]} -eq $SUBJECT_COUNT ]] || die "Provided subject IDs count (${#SUBJECT_IDS[@]}) does not match $SUBJECT_COUNT"
        for idx in "${!SUBJECT_IDS[@]}"; do
            SUBJECT_IDS[$idx]="$(trim "${SUBJECT_IDS[$idx]}")"
            [[ -n "${SUBJECT_IDS[$idx]}" ]] || die "Subject ID at position $((idx + 1)) is empty"
        done
    else
        for ((i = 1; i <= SUBJECT_COUNT; i++)); do
            SUBJECT_IDS+=("subject_$(printf '%02d' "$i")")
        done
    fi
}

run_step() {
    if (( DRY_RUN )); then
        echo "DRY-RUN: $*"
    else
        "$@"
    fi
}

process_subjects() {
    mkdir -p "$OUT_DIR"

    for ((idx = 0; idx < SUBJECT_COUNT; idx++)); do
        local subject_id="${SUBJECT_IDS[$idx]}"
        local subject_root="$OUT_DIR/$subject_id"
        local realign_dir="$subject_root/realign"
        local tool_dir="$subject_root/temp_tool"
        mkdir -p "$realign_dir" "$tool_dir"

        local -a subject_tp_paths=()
        for tp in "${TIMEPOINT_ORDER[@]}"; do
            mapfile -t tp_paths <<< "${TIMEPOINT_DATA[$tp]}"
            subject_tp_paths+=("${tp_paths[$idx]}")
        done

        echo "Processing $subject_id"
        run_step "$REALIGN_SCRIPT" --inverse-consistent --update-headers --inputs "${subject_tp_paths[@]}" --out-dir "$realign_dir" "${REALIGN_ARGS[@]}"

        for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
            local tp_label="${TIMEPOINT_ORDER[$tp_idx]}"
            local tp_path="${subject_tp_paths[$tp_idx]}"
            local tp_out="$tool_dir/$tp_label"
            mkdir -p "$tp_out"
            run_step "$TEMP_TOOL_CMD" "${TEMP_TOOL_ARGS[@]}" "$tp_path" "$tp_out"
        done
    done
}

main() {
    if [[ $# -eq 0 ]]; then
        print_usage
        exit 1
    fi
    parse_args "$@"
    validate_inputs
    process_subjects
}

main "$@"
