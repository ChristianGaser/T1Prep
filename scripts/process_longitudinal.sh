#!/usr/bin/env bash
# Batch longitudinal processing helper
# Groups time points by subject, runs inverse-consistent realignment, then per-time-point tool invocations

set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REALIGN_SCRIPT="$SCRIPT_DIR/realign_longitudinal.sh"
T1PREP_SCRIPT="$SCRIPT_DIR/T1Prep"

OUT_DIR=""
TEMP_TOOL_CMD=""
DRY_RUN=0

declare -a TIMEPOINT_ORDER=()
declare -a TIMEPOINT_DATA=()
declare -a REALIGN_ARGS=()
declare -a TEMP_TOOL_ARGS=()
declare -a SUBJECT_IDS=()
declare -a INPUT_PATHS=()

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

declare -a READ_LINES_RESULT=()
read_lines_to_array() {
    local data="$1"
    READ_LINES_RESULT=()
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        READ_LINES_RESULT+=("$line")
    done <<< "$data"
}

print_usage() {
    cat <<'USAGE'
Batch longitudinal processing.

Usage:
    scripts/process_longitudinal.sh \
        --out-dir /path/to/derivatives \
        [--temp-tool my_tool] [--temp-tool-arg "--flag"] \
        [--realign-arg "--some-option"] [--dry-run] \
        /path/to/tp1.nii.gz /path/to/tp2.nii.gz [...]

Notes:
    - Positional inputs are treated as time points for a single subject.
    - realign_longitudinal.sh must succeed before temp_tool steps run.
    - If --temp-tool is omitted, only the realignment step is performed.
    - Time points are taken in the given order.
    - Additional arguments can be supplied via repeated --*-arg flags.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
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
            --dry-run)
                DRY_RUN=1
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                if [[ "$1" == -* ]]; then
                    print_usage
                    die "Unknown option: $1"
                fi
                break
                ;;
        esac
        shift || true
    done

    if [[ $# -gt 0 ]]; then
        INPUT_PATHS+=("$@")
    fi
}

validate_inputs() {
    [[ ${#INPUT_PATHS[@]} -gt 0 ]] || die "At least one input file is required"

    TIMEPOINT_ORDER=()
    TIMEPOINT_DATA=()
    for ((tp_idx = 0; tp_idx < ${#INPUT_PATHS[@]}; tp_idx++)); do
        local tp_label
        tp_label="tp$(printf '%02d' "$((tp_idx + 1))")"
        TIMEPOINT_ORDER+=("$tp_label")
        TIMEPOINT_DATA+=("${INPUT_PATHS[$tp_idx]}")
    done

    [[ -n "$OUT_DIR" ]] || die "--out-dir is required"
    [[ -x "$REALIGN_SCRIPT" ]] || die "Realignment script not executable: $REALIGN_SCRIPT"
    if [[ -n "$TEMP_TOOL_CMD" ]]; then
        command -v "$TEMP_TOOL_CMD" >/dev/null 2>&1 || die "temp tool command not found: $TEMP_TOOL_CMD"
    fi

    read_lines_to_array "${TIMEPOINT_DATA[0]}"
    SUBJECT_COUNT=${#READ_LINES_RESULT[@]}
    [[ $SUBJECT_COUNT -gt 0 ]] || die "No subjects detected in first time point"

    for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
        local tp_label="${TIMEPOINT_ORDER[$tp_idx]}"
        read_lines_to_array "${TIMEPOINT_DATA[$tp_idx]}"
        [[ ${#READ_LINES_RESULT[@]} -eq $SUBJECT_COUNT ]] || die "Time point '$tp_label' has ${#READ_LINES_RESULT[@]} entries but expected $SUBJECT_COUNT"
    done

    SUBJECT_IDS=()
    for ((i = 1; i <= SUBJECT_COUNT; i++)); do
        SUBJECT_IDS+=("subject_$(printf '%02d' "$i")")
    done
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
        mkdir -p "$realign_dir"

        local -a subject_tp_paths=()
        for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
            read_lines_to_array "${TIMEPOINT_DATA[$tp_idx]}"
            subject_tp_paths+=("${READ_LINES_RESULT[$idx]}")
        done

        echo "Processing $subject_id"
        run_step "$REALIGN_SCRIPT" --use-skullstrip --inverse-consistent --update-headers --inputs "${subject_tp_paths[@]}" --out-dir "$realign_dir" "${REALIGN_ARGS[@]+"${REALIGN_ARGS[@]}"}"

        if [[ -n "$TEMP_TOOL_CMD" ]]; then
            mkdir -p "$tool_dir"
            for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
                local tp_label="${TIMEPOINT_ORDER[$tp_idx]}"
                local tp_path="${subject_tp_paths[$tp_idx]}"
                local tp_out="$tool_dir/$tp_label"
                mkdir -p "$tp_out"
                run_step "$TEMP_TOOL_CMD" "${TEMP_TOOL_ARGS[@]+"${TEMP_TOOL_ARGS[@]}"}" "$tp_path" "$tp_out"
            done
        fi
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
