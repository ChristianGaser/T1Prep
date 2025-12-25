#!/usr/bin/env bash
# Batch longitudinal processing helper
# Groups time points by subject, runs inverse-consistent realignment, then per-time-point tool invocations

set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REALIGN_SCRIPT="$SCRIPT_DIR/realign_longitudinal.sh"
T1PREP_CMD="$SCRIPT_DIR/T1Prep"

# We run with nounset, but utils.sh is shared with scripts that don't.
# Temporarily disable nounset while sourcing to avoid unbound variable errors.
set +u
source "$SCRIPT_DIR/utils.sh"
set -u

T1PREP_VERSION=0.2.4
USE_AMAP=0

OUT_DIR=""
DRY_RUN=0

declare -a T1PREP_ARGS=()
declare -a TIMEPOINT_ORDER=()
declare -a TIMEPOINT_DATA=()
declare -a REALIGN_ARGS=()
declare -a SUBJECT_IDS=()
declare -a INPUT_PATHS=()
INPUT_MODE=""

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
        [--out-dir /path/to/output-or-dataset-root] \
        [--t1prep-arg "--flag"] \
        [--realign-arg "--some-option"] [--dry-run] \
        /path/to/tp1.nii.gz /path/to/tp2.nii.gz [...]

Notes:
    - If inputs are NIfTI files: treated as time points for a single subject.
    - If inputs are text files: each file is a time point list; each line is a subject.
    - realign_longitudinal.sh must succeed before temp_tool steps run.
    - If --t1prep-arg is omitted, only the realignment step is performed.
    - Time points are taken in the given order.
    - Additional arguments can be supplied via repeated --*-arg flags.
    - If --out-dir is omitted, the output folder is derived like scripts/T1Prep.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-dir)
                shift || die "--out-dir requires a path"
                OUT_DIR="$1"
                ;;
            --t1prep-arg)
                shift || die "--t1prep-arg needs a value"
                  T1PREP_ARGS+=("$1")
                ;;
            --realign-arg)
                shift || die "--realign-arg needs a value"
                REALIGN_ARGS+=("$1")
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

is_nifti() {
    [[ "$1" =~ \.nii(\.gz)?$ ]]
}

declare -a READ_TIMEPOINT_RESULT=()
read_timepoint_to_array() {
    local tp_value="$1"
    READ_TIMEPOINT_RESULT=()

    if [[ "$INPUT_MODE" == "nifti" ]]; then
        READ_TIMEPOINT_RESULT+=("$tp_value")
        return 0
    fi

    [[ -f "$tp_value" ]] || die "Time point list file not found: $tp_value"
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="$(trim "$line")"
        [[ -n "$line" ]] || continue
        READ_TIMEPOINT_RESULT+=("$line")
    done < "$tp_value"
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

    if is_nifti "${INPUT_PATHS[0]}"; then
        INPUT_MODE="nifti"
        for p in "${INPUT_PATHS[@]}"; do
            is_nifti "$p" || die "Mixed inputs: all inputs must be NIfTI files, or all must be list files"
        done
    else
        INPUT_MODE="list"
    fi

    [[ -x "$REALIGN_SCRIPT" ]] || die "Realignment script not executable: $REALIGN_SCRIPT"

    read_timepoint_to_array "${TIMEPOINT_DATA[0]}"
    SUBJECT_COUNT=${#READ_TIMEPOINT_RESULT[@]}
    [[ $SUBJECT_COUNT -gt 0 ]] || die "No subjects detected in first time point"

    for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
        local tp_label="${TIMEPOINT_ORDER[$tp_idx]}"
        read_timepoint_to_array "${TIMEPOINT_DATA[$tp_idx]}"
        [[ ${#READ_TIMEPOINT_RESULT[@]} -eq $SUBJECT_COUNT ]] || die "Time point '$tp_label' has ${#READ_TIMEPOINT_RESULT[@]} entries but expected $SUBJECT_COUNT"
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
    for ((idx = 0; idx < SUBJECT_COUNT; idx++)); do
        local -a subject_tp_paths=()
        for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
            read_timepoint_to_array "${TIMEPOINT_DATA[$tp_idx]}"
            subject_tp_paths+=("${READ_TIMEPOINT_RESULT[$idx]}")
        done

        local first_tp="${subject_tp_paths[0]}"
        [[ -f "$first_tp" ]] || die "Input file not found: $first_tp"

        # Derive output folder like scripts/T1Prep.
        t1prep_output_folder_from_input "$first_tp" "${OUT_DIR:-}" "${T1PREP_VERSION}" "${USE_AMAP}"

        local subject_root="$outdir0"
        local mri_dir
        local surf_dir
        if [[ "${use_subfolder}" -eq 0 ]]; then
            mri_dir="$outdir0"
            surf_dir="$outdir0"
        else
            mri_dir="$outdir0/mri"
            surf_dir="$outdir0/surf"
        fi

        mkdir -p "$mri_dir"

        echo "Processing -> $subject_root"

        # Safety for --update-headers: never write back into the input folder.
        for tp_path in "${subject_tp_paths[@]}"; do
            if [[ "$(cd "$(dirname "$tp_path")" && pwd)" == "$(cd "$mri_dir" && pwd)" ]]; then
                die "Refusing to run --update-headers with out-dir equal to input folder: $mri_dir"
            fi
        done

        local -a realign_cmd=(
            "$REALIGN_SCRIPT"
            --use-skullstrip
            --inverse-consistent
            --update-headers
            --inputs
        )
        # Add per-timepoint subfolders to avoid collisions for non-BIDS inputs
        # We use the auto-generated timepoint labels (tp01, tp02, ...) as subfolder names
        #local -a realign_subfolders=("${TIMEPOINT_ORDER[@]}")
        realign_cmd+=("${subject_tp_paths[@]}")

        for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
            if [[ "${use_subfolder}" -eq 1 ]]; then
                realign_subfolders[$tp_idx]=$(dirname "${subject_tp_paths[$tp_idx]}")"/mri"
            else
                realign_subfolders[$tp_idx]=$(dirname "${subject_tp_paths[$tp_idx]}")
            fi
        done
        run_step "${REALIGN_SCRIPT}" --use-skullstrip --inverse-consistent --update-headers --inputs "${subject_tp_paths[@]}" --out-dir "./" --out-subfolders "${realign_subfolders[@]}" "${REALIGN_ARGS[@]+"${REALIGN_ARGS[@]}"}"

        # Compute Mid_surface filename for the first timepoint to use as initial surface
        local first_tp_path="${subject_tp_paths[0]}"
        local first_bname=$(basename "$first_tp_path")
        first_bname="${first_bname%.nii.gz}"
        first_bname="${first_bname%.nii}"
        
        # Determine BIDS naming and hemisphere notation like T1Prep does
        local first_hemi_lh="lh"
        local first_hemi_rh="rh"
        if [[ "${outdir0}" == *"/derivatives/"* ]]; then
            first_hemi_lh="L-"
            first_hemi_rh="R-"
        fi
        
        i=0
        for tp_idx in "${!TIMEPOINT_ORDER[@]}"; do
            local tp_path="${subject_tp_paths[$tp_idx]}"

            # Build T1Prep command cleanly
            local -a t1prep_cmd=("$T1PREP_CMD")
            if [[ -n "${OUT_DIR}" ]]; then
                t1prep_cmd+=(--out-dir "$OUT_DIR")
            fi
            if (( ${#T1PREP_ARGS[@]} )); then
                t1prep_cmd+=("${T1PREP_ARGS[@]}")
            fi

            # Derive per-timepoint long-data location
            local tp_long_data
            if [[ "${use_subfolder}" -eq 0 ]]; then
                tp_long_data="."
            else
                tp_long_data=$(dirname "$tp_path")"/mri/"$(basename "$tp_path")
            fi
            t1prep_cmd+=(--long-data "$tp_long_data" "$tp_path")

            # For subsequent timepoints, seed with initial surface from tp01 and skip segmentation
            if [[ "${i}" -gt 0 ]]; then
                local initial_surf_lh="${surf_dir}/lh.central.${first_bname}.gii"
                t1prep_cmd+=(--initial-surf "$initial_surf_lh")
            fi

            run_step "${t1prep_cmd[@]}"
            ((i++))
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
