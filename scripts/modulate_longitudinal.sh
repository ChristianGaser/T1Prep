#!/usr/bin/env bash
# Longitudinal low-dimensional non-linear registration wrapper
# Ensures the virtual environment is activated before running t1prep.modulate_longitudinal

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

check_environment() {
    if [[ "${VIRTUAL_ENV:-}" == "$ENV_DIR" ]]; then
        return 0
    else
        if [[ ! -d "$ENV_DIR" ]]; then
            echo "❌ Error: Virtual environment not found: $ENV_DIR" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        source "$ENV_DIR/bin/activate"
    fi
}

activate_environment() {
    if [[ ! -d "$ENV_DIR" ]]; then
        echo "❌ Error: Virtual environment directory not found: $ENV_DIR" >&2
        echo "   Please run: python3 -m venv env" >&2
        exit 1
    fi
    if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
        echo "❌ Error: Activation script missing: $ENV_DIR/bin/activate" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"
}

print_usage() {
    cat <<'USAGE'
Longitudinal Jacobian modulation wrapper

Turns the deformations from warp_longitudinal.sh into the modulated tissue maps
a longitudinal VBM analysis is run on -- CAT12's ageing model, except that the
shared tissue map is the mean of the time points' own segmentations rather than
a segmentation of the average image.

Two things are shared across the time points, as CAT12 shares them: one tissue
map, and one spatial normalisation (the mean of the per-time-point 'y_' fields).
Each time point then differs only by its longitudinal Jacobian.

Usage (letting it find the files):
    scripts/modulate_longitudinal.sh \
        --mri-dirs sub-01/ses-1/mri sub-01/ses-2/mri \
        --names tp1 tp2 \
        --out-dir /path/to/output

Usage (explicit paths):
    scripts/modulate_longitudinal.sh \
        --tissue p1tp1.nii p1tp2.nii \
        --displacement tp1_desc-longDisplacement.nii.gz tp2_...nii.gz \
        --log-jacobian tp1_desc-longLogJacobian.nii.gz tp2_...nii.gz \
        --deformation y_tp1.nii y_tp2.nii \
        --out-dir /path/to/output

Writes one modulated map per time point on the MNI grid, named from T1Prep's
own table with the modulation marker doubled, as CAT12's ageing model does:

    mwmwp1<name>.nii                                          (legacy naming)
    <name>_space-..-modulated-modulated_label-GM_probseg.nii  (BIDS)

Doubled because these maps are modulated twice: by the longitudinal Jacobian,
then by the spatial normalisation. It also keeps them distinct from the
cross-sectional 'mwp1<name>.nii' T1Prep writes into the same folder.

Requires:
    - T1Prep run with --p (native segmentations)
    - warp_longitudinal.sh run with --save-displacement

Notes:
    - Wraps Python module: t1prep.modulate_longitudinal
    - Activates ./env before running so dependencies are available
    - All positional/optional arguments are forwarded to the Python CLI
USAGE
}

main() {
    if [[ $# -eq 0 ]]; then
        print_usage
        exit 1
    fi

    if ! check_environment; then
        activate_environment
    fi

    export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

    python -m t1prep.modulate_longitudinal "$@"
}

main "$@"
