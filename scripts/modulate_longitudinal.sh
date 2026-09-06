#!/usr/bin/env bash
# Longitudinal Jacobian modulation wrapper
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
a longitudinal VBM analysis is run on, following CAT12's longitudinal models.

Each time point keeps its own segmentation, as CAT12 does, so individual
anatomy stays visible.  What is shared is the spatial normalisation: the mean of
the per-time-point 'y_' fields, standing in for the average image's warp to MNI
(the average itself is not segmented here -- the one deviation from CAT12).

Models (--model), with CAT12's own output names:
    ageing      (default) longitudinal Jacobian, then spatial normalisation
                -> mwmwp1r<name>.nii   (modulated twice)
    plasticity  spatial normalisation only, anatomy assumed unchanged
                -> mwp1r<name>.nii     (modulated once)
The 'r' marks the realigned input, and keeps both distinct from T1Prep's
cross-sectional 'mwp1<name>.nii' in the same folder.  In BIDS naming the
outputs carry '_desc-long' instead, with '-modulated' doubled for ageing.

Usage (letting it find the files):
    scripts/modulate_longitudinal.sh \
        --mri-dirs  out/mri out/mri \
        --long-dirs data/mri data/mri \
        --names tp1 tp2 \
        --out-dir out/mri [--model plasticity] [--tissue-class WM]

    --mri-dirs  hold T1Prep's outputs (p1/p2 and y_ per time point)
    --long-dirs hold warp_longitudinal.sh's displacement and log-Jacobian;
                the pipeline runs that stage before T1Prep, so those sit
                beside the realigned volumes, not in T1Prep's output folder

Usage (explicit paths):
    scripts/modulate_longitudinal.sh \
        --tissue p1tp1.nii p1tp2.nii \
        --displacement tp1_desc-longDisplacement.nii tp2_desc-longDisplacement.nii \
        --log-jacobian tp1_desc-longLogJacobian.nii tp2_desc-longLogJacobian.nii \
        --deformation y_tp1.nii y_tp2.nii \
        --out-dir /path/to/output

Other options:
    --modulation nonlinear     divide the affine out (controls for head size)
    --tissue-source shared     one tissue map for every time point, so the
                               contrast carries only the Jacobian: quieter,
                               but not CAT12, and it hides individual anatomy

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
