#!/usr/bin/env bash
# Longitudinal low-dimensional non-linear registration wrapper
# Ensures the virtual environment is activated before running t1prep.warp_longitudinal

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
Longitudinal low-dimensional non-linear registration wrapper

Estimates one small, smooth, diffeomorphic deformation per time point towards an
unbiased subject average -- the non-linear half of CAT12's ageing longitudinal
model.  Run it on volumes that have already been rigidly realigned by
realign_longitudinal.sh.

Usage:
    scripts/warp_longitudinal.sh --inputs tp1.nii.gz tp2.nii.gz ... \
        --out-dir /path/to/output [other options]

Example (write the volume-change maps and the subject average):
    scripts/warp_longitudinal.sh --inputs tp1.nii.gz tp2.nii.gz \
        --out-dir /path/to/output --save-template

Outputs per time point:
    <stem>_desc-longLogJacobian.nii[.gz]   log volume ratio vs the average
    <stem>_desc-longDisplacement.nii[.gz]  RAS displacement in mm (--save-displacement)
    <stem>_desc-longWarped.nii[.gz]        time point on the average (--apply)

Notes:
    - Wraps Python module: t1prep.warp_longitudinal
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

    python -m t1prep.warp_longitudinal "$@"
}

main "$@"
