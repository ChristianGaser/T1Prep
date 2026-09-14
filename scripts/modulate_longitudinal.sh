#!/usr/bin/env bash
# Longitudinal Jacobian modulation wrapper
# Resolves the Python environment, then runs t1prep.modulate_longitudinal
#
# Environment resolution is shared with the rest of the bash family (see
# scripts/dice.sh): T1Prep_utils.sh resolves the source-tree vs. installed
# layout -- and, in installed mode, the interpreter that actually has t1prep --
# then activate_t1prep_env() puts the checkout on PYTHONPATH and activates the
# project-managed venv at <repo>/env when there is one.  The module form
# "${python} -m t1prep.modulate_longitudinal" is used so the
# package-relative imports work in both layouts.

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${script_dir}/T1Prep_utils.sh" ]; then
    echo "ERROR: ${script_dir}/T1Prep_utils.sh not found - it must sit next to this script." >&2
    exit 1
fi
# shellcheck source=scripts/T1Prep_utils.sh
source "${script_dir}/T1Prep_utils.sh"

usage() {
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
    - Runs in the project venv (<repo>/env) when present, otherwise in
      the interpreter T1Prep is installed into
    - All positional/optional arguments are forwarded to the Python CLI
USAGE
    echo "Run '$(basename -- "$0") --help' for the full description."
}

main() {
    # Called with nothing at all: the synopsis, as every T1Prep tool does;
    # '--help' and '--version' are forwarded to the Python module
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    activate_t1prep_env

    "${python}" -m t1prep.modulate_longitudinal "$@"
}

main "$@"
