#!/usr/bin/env bash
# Longitudinal rigid realignment wrapper
# Resolves the Python environment, then runs t1prep.realign_longitudinal
#
# Environment resolution is shared with the rest of the bash family (see
# scripts/dice.sh): T1Prep_utils.sh resolves the source-tree vs. installed
# layout -- and, in installed mode, the interpreter that actually has t1prep --
# then activate_t1prep_env() puts the checkout on PYTHONPATH and activates the
# project-managed venv at <repo>/env when there is one.  The module form
# "${python} -m t1prep.realign_longitudinal" is used so the
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
Longitudinal rigid realignment wrapper

Usage:
    scripts/realign_longitudinal.sh --inputs scan1.nii.gz scan2.nii.gz ... \
        --out-dir /path/to/output [other options]

Example (more robust sampling):
    scripts/realign_longitudinal.sh --inputs scan1.nii.gz scan2.nii.gz ... \
        --out-dir /path/to/output --sample-strategy gradient

Notes:
    - Wraps Python module: t1prep.realign_longitudinal
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

    "${python}" -m t1prep.realign_longitudinal "$@"
}

main "$@"
