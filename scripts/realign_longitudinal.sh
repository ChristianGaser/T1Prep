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

# The synopsis, on stderr like that of every T1Prep tool (print_usage)
usage() {
    print_usage "--inputs <tp1.nii.gz> <tp2.nii.gz> ... --out-dir <DIR> [options]" \
        "--inputs <FILE> ...          Time points of one subject" \
        "--out-dir <DIR>              Folder for the outputs" \
        "--sample-strategy <NAME>     grid (default) or gradient" \
        "--save-resampled             Also write the realigned volumes" \
        "--update-headers             Only write the new affines into the headers" \
        "--inverse-consistent         Centre the rigid transforms on their mean" \
        "--use-skullstrip             Estimate on skull-stripped copies" \
        "--verbose                    Print the optimiser's diagnostics" \
        "--help                       Show the full description and every option" \
        "--version                    Show the version"
}

help() {
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
}

main() {
    # Called with nothing at all: the synopsis, as every T1Prep tool does
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    activate_t1prep_env

    # '--help': what this step does, then every option of the Python module
    # ('--version' is answered by the module alone)
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --help | -h)
                help
                echo
                "${python}" -m t1prep.realign_longitudinal --help
                exit $?
                ;;
        esac
    done

    "${python}" -m t1prep.realign_longitudinal "$@"
}

main "$@"
