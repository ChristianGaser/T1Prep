#!/usr/bin/env bash
#
# PURPOSE: Wrapper around the Dice metric CLI (python -m t1prep.dice).
#
# Python discovery and dispatch follow the same strategy as scripts/T1Prep:
# T1Prep_utils.sh resolves the source-tree vs. installed layout (and, in
# installed mode, the interpreter that actually has t1prep), check_python_cmd
# validates/auto-detects the interpreter in source-tree mode, and t1prep.dice
# is imported as part of the package, never run as a file, so package-relative
# imports work in both layouts.
#
# ______________________________________________________________________
#
# Christian Gaser
# Structural Brain Mapping Group (https://neuro-jena.github.io)
# Departments of Neurology and Psychiatry
# Jena University Hospital
# ______________________________________________________________________

# Resolve this script's directory robustly (works even if invoked via symlink)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

synopsis() {
    cat <<'USAGE'
USAGE:
  dice.sh --gt <GT.nii.gz> --pred <PRED.nii.gz> [options]

OPTIONS:
  --gt <FILE>          Ground-truth label or probability volume
  --pred <FILE>        Predicted volume, compared against --gt
  --soft               Soft Dice on the unrounded inputs
  --no-resample        Compare voxel-to-voxel, ignoring the affines
  --save-conf <FILE>   Write the confusion matrix to this CSV
  --verbose            One line per label instead of the single summary line
  --python <FILE>      Interpreter to run t1prep.dice with
  --help               Show the full description
USAGE
    echo "Run '$(basename -- "$0") --help' for the full description."
}

usage() {
    cat <<'USAGE'
Dice-based metric — wrapper

Usage:
    scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz \
        [--soft] [--no-resample] [--save-conf conf.csv] [--verbose] \
        [--python <FILE>]

Notes:
    - Wraps Python module: t1prep.dice
        - --soft computes soft/continuous Dice using unrounded inputs (for
            probability/partial-volume maps); default is rounded labels.
    - The NIfTI affines (sform/qform) are honoured: when --gt and --pred differ
        in shape, voxel size, orientation or rotation, --pred is resampled onto
        the grid of --gt (nearest neighbour for labels, trilinear with --soft)
        and a note is printed to stderr. Use --no-resample to compare
        voxel-to-voxel instead, which ignores the affines and requires
        identical shapes.
    - A brain mask is obtained from ``gt != 0``; all voxels inside this mask
        contribute to the confusion matrix, so disagreements between ``gt`` and
        ``pred`` are fully accounted for.
    - Without --verbose, prints a single line:
          [<dice_label_1>,<dice_label_2>,...] <generalized_dice>,<dice_weighted>
      where the vector order matches the label list
    - With --verbose, prints one line per label, generalized_dice, and dice_weighted
    - --python <FILE> selects the interpreter (or set $T1PREP_PYTHON), exactly
      as for T1Prep. In a source checkout the project venv (../env) is
      activated when it exists and no interpreter was given explicitly; if it
      is absent, the first Python with T1Prep's requirements is used and the
      checkout under ../src takes precedence over an installed t1prep. An
      installed T1Prep is used as-is.
USAGE
}

# ----------------------------------------------------------------------
# Honor an explicit interpreter before anything else, like T1Prep's main():
# "--python <cmd>" / "--python=<cmd>" (or $T1PREP_PYTHON) wins over the
# auto-detection in check_python_cmd.  The flag is consumed here because
# t1prep.dice itself does not know it.
# ----------------------------------------------------------------------
# Called with nothing at all: the synopsis, as every T1Prep tool does;
# '--help' gives the full description
if [ $# -eq 0 ]; then
    synopsis >&2
    exit 1
fi

args=()
_expect_python=0
for _arg in "$@"; do
    if [ "${_expect_python}" -eq 1 ]; then
        python="${_arg}"
        _expect_python=0
        continue
    fi
    case "${_arg}" in
        --python)   _expect_python=1 ;;
        --python=*) python="${_arg#--python=}" ;;
        --help|-h)  usage; exit 0 ;;
        *)          args+=("${_arg}") ;;
    esac
done
python="${python:-${T1PREP_PYTHON:-}}"
# Remember whether the interpreter was chosen explicitly: such a choice must
# survive the venv activation below.
python_explicit="${python}"

# Dual-mode path resolution shared with T1Prep: sets T1PREP_INSTALLED,
# root_dir, src_dir, T1prep_env and (installed mode) python.
if [ ! -f "${script_dir}/T1Prep_utils.sh" ]; then
    echo "ERROR: ${script_dir}/T1Prep_utils.sh not found — it must sit next to this script." >&2
    exit 1
fi
# shellcheck source=scripts/T1Prep_utils.sh
source "${script_dir}/T1Prep_utils.sh"

# Validate/auto-detect the interpreter, put the checkout on PYTHONPATH and
# activate the project-managed venv when there is one.  A missing venv is not
# fatal: any interpreter with T1Prep's requirements works — see
# activate_t1prep_env() in T1Prep_utils.sh.
activate_t1prep_env

# The import is the dependency check, in the process that then computes the
# metric.  t1prep.dice itself needs only numpy/nibabel/scipy, but importing it
# runs the package __init__ first, which pulls in cat_surf and torch: a module
# list kept here goes stale, and a check in a separate process would import
# torch twice.  A failure names the interpreter and the missing module instead
# of ending in a bare traceback.  (In an installed tree T1Prep_utils.sh has
# already settled on an interpreter that imports t1prep.)
# shellcheck disable=SC2016  # $T1PREP_PYTHON is literal text in the message
exec "${python}" -c '
import sys
requirements = sys.argv.pop(1)
try:
    from t1prep.dice import main
except ImportError as exc:
    sys.exit(f"ERROR: {sys.executable} cannot import t1prep: {exc}\n"
             f"Install the requirements ({sys.executable} -m pip install -r {requirements}),\n"
             "or select an interpreter that has them with --python <FILE> / $T1PREP_PYTHON.")
sys.exit(main())
' "${root_dir}/requirements.txt" "${args[@]}"
