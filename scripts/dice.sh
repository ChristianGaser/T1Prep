#!/usr/bin/env bash
#
# PURPOSE: Wrapper around the Dice metric CLI (python -m t1prep.dice).
#
# Python discovery and dispatch follow the same strategy as scripts/T1Prep:
# T1Prep_utils.sh resolves the source-tree vs. installed layout (and, in
# installed mode, the interpreter that actually has t1prep), check_python_cmd
# validates/auto-detects the interpreter in source-tree mode, and the module
# form "${python} -m t1prep.dice" is used so package-relative imports work in
# both layouts.
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
      is absent, any Python providing numpy/nibabel/scipy is used and the
      checkout under ../src takes precedence over an installed t1prep. An
      installed T1Prep is used as-is.
USAGE
}

# Show brief help if no args
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# ----------------------------------------------------------------------
# Honor an explicit interpreter before anything else, like T1Prep's main():
# "--python <cmd>" / "--python=<cmd>" (or $T1PREP_PYTHON) wins over the
# auto-detection in check_python_cmd.  The flag is consumed here because
# t1prep.dice itself does not know it.
# ----------------------------------------------------------------------
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
        -h|--help)  usage; exit 0 ;;
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

check_python_cmd

# ----------------------------------------------------------------------
# Source-tree mode: make the checkout importable and, when available, use the
# project-managed venv.
#
# ${root_dir}/src is prepended to PYTHONPATH unconditionally: PYTHONPATH is
# searched before site-packages, so this guarantees the checkout wins over a
# pip-installed (possibly older) t1prep — otherwise running scripts/dice.sh
# from a source tree can silently execute stale code from site-packages.
#
# A missing venv is *not* fatal here.  Unlike the full T1Prep pipeline this
# wrapper only needs numpy/nibabel/scipy, so any interpreter providing them
# works — including a system Python that T1Prep was pip installed into.  An
# explicit --python / $T1PREP_PYTHON always wins over the venv.
#
# Installed mode already runs inside the environment pip installed into, so
# neither step applies.
# ----------------------------------------------------------------------
if [ "${T1PREP_INSTALLED:-0}" -ne 1 ]; then
    export PYTHONPATH="${root_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

    if [ -z "${python_explicit}" ] && [ -f "${T1prep_env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${T1prep_env}/bin/activate"
        python="${T1prep_env}/bin/python"
    fi
fi

# Fail with an actionable message instead of a bare ImportError further down
if ! missing="$("${python}" -c 'import importlib.util as u; print(" ".join(m for m in ("numpy", "nibabel", "scipy") if u.find_spec(m) is None))' 2>/dev/null)"; then
    echo "${RED}ERROR: cannot run the Python interpreter '${python}'.${NC}" >&2
    echo "Select one with '--python <FILE>' or \$T1PREP_PYTHON." >&2
    exit 1
fi
if [ -n "${missing}" ]; then
    echo "${RED}ERROR: '${python}' is missing required module(s): ${missing}${NC}" >&2
    echo "Install them (${python} -m pip install ${missing}), or select an" >&2
    echo "interpreter that has them with '--python <FILE>' / \$T1PREP_PYTHON." >&2
    exit 1
fi

# Module form so package-relative imports resolve in both layouts
exec "${python}" -m t1prep.dice "${args[@]}"
