#!/usr/bin/env bash
# Kappa CLI wrapper script
# Ensures the virtual environment is activated before running t1prep.dice

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

check_environment() {
    if [[ "$VIRTUAL_ENV" == "$ENV_DIR" ]]; then
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

main() {
    # Show brief help if no args
    if [[ $# -eq 0 ]]; then
        cat <<'USAGE'
Dice-based metric — wrapper

Usage:
    scripts/dice.sh --gt GT.nii.gz --pred PRED.nii.gz \
        [--labels 1,2,3] [--save-conf conf.csv] [--verbose]

Notes:
    - Wraps Python module: t1prep.dice
    - Mask is derived from labels: intersection (default) or gt
    - Without --verbose, prints a single line:
          <generalized_dice> [<dice_label_1>,<dice_label_2>,...]
      where the vector order matches the label list
    - With --verbose, prints labels, generalized_dice, and one line per label
    - Activates ./env before running
USAGE
        exit 1
    fi

    if ! check_environment; then
        activate_environment
    fi

    # Ensure package imports resolve (src on PYTHONPATH)
    export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

    # Run module form to ensure package imports work
    python -m t1prep.dice "$@"
}

main "$@"
