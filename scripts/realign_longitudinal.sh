#!/usr/bin/env bash
# Longitudinal rigid realignment wrapper
# Ensures the virtual environment is activated before running t1prep.realign_longitudinal

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

print_usage() {
    cat <<'USAGE'
Longitudinal rigid realignment wrapper

Usage:
    scripts/realign_longitudinal.sh --inputs scan1.nii.gz scan2.nii.gz ... \
        --out-dir /path/to/output [other options]

Notes:
    - Wraps Python module: t1prep.realign_longitudinal
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

    python -m t1prep.realign_longitudinal "$@"
}

main "$@"
