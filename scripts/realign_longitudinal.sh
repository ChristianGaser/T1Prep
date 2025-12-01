#!/usr/bin/env bash
# Longitudinal rigid realignment wrapper
# Ensures the project virtualenv is active before executing t1prep.realign_longitudinal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

activate_environment() {
    if [[ ! -d "$ENV_DIR" ]]; then
        echo "❌ Error: Virtual environment directory not found: $ENV_DIR" >&2
        echo "   Please create it, e.g. python3 -m venv env" >&2
        exit 1
    fi
    if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
        echo "❌ Error: Activation script missing: $ENV_DIR/bin/activate" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"
}

ensure_environment() {
    if [[ "${VIRTUAL_ENV:-}" != "$ENV_DIR" ]]; then
        activate_environment
    fi
}

print_usage() {
    cat <<'USAGE'
Longitudinal rigid realignment wrapper

Usage:
    scripts/realign_longitudinal.sh --inputs scan1.nii.gz scan2.nii.gz ... \
        --out-dir /path/to/output [additional t1prep.realign_longitudinal args]

Notes:
    - This is a light wrapper around: python -m t1prep.realign_longitudinal
    - The script activates ./env (created via python -m venv env) so that the
      correct dependencies are available.
    - All arguments are forwarded unchanged to the Python CLI. Run with --help
      to see the available options (register-to-mean, inverse-consistent, etc.).
USAGE
}

main() {
    if [[ $# -eq 0 ]]; then
        print_usage
        exit 1
    fi

    ensure_environment

    export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

    python -m t1prep.realign_longitudinal "$@"
}

main "$@"
