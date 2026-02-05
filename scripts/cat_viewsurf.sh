#!/bin/bash
# CAT_ViewSurf Wrapper Script
# Ensures the virtual environment is activated before running cat_viewsurf.py

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

# Function to check if we're in the virtual environment
check_environment() {
    if [[ "$VIRTUAL_ENV" == "$ENV_DIR" ]]; then
        return 0
    else
        if [[ ! -d "$ENV_DIR" ]]; then
            echo "❌ Error: Virtual environment not found: $ENV_DIR"
            exit 1
        fi
        
        source "$ENV_DIR/bin/activate"
    fi
}

# Function to activate the environment
activate_environment() {
    
    if [[ ! -d "$ENV_DIR" ]]; then
        echo "❌ Error: Virtual environment directory not found: $ENV_DIR"
        echo "   Please run: python3 -m venv env"
        exit 1
    fi
    
    if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
        echo "❌ Error: Virtual environment activation script not found: $ENV_DIR/bin/activate"
        exit 1
    fi
    
    # Source the activation script
    source "$ENV_DIR/bin/activate"

    # On macOS, remove quarantine attributes to prevent Gatekeeper blocking
    if [[ "$(uname)" == "Darwin" ]]; then
        xattr -dr com.apple.quarantine "$ENV_DIR" 2>/dev/null || true
    fi
}

# Validate that at least one positional arg is an existing file (skip options)
validate_input_files() {
    local arg
    local found=0
    for arg in "$@"; do
        # Skip options starting with '-'
        [[ "$arg" == -* ]] && continue
        if [[ -e "$arg" ]]; then
            found=1
            break
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo "❌ Error: No existing input files found among provided arguments." >&2
        echo "   Provide at least one existing mesh or overlay file path." >&2
        echo "" >&2
        echo "Usage:" >&2
        echo "  scripts/cat_viewsurf.sh <mesh_or_overlay> [more_overlays...] [options]" >&2
        echo "" >&2
        echo "Examples:" >&2
        echo "  scripts/cat_viewsurf.sh /path/to/lh.central.gii" >&2
        echo "  scripts/cat_viewsurf.sh /path/to/lh.thickness" >&2
        echo "" >&2
        echo "Note: If you passed a wildcard pattern, ensure your shell expands it (avoid quoting *)." >&2
        exit 1
    fi
}

# Main execution
main() {
    # Show help if called without any arguments
    if [[ $# -eq 0 ]]; then
        cat <<'USAGE'
CAT_ViewSurf — wrapper

Usage:
    scripts/cat_viewsurf.sh <mesh_or_overlay> [more_overlays...] [options]

Examples:
    scripts/cat_viewsurf.sh lh.central.gii
    scripts/cat_viewsurf.sh lh.thickness
    scripts/cat_viewsurf.sh lh.thickness lh.pbt -colorbar --title-mode stats

Notes:
    - You can pass one or more overlays; the mesh is auto-derived from the first overlay (e.g., lh.thickness -> lh.central.gii)
    - For full options, run: python src/t1prep/gui/cat_viewsurf.py -h
USAGE
        exit 1
    fi
    
    # Refuse to run if no positional args resolve to existing files
    validate_input_files "$@"
    
    # Check if environment is already activated
    if ! check_environment; then
        activate_environment
    fi

    # Always use the venv Python to avoid system Python/ABI mismatches
    PYTHON_BIN="$ENV_DIR/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "❌ Error: Python executable not found in virtual environment: $PYTHON_BIN" >&2
        exit 1
    fi
    
    # Suppress noisy Qt painter warnings
    export QT_LOGGING_RULES="${QT_LOGGING_RULES:qt.gui.painting=false;qt.qpa.*=false}"

    # Run the Python script by absolute path so user-provided relative paths remain relative to caller's CWD
    export ORIGINAL_CWD="$(pwd)"
    PY_SCRIPT="$PROJECT_DIR/src/t1prep/gui/cat_viewsurf.py"
    # Filter noisy QPainter warnings while preserving other stderr output
    "$PYTHON_BIN" "$PY_SCRIPT" "$@" \
        2> >(grep -v "QPainter::begin: Paint device returned engine == 0, type: 1" >&2)
}

# Run main function with all arguments
main "$@"
