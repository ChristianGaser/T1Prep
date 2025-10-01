#!/bin/bash
# CAT_ViewSurf Wrapper Script
# Ensures the virtual environment is activated before running cat_viewsurf.py

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    
    # Check if environment is already activated
    if ! check_environment; then
        activate_environment
    fi
    
    # Run the Python script by absolute path so user-provided relative paths remain relative to caller's CWD
    export ORIGINAL_CWD="$(pwd)"
    PY_SCRIPT="$PROJECT_DIR/src/t1prep/gui/cat_viewsurf.py"
    python "$PY_SCRIPT" "$@"
}

# Run main function with all arguments
main "$@"
