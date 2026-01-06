#!/bin/bash
# Environment activation script
# Source this script to activate the T1Prep virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

# If we're in the project directory, adjust paths
if [[ "$(basename "$PWD")" == "T1Prep" ]]; then
    PROJECT_DIR="$PWD"
    ENV_DIR="$PROJECT_DIR/env"
fi

# Check if environment exists
if [[ ! -d "$ENV_DIR" ]]; then
    echo "❌ Error: Virtual environment not found: $ENV_DIR"
    echo "   Please run: python3 -m venv env"
    return 1 2>/dev/null || exit 1
fi

# Check if already activated
if [[ "$VIRTUAL_ENV" == "$ENV_DIR" ]]; then
    echo "✅ Virtual environment is already activated: $VIRTUAL_ENV"
    return 0 2>/dev/null || exit 0
fi

# Activate the environment
echo "🔄 Activating T1Prep virtual environment..."
source "$ENV_DIR/bin/activate"

# Verify activation
if [[ "$VIRTUAL_ENV" == "$ENV_DIR" ]]; then
    echo "✅ Virtual environment activated successfully"
    echo "   Python: $(which python)"
    echo "   Python version: $(python --version)"
    echo "   Project directory: $PROJECT_DIR"
    echo ""
    echo "💡 You can now run: scripts/cat_viewsurf.sh <mesh_or_overlay> [options]"
    echo "   Or: python src/t1prep/gui/cat_viewsurf.py -h"
else
    echo "❌ Error: Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi

