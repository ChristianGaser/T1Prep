#!/bin/bash
# Generic script runner that ensures virtual environment activation
# Usage: ./run_with_env.sh <script_name> [arguments...]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"

# Check if script name is provided
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <script_path> [arguments...]"
    echo "Example: $0 src/t1prep/gui/cat_viewsurf.py --help"
    echo "         $0 src/t1prep/segment.py --help"
    exit 0
fi

SCRIPT_NAME="$1"
shift  # Remove first argument, keep the rest

# Check if script exists
if [[ ! -f "$PROJECT_DIR/$SCRIPT_NAME" ]]; then
    echo "❌ Error: Script not found: $PROJECT_DIR/$SCRIPT_NAME"
    exit 1
fi

# Check if environment is activated
if [[ "$VIRTUAL_ENV" != "$ENV_DIR" ]]; then
    echo "🔄 Activating virtual environment..."
    
    if [[ ! -d "$ENV_DIR" ]]; then
        echo "❌ Error: Virtual environment not found: $ENV_DIR"
        exit 1
    fi
    
    source "$ENV_DIR/bin/activate"
    echo "✅ Virtual environment activated"
fi

# Run the script
echo "🎯 Running: $SCRIPT_NAME $*"
cd "$PROJECT_DIR"
python "$SCRIPT_NAME" "$@"
