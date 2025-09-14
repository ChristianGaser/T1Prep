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
        echo "✅ Virtual environment is already activated: $VIRTUAL_ENV"
        return 0
    else
        echo "⚠️  Virtual environment not activated or wrong environment"
        echo "   Current VIRTUAL_ENV: ${VIRTUAL_ENV:-"Not set"}"
        echo "   Expected: $ENV_DIR"
        return 1
    fi
}

# Function to activate the environment
activate_environment() {
    echo "🔄 Activating virtual environment..."
    
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
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" == "$ENV_DIR" ]]; then
        echo "✅ Virtual environment activated successfully"
        echo "   Python: $(which python)"
        echo "   Python version: $(python --version)"
    else
        echo "❌ Error: Failed to activate virtual environment"
        exit 1
    fi
}

# Main execution
main() {
    echo "🚀 CAT_ViewSurf Wrapper Script"
    echo "   Project directory: $PROJECT_DIR"
    echo "   Environment directory: $ENV_DIR"
    echo ""
    
    # Check if environment is already activated
    if ! check_environment; then
        activate_environment
    fi
    
    echo ""
    echo "🎯 Running cat_viewsurf.py with arguments: $*"
    echo ""
    
    # Change to project directory and run the script
    cd "$PROJECT_DIR"
    python src/cat_viewsurf.py "$@"
}

# Run main function with all arguments
main "$@"
