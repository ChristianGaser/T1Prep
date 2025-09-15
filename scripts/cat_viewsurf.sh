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
    
    # Check if environment is already activated
    if ! check_environment; then
        activate_environment
    fi
    
    # Change to project directory and run the script
    cd "$PROJECT_DIR"
    python src/cat_viewsurf.py "$@"
}

# Run main function with all arguments
main "$@"
