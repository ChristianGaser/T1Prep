#!/usr/bin/env bash
# Run T1Prep Web UI (Flask) with environment activation and optional port
# Usage: ./scripts/run_webui.sh [PORT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$PROJECT_DIR/env"
APP_FILE="$PROJECT_DIR/webui/app.py"
PORT="${1:-5555}"

# Activate environment if not already active
if [[ "$VIRTUAL_ENV" != "$ENV_DIR" ]]; then
    source "$ENV_DIR/bin/activate"
fi

cd "$PROJECT_DIR"

# Run Flask app on specified port
exec python "$APP_FILE" --port "$PORT"
