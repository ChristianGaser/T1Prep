#!/bin/bash
# Build macOS application bundles for the T1Prep viewers (source-tree helper).
#
# The implementation lives in src/t1prep/gui/make_apps.py and is installed as
# the `t1prep-make-apps` command; this wrapper only picks an interpreter that
# can reach it.  Options are passed through:
#
#   -o <dir>   where to write the bundles
#   -p <dir>   directory holding CAT_SurfView / CAT_VolView
#   -d         also make them the default for the file types they declare
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"

# 1. installed entry point — the bundles then point at that installation
if command -v t1prep-make-apps >/dev/null 2>&1; then
    exec t1prep-make-apps "$@"
fi

# 2. the virtual environment of a source checkout
if [[ -x "$project_dir/env/bin/python" ]]; then
    exec "$project_dir/env/bin/python" -m t1prep.gui.make_apps "$@"
fi

# 3. plain python3 on the file itself; it needs nothing but the standard
#    library, so this works even without the T1Prep dependencies installed
echo "⚠ T1Prep is not installed; using $(command -v python3)." >&2
echo "  The bundles will launch whichever CAT_SurfView / CAT_VolView is on PATH," >&2
echo "  so make sure they are (or pass -p <bin-dir>)." >&2
exec python3 "$project_dir/src/t1prep/gui/make_apps.py" "$@"
