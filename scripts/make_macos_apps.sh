#!/bin/bash
# Build macOS application bundles for the T1Prep viewers (source-tree helper).
#
# The implementation lives in src/t1prep/gui/make_apps.py and is installed as
# the `t1prep-make-apps` command; this wrapper only runs it with the project
# environment.  Options are passed through:
#
#   -o <dir>   where to write the bundles
#   -p <dir>   directory holding CAT_SurfView / CAT_VolView
#   -d         also make them the default for the file types they declare
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/run_with_env.sh" "$(dirname "$script_dir")/src/t1prep/gui/make_apps.py" "$@"
