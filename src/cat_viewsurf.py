#!/usr/bin/env python3
"""
Legacy wrapper for CAT_ViewSurf.

This module moved to t1prep.gui.cat_viewsurf. Keeping this thin wrapper for
backwards compatibility. It forwards to the new entrypoint.
"""
from __future__ import annotations
import sys

try:
    from t1prep.gui.cat_viewsurf import main as _forward_main
except Exception as e:  # pragma: no cover
    def _forward_main(argv):  # type: ignore
        raise SystemExit(
            "Failed to import t1prep.gui.cat_viewsurf. Ensure T1Prep is installed "
            f"and dependencies are available. Original error: {e}"
        )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    return _forward_main(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

