"""
Back-compat shim for legacy imports.

This module used to contain T1Prep utility functions at the repository root.
The canonical implementations now live in the package module `t1prep.utils`.
We re-export all public symbols so existing imports like `from utils import ...`
continue to work without code changes.
"""

from t1prep.utils import *  # noqa: F401,F403 re-export public API

# Provide a sensible __all__ for tooling and star-import users
try:  # pragma: no cover - defensive
    from t1prep import utils as _t1prep_utils
    __all__ = [n for n in dir(_t1prep_utils) if not n.startswith("_")]
    del _t1prep_utils
except Exception:
    # Fallback: let star-import fall back to all names from t1prep.utils
    pass
