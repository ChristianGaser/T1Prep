import os as _os
import sys as _sys

# MPS environment must be configured *before* torch is imported anywhere in
# the package.  Submodules below (e.g. utils, t1prep, cat_surf) pull torch in
# transitively, and PyTorch checks PYTORCH_ENABLE_MPS_FALLBACK at module-load
# time — setting it later (e.g. inside segment.py) only works for the
# subprocess invocation path, not for `python -m t1prep.segment`.
if _sys.platform == "darwin":
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

__version__ = "0.9.4"

# What the package exports is imported on first use (PEP 562), not here:
# ``utils`` pulls in torch, which costs about a second on every start -- of
# the viewers, the launcher and each ``--help`` as well, none of which runs a
# network.  ``from t1prep import run_t1prep`` and ``t1prep.utils`` work as
# they did.
_EXPORTS = {
    "run_t1prep": "t1prep",
    "progress_bar": "utils",
    "remove_file": "utils",
    "resample_and_save_nifti": "utils",
    "get_resampled_header": "utils",
    "align_brain": "utils",
    "get_filenames": "utils",
    "get_volume_native_space": "utils",
    "compute_dice_nifti": "metrics",
    "apply_LAS": "_intensity",
}


def __getattr__(name):
    """Import an exported name, or a submodule, the first time it is asked for."""
    import importlib

    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
        # ``run_t1prep`` lives in the submodule ``t1prep``: importing that
        # bound the submodule to this name's neighbour, never to the name
        globals()[name] = value
        return value
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        # Only "there is no such submodule" is a missing attribute; a
        # submodule that fails on a missing requirement says so itself
        if exc.name != f"{__name__}.{name}":
            raise
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "cat_surf",
    "unified_seg",
    "run_t1prep",
    "apply_LAS",
    "progress_bar",
    "remove_file",
    "resample_and_save_nifti",
    "get_resampled_header",
    "align_brain",
    "get_filenames",
    "get_volume_native_space",
    "compute_dice_nifti",
]
