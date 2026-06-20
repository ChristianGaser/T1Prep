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

__version__ = "0.4.5"

# Import torch *before* cat_surf.  cat_surf's compiled `_surf` extension
# statically embeds its own LLVM OpenMP runtime, while torch loads its own
# libomp.dylib dynamically.  Whichever OpenMP runtime initializes first owns
# the process-wide OpenMP state; if cat_surf's static libomp goes first,
# torch's libomp collides with it and the process hard-crashes (SIGSEGV in
# __kmp_suspend_64).  Importing torch first is deterministic and avoids the
# crash without needing KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS workarounds.
# This runs for every `python -m t1prep.*` entry point because the package
# __init__ is imported before any submodule.
import torch as _torch  # noqa: F401

from . import cat_surf  # noqa: F401 – expose t1prep.cat_surf namespace
from .t1prep import run_t1prep
from .utils import (
    progress_bar,
    remove_file,
    resample_and_save_nifti,
    get_resampled_header,
    align_brain,
    get_filenames,
    get_volume_native_space,
)
from .metrics import compute_dice_nifti

__all__ = [
    "cat_surf",
    "run_t1prep",
    "progress_bar",
    "remove_file",
    "resample_and_save_nifti",
    "get_resampled_header",
    "align_brain",
    "get_filenames",
    "get_volume_native_space",
    "compute_dice_nifti",
]
