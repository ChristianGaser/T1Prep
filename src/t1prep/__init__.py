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

__version__ = "0.4.6"

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
