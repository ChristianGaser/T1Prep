"""Find the images to process.

A folder of scans usually holds more NIfTI files than there are images to
process once T1Prep has run on it: every result folder is full of them.
:func:`find_images` therefore leaves the folders T1Prep writes alone, so that
adding a study folder a second time does not queue last week's tissue maps.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import List

#: Extensions of an image T1Prep takes
IMAGE_SUFFIXES = (".nii", ".nii.gz")

#: Folders T1Prep writes its results to; never searched for inputs
RESULT_FOLDERS = ("mri", "surf", "report", "label", "derivatives")

#: What "all images" means in a folder
DEFAULT_PATTERN = "*.nii*"


def is_image(path: os.PathLike) -> bool:
    """Whether *path* is named like a NIfTI image."""
    return str(path).lower().endswith(IMAGE_SUFFIXES)


def find_images(folder: os.PathLike, pattern: str = DEFAULT_PATTERN,
                recursive: bool = False, limit: int = 50000) -> List[Path]:
    """The NIfTI images in *folder* whose name matches *pattern*.

    Args:
        folder: Where to look.
        pattern: Shell pattern for the file name, e.g. ``*_T1w.nii.gz``.
        recursive: Also look into subfolders — but not into hidden ones nor
            into :data:`RESULT_FOLDERS`.
        limit: Stop after this many images.

    Returns:
        The images sorted by path.
    """
    pattern = pattern.strip() or DEFAULT_PATTERN
    found: List[Path] = []
    for current, folders, files in os.walk(folder):
        if recursive:
            folders[:] = sorted(
                name for name in folders
                if not name.startswith(".") and name not in RESULT_FOLDERS)
        else:
            folders[:] = []
        for name in sorted(files):
            if (not name.startswith(".") and is_image(name)
                    and fnmatch.fnmatch(name, pattern)):
                found.append(Path(current) / name)
                if len(found) >= limit:
                    return found
    return found
