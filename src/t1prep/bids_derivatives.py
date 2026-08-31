"""BIDS-derivative bookkeeping for the ``--fmriprep`` output mode.

fMRIPrep and sMRIPrep reuse anything they find already computed, but only if
the output directory is a valid BIDS derivative dataset: PyBIDS needs a
``dataset_description.json`` at the dataset root before it will index anything,
and several derivatives carry a JSON sidecar whose fields the specification
requires (``Type`` for masks, ``SkullStripped`` for anatomicals).

Without these the files are simply invisible, no matter how correct they are —
which is why T1Prep's outputs are written but not picked up.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from . import __version__

__all__ = [
    "derivatives_root",
    "write_dataset_description",
    "write_sidecar",
]

_SUBJECT = re.compile(r"^sub-[A-Za-z0-9]+$")
_SESSION = re.compile(r"^ses-[A-Za-z0-9]+$")
#: Directory names that sit between the dataset root and the derivative files.
_DATATYPES = {"anat", "func", "fmap", "dwi", "perf"}


def derivatives_root(out_dir: str) -> str:
    """Find the dataset root that ``out_dir`` belongs to.

    ``dataset_description.json`` must sit at the root of the derivative
    dataset, not beside the files.  T1Prep writes everything into one flat
    directory, so walk up past any ``anat/``, ``ses-*`` and ``sub-*``
    components the caller has arranged around it.

    Args:
        out_dir: Directory the derivative files are written to.

    Returns:
        The dataset root, or ``out_dir`` itself when it is not laid out as a
        BIDS tree.
    """
    root = os.path.abspath(out_dir)
    for pattern in (_DATATYPES.__contains__, _SESSION.match, _SUBJECT.match):
        parent, name = os.path.split(root)
        if parent and parent != root and pattern(name):
            root = parent
    return root


def write_dataset_description(
    out_dir: str, source_datasets: Iterable[str] = (), overwrite: bool = False
) -> str:
    """Write ``dataset_description.json`` for the derivative dataset.

    Args:
        out_dir: Directory the derivative files are written to; the description
            is placed at the dataset root found by :func:`derivatives_root`.
        source_datasets: Raw BIDS datasets the derivatives were computed from.
        overwrite: Replace an existing description.  Off by default so a run
            over a second subject does not clobber the first one's file.

    Returns:
        Path to the description that is now on disk.
    """
    path = os.path.join(derivatives_root(out_dir), "dataset_description.json")
    if os.path.exists(path) and not overwrite:
        return path

    description: dict[str, Any] = {
        "Name": "T1Prep - T1-weighted MRI preprocessing and surface reconstruction",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "T1Prep",
                "Version": __version__,
                "CodeURL": "https://github.com/ChristianGaser/T1Prep",
            }
        ],
        "HowToAcknowledge": (
            "Please cite T1Prep (https://github.com/ChristianGaser/T1Prep)."
        ),
    }
    sources = [str(s) for s in source_datasets if s]
    if sources:
        description["SourceDatasets"] = [{"URL": s} for s in sources]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(description, fh, indent=2)
        fh.write("\n")
    return path


def write_sidecar(data_path: str, **fields: Any) -> str | None:
    """Write the JSON sidecar that accompanies a derivative file.

    Args:
        data_path: The derivative the sidecar describes.  Its BIDS suffix and
            extension are replaced with ``.json``, so ``.nii.gz``,
            ``.surf.gii`` and ``.shape.gii`` all resolve correctly.
        **fields: Sidecar contents.  Keys whose value is ``None`` are dropped,
            which lets callers pass optional provenance unconditionally.

    Returns:
        Path to the sidecar, or ``None`` when ``data_path`` is empty.
    """
    if not data_path:
        return None
    base = os.path.basename(data_path)
    # Strip every extension: sub-01_desc-brain_mask.nii.gz -> ..._mask
    stem = base.split(".")[0]
    path = os.path.join(os.path.dirname(data_path), f"{stem}.json")

    payload = {key: value for key, value in fields.items() if value is not None}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path
