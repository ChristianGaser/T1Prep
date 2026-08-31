"""Reading and writing ITK/ANTs plain-text transform files.

The ``#Insight Transform File V1.0`` format fMRIPrep uses for its rigid and
affine transforms.  Kept apart from :mod:`t1prep.segment` so the small
transform utilities — and the ``t1prep-bbreg`` console script — do not have to
pull in torch and the segmentation models.
"""

from __future__ import annotations

import numpy as np

__all__ = ["save_affine_itk_txt", "load_affine_itk_txt"]


def save_affine_itk_txt(affine_ras: np.ndarray, out_path: str) -> None:
    """Save a 4x4 RAS affine matrix as an ITK/ANTs plain-text transform file.

    Writes the ``#Insight Transform File V1.0`` format used by ANTs, ITK,
    and fMRIPrep.  Handles the RAS→LPS coordinate-system conversion: the
    3x3 rotation/scaling block and the translation vector are both negated
    on their x and y components before writing.

    Args:
        affine_ras: 4x4 affine matrix in RAS coordinates (e.g. the T1w-to-MNI
            registration matrix returned by deepmriprep).
        out_path: Output file path (should end with ``.txt``).
    """
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    M_lps = ras2lps @ affine_ras[:3, :3] @ ras2lps
    T_lps = ras2lps @ affine_ras[:3, 3]
    params = np.concatenate([M_lps.ravel(order="C"), T_lps])
    with open(out_path, "w") as fh:
        fh.write("#Insight Transform File V1.0\n")
        fh.write("#Transform 0\n")
        fh.write("Transform: AffineTransform_float_3_3\n")
        fh.write("Parameters: " + " ".join(f"{v:.10g}" for v in params) + "\n")
        fh.write("FixedParameters: 0 0 0\n")



def load_affine_itk_txt(path: str) -> np.ndarray:
    """Read an ITK plain-text affine back into a 4x4 RAS matrix.

    Args:
        path: An ``#Insight Transform File V1.0`` file holding a single
            affine or rigid transform.

    Returns:
        The 4x4 transform in RAS coordinates.

    Raises:
        ValueError: If the file carries no ``Parameters:`` line.
    """
    parameters = center = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("Parameters:"):
                parameters = np.array(line.split(":", 1)[1].split(), dtype=np.float64)
            elif line.startswith("FixedParameters:"):
                center = np.array(line.split(":", 1)[1].split(), dtype=np.float64)
    if parameters is None:
        raise ValueError(f"No 'Parameters:' line in {path}")

    matrix, translation = parameters[:9].reshape(3, 3), parameters[9:12]
    center = np.zeros(3) if center is None or center.size < 3 else center[:3]
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    out = np.eye(4)
    out[:3, :3] = ras2lps @ matrix @ ras2lps
    # ITK rotates about `center`; fold that back into a plain translation.
    out[:3, 3] = ras2lps @ (center + translation - matrix @ center)
    return out
