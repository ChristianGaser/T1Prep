"""Metrics utilities: Cohen's kappa for 2- or 3-class NIfTI label maps.

This module provides a function to compute Cohen's kappa between an integer
label ground truth and a test label map restricted to a mask and a selected set
of label values (e.g., [1, 2] or [1, 2, 3]).

Inputs may be file paths to NIfTI images or nibabel Nifti1Image objects.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib

NiftiLike = Union[str, nib.Nifti1Image]

__all__ = ["compute_cohen_kappa_nifti"]


def _load_labels(img: NiftiLike) -> np.ndarray:
    if isinstance(img, nib.Nifti1Image):
        data = img.get_fdata()
    else:
        data = nib.load(str(img)).get_fdata()
    # Convert to integer labels; round to nearest to be robust to tiny float stores
    return np.rint(data).astype(np.int64, copy=False)


def _to_bool_mask_from_labels(arr: np.ndarray, labels_arr: np.ndarray) -> np.ndarray:
    """Return a boolean mask where arr is one of the given labels."""
    return np.isin(arr, labels_arr)


def compute_cohen_kappa_nifti(
    gt: NiftiLike,
    pred: NiftiLike,
    labels: Optional[Sequence[int]] = None,
    mask_mode: str = "intersection",
) -> Tuple[float, np.ndarray, Sequence[int]]:
    """Compute Cohen's kappa for 2- or 3-class integer labels from NIfTI images.

    Parameters
    ----------
    gt : str | nib.Nifti1Image
        Ground-truth label image (NIfTI). Integer labels (2 or 3 classes).
    pred : str | nib.Nifti1Image
        Test/predicted label image (NIfTI). Integer labels (2 or 3 classes).
    labels : Sequence[int] | None
        Which label values to include, e.g., [1, 2] or [1, 2, 3]. If None,
        labels are inferred as the sorted unique intersection of labels present
        in gt and pred (excluding 0).
    mask_mode : {"intersection", "gt"}
        How to define the evaluation mask based on labels:
        - "intersection" (default): voxels where both gt and pred are in the selected labels
        - "gt": voxels where gt is in the selected labels

    Returns
    -------
    kappa : float
        Cohen's kappa coefficient (unweighted). NaN if undefined.
    confusion : np.ndarray
        KxK confusion matrix with rows gt and columns pred, for K=len(labels).
    labels_order : Sequence[int]
        The label values corresponding to the rows/cols of confusion.

    Notes
    -----
    - This is nominal (unweighted) Cohen's kappa.
    - Handles degenerate cases by returning NaN when denominator is zero or when
      there are no valid voxels after masking/label filtering.
    - Input labels are rounded before casting to integer to tolerate minor float
      imprecision in saved NIfTI files.
    """
    y_true = _load_labels(gt)
    y_pred = _load_labels(pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: gt {y_true.shape} vs pred {y_pred.shape}")
    # Determine labels to consider
    if labels is None:
        # Use intersection of labels present (excluding background 0)
        present = np.intersect1d(np.unique(y_true), np.unique(y_pred))
        labels_arr = present[present != 0]
        if labels_arr.size == 0:
            return float("nan"), np.zeros((0, 0), dtype=np.int64), []
    else:
        labels_arr = np.asarray(labels, dtype=np.int64)

    # Build mapping from label value to contiguous index 0..K-1
    label_to_idx = {int(v): i for i, v in enumerate(labels_arr.tolist())}
    K = len(labels_arr)

    # Build label-based masks and select valid voxels per mask_mode
    m_gt = _to_bool_mask_from_labels(y_true, labels_arr)
    m_pred = _to_bool_mask_from_labels(y_pred, labels_arr)
    if mask_mode not in ("intersection", "gt"):
        raise ValueError("mask_mode must be 'intersection' or 'gt'")
    mask_valid = (m_gt & m_pred) if mask_mode == "intersection" else m_gt
    yt = y_true[mask_valid]
    yp = y_pred[mask_valid]
    # Ensure both yt and yp are within labels (intersection mode already guarantees both)
    valid = np.isin(yt, labels_arr) & np.isin(yp, labels_arr)
    yt = yt[valid]
    yp = yp[valid]
    if yt.size == 0:
        return float("nan"), np.zeros((K, K), dtype=np.int64), labels_arr.tolist()

    # Map to indices
    yt_idx = np.fromiter((label_to_idx[int(v)] for v in yt), count=yt.size, dtype=np.int64)
    yp_idx = np.fromiter((label_to_idx[int(v)] for v in yp), count=yp.size, dtype=np.int64)

    # Confusion matrix via bincount
    idx = yt_idx * K + yp_idx
    conf = np.bincount(idx, minlength=K * K).reshape(K, K)

    N = conf.sum()
    if N == 0:
        return float("nan"), conf, labels_arr.tolist()
    po = np.trace(conf) / N
    row = conf.sum(axis=1)
    col = conf.sum(axis=0)
    pe = float(np.dot(row, col)) / (N * N)
    denom = 1.0 - pe
    kappa = (po - pe) / denom if denom != 0.0 else float("nan")
    return float(kappa), conf, labels_arr.tolist()
