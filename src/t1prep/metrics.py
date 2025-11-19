"""Metrics utilities: Dice-based metrics for 2- or 3-class NIfTI label maps.

This module provides a function to compute Dice-based metrics between an integer
label ground truth and a test label map restricted to a mask and a selected set
of label values (e.g., [1, 2] or [1, 2, 3]).

Inputs may be file paths to NIfTI images or nibabel Nifti1Image objects.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import nibabel as nib

NiftiLike = Union[str, nib.Nifti1Image]

__all__ = ["compute_dice_nifti"]


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


def compute_dice_nifti(
    gt: NiftiLike,
    pred: NiftiLike,
) -> Tuple[np.ndarray, Sequence[int], np.ndarray, float, float]:
    """Compute Dice-based metrics for 2- or 3-class integer labels from NIfTI images.

    Parameters
    ----------
    gt : str | nib.Nifti1Image
        Ground-truth label image (NIfTI). Integer labels (2 or 3 classes).
    pred : str | nib.Nifti1Image
        Test/predicted label image (NIfTI). Integer labels (2 or 3 classes).
        Notes
        -----
        - All **non-zero** integer labels present in the ground truth ``gt`` are
            evaluated. The prediction ``pred`` does **not** control which labels are
            included, so missing classes in ``pred`` are treated as Dice = 0 rather
            than being silently ignored.

    Returns
    -------
    confusion : np.ndarray
        KxK confusion matrix with rows gt and columns pred, for K=len(labels).
    labels_order : Sequence[int]
        The label values corresponding to the rows/cols of ``confusion``.
    dice_per_label : np.ndarray
        Per-label Dice coefficients of length K, computed as
        2*TP / (2*TP + FP + FN) for each class.
    dice_weighted : float
        Volume-weighted Dice coefficient: sum_c w_c * Dice_c,
        where w_c is proportional to the number of gt voxels of class c.
    generalized_dice : float
        Generalized Dice coefficient using inverse-squared class volumes as
        weights:

            GDC = 2 * sum_c w_c * TP_c / sum_c w_c * (2*TP_c + FP_c + FN_c),

        with w_c = 1 / (V_c^2 + eps) and V_c = #gt voxels of class c.

    Notes
    -----
    - Dice-based metrics ignore true negatives (TN) by construction.
    - A brain mask is obtained from ``gt != 0``; all voxels inside this mask
        contribute to the confusion matrix, so disagreements between ``gt`` and
        ``pred`` are fully accounted for.
    - Handles degenerate cases by returning NaN when denominators are zero or
        when there are no valid voxels after masking.
    - Input labels are rounded before casting to integer to tolerate minor
        float imprecision in saved NIfTI files.
    """
    # Load integer labels
    y_true = _load_labels(gt)
    y_pred = _load_labels(pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: gt {y_true.shape} vs pred {y_pred.shape}")

    # Determine labels to consider from ground truth only (exclude background 0)
    labels_arr = np.unique(y_true)
    labels_arr = labels_arr[labels_arr != 0]
    if labels_arr.size == 0:
        return (
            np.zeros((0, 0), dtype=np.int64),
            [],
            np.zeros((0,), dtype=float),
            float("nan"),
            float("nan"),
        )

    # Build mapping from label value to contiguous index 0..K-1
    label_to_idx = {int(v): i for i, v in enumerate(labels_arr.tolist())}
    K = len(labels_arr)

    # Build a brain mask from ground truth and select valid voxels
    # All non-zero gt voxels contribute; disagreements in pred are preserved.
    brain_mask = y_true != 0

    yt = y_true[brain_mask]
    yp = y_pred[brain_mask]

    # Ensure both yt and yp are within the label set (in case pred has
    # unexpected labels); voxels with out-of-set labels are ignored.
    valid = np.isin(yt, labels_arr) & np.isin(yp, labels_arr)
    yt = yt[valid]
    yp = yp[valid]
    if yt.size == 0:
        return (
            np.zeros((K, K), dtype=np.int64),
            labels_arr.tolist(),
            np.full((K,), np.nan, dtype=float),
            float("nan"),
            float("nan"),
        )

    # Map to indices
    yt_idx = np.fromiter((label_to_idx[int(v)] for v in yt), count=yt.size, dtype=np.int64)
    yp_idx = np.fromiter((label_to_idx[int(v)] for v in yp), count=yp.size, dtype=np.int64)

    # Confusion matrix via bincount
    idx = yt_idx * K + yp_idx
    conf = np.bincount(idx, minlength=K * K).reshape(K, K)

    N = conf.sum()
    if N == 0:
        return (
            conf,
            labels_arr.tolist(),
            np.full((K,), np.nan, dtype=float),
            float("nan"),
            float("nan"),
        )

    # Row/col sums (gt / pred volumes per class)
    row = conf.sum(axis=1).astype(float)  # gt counts per class
    col = conf.sum(axis=0).astype(float)  # pred counts per class

    # ------------------------------------------------------------------
    # Dice per label
    # ------------------------------------------------------------------
    dice_per = np.full((K,), np.nan, dtype=float)

    tp = np.diag(conf).astype(float)
    fn = row - tp
    fp = col - tp
    denom_dice = 2.0 * tp + fp + fn

    valid_dice = denom_dice > 0
    dice_per[valid_dice] = 2.0 * tp[valid_dice] / denom_dice[valid_dice]

    # ------------------------------------------------------------------
    # Volume-weighted Dice (weights based on gt volumes)
    # ------------------------------------------------------------------
    valid_vol = row > 0
    if np.any(valid_vol) and np.any(valid_dice):
        valid_combined = valid_vol & valid_dice
        if np.any(valid_combined):
            weights = row[valid_combined]
            weights = weights / weights.sum()
            dice_weighted = float(np.sum(weights * dice_per[valid_combined]))
        else:
            dice_weighted = float("nan")
    else:
        dice_weighted = float("nan")

    # ------------------------------------------------------------------
    # Generalized Dice (inverse-squared volume weights)
    # ------------------------------------------------------------------
    eps = np.finfo(float).eps
    valid_gd = row > 0
    if np.any(valid_gd):
        w = np.zeros_like(row)
        w[valid_gd] = 1.0 / (row[valid_gd] ** 2 + eps)

        num = 2.0 * np.sum(w * tp)
        den = np.sum(w * (2.0 * tp + fp + fn))
        generalized_dice = float(num / den) if den > 0.0 else float("nan")
    else:
        generalized_dice = float("nan")

    return (
        conf,
        labels_arr.tolist(),
        dice_per,
        float(dice_weighted),
        float(generalized_dice),
    )
