"""Metrics utilities: Dice-based metrics for 2- or 3-class NIfTI label maps.

This module provides a function to compute Dice-based metrics between an integer
label ground truth and a test label map restricted to a mask and a selected set
of label values (e.g., [1, 2] or [1, 2, 3]).

Inputs may be file paths to NIfTI images or nibabel Nifti1Image objects.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib

NiftiLike = Union[str, nib.Nifti1Image]

__all__ = ["compute_dice_nifti", "dice_cli_main"]


def _load_labels(img: NiftiLike, round_labels: bool) -> np.ndarray:
    if isinstance(img, nib.Nifti1Image):
        data = img.get_fdata()
    else:
        data = nib.load(str(img)).get_fdata()
    if round_labels:
        # Convert to integer labels; round to nearest to be robust to tiny float stores
        return np.rint(data).astype(np.int64, copy=False)
    # Preserve original floating values for soft/continuous Dice
    return np.asarray(data, dtype=float)


def _to_bool_mask_from_labels(arr: np.ndarray, labels_arr: np.ndarray) -> np.ndarray:
    """Return a boolean mask where arr is one of the given labels."""
    return np.isin(arr, labels_arr)


def compute_dice_nifti(
    gt: NiftiLike,
    pred: NiftiLike,
    *,
    round_labels: bool = True,
) -> Tuple[np.ndarray, Sequence[int], np.ndarray, float, float]:
    """Compute Dice-based metrics for 2- or 3-class integer labels from NIfTI images.

    Parameters
    ----------
    gt : str | nib.Nifti1Image
        Ground-truth label image (NIfTI). Integer labels (2 or 3 classes).
    pred : str | nib.Nifti1Image
        Test/predicted label image (NIfTI). Integer labels (2 or 3 classes).
    round_labels : bool, optional
        When True (default), round both inputs to the nearest integer before
        computing Dice scores (original behaviour). When False, compute soft /
        continuous Dice using the unrounded arrays. In soft mode, per-class
        volumes and overlaps are derived from the continuous values, while the
        class set and confusion matrix still follow the rounded label indices.

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
        weights. When ``round_labels`` is True, uses

            GDC = 2 * sum_c w_c * TP_c / sum_c w_c * (2*TP_c + FP_c + FN_c),

        with w_c = 1 / (V_c^2 + eps) and V_c = #gt voxels of class c. When
        ``round_labels`` is False, TP/FP/FN are replaced by continuous overlaps
        and volumes derived from the unrounded maps.

    Notes
    -----
    - Dice-based metrics ignore true negatives (TN) by construction.
    - A brain mask is obtained from ``gt != 0`` (using raw values when
        ``round_labels`` is False); all voxels inside this mask contribute to
        the confusion matrix, so disagreements between ``gt`` and ``pred`` are
        fully accounted for.
    - In soft mode, per-class memberships are clipped to [0, 1] to keep Dice
        values bounded.
    - Handles degenerate cases by returning NaN when denominators are zero or
        when there are no valid voxels after masking.
    """
    # Load labels (rounded or continuous depending on mode)
    y_true = _load_labels(gt, round_labels)
    y_pred = _load_labels(pred, round_labels)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: gt {y_true.shape} vs pred {y_pred.shape}")

    # Keep an unrounded copy when we need soft/continuous Dice
    if round_labels:
        y_true_raw: Optional[np.ndarray] = None
        y_pred_raw: Optional[np.ndarray] = None
    else:
        y_true_raw = y_true
        y_pred_raw = y_pred
        # For confusion/label discovery we still need integer views
        y_true_round = np.rint(y_true_raw).astype(np.int64, copy=False)
        y_pred_round = np.rint(y_pred_raw).astype(np.int64, copy=False)
        # Reuse the rounded arrays for confusion while keeping raw for soft Dice
        y_true_conf = y_true_round
        y_pred_conf = y_pred_round
    if round_labels:
        y_true_conf = y_true
        y_pred_conf = y_pred

    # Determine labels to consider from ground truth only (exclude background 0)
    if y_true_conf.ndim > 3:
        # Multi-channel probability maps: infer labels from channel count
        labels_arr = np.arange(1, y_true_conf.shape[-1] + 1)
    else:
        labels_arr = np.unique(y_true_conf)
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
    if not round_labels and y_true_raw is not None:
        source_for_mask = y_true_raw
    else:
        source_for_mask = y_true_conf

    if source_for_mask.ndim > 3:
        brain_mask = np.any(source_for_mask != 0, axis=-1)
    else:
        brain_mask = source_for_mask != 0

    if y_true_conf.ndim > 3:
        yt_conf = np.argmax(y_true_conf, axis=-1) + 1
        yp_conf = np.argmax(y_pred_conf, axis=-1) + 1
    else:
        yt_conf = y_true_conf
        yp_conf = y_pred_conf

    yt_conf = yt_conf[brain_mask]
    yp_conf = yp_conf[brain_mask]

    # Ensure both yt and yp are within the label set (in case pred has
    # unexpected labels); voxels with out-of-set labels are ignored.
    valid = np.isin(yt_conf, labels_arr) & np.isin(yp_conf, labels_arr)
    yt_conf = yt_conf[valid]
    yp_conf = yp_conf[valid]
    if yt_conf.size == 0:
        return (
            np.zeros((K, K), dtype=np.int64),
            labels_arr.tolist(),
            np.full((K,), np.nan, dtype=float),
            float("nan"),
            float("nan"),
        )

    # Map to indices
    yt_idx = np.fromiter((label_to_idx[int(v)] for v in yt_conf), count=yt_conf.size, dtype=np.int64)
    yp_idx = np.fromiter((label_to_idx[int(v)] for v in yp_conf), count=yp_conf.size, dtype=np.int64)

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

    # Row/col sums (gt / pred volumes per class) from confusion
    row = conf.sum(axis=1).astype(float)  # gt counts per class
    col = conf.sum(axis=0).astype(float)  # pred counts per class

    # ------------------------------------------------------------------
    # Dice per label
    # ------------------------------------------------------------------
    dice_per = np.full((K,), np.nan, dtype=float)

    if round_labels:
        tp = np.diag(conf).astype(float)
        fn = row - tp
        fp = col - tp
        denom_dice = 2.0 * tp + fp + fn

        valid_dice = denom_dice > 0
        dice_per[valid_dice] = 2.0 * tp[valid_dice] / denom_dice[valid_dice]
    else:
        # Soft/continuous Dice: operate on raw arrays mapped to per-class weights
        if y_true_raw is None or y_pred_raw is None:
            raise RuntimeError("Soft Dice requires access to raw arrays.")

        if y_true_raw.ndim > 3:
            yt_flat = y_true_raw[brain_mask]
            yp_flat = y_pred_raw[brain_mask]
            if yt_flat.ndim != 2:
                yt_flat = yt_flat.reshape(-1, yt_flat.shape[-1])
            if yp_flat.shape != yt_flat.shape:
                raise ValueError("Soft Dice expects matching channel dimensions.")
            if yt_flat.shape[1] != K:
                raise ValueError(
                    "Soft Dice channel count mismatch: "
                    f"expected {K}, got {yt_flat.shape[1]}"
                )
            gt_stack = yt_flat.T
            pred_stack = yp_flat.T
        else:
            yt_flat = y_true_raw[brain_mask].reshape(-1)
            yp_flat = y_pred_raw[brain_mask].reshape(-1)
            assignments = np.fromiter(
                (label_to_idx.get(int(v), -1) for v in np.rint(yt_flat)),
                count=yt_flat.size,
                dtype=np.int64,
            )
            pred_assign = np.fromiter(
                (label_to_idx.get(int(v), -1) for v in np.rint(yp_flat)),
                count=yp_flat.size,
                dtype=np.int64,
            )
            valid_soft = (assignments >= 0) & (pred_assign >= 0)
            yt_flat = yt_flat[valid_soft]
            yp_flat = yp_flat[valid_soft]
            assignments = assignments[valid_soft]
            pred_assign = pred_assign[valid_soft]
            gt_stack = np.zeros((K, yt_flat.size), dtype=float)
            pred_stack = np.zeros_like(gt_stack)
            for idx_label in range(K):
                gt_sel = assignments == idx_label
                pred_sel = pred_assign == idx_label
                gt_stack[idx_label, gt_sel] = yt_flat[gt_sel]
                pred_stack[idx_label, pred_sel] = yp_flat[pred_sel]

            # Clamp membership weights to [0, 1] to keep Dice bounded
            np.clip(gt_stack, 0.0, 1.0, out=gt_stack)
            np.clip(pred_stack, 0.0, 1.0, out=pred_stack)

        num = 2.0 * np.sum(gt_stack * pred_stack, axis=1)
        den = np.sum(gt_stack * gt_stack + pred_stack * pred_stack, axis=1)
        valid_dice = den > 0
        dice_per[valid_dice] = num[valid_dice] / den[valid_dice]

    # ------------------------------------------------------------------
    # Volume-weighted Dice (weights based on gt volumes)
    # ------------------------------------------------------------------
    if round_labels:
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
    else:
        vol_gt_soft = np.sum(gt_stack, axis=1)
        valid_vol = vol_gt_soft > 0
        valid_combined = valid_vol & valid_dice
        if np.any(valid_combined):
            weights = vol_gt_soft[valid_combined]
            weights = weights / weights.sum()
            dice_weighted = float(np.sum(weights * dice_per[valid_combined]))
        else:
            dice_weighted = float("nan")

    # ------------------------------------------------------------------
    # Generalized Dice (inverse-squared volume weights)
    # ------------------------------------------------------------------
    eps = np.finfo(float).eps
    if round_labels:
        valid_gd = row > 0
        if np.any(valid_gd):
            w = np.zeros_like(row)
            w[valid_gd] = 1.0 / (row[valid_gd] ** 2 + eps)

            num_gd = 2.0 * np.sum(w * tp)
            den_gd = np.sum(w * (2.0 * tp + fp + fn))
            generalized_dice = float(num_gd / den_gd) if den_gd > 0.0 else float("nan")
        else:
            generalized_dice = float("nan")
    else:
        vol_gt_soft = np.sum(gt_stack, axis=1)
        vol_pred_soft = np.sum(pred_stack, axis=1)
        tp_soft = np.sum(gt_stack * pred_stack, axis=1)
        valid_gd = vol_gt_soft > 0
        if np.any(valid_gd):
            w = np.zeros_like(vol_gt_soft)
            w[valid_gd] = 1.0 / (vol_gt_soft[valid_gd] ** 2 + eps)
            num_gd = 2.0 * np.sum(w * tp_soft)
            den_gd = np.sum(w * (vol_gt_soft + vol_pred_soft))
            generalized_dice = float(num_gd / den_gd) if den_gd > 0.0 else float("nan")
        else:
            generalized_dice = float("nan")

    return (
        conf,
        labels_arr.tolist(),
        dice_per,
        float(dice_weighted),
        float(generalized_dice),
    )


# ---------------------------------------------------------------------------
# CLI entry point (merged from dice.py)
# ---------------------------------------------------------------------------


def _parse_dice_args(argv=None):
    """Parse command-line arguments for Dice CLI."""
    import argparse

    p = argparse.ArgumentParser(
        description="Compute Dice-based metrics for NIfTI label maps (2-3 classes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gt", required=True, help="Ground truth NIfTI path")
    p.add_argument("--pred", required=True, help="Test/prediction NIfTI path")
    p.add_argument(
        "--soft",
        action="store_true",
        help=(
            "Use soft/continuous Dice without rounding labels. Applicable when "
            "inputs store probability/partial-volume values."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print detailed output (labels, overall and per-label Dice metrics). "
            "If not set, prints a single CSV-like line with per-label Dice as a vector, "
            "followed by generalized_dice and dice_weighted."
        ),
    )
    p.add_argument("--save-conf", help="Optional path to save confusion matrix as CSV")
    return p.parse_args(argv)


def dice_cli_main(argv=None) -> int:
    """CLI main function for computing Dice metrics.
    
    Returns 0 on success, non-zero on error.
    """
    import sys

    args = _parse_dice_args(argv)
    conf, order, dice_per, dice_weighted, generalized_dice = compute_dice_nifti(
        args.gt, args.pred, round_labels=not args.soft
    )

    if args.verbose:
        if dice_per.size:
            for lab, kv in zip(order, dice_per):
                kv_str = "nan" if not np.isfinite(kv) else f"{kv:.6f}"
                print(f"dice_per[{lab}]: {kv_str}")
        print(f"generalized_dice: {generalized_dice:.6f}")
        print(f"dice_weighted:    {dice_weighted:.6f}")
    else:
        if dice_per.size:
            vec_str = ",".join(
                "nan" if not np.isfinite(kv) else f"{kv:.6f}" for kv in dice_per
            )
            print(f"{vec_str},{generalized_dice:.6f},{dice_weighted:.6f}")
        else:
            print(f"{generalized_dice:.6f},{dice_weighted:.6f}")

    if args.save_conf:
        try:
            np.savetxt(args.save_conf, conf.astype(int), fmt="%d", delimiter=",")
            print(f"confusion saved to: {args.save_conf}")
        except Exception as e:
            print(f"Failed to save confusion: {e}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(dice_cli_main())
