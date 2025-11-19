#!/usr/bin/env python3
"""CLI to compute Dice-based metrics for 2- or 3-class integer NIfTI labels.

Example:
    python -m t1prep.dice --gt path/to/gt.nii.gz --pred path/to/pred.nii.gz \
            --save-conf conf.csv
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

from .metrics import compute_dice_nifti


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute Dice-based metrics for NIfTI label maps (2-3 classes)")
    p.add_argument("--gt", required=True, help="Ground truth NIfTI path")
    p.add_argument("--pred", required=True, help="Test/prediction NIfTI path")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output (labels, overall and per-label Dice metrics). If not set, prints a single line with per-label Dice as a vector, generalized_dice and dice_weighted.",
    )
    p.add_argument("--save-conf", help="Optional path to save confusion matrix as CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # compute_dice_nifti returns:
    # conf: confusion matrix (ndarray)
    # order: label order (list or ndarray)
    # dice_per: per-label Dice scores (ndarray)
    # dice_weighted: weighted average Dice score (float)
    # generalized_dice: generalized Dice score (float)
    conf, order, dice_per, dice_weighted, generalized_dice = compute_dice_nifti(
        args.gt, args.pred
    )
    if args.verbose:
        if dice_per.size:
            for lab, kv in zip(order, dice_per):
                kv_str = "nan" if not np.isfinite(kv) else f"{kv:.6f}"
                print(f"dice_per[{lab}]: {kv_str}")
        print(f"generalized_dice: {generalized_dice:.6f}")
        print(f"dice_weighted:    {dice_weighted:.6f}")
    else:
        # Compact single-line output: overall followed by per-label Dice as a vector
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
    raise SystemExit(main())
