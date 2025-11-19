#!/usr/bin/env python3
"""CLI to compute Dice-based metrics for 2- or 3-class integer NIfTI labels.

Example:
    python -m t1prep.dice --gt path/to/gt.nii.gz --pred path/to/pred.nii.gz \
            --labels 1,2,3 --mask-mode intersection --save-conf conf.csv
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
        "--labels",
        required=False,
        help="Comma-separated label values to include (e.g., 1,2 or 1,2,3). If omitted, uses intersection of labels present (excluding 0)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output (labels, overall and per-label Dice metrics). If not set, prints a single line with generalized_dice and per-label Dice as a vector.",
    )
    p.add_argument("--save-conf", help="Optional path to save confusion matrix as CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    labels = None
    if args.labels:
        try:
            labels = [int(s) for s in args.labels.split(",") if s.strip() != ""]
        except Exception as e:
            print(f"Failed to parse --labels: {e}", file=sys.stderr)
            return 2
    conf, order, dice_per, dice_weighted, generalized_dice = compute_dice_nifti(
        args.gt, args.pred, labels
    )
    if args.verbose:
        print(f"labels: {order}")
        print(f"generalized_dice: {generalized_dice:.6f}")
        if dice_per.size:
            for lab, kv in zip(order, dice_per):
                kv_str = "nan" if not np.isfinite(kv) else f"{kv:.6f}"
                print(f"dice_per[{lab}]: {kv_str}")
    else:
        # Compact single-line output: overall followed by per-label Dice as a vector
        # Example: "0.800000 [0.750000,0.820000,0.830000]"
        if dice_per.size:
            vec_str = ",".join(
                "nan" if not np.isfinite(kv) else f"{kv:.6f}" for kv in dice_per
            )
            print(f"{generalized_dice:.6f} [{vec_str}]")
        else:
            print(f"{generalized_dice:.6f} []")
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
