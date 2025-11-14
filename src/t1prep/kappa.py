#!/usr/bin/env python3
"""CLI to compute Cohen's kappa for 2- or 3-class integer NIfTI labels.

Example:
    python -m t1prep.kappa --gt path/to/gt.nii.gz --pred path/to/pred.nii.gz \
            --labels 1,2,3 --mask-mode intersection --save-conf conf.csv
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

from .metrics import compute_cohen_kappa_nifti


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute Cohen's kappa for NIfTI label maps (2-3 classes)")
    p.add_argument("--gt", required=True, help="Ground truth NIfTI path")
    p.add_argument("--pred", required=True, help="Test/prediction NIfTI path")
    # Mask is derived from labels; choose the strategy
    p.add_argument(
        "--mask-mode",
        choices=["intersection", "gt"],
        default="intersection",
        help="How to define the evaluation mask from labels: 'intersection' (gt AND pred in labels) or 'gt' (gt in labels)",
    )
    p.add_argument(
        "--labels",
        required=False,
        help="Comma-separated label values to include (e.g., 1,2 or 1,2,3). If omitted, uses intersection of labels present (excluding 0)",
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
    kappa, conf, order = compute_cohen_kappa_nifti(args.gt, args.pred, labels, args.mask_mode)
    print(f"kappa: {kappa:.6f}")
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
