#!/usr/bin/env python3
"""CLI to compute Dice-based metrics for 2- or 3-class integer NIfTI labels.

This module is a thin wrapper around metrics.dice_cli_main() for backward
compatibility with `python -m t1prep.dice`.

Example:
    python -m t1prep.dice --gt path/to/gt.nii.gz --pred path/to/pred.nii.gz \\
            --save-conf conf.csv
"""
from __future__ import annotations

from .metrics import dice_cli_main

# Re-export main for direct calls
main = dice_cli_main

if __name__ == "__main__":
    raise SystemExit(dice_cli_main())
