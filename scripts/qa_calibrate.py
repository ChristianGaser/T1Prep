#!/usr/bin/env python3
"""Derive the QA rating bounds in ``t1prep.qa`` from BrainWeb Phantom runs.

The rating scale of every quality measure maps a "best" value to mark 1 and
a "worst" value to mark 6 (see ``_RATING_BOUNDS`` in
:mod:`t1prep.qa`).  Those two anchors are calibrated on the BrainWeb
Phantom (BWP), which provides the same anatomy with simulated noise (``pn``)
and intensity inhomogeneity (``rf``) levels.  A robust line is fitted
through the measure as a function of the simulated degradation level and
evaluated at level 1 and level 6, following the CAT12 procedure
(``calc_limits_QA.m``).

Usage::

    # process the BWP images first, then
    python scripts/qa_calibrate.py /path/to/BWP/report

    # only the noise anchors, from a subset
    python scripts/qa_calibrate.py /path/to/BWP/report --measure NCR

File names are expected to contain ``pn<level>`` and ``rf<level>`` (e.g.
``BWPC_HC_T1_pn3_rf040pA_vx100x100x100``).  ``pn`` levels 1..9 and ``rf``
levels 20..100 are mapped linearly onto the 1..5 degradation scale, so that
extrapolating to 6 gives the "worst" anchor.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Measures that scale with the simulated noise level, and those that scale
# with the simulated inhomogeneity level.
NOISE_MEASURES = ("NCR", "res_ECR")
BIAS_MEASURES = ("ICR",)

_PATTERN = re.compile(r"pn(\d+)_rf(\d+)")


def robust_fit(x: np.ndarray, y: np.ndarray, tune: float = 4.685,
               iterations: int = 50) -> tuple[float, float]:
    """Bisquare IRLS line fit (equivalent to MATLAB ``robustfit``).

    Args:
        x: Predictor values.
        y: Response values.
        tune: Bisquare tuning constant.
        iterations: Maximum number of reweighting steps.

    Returns:
        Tuple ``(intercept, slope)``.
    """
    design = np.vstack([np.ones_like(x), x]).T
    weights = np.ones_like(y)
    coef = np.zeros(2)
    for _ in range(iterations):
        sqrt_w = np.sqrt(weights)[:, None]
        new_coef, *_ = np.linalg.lstsq(design * sqrt_w, y * sqrt_w[:, 0],
                                       rcond=None)
        if np.allclose(new_coef, coef, rtol=1e-10, atol=1e-12):
            coef = new_coef
            break
        coef = new_coef
        residual = y - design @ coef
        scale = np.median(np.abs(residual - np.median(residual))) / 0.6745
        if scale < 1e-12:
            break
        u = residual / (tune * scale)
        weights = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return float(coef[0]), float(coef[1])


def collect(report_dir: Path) -> list[dict]:
    """Read all BWP report JSONs below *report_dir*."""
    rows = []
    for path in sorted(report_dir.rglob("*.json")):
        match = _PATTERN.search(path.name)
        if not match:
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        measures = data.get("qualitymeasures", {})
        if not measures:
            continue
        row = {
            "pn": int(match.group(1)),
            "rf": int(match.group(2)),
            "file": path.name,
        }
        for name, entry in measures.items():
            if isinstance(entry, dict) and isinstance(
                entry.get("value"), (int, float)
            ):
                row[name] = float(entry["value"])
        rows.append(row)
    return rows


def level(values: np.ndarray, kind: str) -> np.ndarray:
    """Map simulated ``pn``/``rf`` settings onto the 1..5 degradation scale."""
    if kind == "noise":          # pn 1,3,5,7,9  ->  1,2,3,4,5
        return (values - 1.0) / 8.0 * 4.0 + 1.0
    return (values - 20.0) / 80.0 * 4.0 + 1.0  # rf 20..100 -> 1..5


def calibrate(rows: list[dict], measure: str, kind: str) -> None:
    """Fit *measure* against the degradation level and print the anchors."""
    factor = "pn" if kind == "noise" else "rf"
    pairs = [(r[factor], r[measure]) for r in rows
             if measure in r and np.isfinite(r[measure])]
    if len(pairs) < 6:
        print(f"{measure:>10}: not enough data ({len(pairs)} values)")
        return

    x = level(np.array([p[0] for p in pairs], dtype=float), kind)
    y = np.array([p[1] for p in pairs], dtype=float)

    intercept, slope = robust_fit(x, y)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Sensitivity to the *other* factor should be small: it quantifies how
    # much of the unrelated degradation leaks into this measure.
    other = "rf" if kind == "noise" else "pn"
    other_levels = sorted({r[other] for r in rows if measure in r})
    means = [np.mean([r[measure] for r in rows
                      if r.get(other) == lev and measure in r])
             for lev in other_levels]
    leak = (max(means) - min(means)) / abs(slope * 4.0) if slope else np.nan

    best, worst = intercept + slope, intercept + slope * 6.0
    print(f"{measure:>10}: {intercept:+.4f} {slope:+.4f}*level  "
          f"R2={r2:5.3f}  n={len(y):3d}  {other}-leakage={leak * 100:5.1f}%")
    print(f"{'':>10}  \"{measure}\": ({best:.4f}, {worst:.4f}),")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the T1Prep QA rating bounds on the BWP."
    )
    parser.add_argument("report_dir", type=Path,
                        help="directory with the BWP report JSON files")
    parser.add_argument("--measure", action="append", default=None,
                        help="restrict to this measure (repeatable)")
    args = parser.parse_args()

    rows = collect(args.report_dir)
    if not rows:
        raise SystemExit(f"no BWP reports found below {args.report_dir}")
    print(f"# {len(rows)} BWP reports")
    print("# rating bounds (best = level 1, worst = level 6):\n")

    for measure in NOISE_MEASURES:
        if args.measure and measure not in args.measure:
            continue
        calibrate(rows, measure, "noise")
    for measure in BIAS_MEASURES:
        if args.measure and measure not in args.measure:
            continue
        calibrate(rows, measure, "bias")


if __name__ == "__main__":
    main()
