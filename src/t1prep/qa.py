"""Image quality assurance measures for T1-weighted MRI.

This module provides functions to estimate image quality measures following
the CAT12 ``cat_vol_qa201901x`` / ``cat_stat_marks`` framework.  The
measures quantify noise, spatial intensity inhomogeneity, tissue contrast,
and effective resolution, and are combined into an overall *Image Quality
Rating* (IQR).

Typical entry point
-------------------
Call :func:`estimate_qa` with a segmentation label map, bias-corrected
intensity image, and voxel dimensions.  It returns a dictionary of
quality measures and a 1--6 school-mark style rating.

References
----------
- Gaser C, Dahnke R et al., *CAT – A Computational Anatomy Toolbox for
  the Analysis of Structural MRI Data*, 2024.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    uniform_filter,
    zoom,
)


# ---------------------------------------------------------------------------
# Rating scale (linear mapping matching CAT12 cat_stat_marks defaults)
# ---------------------------------------------------------------------------
# Each entry maps a measure name to (best_value, worst_value).
# A value equal to *best* → mark 1 (excellent), equal to *worst* → mark 6.
# Bounds from CAT12 cat_stat_marks.m (rev 2577).
_RATING_BOUNDS: dict[str, tuple[float, float]] = {
    "NCR": (0.0466, 0.3949),
    "ICR": (0.2178, 2.2338),          # cat_stat_marks: 1.1169 * 2
    "contrastr": (1.0 / 3.0, 0.0),
    "res_RMS": (0.50, 3.00),
    "res_ECR": (0.125, 1.00),
}


def _mark(value: float, best: float, worst: float) -> float:
    """Map *value* to a 1-6 school-mark scale.

    1 = excellent, 6 = very poor.  Clamped to [0.5, 10.5].

    Args:
        value: The measured quantity.
        best: Value that corresponds to mark 1.
        worst: Value that corresponds to mark 6.

    Returns:
        Continuous mark in [0.5, 10.5].
    """
    span = abs(worst - best)
    if span < 1e-12:
        return 1.0
    mark = (value - best) / (worst - best) * 5.0 + 1.0
    return float(np.clip(mark, 0.5, 10.5))


def _iqr(marks: list[float], power: int = 8) -> float:
    """Compute Image Quality Rating from individual marks.

    Uses a generalised power mean so that bad scores dominate.
    CAT12 uses power=8 by default.

    Args:
        marks: List of individual measure marks (1--6+ scale).
        power: Exponent for the power mean (default 8).

    Returns:
        Combined mark (lower is better).
    """
    arr = np.asarray(marks, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(max(0.0, np.mean(arr ** power) ** (1.0 / power)))


def mark_to_grade(mark: float) -> str:
    """Convert a numeric mark to a letter grade.

    Args:
        mark: Numeric mark on [0.5, 10.5] scale.

    Returns:
        Letter grade string (e.g. ``'A'``, ``'B+'``, ``'D-'``).
    """
    if mark <= 1.5:
        return "A+"
    if mark <= 2.0:
        return "A"
    if mark <= 2.5:
        return "A-"
    if mark <= 3.0:
        return "B+"
    if mark <= 3.5:
        return "B"
    if mark <= 4.0:
        return "B-"
    if mark <= 4.5:
        return "C+"
    if mark <= 5.0:
        return "C"
    if mark <= 5.5:
        return "C-"
    if mark <= 6.0:
        return "D"
    if mark <= 7.0:
        return "D-"
    if mark <= 8.0:
        return "E"
    return "F"


def mark_to_rps(mark: float) -> float:
    """Convert a numeric mark to a *Rating Percentage Score* (RPS).

    Args:
        mark: Numeric mark on [0.5, 10.5] scale.

    Returns:
        Percentage score in [0, 100] (higher is better).
    """
    return float(np.clip(105.0 - mark * 10.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Individual quality measures
# ---------------------------------------------------------------------------

def _tissue_masks(p0: np.ndarray):
    """Create binary tissue masks from a label map.

    Args:
        p0: Segmentation label map (0=BG, 1=CSF, 2=GM, 3=WM).

    Returns:
        Tuple of binary masks (csf, gm, wm, brain).
    """
    p0r = np.round(p0)
    csf = p0r == 1
    gm = p0r == 2
    wm = p0r == 3
    brain = p0r >= 1
    return csf, gm, wm, brain


def _masked_local_sd(
    intensity: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Compute local standard deviation using only masked voxels.

    Mirrors CAT12 ``cat_vol_localstat(Ym, mask, nb, 4)`` where only
    voxels inside *mask* contribute to the neighbourhood statistics.

    Args:
        intensity: Input image.
        mask: Binary mask — only these voxels contribute.
        radius: Neighbourhood half-width in voxels (CAT12 uses nb=1).

    Returns:
        Array of per-voxel local SD (non-masked voxels are 0).
    """
    size = 2 * radius + 1
    mask_f = mask.astype(np.float64)
    im = intensity.astype(np.float64) * mask_f

    # Number of masked neighbours (fraction → count via kernel volume)
    count = uniform_filter(mask_f, size=size, mode="constant")
    count = np.maximum(count, 1e-10)

    local_mean = uniform_filter(im, size=size, mode="constant") / count
    local_sq_mean = (
        uniform_filter(im ** 2, size=size, mode="constant") / count
    )
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    return np.sqrt(local_var)


def estimate_noise(
    intensity: np.ndarray,
    wm_mask: np.ndarray,
    vx_vol: np.ndarray,
    contrast: float,
) -> float:
    """Estimate Noise-to-Contrast Ratio (NCR).

    Computes the masked local standard deviation within white matter
    (only WM neighbours contribute) and normalises by the tissue
    contrast, following the CAT12 ``cat_vol_localstat`` approach with
    ``nb=1``.

    Args:
        intensity: Bias-corrected intensity image.
        wm_mask: Binary white-matter mask.
        vx_vol: Voxel dimensions in mm (length-3 array).
        contrast: Absolute tissue contrast (WM median − GM median).

    Returns:
        NCR value (lower is better).
    """
    if contrast < 1e-6 or np.sum(wm_mask) < 100:
        return float("nan")

    # Masked local SD with radius=1 voxel (matching CAT12 nb=1)
    local_sd = _masked_local_sd(intensity, wm_mask, radius=1)

    # Median of local SD within WM (only where local_sd > 0)
    sd_vals = local_sd[wm_mask]
    sd_vals = sd_vals[sd_vals > 0]
    if len(sd_vals) == 0:
        return float("nan")
    noise = float(np.median(sd_vals))
    return noise / contrast


def estimate_bias(
    intensity: np.ndarray,
    wm_mask: np.ndarray,
    vx_vol: np.ndarray,
    contrast: float,
) -> float:
    """Estimate Inhomogeneity-to-Contrast Ratio (ICR).

    Measures the spatial variation of white-matter intensities after
    low-pass filtering to isolate the bias-field component from noise.
    Following CAT12, the WM map is downsampled to ~4 mm resolution
    before computing the standard deviation.

    Args:
        intensity: Bias-corrected intensity image.
        wm_mask: Binary white-matter mask.
        vx_vol: Voxel dimensions in mm (length-3 array).
        contrast: Absolute tissue contrast (WM median − GM median).

    Returns:
        ICR value (lower is better).
    """
    if contrast < 1e-6 or np.sum(wm_mask) < 100:
        return float("nan")

    # CAT12: downsample WM intensities to ~4 mm to remove noise,
    # keeping only the slowly-varying bias field.
    target_res = 4.0
    vx = np.asarray(vx_vol, dtype=np.float64)
    zoom_factor = vx / target_res  # < 1 → shrink

    wm_f = wm_mask.astype(np.float64)
    im_wm = intensity.astype(np.float64) * wm_f

    # Downsample using order=1 (bilinear)
    im_ds = zoom(im_wm, zoom_factor, order=1)
    mask_ds = zoom(wm_f, zoom_factor, order=1)

    # Recover WM-only mean values (avoid PVE division-by-zero)
    valid = mask_ds > 0.5
    if np.sum(valid) < 20:
        return float("nan")
    im_ds[valid] /= mask_ds[valid]

    # One pass of local-mean smoothing (CAT12: cat_vol_localstat nb=1)
    sm = _masked_local_mean(im_ds, valid, radius=1)
    vals = sm[valid & (sm > 0)]
    if len(vals) < 20:
        return float("nan")

    return float(np.std(vals) / contrast)


def _masked_local_mean(
    intensity: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Compute local mean using only masked voxels.

    Args:
        intensity: Input image.
        mask: Binary mask.
        radius: Neighbourhood half-width in voxels.

    Returns:
        Array of local means (non-masked voxels are 0).
    """
    size = 2 * radius + 1
    mask_f = mask.astype(np.float64)
    im = intensity.astype(np.float64) * mask_f

    count = uniform_filter(mask_f, size=size, mode="constant")
    count = np.maximum(count, 1e-10)
    result = uniform_filter(im, size=size, mode="constant") / count
    result[~mask] = 0.0
    return result


def estimate_contrast(
    intensity: np.ndarray,
    csf_mask: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
) -> tuple[float, float]:
    """Estimate tissue contrast and *contrast ratio* (contrastr).

    The contrast is the minimum of |WM−GM| and |GM−CSF| normalised
    by max(WM, GM), with a correction to avoid over-optimisation
    (matching CAT12 ``cat_vol_qa201901x``).

    Args:
        intensity: Bias-corrected intensity image.
        csf_mask: Binary CSF mask.
        gm_mask: Binary GM mask.
        wm_mask: Binary WM mask.

    Returns:
        Tuple ``(contrast_abs, contrastr)`` where *contrast_abs* is
        the unnormalised absolute contrast and *contrastr* is the
        normalised contrast ratio.
    """
    if np.sum(wm_mask) < 50 or np.sum(gm_mask) < 50:
        return (float("nan"), float("nan"))

    wm_med = float(np.median(intensity[wm_mask]))
    gm_med = float(np.median(intensity[gm_mask]))
    csf_med = (
        float(np.median(intensity[csf_mask]))
        if np.sum(csf_mask) > 50
        else 0.0
    )

    signal = max(wm_med, gm_med)
    if signal < 1e-6:
        return (float("nan"), float("nan"))

    # Relative contrast (CAT12 formula)
    contrast_rel = (
        min(abs(wm_med - gm_med), abs(gm_med - csf_med)) / signal
    )
    # Avoid over-optimisation (CAT12: contrast + min(0, 13/36 - c) * 1.2)
    contrast_rel = contrast_rel + min(0.0, 13.0 / 36.0 - contrast_rel) * 1.2

    contrast_abs = contrast_rel * signal
    return contrast_abs, max(0.0, contrast_rel)


def estimate_res_rms(vx_vol: np.ndarray) -> float:
    """Compute the RMS voxel dimension as a resolution indicator.

    Args:
        vx_vol: Voxel dimensions in mm (length-3 array).

    Returns:
        RMS resolution in mm (lower is better).
    """
    return float(np.sqrt(np.mean(np.asarray(vx_vol, dtype=np.float64) ** 2)))


def estimate_res_ecr(
    intensity: np.ndarray,
    p0: np.ndarray,
    vx_vol: np.ndarray,
) -> float:
    """Estimate *Effective Contrast Resolution* (res_ECR).

    Quantifies anatomical detail by measuring the gradient magnitude at
    GM/WM boundaries, following the CAT12 ``estimateECR0`` approach.

    The raw median gradient ``ecr0`` is transformed via
    ``abs(2.5 − ecr0 × 10)`` to match the CAT12 ECR scale.

    Args:
        intensity: Bias-corrected intensity image.
        p0: Segmentation label map (0=BG, 1=CSF, 2=GM, 3=WM).
        vx_vol: Voxel dimensions in mm (length-3 array).

    Returns:
        res_ECR value (lower is better).
    """
    vx = np.asarray(vx_vol, dtype=np.float64)

    # Normalise intensity so that WM ≈ 1, clip to [2/3, 1]
    brain = p0 > 0.5
    if np.sum(brain) < 100:
        return float("nan")

    im = intensity.astype(np.float64).copy()
    wm_med = (
        float(np.median(im[np.round(p0) == 3]))
        if np.sum(np.round(p0) == 3) > 50
        else 1.0
    )
    if wm_med > 1e-6:
        im /= wm_med
    im = np.clip(im, 2.0 / 3.0, 1.0)

    # GM/WM boundary region (p0 ∈ [2.05, 2.95])
    wm_boundary = (p0 > 2.05) & (p0 < 2.95)
    if np.sum(wm_boundary) < 50:
        return float("nan")

    # Gradient magnitude with sqrt(vx_vol) spacing (CAT12 convention)
    spacing = np.sqrt(vx)
    gx = sobel(im, axis=0) / (spacing[0] * 2.0)
    gy = sobel(im, axis=1) / (spacing[1] * 2.0)
    gz = sobel(im, axis=2) / (spacing[2] * 2.0)
    grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    ecr0 = float(np.median(grad[wm_boundary]))

    # CAT12 transform: res_ECR = abs(2.5 - ecr0 * 10)
    return abs(2.5 - ecr0 * 10.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_qa(
    p0: np.ndarray,
    intensity: np.ndarray,
    vx_vol: np.ndarray,
    vx_vol_orig: np.ndarray | None = None,
) -> dict:
    """Estimate all image quality measures.

    Args:
        p0: Segmentation label map (0=BG, 1=CSF, 2=GM, 3=WM).  Can be a
            continuous map (values in [0, 3]).
        intensity: Bias-corrected intensity image (same shape as *p0*).
        vx_vol: Voxel dimensions in mm of the data arrays (length-3).
        vx_vol_orig: Original acquisition voxel dimensions in mm.  Used
            only for ``res_RMS``.  Falls back to *vx_vol* if not given.

    Returns:
        Dictionary with the following structure::

            {
                "qualitymeasures": {
                    "NCR":      {"value": …, "mark": …, "desc": "…"},
                    "ICR":      {"value": …, "mark": …, "desc": "…"},
                    "contrastr": {"value": …, "mark": …, "desc": "…"},
                    "res_RMS":  {"value": …, "mark": …, "desc": "…"},
                    "res_ECR":  {"value": …, "mark": …, "desc": "…"},
                    "IQR":      {"value": …, "grade": "…", "desc": "…"},
                },
            }
    """
    vx = np.asarray(vx_vol, dtype=np.float64)
    vx_orig = (
        np.asarray(vx_vol_orig, dtype=np.float64)
        if vx_vol_orig is not None
        else vx
    )

    # Tissue masks
    csf_mask, gm_mask, wm_mask, brain_mask = _tissue_masks(p0)

    # Contrast
    contrast_abs, contrastr = estimate_contrast(
        intensity, csf_mask, gm_mask, wm_mask
    )

    # Noise (masked local SD, nb=1)
    ncr = estimate_noise(intensity, wm_mask, vx, contrast_abs)

    # Bias / inhomogeneity (downsampled to ~4 mm)
    icr = estimate_bias(intensity, wm_mask, vx, contrast_abs)

    # Resolution (uses original acquisition voxel dims)
    res_rms = estimate_res_rms(vx_orig)

    # Effective contrast resolution
    res_ecr = estimate_res_ecr(intensity, p0, vx)

    # Individual marks
    marks: dict[str, float] = {}
    for name, value in [
        ("NCR", ncr),
        ("ICR", icr),
        ("contrastr", contrastr),
        ("res_RMS", res_rms),
        ("res_ECR", res_ecr),
    ]:
        best, worst = _RATING_BOUNDS[name]
        marks[name] = _mark(value, best, worst)

    # Overall IQR — CAT12 uses only NCR + res_RMS with power 8
    iqr_value = _iqr(
        [marks["NCR"], marks["res_RMS"]], power=8
    )
    iqr_grade = mark_to_grade(iqr_value)

    # Descriptions
    descs = {
        "NCR": "Noise-to-Contrast Ratio (lower is better)",
        "ICR": "Inhomogeneity-to-Contrast Ratio (lower is better)",
        "contrastr": "Tissue contrast ratio (closer to 0.33 is better)",
        "res_RMS": "RMS voxel dimension in mm (lower is better)",
        "res_ECR": "Effective Contrast Resolution (lower is better)",
        "IQR": "Overall Image Quality Rating (1=excellent, 6=poor)",
    }

    result: dict = {}
    for name, value in [
        ("NCR", ncr),
        ("ICR", icr),
        ("contrastr", contrastr),
        ("res_RMS", res_rms),
        ("res_ECR", res_ecr),
    ]:
        result[name] = {
            "value": round(value, 4) if np.isfinite(value) else None,
            "mark": round(marks[name], 2) if np.isfinite(marks[name]) else None,
            "desc": descs[name],
        }

    result["IQR"] = {
        "value": round(iqr_value, 2) if np.isfinite(iqr_value) else None,
        "grade": iqr_grade,
        "desc": descs["IQR"],
    }

    return {"qualitymeasures": result}
