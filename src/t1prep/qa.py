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
    generate_binary_structure,
    binary_erosion,
    gaussian_filter,
    generic_filter,
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
    "NCR":      (0.0183, 0.0868),   # cat_vol_qa201901x: ndef.noise
    "CNR":      (0.0183, 0.0868),   # cat_vol_qa201901x: ndef.noise
    "contrastr": (1.0 / 3.0, 0.0), # cat_stat_marks default: CM=[1/3 0]
    "res_RMS":  (0.50, 3.00),       # cat_stat_marks default
    "res_ECR":  (0.0202, 0.1003),   # cat_vol_qa201901x: ndef.ECR
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


def _downsample_volume(
    volume: np.ndarray,
    mask: np.ndarray,
    vx_vol: np.ndarray,
    target_res: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample *volume* to *target_res* mm using PyTorch grid_sample.

    Computes mean-of-masked voxels per downsampled cell (matching CAT12
    ``cat_vol_resize(..., 'meanm')``): the masked image and mask are both
    resampled with trilinear interpolation, then the image is divided by
    the mask weight so boundary bias from zero-padded non-WM voxels is
    removed.

    An identity affine with a smaller output size produces uniform
    downsampling over the full input extent (``align_corners=True``).

    Args:
        volume: 3-D intensity image (WM-masked, zeros outside WM).
        mask: Binary WM mask (same shape as *volume*).
        vx_vol: Current voxel dimensions in mm (length-3).
        target_res: Target isotropic resolution in mm (default 2.0).

    Returns:
        Tuple ``(vol_ds, mask_ds)`` — downsampled intensity and binary
        mask at *target_res* mm resolution.
    """
    import torch
    import torch.nn.functional as F

    vx = np.asarray(vx_vol, dtype=np.float64)
    out_d = int(max(2, round(volume.shape[0] * vx[0] / target_res)))
    out_h = int(max(2, round(volume.shape[1] * vx[1] / target_res)))
    out_w = int(max(2, round(volume.shape[2] * vx[2] / target_res)))
    out_shape = (out_d, out_h, out_w)

    mask_f = mask.astype(np.float32)
    vol_masked = (volume * mask_f).astype(np.float32)

    def _pool(arr: np.ndarray) -> np.ndarray:
        # adaptive_avg_pool3d = mean pooling, matching CAT12 cat_vol_resize 'meanm'
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        return F.adaptive_avg_pool3d(t, out_shape).squeeze().numpy()

    vol_ds = _pool(vol_masked)
    mask_ds = _pool(mask_f)

    # Recover per-cell mean of masked values (matching CAT12 'meanm' with masking)
    valid = mask_ds > 0.5
    vol_ds[valid] /= mask_ds[valid]
    vol_ds[~valid] = 0.0

    return vol_ds, valid


def _masked_local_sd(
    intensity: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Fast masked local SD via uniform_filter (approximation).

    Uses the identity ``Var = E[x²] − E[x]²`` with masked uniform
    filters — equivalent to ``cat_vol_localstat(Ym, mask, nb, 4)``
    for interior mask voxels.  Very fast (O(N) C-level convolutions).

    Args:
        intensity: Input image.
        mask: Binary mask — only these voxels contribute.
        radius: Neighbourhood half-width in voxels (CAT12 uses nb=1).

    Returns:
        Array of per-voxel local SD; zero outside mask.
    """
    size = 2 * radius + 1
    mask_f = mask.astype(np.float64)
    im = intensity.astype(np.float64) * mask_f

    count = uniform_filter(mask_f, size=size, mode="constant")
    count = np.maximum(count, 1e-10)

    local_mean = uniform_filter(im, size=size, mode="constant") / count
    local_sq_mean = uniform_filter(im ** 2, size=size, mode="constant") / count
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    return np.sqrt(local_var)


def _masked_local_sd_exact(
    intensity: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Exact masked local SD matching ``localstat_double`` (CAT_Vol.c).

    Replicates ``cat_vol_localstat(Ym, mask, 1, F_STD)`` exactly:

    * 3×3×3 block neighbourhood (no Euclidean restriction, dist=1).
    * Only mask voxels contribute; non-mask voxels are treated as NaN.
    * Sample standard deviation (``ddof=1``), matching ``get_std_double``
      in ``CAT_Math.c`` which divides by ``n-1``.
    * Returns 0 (not NaN) for voxels with ≤1 valid neighbours so that
      the subsequent median step can filter them with ``sd > 0``.

    This uses ``scipy.ndimage.generic_filter`` with a Python callback,
    which is correct but **~100× slower** than ``_masked_local_sd`` at
    native resolution.  Prefer this only on 2 mm downsampled data where
    volume sizes are small (~640 K voxels).

    Args:
        intensity: Input image (non-mask voxels should be 0 or NaN).
        mask: Binary mask.

    Returns:
        Array of per-voxel sample SD; zero outside mask.
    """
    masked_vol = np.where(mask, intensity.astype(np.float32), np.nan)

    def _local_std(values: np.ndarray) -> float:
        v = values[np.isfinite(values)]
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    result = generic_filter(
        masked_vol, _local_std, size=3, mode="constant", cval=np.nan
    )
    result[~mask] = 0.0
    return result


def estimate_noise(
    intensity: np.ndarray,
    tissue_mask: np.ndarray,
    vx_vol: np.ndarray,
    contrast: float,
) -> float:
    """Estimate Noise-to-Contrast Ratio (NCR).

    Follows CAT12 ``cat_vol_qa201901x`` / ``estimateNoiseLevel``:

    1. Gaussian pre-smoothing (sigma = 0.8 + 0.5/vx_mm voxels) filling
       non-tissue with tissue mean, matching CAT12's ``Ymx`` computation.
    2. Downsample to **2 mm** isotropic via mean pooling (``adaptive_avg_pool3d``),
       matching CAT12 ``cat_vol_resize(..., 'reduceV', vx_vol, 2, ..., 'meanm')``.
    3. Compute masked local SD in the 3×3×3 neighbourhood via
       ``_masked_local_sd_exact`` (matching ``cat_vol_localstat(Ym, YM, 1, 4)``).
    4. Median of the per-voxel SDs within tissue, normalised by absolute
       tissue contrast.

    Args:
        intensity: Bias-corrected T1w image in native space.
        tissue_mask: Binary tissue mask.
        vx_vol: Voxel dimensions in mm (length-3 array).
        contrast: Absolute tissue contrast (i.e. tissue median − GM median).

    Returns:
        NCR value (lower is better).
    """
    if contrast < 1e-6 or np.sum(tissue_mask) < 100:
        return float("nan")

    # Step 1: Gaussian pre-smoothing matching CAT12's Ymx computation.
    # CAT12: Yos = Ymx.*Ywm + (1-Ywm).*T1th(3); spm_smooth(Yos,Yos,.8+.5./vx_vol)
    # Fill non-tissue with tissue mean, smooth, then restore tissue values only.
    vx = np.asarray(vx_vol, dtype=np.float64)
    sigma_vox = 0.8 + 0.5 / float(np.mean(vx))  # sigma in voxels
    tissue_mean = float(np.mean(intensity[tissue_mask]))
    im_padded = np.where(tissue_mask, intensity, tissue_mean)
    im_smoothed = gaussian_filter(im_padded, sigma=sigma_vox).astype(np.float32)
    # Only use smoothed values inside tissue (matching Ymx(Ywm>0) = Yos(Ywm>0))
    im_for_noise = np.where(tissue_mask, im_smoothed, 0.0).astype(np.float32)

    # Step 2: downsample to 1 mm using mean pooling (CAT12: reduceV 2mm 'meanm').
    im_ds, mask_ds = _downsample_volume(im_for_noise, tissue_mask, vx_vol, target_res=2)

    if np.sum(mask_ds) < 20:
        return float("nan")

    # Step 3: masked local SD at 2 mm (cat_vol_localstat(Ym, YM, 1, 4))
    local_sd = _masked_local_sd_exact(im_ds, mask_ds)

    # Step 4: median within tissue / absolute contrast
    sd_vals = local_sd[mask_ds]
    sd_vals = sd_vals[sd_vals > 0]
    if len(sd_vals) == 0:
        return float("nan")
    noise = float(np.median(sd_vals))
    return noise / contrast


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
    # Avoid over-optimisation (CAT12: contrastr + min(0, 1/3 - c) * 1.1)
    contrast_rel = contrast_rel + min(0.0, 1.0 / 3.0 - contrast_rel) * 1.1

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

    # CAT12 formula (cat_vol_qa201901x, estimateECR0old):
    #   res_ECR = max(0, 1/4 - ecr0)
    # High boundary gradient (sharp GM/WM edge) → ecr0 ≈ 0.25 → res_ECR ≈ 0
    # Low gradient (blurry edge) → ecr0 ≈ 0 → res_ECR ≈ 0.25
    return float(max(0.0, 0.25 - ecr0))


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
        intensity: Original (uncorrected) intensity image (same shape as *p0*).
            CAT12 reads the raw file and normalises to WM≈1 via
            ``cat_vol_approx``; any consistent scaling is acceptable here
            since all QA measures use ratios.
        vx_vol: Voxel dimensions in mm of the data arrays (length-3).
        vx_vol_orig: Original acquisition voxel dimensions in mm.  Used
            only for ``res_RMS``.  Falls back to *vx_vol* if not given.

    Returns:
        Dictionary with the following structure::

            {
                "qualitymeasures": {
                    "NCR":      {"value": …, "mark": …, "desc": "…"},
                    "CNR":      {"value": …, "mark": …, "desc": "…"},
                    "contrastr": {"value": …, "mark": …, "desc": "…"},
                    "res_RMS":  {"value": …, "mark": …, "desc": "…"},
                    "res_ECR":  {"value": …, "mark": …, "desc": "…"},
                    "IQR":      {"value": …, "grade": "…", "desc": "…"},
                    "SIQR":     {"value": …, "grade": "…", "desc": "…"},
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

    # Erode masks for WM and CSF
    wm_mask = binary_erosion(wm_mask, generate_binary_structure(3, 3), 2)
    csf_mask = binary_erosion(csf_mask, generate_binary_structure(3, 3), 1)

    # Contrast
    contrast_abs, contrastr = estimate_contrast(
        intensity, csf_mask, gm_mask, wm_mask
    )

    # Noise-to-Contrast Ratio in WM and CSF
    ncr_wm = estimate_noise(intensity, wm_mask, vx, contrast_abs)
    ncr_csf = estimate_noise(intensity, csf_mask, vx, contrast_abs)
        
    # Finally use the smaller values since noise can be overestimated sometime in
    # WM due to WMHs
    ncr = min(ncr_wm, ncr_csf)

    # Resolution (uses original acquisition voxel dims)
    res_rms = estimate_res_rms(vx_orig)

    # Effective contrast resolution
    res_ecr = estimate_res_ecr(intensity, p0, vx)

    # Individual marks
    marks: dict[str, float] = {}
    for name, value in [
        ("NCR", ncr),
        ("CNR", ncr),
        ("contrastr", contrastr),
        ("res_RMS", res_rms),
        ("res_ECR", res_ecr),
    ]:
        best, worst = _RATING_BOUNDS[name]
        marks[name] = _mark(value, best, worst)

    # IQR — CAT12: power mean of NCR + res_RMS with power 8
    iqr_value = _iqr([marks["NCR"], marks["res_RMS"]], power=8)
    iqr_grade = mark_to_grade(iqr_value)

    # SIQR — CAT12: power mean of NCR + res_RMS + res_ECR with power 4
    siqr_value = _iqr(
        [marks["NCR"], marks["res_RMS"], marks["res_ECR"]], power=4
    )
    siqr_grade = mark_to_grade(siqr_value)

    # Descriptions
    descs = {
        "NCR": "Noise-to-Contrast Ratio (lower is better)",
        "CNR": "Contrast-to-Noise Ratio (inverse of NCR, higher is better)",
        "contrastr": "Tissue contrast ratio (closer to 0.33 is better)",
        "res_RMS": "RMS voxel dimension in mm (lower is better)",
        "res_ECR": "Effective Contrast Resolution (lower is better)",
        "IQR": "Image Quality Rating: NCR+res_RMS, power 8 (1=excellent, 6=poor)",
        "SIQR": "Structural IQR: NCR+res_RMS+res_ECR, power 4 (no FEC)",
    }

    result: dict = {}
    for name, value in [
        ("NCR", ncr),
        ("CNR", 1/ncr),
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

    result["SIQR"] = {
        "value": round(siqr_value, 2) if np.isfinite(siqr_value) else None,
        "grade": siqr_grade,
        "desc": descs["SIQR"],
    }

    return {"qualitymeasures": result}
