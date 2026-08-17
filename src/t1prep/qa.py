"""Image quality assurance measures for T1-weighted MRI.

This module provides functions to estimate image quality measures following
the CAT12 ``cat_vol_qa201901x`` / ``cat_stat_marks`` framework.  The
measures quantify noise, spatial intensity inhomogeneity, tissue contrast,
and effective resolution, and are combined into an overall *Image Quality
Rating* (IQR).

The implementation mirrors the CAT12 processing chain step by step, because
the individual measures are only comparable across sites when the
pre-processing (bias correction, tissue masking, smoothing and spatial
down-sampling) is identical:

1.  Resolution standardisation to 1 mm for high-resolution input.
2.  Bias-field removal by approximating the WM intensity across the volume
    (``cat_vol_approx``) so that the noise estimate is not contaminated by
    intensity inhomogeneity.
3.  PVE-free tissue masks obtained by distance erosion, plus a "deep CSF"
    restriction for the ventricular CSF mask.
4.  Tissue-wise Gaussian smoothing with the CAT12 kernel (FWHM in voxels),
    block-mean reduction to 2.3 mm, and a local standard deviation in the
    6-neighbourhood (``cat_vol_localstat``).

Typical entry point
-------------------
Call :func:`estimate_qa` with a segmentation label map, the *original*
(uncorrected) intensity image, and voxel dimensions.  It returns a
dictionary of quality measures and a 1--6 school-mark style rating.

References
----------
- Gaser C, Dahnke R et al., *CAT – A Computational Anatomy Toolbox for
  the Analysis of Structural MRI Data*, 2024.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    correlate1d,
    distance_transform_edt,
    gaussian_filter,
    label as cc_label,
    zoom,
)


# Conversion between the FWHM used by ``spm_smooth`` (in voxels when it is
# called on a plain array) and the sigma expected by ``gaussian_filter``.
_FWHM2SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))

# 3x3x3 box used by ``cat_vol_morph(...,'d'|'e')`` (dtype 'c' -> convn box).
_BOX3 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# Rating scale (linear mapping matching CAT12 cat_stat_marks defaults)
# ---------------------------------------------------------------------------
# Each entry maps a measure name to (best_value, worst_value).
# A value equal to *best* → mark 1 (excellent), equal to *worst* → mark 6.
#
# NCR and ICR were calibrated on the BrainWeb Phantom (BWP, 75 volumes with
# simulated noise 1-9 % and inhomogeneity 20-100 %) by a robust fit of the
# measure against the simulated degradation level, evaluated at level 1 and
# level 6 — the procedure CAT12 uses in ``calc_limits_QA.m``.  ``res_ECR`` is
# anchored on the *effective* resolution instead, so that a scan whose
# structural detail corresponds to a voxel size of 0.5 / 3.0 mm receives the
# same mark as ``res_RMS`` does for those voxel sizes.
# Re-derive all bounds with ``scripts/qa_calibrate.py``.
_RATING_BOUNDS: dict[str, tuple[float, float]] = {
    "NCR":       (0.0232, 0.1585),   # BWP noise level 1 .. 6
    "CNR":       (0.0232, 0.1585),   # rated via NCR
    "ICR":       (0.3397, 0.9579),   # BWP bias level 1 .. 6
    "contrastr": (1.0 / 3.0, 0.0),   # cat_stat_marks default: CM=[1/3 0]
    "res_RMS":   (0.50, 3.00),       # cat_stat_marks default
    "res_ECR":   (0.0081, 0.0481),   # effective resolution of 0.5 .. 3.0 mm
    "EC_abs":    (21.1904, 128.6649),
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
    if not np.isfinite(value):
        return float("nan")
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
    if not np.isfinite(mark):
        return "NA"
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
# Low-level helpers mirroring the CAT12 building blocks
# ---------------------------------------------------------------------------

def _smooth_fwhm(volume: np.ndarray, fwhm_vox) -> np.ndarray:
    """Gaussian smoothing with a FWHM given in *voxels*.

    ``spm_smooth(Y, Y, s)`` interprets ``s`` as FWHM in voxels when it is
    applied to a plain array (``VOX = [1 1 1]``).  CAT12 relies on that
    behaviour throughout ``cat_vol_qa201901x``, so the conversion to the
    sigma expected by :func:`scipy.ndimage.gaussian_filter` has to be done
    explicitly — using the FWHM as a sigma would over-smooth by a factor of
    2.35 and largely destroy the very noise the measures try to quantify.

    Args:
        volume: Input image.
        fwhm_vox: Scalar or length-3 FWHM in voxels.

    Returns:
        Smoothed copy of *volume*.
    """
    sigma = np.atleast_1d(np.asarray(fwhm_vox, dtype=np.float64)) * _FWHM2SIGMA
    if sigma.size == 1:
        sigma = np.repeat(sigma, 3)
    sigma = np.maximum(sigma, 0.0)
    if np.all(sigma < 1e-3):
        return volume.astype(np.float32, copy=True)
    return gaussian_filter(volume.astype(np.float32), sigma=tuple(sigma))


def _smooth_sigma_fast(volume: np.ndarray, sigma_vox: float) -> np.ndarray:
    """Heavy Gaussian smoothing (sigma in voxels) via a coarse grid.

    Mirrors ``cat_vol_smooth3X`` for large filter widths, which recursively
    halves the resolution instead of convolving at full resolution.

    Args:
        volume: Input image.
        sigma_vox: Gaussian sigma in voxels.

    Returns:
        Smoothed image with the shape of *volume*.
    """
    vol = volume.astype(np.float32)
    if sigma_vox <= 1.0:
        return gaussian_filter(vol, sigma=sigma_vox)
    factor = max(1, int(2 ** np.floor(np.log2(sigma_vox / 2.0))))
    if factor == 1:
        return gaussian_filter(vol, sigma=sigma_vox)
    small = zoom(vol, 1.0 / factor, order=1)
    small = gaussian_filter(small, sigma=sigma_vox / factor)
    out = zoom(small, np.array(vol.shape) / np.array(small.shape), order=1)
    return _match_shape(out, vol.shape)


def _match_shape(volume: np.ndarray, shape) -> np.ndarray:
    """Pad/crop *volume* (edge replication) so that it has *shape*."""
    if volume.shape == tuple(shape):
        return volume
    out = volume
    for axis, target in enumerate(shape):
        cur = out.shape[axis]
        if cur > target:
            out = np.take(out, np.arange(target), axis=axis)
        elif cur < target:
            pad = [(0, 0)] * out.ndim
            pad[axis] = (0, target - cur)
            out = np.pad(out, pad, mode="edge")
    return out


def _smooth3x_small(volume: np.ndarray, weight: float) -> np.ndarray:
    """``cat_vol_smooth3X`` for ``0 < s < 0.5`` (blend with a 3^3 kernel)."""
    kernel_1d = np.exp(-0.5 * (np.arange(-1, 2) / 0.5) ** 2)
    kernel_1d /= kernel_1d.sum()
    smoothed = volume.astype(np.float32)
    for axis in range(3):
        smoothed = correlate1d(smoothed, kernel_1d, axis=axis, mode="nearest")
    return (smoothed * weight + volume * (1.0 - weight)).astype(np.float32)


def _disterode(mask: np.ndarray, dist_mm: float, vx_vol) -> np.ndarray:
    """``cat_vol_morph(mask, 'de', dist_mm, vx_vol)``.

    Distance erosion: keep only voxels whose Euclidean distance to the
    mask boundary exceeds *dist_mm*.

    Args:
        mask: Binary input mask.
        dist_mm: Erosion distance in mm.
        vx_vol: Voxel dimensions in mm.

    Returns:
        Eroded binary mask.
    """
    if not mask.any():
        return mask
    return distance_transform_edt(mask, sampling=np.asarray(vx_vol, float)) > dist_mm


def _cdilate(mask: np.ndarray, dist_mm: float, vx_vol) -> np.ndarray:
    """``cat_vol_morph(mask, 'd', dist_mm, vx_vol)`` (box structuring element)."""
    radius = np.maximum(1, np.round(dist_mm / np.asarray(vx_vol, float)).astype(int))
    struct = np.ones(2 * radius + 1, dtype=bool)
    return binary_dilation(mask, structure=struct)


def _reduce_factor(vx_vol, target_res: float, min_size: int = 32,
                   shape=None) -> np.ndarray:
    """Integer down-sampling factor used by ``cat_vol_resize(...,'reduceV')``."""
    vx = np.asarray(vx_vol, dtype=np.float64)
    target = np.maximum(np.full(3, float(target_res)), vx)
    step = np.floor(target / vx).astype(int)
    step = np.maximum(step, 1)
    if shape is not None:
        shape = np.asarray(shape, dtype=int)
        reduced = np.floor(shape / step).astype(int)
        step = np.floor(shape / np.maximum(reduced, min_size)).astype(int)
        step = np.maximum(step, 1)
    return step


def _reduce_meanm(volume: np.ndarray, step) -> np.ndarray:
    """Block mean over non-zero voxels (``cat_vol_resize(...,'meanm')``).

    Voxels of the reduced grid that receive fewer than ``max(2, mean(step)/2)``
    non-zero inputs are set to zero, exactly as CAT12 does, so that boundary
    cells with mostly background do not bias the result.

    Args:
        volume: Input image (zeros are treated as "no data").
        step: Length-3 integer reduction factor.

    Returns:
        Reduced image.
    """
    step = np.asarray(step, dtype=int)
    if np.all(step == 1):
        return volume.astype(np.float32, copy=True)

    vol = volume.astype(np.float32)
    # CAT12 replicates the last slice for odd dimensions before reducing.
    pad = [(0, 1) if (vol.shape[a] % 2 == 1 and step[a] > 1) else (0, 0)
           for a in range(3)]
    if any(p[1] for p in pad):
        vol = np.pad(vol, pad, mode="edge")

    new_shape = np.floor(np.array(vol.shape) / step).astype(int)
    crop = new_shape * step
    vol = vol[:crop[0], :crop[1], :crop[2]]
    blocks = vol.reshape(
        new_shape[0], step[0], new_shape[1], step[1], new_shape[2], step[2]
    ).transpose(0, 2, 4, 1, 3, 5).reshape(*new_shape, -1)

    valid = (blocks != 0) & np.isfinite(blocks)
    count = valid.sum(axis=-1)
    total = np.where(valid, blocks, 0.0).sum(axis=-1)

    min_count = max(2.0, float(np.mean(step)) / 2.0)
    out = np.zeros(tuple(new_shape), dtype=np.float32)
    keep = count >= min_count
    out[keep] = total[keep] / count[keep]
    return out


# Offsets of the 7 voxels with Euclidean distance <= 1 used by
# ``cat_vol_localstat(..., nb=1, ...)``.  Note that this is *not* the full
# 3x3x3 block: CAT_Vol.c restricts the box to ``sqrt(i^2+j^2+k^2) <= nb``.
_NB1_OFFSETS = (
    (0, 0, 0),
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)


def _shift(volume: np.ndarray, offset) -> np.ndarray:
    """Shift *volume* by *offset* voxels, filling the border with zeros."""
    if offset == (0, 0, 0):
        return volume
    out = np.zeros_like(volume)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for axis, off in enumerate(offset):
        if off > 0:
            src[axis] = slice(off, None)
            dst[axis] = slice(None, -off)
        elif off < 0:
            src[axis] = slice(None, off)
            dst[axis] = slice(-off, None)
    out[tuple(dst)] = volume[tuple(src)]
    return out


def _localstat_sd(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``cat_vol_localstat(volume, mask, 1, 4)`` — masked local sample SD.

    Uses the 7-voxel neighbourhood (Euclidean distance <= 1) and the sample
    standard deviation (``ddof=1``) of ``get_std_double`` in CAT_Math.c.

    Args:
        volume: Input image.
        mask: Binary mask; only these voxels contribute and are evaluated.

    Returns:
        Per-voxel SD, zero outside *mask* and where fewer than two
        neighbours are available.
    """
    val = np.where(mask, volume.astype(np.float64), 0.0)
    wgt = mask.astype(np.float64)

    count = np.zeros_like(val)
    total = np.zeros_like(val)
    total_sq = np.zeros_like(val)
    for offset in _NB1_OFFSETS:
        shifted_w = _shift(wgt, offset)
        shifted_v = _shift(val, offset)
        count += shifted_w
        total += shifted_v
        total_sq += shifted_v * shifted_v

    out = np.zeros_like(val)
    ok = mask & (count > 1.5)
    n = count[ok]
    var = (total_sq[ok] - total[ok] ** 2 / n) / (n - 1.0)
    out[ok] = np.sqrt(np.maximum(var, 0.0))
    return out


def _localstat_max(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``cat_vol_localstat(volume, mask, 1, 3)`` — masked local maximum."""
    val = np.where(mask, volume.astype(np.float32), -np.inf)
    out = volume.astype(np.float32).copy()
    for offset in _NB1_OFFSETS[1:]:
        shifted = _shift_fill(val, offset, -np.inf)
        out = np.maximum(out, shifted)
    out[~mask] = 0.0
    return out


def _shift_fill(volume: np.ndarray, offset, fill: float) -> np.ndarray:
    """Shift *volume*, filling the border with *fill*."""
    out = np.full_like(volume, fill)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for axis, off in enumerate(offset):
        if off > 0:
            src[axis] = slice(off, None)
            dst[axis] = slice(None, -off)
        elif off < 0:
            src[axis] = slice(None, off)
            dst[axis] = slice(-off, None)
    out[tuple(dst)] = volume[tuple(src)]
    return out


def _median3(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``cat_vol_median3(volume, mask, mask)`` — masked 3^3 median filter.

    Only voxels inside *mask* are filtered; neighbours outside the mask are
    replaced by the value of the centre voxel, exactly as CAT_Vol.c does
    (``ni = ind``).  The filter is evaluated on the mask voxels only, which
    keeps the memory footprint proportional to the mask size.

    Args:
        volume: Input image.
        mask: Voxels to filter (and to use as neighbours).

    Returns:
        Filtered copy of *volume* (unchanged outside *mask*).
    """
    out = volume.astype(np.float32, copy=True)
    inner = mask.copy()
    inner[[0, -1], :, :] = False
    inner[:, [0, -1], :] = False
    inner[:, :, [0, -1]] = False
    index = np.flatnonzero(inner)
    if index.size == 0:
        return out

    flat_vol = volume.astype(np.float32).ravel()
    flat_mask = mask.ravel()
    stride = np.array(
        [volume.shape[1] * volume.shape[2], volume.shape[2], 1], dtype=np.int64
    )
    offsets = np.array(
        [i * stride[0] + j * stride[1] + k * stride[2]
         for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=np.int64,
    )

    result = np.empty(index.size, dtype=np.float32)
    chunk = max(1, int(4e6 // offsets.size))
    for start in range(0, index.size, chunk):
        sel = index[start:start + chunk]
        neigh = sel[:, None] + offsets[None, :]
        vals = flat_vol[neigh]
        centre = flat_vol[sel][:, None]
        vals = np.where(flat_mask[neigh], vals, centre)
        result[start:start + sel.size] = np.median(vals, axis=1)

    out.ravel()[index] = result
    return out


def _grad_abs(volume: np.ndarray, vx_vol) -> np.ndarray:
    """``cat_vol_grad`` with the default method (absolute sum of gradients).

    Uses the central differences of ``cat_vol_gradient3`` (divided by two,
    replicating the value at the volume border).

    Args:
        volume: Input image.
        vx_vol: Length-3 divisor per axis (CAT12 passes powers of the voxel
            size here, e.g. ``vx_vol**0.5``).

    Returns:
        Gradient magnitude image (absolute sum over the three axes).
    """
    vx = np.asarray(vx_vol, dtype=np.float64)
    vol = volume.astype(np.float32)
    grad = np.zeros_like(vol)
    for axis in range(3):
        fwd = _shift_edge(vol, axis, 1)
        bwd = _shift_edge(vol, axis, -1)
        grad += np.abs((fwd - bwd) / 2.0 / vx[axis])
    return grad


def _shift_edge(volume: np.ndarray, axis: int, offset: int) -> np.ndarray:
    """Shift along *axis* replicating the border value (as cat_vol_gradient3)."""
    out = np.roll(volume, -offset, axis=axis)
    idx = [slice(None)] * 3
    if offset > 0:
        idx[axis] = slice(-offset, None)
    else:
        idx[axis] = slice(None, -offset)
    ref = [slice(None)] * 3
    ref[axis] = slice(-offset, None) if offset > 0 else slice(None, -offset)
    out[tuple(idx)] = volume[tuple(ref)]
    return out


def _laplace_fill(volume: np.ndarray, unknown: np.ndarray,
                  iterations: int = 60) -> np.ndarray:
    """Dirichlet Laplace relaxation (``cat_vol_laplace3R``).

    Known voxels stay fixed while *unknown* voxels are iteratively replaced
    by the mean of their 6 neighbours.

    Args:
        volume: Image with valid values at the known positions.
        unknown: Voxels to be relaxed.
        iterations: Number of Jacobi sweeps.

    Returns:
        Relaxed image.
    """
    out = volume.astype(np.float32).copy()
    if not unknown.any():
        return out
    for _ in range(iterations):
        acc = np.zeros_like(out)
        for offset in _NB1_OFFSETS[1:]:
            acc += _shift_edge_offset(out, offset)
        out[unknown] = (acc[unknown] / 6.0)
    return out


def _shift_edge_offset(volume: np.ndarray, offset) -> np.ndarray:
    """Shift by *offset* with edge replication (Neumann boundary)."""
    out = volume
    for axis, off in enumerate(offset):
        if off != 0:
            out = _shift_edge(out, axis, off)
    return out


def _approx(volume: np.ndarray, vx_vol, res: float = 4.0) -> np.ndarray:
    """Smooth approximation of missing (zero) values — ``cat_vol_approx``.

    Reproduces the structure of the classic CAT12 ``'nn'`` method: reduce to
    ``res`` mm using a block mean over the defined voxels, fill the
    undefined voxels with their nearest neighbour, relax with a Laplace
    filter (known values fixed) and resample back.

    Args:
        volume: Image with zeros marking undefined voxels.
        vx_vol: Voxel dimensions in mm.
        res: Working resolution in mm (default 4).

    Returns:
        Fully defined smooth image with the shape of *volume*.
    """
    vol = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(vol != 0):
        return np.zeros_like(vol)

    step = _reduce_factor(vx_vol, res, min_size=16, shape=vol.shape)
    small = _reduce_meanm(vol, step)
    known = small != 0
    if not known.any():
        return np.zeros_like(vol)

    # nearest-neighbour fill (cat_vbdist)
    _, idx = distance_transform_edt(~known, return_indices=True)
    filled = small[tuple(idx)]

    # Laplace relaxation with Dirichlet boundary at the known voxels.
    filled = _laplace_fill(filled, ~known)
    filled = gaussian_filter(filled, sigma=1.0)

    factor = np.array(vol.shape, dtype=np.float64) / np.array(filled.shape)
    out = zoom(filled, factor, order=1)
    return _match_shape(out, vol.shape).astype(np.float32)


def _gintnorm(volume: np.ndarray, t1th) -> np.ndarray:
    """``cat_main_gintnorm`` with ``T3thx = [0 1 2 3 6]`` (divided by 3).

    Maps background → 0, CSF → 1/3, GM → 2/3, WM → 1 by piecewise linear
    interpolation, so that the result is directly comparable across scans.

    Args:
        volume: Bias-corrected intensity image.
        t1th: Median intensities of [CSF, GM, WM].

    Returns:
        Intensity-normalised image (WM ≈ 1).
    """
    t1th = np.asarray(t1th, dtype=np.float64)
    knots_x = np.array([0.0, t1th[0], t1th[1], t1th[2], 2.0 * t1th[2]])
    knots_y = np.array([0.0, 1.0, 2.0, 3.0, 6.0]) / 3.0
    order = np.argsort(knots_x)
    knots_x, knots_y = knots_x[order], knots_y[order]
    # Strictly increasing knots are required for a stable interpolation.
    for i in range(1, len(knots_x)):
        knots_x[i] = max(knots_x[i], knots_x[i - 1] + 1e-6)

    out = np.interp(volume.astype(np.float64), knots_x, knots_y)
    # Linear extrapolation beyond the last knot (CAT12 folds these values
    # back, which is not monotone and irrelevant for the measures here).
    high = volume > knots_x[-1]
    if high.any():
        slope = (knots_y[-1] - knots_y[-2]) / (knots_x[-1] - knots_x[-2])
        out[high] = knots_y[-1] + (volume[high] - knots_x[-1]) * slope
    return out.astype(np.float32)


def _largest_components(mask: np.ndarray, rel_size: float = 0.5) -> np.ndarray:
    """Keep connected components of at least *rel_size* of the largest one."""
    if not mask.any():
        return mask
    labels, count = cc_label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.flatnonzero(sizes >= rel_size * sizes.max())
    return np.isin(labels, keep)


# ---------------------------------------------------------------------------
# Individual quality measures
# ---------------------------------------------------------------------------

def estimate_res_rms(vx_vol) -> float:
    """Compute the RMS voxel dimension as a resolution indicator.

    Args:
        vx_vol: Voxel dimensions in mm (length-3 array).

    Returns:
        RMS resolution in mm (lower is better).
    """
    return float(np.sqrt(np.mean(np.asarray(vx_vol, dtype=np.float64) ** 2)))


def _estimate_ecr0(intensity: np.ndarray, p0: np.ndarray, vx_vol,
                   recurse: bool = True) -> float:
    """Raw edge contrast ratio — ``estimateECR0old`` of ``cat_vol_qa201901x``.

    Quantifies the anatomical detail that is actually present in the image by
    the normalised edge strength at the GM/WM boundary, corrected for the
    "edge" strength caused by noise inside the WM.

    Args:
        intensity: Intensity-normalised image (WM ≈ 1).
        p0: Segmentation label map (0=BG, 1=CSF, 2=GM, 3=WM).
        vx_vol: Voxel dimensions in mm.
        recurse: Re-evaluate with an eroded and a dilated WM boundary and
            return the maximum, which makes the measure robust against
            systematic segmentation offsets.

    Returns:
        Raw ECR value (higher = sharper boundary).
    """
    vx = np.asarray(vx_vol, dtype=np.float64)
    p0s = _smooth3x_small(p0, 0.2)
    wm = p0s > 2.5
    if np.count_nonzero(wm) < 50:
        return float("nan")

    noise = float(np.std(intensity[wm]))
    p0s = np.where(wm, 3.0, p0s)

    ims = np.maximum(2.0 / 3.0, intensity)
    ims = _smooth_fwhm(ims, np.maximum(0.0, noise / vx ** 2))

    p0r = np.round(p0s)
    wmr = p0r > 2.5
    edge = binary_dilation(wmr, _BOX3) & ~binary_erosion(wmr, _BOX3)
    if np.count_nonzero(edge) < 50:
        return float("nan")

    bad = _cdilate(
        (np.abs(p0s / 3.0 - ims) > 0.5) | ~np.isfinite(intensity) | (p0s == 0),
        1.0, vx,
    )
    sel = edge & ~bad
    if np.count_nonzero(sel) < 50:
        sel = edge

    grad = _grad_abs(np.clip(ims, 2.0 / 3.0, 1.0), vx ** 0.5)
    grad = _localstat_max(grad, edge)
    grad = _approx(grad, vx)

    grad_wm = _grad_abs(np.maximum(1.0, ims), vx ** 0.75)
    grad_wm = _localstat_max(grad_wm, wm)
    grad_wm = _approx(grad_wm, vx)

    ecr = float(np.median(grad[sel]) - np.median(grad_wm[sel]))

    if recurse:
        base = np.minimum(2.0, p0r)
        candidates = [ecr]
        for morphed in (binary_erosion(wmr, _BOX3), binary_dilation(wmr, _BOX3)):
            val = _estimate_ecr0(intensity, base + morphed, vx, recurse=False)
            if np.isfinite(val):
                candidates.append(val)
        ecr = float(max(candidates))
    return ecr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_qa(
    p0: np.ndarray,
    intensity: np.ndarray,
    vx_vol,
    vx_vol_orig=None,
) -> dict:
    """Estimate all image quality measures.

    Args:
        p0: Segmentation label map (0=BG, 1=CSF, 2=GM, 3=WM).  Can be a
            continuous map (values in [0, 3]) — the partial volume
            information is used for the PVE-free tissue masks.
        intensity: Original (uncorrected) intensity image (same shape as
            *p0*).  The bias field is removed internally, mirroring
            ``cat_vol_qa201901x``, so the raw image must be passed here.
        vx_vol: Voxel dimensions in mm of the data arrays (length-3).
        vx_vol_orig: Original acquisition voxel dimensions in mm.  Used
            only for ``res_RMS``.  Falls back to *vx_vol* if not given.

    Returns:
        Dictionary with the following structure::

            {
                "qualitymeasures": {
                    "NCR":       {"value": …, "mark": …, "desc": "…"},
                    "CNR":       {"value": …, "mark": …, "desc": "…"},
                    "ICR":       {"value": …, "mark": …, "desc": "…"},
                    "contrastr": {"value": …, "mark": …, "desc": "…"},
                    "res_RMS":   {"value": …, "mark": …, "desc": "…"},
                    "res_ECR":   {"value": …, "mark": …, "desc": "…"},
                    "IQR":       {"value": …, "grade": "…", "desc": "…"},
                    "SIQR":      {"value": …, "grade": "…", "desc": "…"},
                },
            }
    """
    vx = np.asarray(vx_vol, dtype=np.float64).copy()
    vx_orig = (
        np.asarray(vx_vol_orig, dtype=np.float64)
        if vx_vol_orig is not None
        else vx.copy()
    )

    seg = np.asarray(p0, dtype=np.float32).copy()
    org = np.asarray(intensity, dtype=np.float32).copy()

    # ---------------------------------------------------------------
    # 1. Resolution standardisation to 1 mm (CAT12 RD202411)
    # ---------------------------------------------------------------
    if np.any(vx < 0.8):
        fwhm = np.clip((1.0 - vx) ** 2, 0.2, 2.0)
        seg = _smooth_fwhm(seg, fwhm)
        org = _smooth_fwhm(org, fwhm)
        seg = zoom(seg, vx, order=1)
        org = zoom(org, vx, order=1)
        vx = np.ones(3)

    # ---------------------------------------------------------------
    # 2. Background / lesion handling and cropping to the brain
    # ---------------------------------------------------------------
    # CAT12 also drops opened zero-intensity blobs so that masked/defaced
    # regions inside the segmentation do not enter the measures.
    background = (
        (seg <= 0)
        | ~np.isfinite(org)
        | binary_dilation(binary_erosion(org == 0, _BOX3), _BOX3)
    )
    org = np.nan_to_num(org, nan=0.0, posinf=0.0, neginf=0.0)
    org[background] = 0.0
    seg[background] = 0.0

    brain = seg > 1.5
    if np.count_nonzero(brain) < 1000:
        return {"qualitymeasures": _empty_result()}

    bbox = []
    for axis in range(3):
        proj = np.any(brain, axis=tuple(i for i in range(3) if i != axis))
        idx = np.flatnonzero(proj)
        lo = max(0, int(idx[0]) - 4)
        hi = min(brain.shape[axis], int(idx[-1]) + 5)
        bbox.append(slice(lo, hi))
    bbox = tuple(bbox)
    seg, org = seg[bbox], org[bbox]

    # ---------------------------------------------------------------
    # 3. Refine the segmentation (drop detached fragments)
    # ---------------------------------------------------------------
    step2 = _reduce_factor(vx, 2.0, shape=seg.shape)
    seg_small = _reduce_meanm(seg, step2)
    core = binary_erosion(seg_small > 0.9, _BOX3)
    core = _largest_components(core, 0.5)
    core = binary_dilation(core, _BOX3)
    core_full = _match_shape(
        zoom(core.astype(np.float32),
             np.array(seg.shape) / np.array(core.shape), order=1),
        seg.shape,
    )
    seg = seg * (core_full > 0.5)

    # ---------------------------------------------------------------
    # 4. Bias correction (cat_vol_qa201901x lines 293-298)
    #    Without this the noise estimate is dominated by the intensity
    #    inhomogeneity rather than by the noise.
    # ---------------------------------------------------------------
    wm_seed = (seg > 2.95) | binary_erosion(seg > 2.25, _BOX3)
    if np.count_nonzero(wm_seed) < 100:
        return {"qualitymeasures": _empty_result()}

    offset = float(np.min(org))
    field = _approx(np.where(wm_seed, org - offset, 0.0), vx) + offset
    wm_ref = float(np.median(org[wm_seed]))
    if not np.isfinite(wm_ref) or abs(wm_ref) < 1e-9:
        return {"qualitymeasures": _empty_result()}
    field = field / wm_ref
    img = org / np.maximum(field, np.finfo(np.float32).eps)
    img[seg <= 0] = 0.0

    # ---------------------------------------------------------------
    # 5. Rough tissue medians and global noise level
    # ---------------------------------------------------------------
    t1th = np.array([
        float(np.median(img[np.abs(seg - c) < 0.1])) if np.count_nonzero(
            np.abs(seg - c) < 0.1) > 50 else np.nan
        for c in (1.0, 2.0, 3.0)
    ])
    if not np.all(np.isfinite(t1th)):
        return {"qualitymeasures": _empty_result()}
    tissue_step = float(np.min(np.abs(np.diff(t1th))))
    if tissue_step < 1e-9:
        return {"qualitymeasures": _empty_result()}

    noise = float(np.clip(np.std(img[seg > 2.9]) / tissue_step, 0.0, 1.0)) / 3.0
    img_s = _smooth_fwhm(img, noise * 4.0)

    # ---------------------------------------------------------------
    # 6. PVE-free tissue masks
    # ---------------------------------------------------------------
    tiv_ml = float(np.count_nonzero(seg > 0) * np.prod(vx) / 1000.0)
    radius = (tiv_ml / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0) / float(np.mean(vx))
    deep = 1.0 - _smooth_sigma_fast(
        ((seg < 1.0) | (org == 0)).astype(np.float32),
        float(np.clip(radius * 2.0, 16.0, 24.0)),
    )

    csf_mask = ((seg > 0.75) & (seg < 1.25)
                & _disterode((seg > 0.25) & (seg < 1.75), 1.0, vx)
                & (deep > 0.75))
    gm_mask = ((seg > 1.75) & (seg < 2.25)
               & _disterode((seg > 1.25) & (seg < 2.75), 1.0, vx))
    wm_mask = ((seg > 2.75) & (seg < 3.25)
               & _disterode((seg > 2.25) & (seg < 3.75), 1.0, vx))

    # Drop WM voxels that deviate from their local median (PVE, vessels, PVS).
    if wm_mask.any():
        wm_med = _median3(img_s, wm_mask)
        wm_mask = wm_mask & ~((wm_med - img_s) > noise * t1th[2] * 2.0)

    if np.count_nonzero(wm_mask) < 100 or np.count_nonzero(gm_mask) < 100:
        return {"qualitymeasures": _empty_result()}

    # ---------------------------------------------------------------
    # 7. Effective (structural) resolution
    # ---------------------------------------------------------------
    img_norm = _gintnorm(img_s, t1th)
    if np.all(vx < 1.5):
        vessels = (img_norm > 1.15) & (seg < 2) & ~wm_mask
        if vessels.any():
            _, idx = distance_transform_edt(vessels, return_indices=True)
            img_norm = np.minimum(img_norm, img_norm[tuple(idx)])
    ecr0 = _estimate_ecr0(img_norm, seg, vx)
    res_ecr = (float(max(0.0, 0.25 - (ecr0 + 0.25 * noise)))
               if np.isfinite(ecr0) else float("nan"))

    # ---------------------------------------------------------------
    # 8. Tissue-wise smoothing (CAT12 "Ymx")
    # ---------------------------------------------------------------
    fwhm_vox = 0.8 + 0.5 / vx
    img_mx = img.copy()
    for mask, level in ((wm_mask, t1th[2]), (gm_mask, t1th[1]),
                        (csf_mask, t1th[0])):
        if not mask.any():
            continue
        smoothed = _smooth_fwhm(np.where(mask, img_mx, level), fwhm_vox)
        img_mx[mask] = smoothed[mask]

    # ---------------------------------------------------------------
    # 9. Reduction to 2.3 mm and tissue thresholds
    # ---------------------------------------------------------------
    step = _reduce_factor(vx, 2.3, shape=seg.shape)
    vx_red = vx * step

    csf_r = _reduce_meanm(csf_mask.astype(np.float32), step)
    gm_r = _reduce_meanm(gm_mask.astype(np.float32), step)
    wm_r = _reduce_meanm(wm_mask.astype(np.float32), step)
    csf_i = _reduce_meanm(img_mx * csf_mask, step) * (csf_r >= 0.5)
    gm_i = _reduce_meanm(img_mx * gm_mask, step) * (gm_r >= 0.5)
    wm_i = _reduce_meanm(img_mx * wm_mask, step) * (wm_r >= 0.5)
    seg_r = _reduce_meanm(seg, step)
    field_r = _reduce_meanm(field, step)

    def _median_nonzero(vol):
        vals = vol[(vol != 0) & np.isfinite(vol)]
        return float(np.median(vals)) if vals.size else float("nan")

    wm_th = _median_nonzero(wm_i)
    gm_th = _median_nonzero(gm_i)
    csf_th = _median_nonzero(csf_i)
    if not np.isfinite(wm_th) or not np.isfinite(gm_th):
        return {"qualitymeasures": _empty_result()}
    if not np.isfinite(csf_th):
        csf_th = 0.0

    # ---------------------------------------------------------------
    # 10. Signal and contrast
    # ---------------------------------------------------------------
    signal = abs(max(wm_th, gm_th) - min(csf_th, 0.0))
    if signal < 1e-9:
        return {"qualitymeasures": _empty_result()}
    contrastr = min(abs(gm_th - csf_th), abs(wm_th - gm_th)) / signal
    # Avoid over-optimisation by unusually high contrasts.
    contrastr = contrastr + min(0.0, 1.0 / 3.0 - contrastr) * 1.1
    contrast_abs = contrastr * signal
    if contrast_abs < 1e-9:
        return {"qualitymeasures": _empty_result()}

    # ---------------------------------------------------------------
    # 11. Noise-to-contrast ratio (volume-weighted WM/CSF mixture)
    # ---------------------------------------------------------------
    voxel_ml = float(np.prod(vx_red))
    vol_wm = float(np.count_nonzero(wm_i > 0)) * voxel_ml
    vol_csf = float(np.count_nonzero(csf_i > 0)) * voxel_ml

    ncr_wm = _noise_level(wm_i, wm_i > 0) / contrast_abs

    csf_th_ml = 200.0
    if csf_th < gm_th and vol_csf > csf_th_ml:
        ncr_csf = _noise_level(csf_i, csf_i > 0) / contrast_abs
    else:
        ncr_csf = 0.0
        vol_csf = 0.0

    w_csf = min(csf_th_ml, max(0.0, vol_csf - csf_th_ml))
    w_wm = max(0.0, min(csf_th_ml, vol_wm) - w_csf)
    if w_csf < 3 * csf_th_ml and w_wm < 10 * csf_th_ml and ncr_csf > 0:
        # CSF noise can be overestimated (flow, PVE) — cap it by the WM value.
        ncr_csf = min(ncr_csf, ncr_wm)
    denom = w_wm + w_csf
    ncr = ((ncr_wm * w_wm + ncr_csf * w_csf) / denom
           if denom > 0 else ncr_wm)

    # ---------------------------------------------------------------
    # 12. Inhomogeneity-to-contrast ratio
    # ---------------------------------------------------------------
    field_vals = field_r[(seg_r > 0) & np.isfinite(field_r) & (field_r != 0)]
    icr = (float(np.std(field_vals)) / contrastr
           if field_vals.size > 10 and contrastr > 1e-9 else float("nan"))

    # ---------------------------------------------------------------
    # 13. Resolution and marks
    # ---------------------------------------------------------------
    res_rms = estimate_res_rms(vx_orig)

    values = {
        "NCR": ncr,
        "CNR": 1.0 / ncr if ncr > 1e-9 else float("nan"),
        "ICR": icr,
        "contrastr": contrastr,
        "res_RMS": res_rms,
        "res_ECR": res_ecr,
    }
    marks = {
        name: _mark(ncr if name == "CNR" else values[name],
                    *_RATING_BOUNDS[name])
        for name in ("NCR", "CNR", "ICR", "contrastr", "res_RMS", "res_ECR")
    }

    # IQR — CAT12: power mean of NCR + res_RMS with power 8
    iqr_value = _iqr([marks["NCR"], marks["res_RMS"]], power=8)
    # SIQR — CAT12: power mean of NCR + res_RMS + res_ECR with power 4
    siqr_value = _iqr(
        [marks["NCR"], marks["res_RMS"], marks["res_ECR"]], power=4
    )

    result: dict = {}
    for name in ("NCR", "CNR", "ICR", "contrastr", "res_RMS", "res_ECR"):
        value = values[name]
        mark = marks[name]
        result[name] = {
            "value": round(value, 4) if np.isfinite(value) else None,
            "mark": round(mark, 2) if np.isfinite(mark) else None,
            "desc": _DESCRIPTIONS[name],
        }

    result["IQR"] = {
        "value": round(iqr_value, 2) if np.isfinite(iqr_value) else None,
        "grade": mark_to_grade(iqr_value),
        "desc": _DESCRIPTIONS["IQR"],
    }
    result["SIQR"] = {
        "value": round(siqr_value, 2) if np.isfinite(siqr_value) else None,
        "grade": mark_to_grade(siqr_value),
        "desc": _DESCRIPTIONS["SIQR"],
    }
    return {"qualitymeasures": result}


def _noise_level(volume: np.ndarray, mask: np.ndarray) -> float:
    """Median local SD within *mask* (``estimateNoiseLevel``)."""
    if np.count_nonzero(mask) < 20:
        return float("nan")
    sd = _localstat_sd(volume, mask)
    vals = sd[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if vals.size else float("nan")


_DESCRIPTIONS = {
    "NCR": "Noise-to-Contrast Ratio (lower is better)",
    "CNR": "Contrast-to-Noise Ratio (inverse of NCR, higher is better)",
    "ICR": "Inhomogeneity-to-Contrast Ratio (lower is better)",
    "contrastr": "Tissue contrast ratio (closer to 0.33 is better)",
    "res_RMS": "RMS voxel dimension in mm (lower is better)",
    "res_ECR": "Effective Contrast Resolution (lower is better)",
    "IQR": "Image Quality Rating: NCR+res_RMS, power 8 (1=excellent, 6=poor)",
    "SIQR": "Structural IQR: NCR+res_RMS+res_ECR, power 4 (no FEC)",
    "EC_abs": "Absolute Euler number for both hemispheres",
}


def _empty_result() -> dict:
    """Return a result dictionary with all measures undefined."""
    result = {
        name: {"value": None, "mark": None, "desc": _DESCRIPTIONS[name]}
        for name in ("NCR", "CNR", "ICR", "contrastr", "res_RMS", "res_ECR")
    }
    for name in ("IQR", "SIQR"):
        result[name] = {"value": None, "grade": "NA",
                        "desc": _DESCRIPTIONS[name]}
    return result
