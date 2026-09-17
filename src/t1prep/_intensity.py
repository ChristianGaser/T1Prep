"""Intensity normalisation and bias-field correction.

The spline-based bias-field fits, the CAT12-style global intensity
normalisation and the Local Adaptive Segmentation (LAS) built on them.
"""

import math

import nibabel as nib
import numpy as np
import torch
from nxbc.filter import (
    Eu_v,
    distrib_kde,
    kernelfntri,
    map_Eu_v,
    symGaussFilt,
    wiener_filter_withpad,
)
from scipy.ndimage import binary_closing, generate_binary_structure, median_filter
from SplineSmooth3D.SplineSmooth3D import (
    SplineSmooth3D,
    SplineSmooth3DUnregularized,
)

from .utils import find_largest_cluster


def scale_intensity(x, low=.5, high=99.5):
    """Rescale ``x`` to the given percentile range, compressing the top tail.

    Boolean-mask indexing is deliberately avoided on the accelerator.  Both
    ``x[mask]`` and ``x[mask] = ...`` resolve the mask through ``nonzero``, so
    the mask is materialised once per occurrence; the write form additionally
    has to agree with the value tensor produced by the read form.  On MPS
    those resolutions have been observed to disagree, which aborts the run
    with "shape mismatch: value tensor of shape [N] cannot be broadcast to
    indexing result of shape [M]".  ``torch.where`` computes the same result
    in a single elementwise pass with no mask materialisation at all.
    """
    # Mask on the host: CPU nonzero is not subject to the disagreement above,
    # where a short read would silently skew the percentiles instead of
    # raising.
    x_host = x.cpu()
    x_nonzero = x_host[x_host > 0]
    low = np.percentile(x_nonzero, low)
    high = np.percentile(x_nonzero, high)
    x = (x - low) / (high - low)
    # clamp(min=1) only guards the discarded branch: where x > 1 it is the
    # identity, so the retained values are bit-for-bit what log10(x) gave
    # before, while values <= 1 no longer feed -inf/NaN into the unused side.
    return torch.where(x > 1, 1 + torch.log10(x.clamp(min=1)), x)


def piecewise_linear_scaling(input_img, label_img):
    """Piecewise linear scaling of an intensity image."""
    target_values = np.arange(0, 5)
    Ym = input_img.copy().astype(float)
    median_input = {}
    for k in [1, 2, 3]:
        mask = np.abs(label_img - k) < 0.01
        median_input[k] = np.median(input_img[mask])
    mask = (label_img == 0) & (input_img < 0.9 * median_input[1])
    median_input[0] = np.median(input_img[mask])
    median_input[4] = median_input[3] + (median_input[3] - median_input[2])
    for i in range(1, len(target_values)):
        mask = (input_img > median_input[i - 1]) & (input_img <= median_input[i])
        Ym[mask] = target_values[i - 1] + (input_img[mask] - median_input[i - 1]) / (
            median_input[i] - median_input[i - 1]
        ) * (target_values[i] - target_values[i - 1])
    mask = input_img >= median_input[4]
    slope = (target_values[4] - target_values[3]) / (median_input[4] - median_input[3])
    Ym[mask] = target_values[4] + (input_img[mask] - median_input[4]) * slope
    return Ym / 3


def correct_bias_field(brain, seg=None, steps=1000, spacing=1.0, get_discrepancy=False):
    """Apply bias field correction to a brain image."""
    subdivide = True
    bcl = True
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.2
    subsamp = 5
    stopthr = 5e-4

    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata().copy()

    if seg is not None:
        seg0 = seg.get_fdata().copy()
        max_seg = np.max(seg0)
        mask = seg0 >= (2.75 / 3.0 * max_seg)
    else:
        # Obtain gradient and its magnitude
        gx, gy, gz = np.gradient(brain0)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Mask out regions with high gradient (i.e. GM, sulci, vessels)
        mask = brain0 * ((grad_mag / brain0) < 0.1)

        # Remove low intensity areas that are rather GM
        thresh = np.quantile(mask[mask != 0], 0.3)
        mask0 = mask > thresh

        # Close remaining holes using morphol. operations and remove filled areas
        # from mask that are rather subcortical structures
        mask0 = ~mask0 & binary_closing(mask0, generate_binary_structure(3, 3), 10)
        mask[mask0] = 0

        # Remove thin structures by median filtering and finally create mask
        mask = median_filter(mask, size=2)
        mask = mask > 0

        mask = find_largest_cluster(mask)

    if subsamp:
        offset = 0
        dataSub = brain0[offset::subsamp, offset::subsamp, offset::subsamp]
        wm_mask = mask[offset::subsamp, offset::subsamp, offset::subsamp]
        dataSubVoxSize = dataVoxSize * subsamp
    else:
        dataSub = brain0
        wm_mask = mask
        
    dataSubVoxSize = 1 / (np.array(dataSub.shape) - 1)
    dataVoxSize = dataSubVoxSize / subsamp

    datalog = dataSub.astype(np.float32)
    
    datalog[wm_mask] = np.log(datalog[wm_mask])
    if seg is None and np.sum(np.size(datalog[wm_mask])) < 100:
        print("Warning: Stopped initial bias field correction since estimated WM mask is too small.")
        return brain
        
    datalog[np.logical_not(wm_mask) | ~np.isfinite(datalog)] = 0
    datalogmasked = datalog[wm_mask]
    fit_data = np.zeros_like(datalog)
    datalogmaskedcur = np.copy(datalogmasked)

    levels = [lvl for lvl in range(maxlevel) for _ in range(steps)]
    levelfwhm = (
        fwhm / (np.arange(maxlevel) + 1) if not subdivide else fwhm * np.ones(maxlevel)
    )

    splsm3d = SplineSmooth3DUnregularized(
        datalog, dataSubVoxSize, spacing, domainMethod="minc", mask=wm_mask
    )
    predictor = SplineSmooth3D(
        brain0, dataVoxSize, spacing, knts=splsm3d.kntsArr, dofit=False
    )
    datalogcur = np.copy(datalog)
    nextlevel = 0
    controlField = None
    chosenkernelfn = kernelfntri

    for N in range(len(levels)):
        if levels[N] < nextlevel:
            continue
        hist, histvaledge, histval, histbinwidth = distrib_kde(
            datalogmaskedcur, Nbins, kernfn=chosenkernelfn, binCentreLimits=bcl
        )
        thisFWHM = levelfwhm[levels[N]]
        thisSD = thisFWHM / math.sqrt(8 * math.log(2))
        mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)
        histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
        histfiltclip = np.clip(histfilt, 0, None)
        uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
        datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
        logbc = datalogmaskedcur - datalogmaskedupd
        logbc = logbc - np.mean(logbc)
        fit_data[wm_mask] = logbc
        splsm3d.fit(fit_data, reportingLevel=0)
        log_bias_field = splsm3d.predict()
        log_bias_masked = log_bias_field[wm_mask]
        bcratio = np.exp(log_bias_masked)
        conv = bcratio.std() / bcratio.mean()
        datalogmaskedcur = datalogmaskedcur - log_bias_masked
        if controlField is None:
            controlField = splsm3d.P.copy()
        else:
            controlField += splsm3d.P
        datalogcur[wm_mask] = datalogmaskedcur
        if conv < stopthr:
            nextlevel = levels[N] + 1
        if (
            subdivide
            and (N + 1) < len(levels)
            and (nextlevel > levels[N] or levels[N + 1] != levels[N])
        ):
            splsm3d.P = controlField
            splsm3d = splsm3d.promote()
            predictor = predictor.promote()
            controlField = splsm3d.P

    splsm3d.P = controlField
    predictor.P = splsm3d.P

    bias0 = np.exp(predictor.predict())
    tissue_idx = bias0 != 0
    brain0[tissue_idx] /= bias0[tissue_idx]
    if seg is not None:
        brain0 = piecewise_linear_scaling(brain0, seg0)
    return nib.Nifti1Image(brain0, brain.affine, brain.header)


def _gradient_magnitude(vol, vx=(1.0, 1.0, 1.0)):
    """Gradient magnitude in physical units (mm^-1)."""
    g = np.gradient(vol, vx[0], vx[1], vx[2])
    return np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)


def _divergence(vol, vx=(1.0, 1.0, 1.0)):
    """Divergence of the normalised gradient, as in CAT12's cat_vol_div.

    Thin bright structures -- blood vessels, meninges -- have a strongly
    negative divergence, while the interior of a tissue does not.  This is
    what lets them be excluded from the intensity peak estimation instead of
    dragging the CSF and GM peaks upwards.
    """
    eps = np.finfo(np.float32).eps
    g = np.gradient(vol, vx[0], vx[1], vx[2])
    n = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + eps
    return sum(np.gradient(g[i] / n, vx[i], axis=i) for i in range(3))


def _tissue_masks(seg0):
    """High-confidence, non-overlapping tissue masks from a PVE label map.

    A label of 1/2/3 is pure CSF/GM/WM, so a tissue fraction of at least 75 %
    -- the >192/255 class threshold CAT12 uses -- becomes a +-0.25 window
    around the pure value.  The windows do not overlap, unlike the label
    ranges these replace, where the GM range ran to 2.85 and so was fitted
    largely on GM/WM partial-volume voxels.
    """
    return (
        (seg0 > 0.75) & (seg0 <= 1.25),      # CSF
        (seg0 >= 1.75) & (seg0 <= 2.25),     # GM
        (seg0 >= 2.75),                      # WM
    )


def global_intensity_norm(brain, seg, vx=None, verbose=False):
    """Global intensity normalisation, after CAT12's cat_main_gintnorm.

    Estimates one intensity peak per tissue and maps the image through those
    peaks onto the scale AMAP expects (CSF=1/3, GM=2/3, WM=1).  Three ideas
    are taken from cat_main_gintnorm and are the reason this is more robust
    than taking a plain mean per label range:

    * the peak is a **median** over voxels that are at least 75 % one tissue,
    * restricted to **low local gradient**, which drops partial-volume and
      edge voxels (a tighter bound for WM than for CSF/GM), and additionally
      **low divergence**, which drops vessels and meninges,
    * CSF is additionally capped below the GM level, because vessels and
      meninges have GM-like intensity and otherwise pull the CSF peak up.

    The mapping runs through a node list rather than a single linear scale,
    with a node above WM, so hyperintensities are compressed into a bounded
    range instead of extrapolating away.

    Returns
    -------
    (normalised nifti, stats dict) with the peaks, the contrast, and a
    noise estimate (local sigma in WM and CSF over the smallest tissue gap),
    which is the contrast-to-noise figure CAT12 uses to decide how hard to
    filter.
    """
    eps = np.finfo(np.float32).eps
    src = brain.get_fdata().astype(np.float32)
    seg0 = seg.get_fdata()
    if vx is None:
        vx = tuple(float(z) for z in brain.header.get_zooms()[:3])

    m_csf, m_gm, m_wm = _tissue_masks(seg0)

    def _median(mask, fallback):
        return float(np.median(src[mask])) if mask.sum() >= 100 else fallback

    # Pass 1: crude peaks, needed only to normalise the image so that the
    # gradient and the CSF ceiling below are on a known scale.
    c0 = _median(m_csf, 0.0)
    g0 = _median(m_gm, 0.0)
    w0 = _median(m_wm, float(np.max(src)) if src.size else 1.0)
    prov = _map_through_nodes(src, c0, g0, w0)

    # Pass 2: refine with the gradient / divergence / ceiling guards.
    yg = _gradient_magnitude(prov, vx) / np.maximum(prov, eps)
    ydiv = _divergence(prov, vx)
    div_lim = float(np.percentile(np.abs(ydiv), 95)) if ydiv.size else np.inf

    clean = np.abs(ydiv) < div_lim
    sel_csf = m_csf & (yg < 0.20) & (prov < 0.45) & clean
    sel_gm = m_gm & (yg < 0.20) & clean
    sel_wm = m_wm & (yg < 0.10) & clean

    csf = _median(sel_csf, c0)
    gm = _median(sel_gm, g0)
    wm = _median(sel_wm, w0)

    # Contrast-to-noise, as in cat_main_gintnorm: local sigma inside the two
    # most homogeneous tissues over the smallest tissue gap.
    gaps = [abs(gm - csf), abs(wm - gm)]
    contrast = min(gaps) / max(abs(wm - csf), eps)
    noise = np.nan
    if sel_wm.sum() >= 100 and sel_csf.sum() >= 100:
        sd = min(float(np.std(src[sel_wm])), float(np.std(src[sel_csf])))
        noise = sd / max(min(gaps), eps)

    if not (csf < gm < wm):
        # Non-monotonic peaks mean the contrast assumption is violated (bad
        # segmentation, inverted contrast).  Fall back to the crude peaks
        # rather than building a non-monotonic mapping.
        if verbose:
            print(f"gintnorm: non-monotonic peaks ({csf:.3f},{gm:.3f},{wm:.3f}), "
                  "falling back to unguarded estimates")
        csf, gm, wm = c0, g0, w0
    if verbose:
        print(f"gintnorm: peaks CSF/GM/WM = {csf:.3f}/{gm:.3f}/{wm:.3f}, "
              f"contrast = {contrast:.3f}, noise = {noise:.4f}")

    out = _map_through_nodes(src, csf, gm, wm)
    stats = {"csf": csf, "gm": gm, "wm": wm,
             "contrast": float(contrast), "noise": float(noise)}
    return nib.Nifti1Image(out, brain.affine, brain.header), stats


def _map_through_nodes(src, csf, gm, wm):
    """Piecewise-linear map through the tissue peaks onto the 1/3, 2/3, 1 scale.

    The node above WM (``wm + (wm - csf) / 2`` -> 4/3) is what bounds
    hyperintense structures: everything from there to the image maximum is
    compressed into 4/3..5/3 rather than extrapolated linearly.
    """
    eps = np.finfo(np.float32).eps
    wm_plus = wm + 0.5 * (wm - csf)
    imax = max(float(np.max(src)) if src.size else wm_plus, wm_plus + eps)
    xs = [0.0, csf, gm, wm, wm_plus, imax]
    # np.interp needs strictly increasing nodes
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32) / 3.0
    return np.interp(src, xs, ys).astype(np.float32)


def fit_intensity_field(
    brain, seg, limit=None, steps=1000, spacing=1.0, stopthr=5e-4, use_prctile=3,
    exclude=None
):
    """Estimate a smooth bias-like intensity field."""
    if limit is None:
        limit = [2.75, 3]
    if not (isinstance(limit, (list, tuple)) and len(limit) == 2):
        raise ValueError("limit must be a 2-element list or tuple")

    subdivide = True
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.2
    subsamp = 5

    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata().copy()
    seg0 = seg.get_fdata().copy()
    mask = (seg0 > limit[0]) & (seg0 <= limit[1])
    if exclude is not None:
        # Drop edge / partial-volume / vessel voxels before fitting, so the
        # field describes the tissue and not the structures next to it.
        # Keep the unguarded mask if the guard leaves too little to fit.
        guarded = mask & ~exclude
        if guarded.sum() >= max(1000, 0.05 * mask.sum()):
            mask = guarded

    if subsamp:
        offset = 0
        dataSub = brain0[offset::subsamp, offset::subsamp, offset::subsamp]
        maskSub = mask[offset::subsamp, offset::subsamp, offset::subsamp]
        dataSubVoxSize = dataVoxSize * subsamp
    else:
        dataSub = brain0
        maskSub = mask

    dataSubVoxSize = 1 / (np.array(dataSub.shape) - 1)
    dataVoxSize = dataSubVoxSize / subsamp

    if use_prctile == 3:
        p5, p95 = np.percentile(dataSub[maskSub], [5, 95])
        maskSub = maskSub & (dataSub > p5) & (dataSub < p95)
    elif use_prctile == 2:
        p95 = np.percentile(dataSub[maskSub], 95)
        maskSub = maskSub & (dataSub < p95)
    elif use_prctile == 1:
        p5 = np.percentile(dataSub[maskSub], 5)
        maskSub = maskSub & (dataSub > p5)

    datalog = dataSub.astype(np.float32)
    if np.any(datalog[maskSub] <= 0):
        raise ValueError(
            "Non-positive values found in the masked data. Adjust mask or preprocess the image."
        )
    datalog[maskSub] = np.log(datalog[maskSub])
    datalog[np.logical_not(maskSub) | ~np.isfinite(datalog)] = 0
    datalogmasked = datalog[maskSub]
    fit_data = np.zeros_like(datalog)
    datalogmaskedcur = np.copy(datalogmasked)

    levels = [lvl for lvl in range(maxlevel) for _ in range(steps)]
    levelfwhm = (
        fwhm / (np.arange(maxlevel) + 1) if not subdivide else fwhm * np.ones(maxlevel)
    )

    splsm3d = SplineSmooth3DUnregularized(
        datalog, dataSubVoxSize, spacing, domainMethod="minc", mask=maskSub
    )
    predictor = SplineSmooth3D(
        brain0, dataVoxSize, spacing, knts=splsm3d.kntsArr, dofit=False
    )

    datalogcur = np.copy(datalog)
    nextlevel = 0
    controlField = None
    chosenkernelfn = kernelfntri

    for N in range(len(levels)):
        if levels[N] < nextlevel:
            continue
        hist, histvaledge, histval, histbinwidth = distrib_kde(
            datalogmaskedcur, Nbins, kernfn=chosenkernelfn, binCentreLimits=True
        )
        thisFWHM = levelfwhm[levels[N]]
        thisSD = thisFWHM / np.sqrt(8 * np.log(2))
        mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)
        histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
        histfiltclip = np.clip(histfilt, 0, None)
        uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
        datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
        diff = datalogmaskedcur - datalogmaskedupd
        fit_data[maskSub] = diff
        splsm3d.fit(fit_data, reportingLevel=0)
        diff_field = splsm3d.predict()
        diff_masked = diff_field[maskSub]
        bcratio = np.exp(diff_masked)
        conv = bcratio.std() / bcratio.mean()
        datalogmaskedcur = datalogmaskedcur - diff_masked
        if controlField is None:
            controlField = splsm3d.P.copy()
        else:
            controlField += splsm3d.P
        if conv < stopthr:
            nextlevel = levels[N] + 1
        if (
            subdivide
            and (N + 1) < len(levels)
            and (nextlevel > levels[N] or levels[N + 1] != levels[N])
        ):
            splsm3d.P = controlField
            splsm3d = splsm3d.promote()
            predictor = predictor.promote()
            controlField = splsm3d.P

    splsm3d.P = controlField
    predictor.P = splsm3d.P
    field = np.exp(predictor.predict())
    mean_raw = np.median(brain0[mask])
    mean_field = np.median(field[mask])
    field = field * (mean_raw / mean_field)
    return field


def apply_LAS(t1, label, verbose=False):
    """Apply Local Adaptive Segmentation to T1 images.

    Runs in two stages, following CAT12's order of gintnorm -> LAS: a global
    intensity normalisation fixes the tissue peaks and bounds hyperintense
    structures, then the smooth per-tissue fields below capture what is left,
    which is the spatial variation LAS is actually for.

    The fields are fitted on high-confidence, non-overlapping label windows
    with the edge / vessel voxels excluded.  Previously the GM window ran to
    2.85 and so was fitted largely on GM/WM partial-volume voxels, which
    biased the GM reference towards WM and compressed the GM/WM contrast.
    """
    eps = np.finfo(float).eps
    stopthr = 5e-4
    spacing = 1.0

    t1, stats = global_intensity_norm(t1, label, verbose=verbose)

    Ysrc = t1.get_fdata().copy()
    minYsrc = np.min(Ysrc)

    # Edge / partial-volume / vessel voxels, excluded from every fit below.
    vx = tuple(float(z) for z in t1.header.get_zooms()[:3])
    yg = _gradient_magnitude(Ysrc, vx) / np.maximum(Ysrc, eps)
    ydiv = _divergence(Ysrc, vx)
    div_lim = float(np.percentile(np.abs(ydiv), 95)) if ydiv.size else np.inf
    exclude_soft = (yg >= 0.20) | (np.abs(ydiv) >= div_lim)
    exclude_hard = (yg >= 0.10) | (np.abs(ydiv) >= div_lim)

    fit_csf = fit_intensity_field(
        t1, label, limit=[0.75, 1.25], spacing=spacing, stopthr=stopthr,
        use_prctile=3, exclude=exclude_soft
    )
    fit_gm = fit_intensity_field(
        t1, label, limit=[1.75, 2.25], spacing=spacing, stopthr=stopthr,
        use_prctile=3, exclude=exclude_soft
    )
    fit_wm = fit_intensity_field(
        t1, label, limit=[2.75, 3], spacing=spacing, stopthr=stopthr,
        use_prctile=3, exclude=exclude_hard
    )

    Yml = np.zeros_like(Ysrc, dtype=np.float32)
    if not (fit_csf.shape == fit_gm.shape == fit_wm.shape == Ysrc.shape):
        raise ValueError("All fitted fields and source image must have the same shape.")

    # Above WM the slope must match the GM->WM slope, otherwise the map has a
    # kink at the WM level (it used fit_wm - fit_csf here, roughly twice the
    # GM->WM span, so the slope halved).  Yml is bounded at 5 afterwards,
    # which is the same ceiling the global node mapping applies.
    mask_wm = Ysrc >= fit_wm
    Yml += mask_wm * (3 + (Ysrc - fit_wm) / np.maximum(eps, fit_wm - fit_gm))

    mask_gm = (Ysrc >= fit_gm) & (Ysrc < fit_wm)
    Yml += mask_gm * (2 + (Ysrc - fit_gm) / np.maximum(eps, fit_wm - fit_gm))

    mask_csf = (Ysrc >= fit_csf) & (Ysrc < fit_gm)
    Yml += mask_csf * (1 + (Ysrc - fit_csf) / np.maximum(eps, fit_gm - fit_csf))

    mask_bg = Ysrc < fit_csf
    Yml += mask_bg * ((Ysrc - minYsrc) / np.maximum(eps, fit_csf - minYsrc))

    Yml[Yml < 0.25] = 0
    np.clip(Yml, 0.0, 5.0, out=Yml)
    return nib.Nifti1Image(Yml / 3, t1.affine, t1.header)


def correct_label_map(brain, seg):
    """Correct a label map based on local intensity discrepancies."""
    brain0 = brain.get_fdata().copy()
    seg0 = seg.get_fdata().copy()

    discrepancy0 = (1 + brain0 * 3) / (1 + seg0)
    discrepancy0 = median_filter(discrepancy0, size=3)

    wm_mask = (seg0 > 2.5) & (discrepancy0 < 1)
    seg0[wm_mask] *= discrepancy0[wm_mask] ** 2

    csf_mask = (seg0 < 1.5) & (discrepancy0 > 1) & (brain0 > 1.5 / 3)
    brain0[csf_mask] /= discrepancy0[csf_mask] ** 2

    gm_mask1 = (seg0 > 1.5) & (seg0 <= 2)
    brain0[gm_mask1 & (brain0 > 1.4 / 3) & (brain0 <= 1.6 / 3)] = 1.6 / 3
    gm_mask2 = (seg0 > 2) & (seg0 <= 2.5)
    brain0[gm_mask2 & (brain0 > 2.4 / 3) & (brain0 <= 2.6 / 3)] = 2.4 / 3

    seg_corrected = nib.Nifti1Image(seg0, seg.affine, seg.header)
    brain_corrected = nib.Nifti1Image(brain0, brain.affine, brain.header)
    return seg_corrected, brain_corrected
