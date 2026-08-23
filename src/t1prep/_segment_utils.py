import math
import torch
import os
import cat_surf
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import numpy as np

from scipy.ndimage import (
    binary_opening,
    binary_dilation,
    binary_closing,
    binary_erosion,
    convolve,
    distance_transform_edt,
    generate_binary_structure,
    gaussian_laplace,
    grey_opening,
    median_filter,
)
from .utils import (
    DATA_PATH_T1PREP,
    TEMPLATE_PATH_T1PREP, 
    find_largest_cluster, 
    remove_file,
)
from SplineSmooth3D.SplineSmooth3D import (
    SplineSmooth3D, 
    SplineSmooth3DUnregularized,
)
from scipy.ndimage import label as label_image
from nxbc.filter import *

from torchreg.utils import smooth_kernel
from deepmriprep.utils import DEVICE, nifti_to_tensor
from deepmriprep.atlas import shape_from_to, AtlasRegistration
from typing import Union, Tuple


def _resolve_template_file(name: str, ext: str) -> str:
    """Return the full path for *name* + *ext* inside TEMPLATE_PATH_T1PREP.

    Performs an exact lookup first and falls back to a case-insensitive scan
    of the directory so that callers using different capitalisation (e.g.
    ``"ibsr"`` vs the on-disk ``"IBSR"``) still resolve correctly.

    Parameters
    ----------
    name:
        Base name without extension (e.g. ``"IBSR"`` or ``"neuromorphometrics"``).
    ext:
        File extension including the leading dot (e.g. ``".csv"`` or
        ``".nii.gz"``).

    Returns
    -------
    str
        Absolute path to the resolved file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found in the template directory.
    """
    # Fast exact path
    candidate = os.path.join(TEMPLATE_PATH_T1PREP, f"{name}{ext}")
    if os.path.exists(candidate):
        return candidate

    # Case-insensitive fallback
    target = f"{name.lower()}{ext.lower()}"
    for fname in os.listdir(TEMPLATE_PATH_T1PREP):
        if fname.lower() == target:
            return os.path.join(TEMPLATE_PATH_T1PREP, fname)

    raise FileNotFoundError(
        f"Template file '{name}{ext}' not found in {TEMPLATE_PATH_T1PREP}"
    )

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

def normalize_to_sum1(
    data1: Union[np.ndarray, nib.Nifti1Image],
    data2: Union[np.ndarray, nib.Nifti1Image],
    data3: Union[np.ndarray, nib.Nifti1Image],
):
    """
    Normalize three input arrays or Nifti1Images so that their sum is 1 at each voxel/data point.

    Each output is calculated as:
        paraX_norm = paraX / (data1 + data2 + data3)   (element-wise)

    Division by zero is handled by setting the sum to 1 at voxels/data points where it is zero.

    Parameters
    ----------
    data1 : np.ndarray or nibabel.Nifti1Image
        First input parameter (array or Nifti image).
    data2 : np.ndarray or nibabel.Nifti1Image
        Second input parameter (array or Nifti image).
    data3 : np.ndarray or nibabel.Nifti1Image
        Third input parameter (array or Nifti image).

    Returns
    -------
    norm1 : np.ndarray or nibabel.Nifti1Image
        Normalized first parameter. If input was Nifti1Image, output will be Nifti1Image.
    norm2 : np.ndarray or nibabel.Nifti1Image
        Normalized second parameter (same type as input).
    norm3 : np.ndarray or nibabel.Nifti1Image
        Normalized third parameter (same type as input).

    Notes
    -----
    - The type of the output for each parameter matches its input type:
        - If the input is a Nifti1Image, the output is a Nifti1Image (with the same affine and header).
        - If the input is an array, the output is an array.
    - Mixed types are supported (e.g., two arrays and one Nifti1Image).
    - For voxels/data points where data1 + data2 + data3 == 0, the denominator is set to 1 to avoid division by zero (output is then 0 for all three).
    - The input data are first clipped to a range of 0..1

    """

    def extract_data(x):
        if isinstance(x, nib.Nifti1Image):
            return x.get_fdata(), x
        else:
            return np.asarray(x), None

    data1, nifti1 = extract_data(data1)
    data2, nifti2 = extract_data(data2)
    data3, nifti3 = extract_data(data3)

    # Clip data first to a range of 0..1
    data1 = np.clip(data1, 0, 1)
    data2 = np.clip(data2, 0, 1)
    data3 = np.clip(data3, 0, 1)

    sum_data = data1 + data2 + data3
    sum_data[sum_data == 0] = 1  # Prevent division by zero

    norm1 = data1 / sum_data
    norm2 = data2 / sum_data
    norm3 = data3 / sum_data

    def wrap_nifti(norm, ref_nifti):
        if ref_nifti is not None:
            return nib.Nifti1Image(norm, ref_nifti.affine, ref_nifti.header)
        else:
            return norm

    return (
        wrap_nifti(norm1, nifti1),
        wrap_nifti(norm2, nifti2),
        wrap_nifti(norm3, nifti3),
    )


def cleanup_vessels(
    gm0: nib.Nifti1Image,
    wm0: nib.Nifti1Image,
    csf0: nib.Nifti1Image,
    mri_dir: str,
    out_name: str,
    ext: str,
    debug:bool,
    cerebellum=None,
):
    """Blood vessel correction for PVE-based tissue probability maps.

    Detects and corrects blood vessel misclassifications using the PVE
    label map approach from ``blood_vessel_correction_pve_float``
    (CAT_Vol.c).  Optional a cerebellum mask can refine the detection.

    Parameters
    ----------
    gm0, wm0, csf0 : nib.Nifti1Image
        GM, WM, CSF probability maps (values in [0, 1]).
    strength : float, optional
        Correction strength (default 1).  Higher values increase the
        opening radii and lower the vessel-detection threshold.
    cerebellum : np.ndarray or None, optional
        Binary cerebellum mask in the same space.  Vessel detection is
        suppressed inside the cerebellum and cerebellar WM voxels are
        added to the seed region.

    Returns
    -------
    label : nib.Nifti1Image
        Soft PVE label map (``csf + 2*gm + 3*wm``).
    gm, wm, csf : nib.Nifti1Image
        Corrected tissue probability maps.
    """
    gm = gm0.get_fdata().copy().astype(np.float32)
    wm = wm0.get_fdata().copy().astype(np.float32)
    csf = csf0.get_fdata().copy().astype(np.float32)

    # Normalise probabilities.
    total = gm + wm + csf
    total[total == 0] = 1.0
    gm /= total
    wm /= total
    csf /= total

    # PVE label map (CSF=1, GM=2, WM=3 with partial volumes).
    label_in = (csf + 2.0 * gm + 3.0 * wm).astype(np.float32)
    mask = label_in > 0

    # Blood vessel correction via cat_surf Python binding (in-process)
    vx = gm0.header.get_zooms()[:3]
    label_out = cat_surf.vol_blood_vessel_correction(label_in, voxelsize=vx)

    if cerebellum is not None:
        mask = mask & (cerebellum == 0)

    # Rescue original label values outside mask
    label_out[~mask] = label_in[~mask]

    # Get single tissue segmentations
    csf_new = 1 - np.minimum(1, np.abs(label_out - 1))
    gm_new = 1 - np.minimum(1, np.abs(label_out - 2))
    wm_new = 1 - np.minimum(1, np.abs(label_out - 3))

    # Rescue original tissue segmentations outside mask
    csf_new[~mask] = csf[~mask]
    gm_new[~mask] = gm[~mask]
    wm_new[~mask] = wm[~mask]

    gm_new, wm_new, csf_new = normalize_to_sum1(gm_new, wm_new, csf_new)
    label_out = (csf_new + 2.0 * gm_new + 3.0 * wm_new).astype(np.float32)

    if debug:
        post_name = f"{mri_dir}/{out_name}_p0_large_post_vessel_cleanup_tmp.{ext}"
        nib.save(nib.Nifti1Image(label_out, gm0.affine, gm0.header), post_name)

    return (
        nib.Nifti1Image(label_out, gm0.affine, gm0.header),
        nib.Nifti1Image(gm_new, gm0.affine, gm0.header),
        nib.Nifti1Image(wm_new, wm0.affine, wm0.header),
        nib.Nifti1Image(csf_new, csf0.affine, csf0.header),
    )


def laplacian_3d(f, spacing=(1.0, 1.0, 1.0)):
    dz, dy, dx = spacing
    grad = np.gradient(f, dz, dy, dx)
    lap = sum(np.gradient(grad[i], (dz, dy, dx)[i], axis=i) for i in range(3))
    return lap
    

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


def unsmooth_kernel(factor=3.0, sigma=0.6, device="cpu"):
    kernel = -factor * smooth_kernel(
        kernel_size=3 * [3], sigma=torch.tensor(3 * [sigma], device=device)
    )
    kernel[1, 1, 1] = 0
    kernel[1, 1, 1] = 1 - kernel.sum()
    return kernel


def handle_lesions(
    t1: nib.Nifti1Image,
    affine,
    brain_large: nib.Nifti1Image,
    p0_large: nib.Nifti1Image,
    p0_large_orig: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    affine_resamp_reordered,
    header_resamp_reordered,
    mri_dir: str,
    out_name: str,
    ext: str,
    use_amap: bool,
    debug: bool,
    device: torch.device,
) -> tuple[
    nib.Nifti1Image,
    nib.Nifti1Image,
    nib.Nifti1Image,
    np.ndarray,
    np.ndarray,
]:
    """Detect lesions and correct tissue probability maps."""

    p0_value = p0_large_orig.get_fdata().copy()
    wm = p0_value >= 2.5
    # Fill WM holes to close potential WMH lesions
    wm = binary_closing(wm, generate_binary_structure(3, 3), 3)
    # Get a conservative WM mask
    wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
    gm = (p0_value >= 1.5) & (p0_value < 2.5)
    csf = (p0_value < 1.5) & (p0_value > 0)

    if use_amap:
        p0_large_diff_value = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
            - p0_large_orig.get_fdata().copy()
        )

        # Keep the untouched AMAP maps.  Nifti1Image objects are rebound
        # rather than mutated below, but bind them explicitly so an in-place
        # edit later cannot silently corrupt the reference.
        p1_large_uncorr = nib.Nifti1Image(
            p1_large.get_fdata().copy(), p1_large.affine, p1_large.header
        )
        p2_large_uncorr = nib.Nifti1Image(
            p2_large.get_fdata().copy(), p2_large.affine, p2_large.header
        )
        p3_large_uncorr = nib.Nifti1Image(
            p3_large.get_fdata().copy(), p3_large.affine, p3_large.header
        )

        # Reference GM map built from the deepmriprep label.  Inside the
        # conservative WM mask (and in CSF) it is exactly zero, which is what
        # makes `wmh_value` below equal AMAP's GM probability in deep WM --
        # the lesion signal.  Outside those masks it is only a ramp above the
        # CSF/GM threshold, not a probability, so it must not be used for
        # anything that is not restricted to `deep_wm`.
        p0_value = p0_large_orig.get_fdata().copy()
        p0_value[csf | wm] = 1.5
        p0_value -= 1.5
        p1_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )

        # Reference CSF map, on the probability scale: the label value 1 is
        # pure CSF, 2 is pure GM, so the CSF fraction is 2 - p0 clipped to
        # [0, 1].  Using the raw label value here would make the reference
        # *rise* across the CSF/GM partial-volume band while the true CSF
        # fraction falls, which turned the comparison below into a systematic
        # brain-wide offset instead of a discrepancy.
        p0_value = np.clip(2.0 - p0_large_orig.get_fdata(), 0.0, 1.0)
        p0_value[~csf] = 0
        p3_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )
        wmh_value = p1_large_uncorr.get_fdata().copy() - p1_large.get_fdata().copy()
    else:
        # brain_large is for the deepmriprep method the LAS corrected orignal
        # image which can be used here as proxi for p0_large_orig from AMAP
        p0_large_diff_value = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
            - 3 * brain_large.get_fdata().copy()
        )

        # WMH are where p0_large_diff_value shows a positive difference in WM
        wmh_value = np.zeros_like(p0_value)
        wmh_mask = wm & (p0_large_diff_value > 0)
        wmh_value[wmh_mask] = p0_large_diff_value[wmh_mask]

    # Apply median filter to remove noise
    wmh_value = median_filter(wmh_value, size=3)
    p0_large_diff_value = median_filter(p0_large_diff_value, size=3)
    wmh_value = np.clip(wmh_value, -1, 1)
    p0_large_diff_value = np.clip(p0_large_diff_value, -1, 1)
    p0_large_diff = nib.Nifti1Image(
        p0_large_diff_value, affine_resamp_reordered, header_resamp_reordered
    )

    deep_wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
    gm_border = binary_dilation(gm, generate_binary_structure(3, 3), 2)

    atlas = get_atlas(
        t1,
        affine,
        p0_large.header,
        p0_large.affine,
        "cat_wmh",
        None,
        device,
        is_label_atlas=False,
    )
    wmh_tpm = atlas.get_fdata().copy()
    wmh_tpm /= np.max(wmh_tpm)

    ind_wmh = ((wmh_value * wmh_tpm) > 0.025) & deep_wm & (~gm_border)

    # Drop small clusters.  Use the same 26-connectivity as the morphology
    # above, otherwise a diagonally connected lesion is split into pieces that
    # can each fall below the threshold.  The size limit is a volume, not a
    # voxel count, so it keeps its meaning if the working grid ever changes.
    min_lesion_mm3 = 62.5
    vx_vol = float(np.prod(p0_large_orig.header.get_zooms()[:3]))
    min_lesion_size = max(1, int(round(min_lesion_mm3 / max(vx_vol, 1e-6))))
    label_map, _ = label_image(ind_wmh, structure=generate_binary_structure(3, 3))
    sizes = np.bincount(label_map.ravel())
    remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
    ind_wmh[remove] = 0

    wmh_value[~ind_wmh] = 0

    if use_amap:
        csf_discrep_large = (
            p3_large_uncorr.get_fdata().copy() - p3_large.get_fdata().copy()
        )
        csf_discrep_large = median_filter(csf_discrep_large, size=3)

        # Act only where AMAP and the reference disagree by a meaningful
        # amount, and only inside the CSF band the reference is defined on.
        # Without a threshold this fired on every voxel with even a rounding
        # difference, and the correction is applied brain-wide -- unlike the
        # WMH one, which is confined by deep_wm/gm_border and a size filter.
        min_csf_discrep = 0.05
        ind_csf_discrep = (csf_discrep_large < -min_csf_discrep) & csf

        # Direction: AMAP found less CSF (more GM) than deepmriprep here.
        # deepmriprep is the map that tends to miss lesions and underestimate
        # GM, so AMAP's finding is kept and reinforced rather than pulled back
        # towards the reference.  Flip the two signs below to instead correct
        # AMAP towards deepmriprep.
        tmp_p1 = p1_large_uncorr.get_fdata().copy()
        tmp_p1[ind_wmh] -= wmh_value[ind_wmh]
        tmp_p1[ind_csf_discrep] -= csf_discrep_large[ind_csf_discrep]

        tmp_p2 = p2_large_uncorr.get_fdata().copy()
        tmp_p2[ind_wmh] += wmh_value[ind_wmh]

        tmp_p3 = p3_large_uncorr.get_fdata().copy()
        tmp_p3[ind_csf_discrep] += csf_discrep_large[ind_csf_discrep]

        # We have to normalize all tissue values to overall sum of one
        tmp_p1, tmp_p2, tmp_p3 = normalize_to_sum1(tmp_p1, tmp_p2, tmp_p3)

        # Convert back to nifti
        p1_large = nib.Nifti1Image(
            tmp_p1, affine_resamp_reordered, header_resamp_reordered
        )
        p2_large = nib.Nifti1Image(
            tmp_p2, affine_resamp_reordered, header_resamp_reordered
        )
        p3_large = nib.Nifti1Image(
            tmp_p3, affine_resamp_reordered, header_resamp_reordered
        )

    return p1_large, p2_large, p3_large, p0_large_diff, wmh_value, ind_wmh


def _resample_to(img, target_affine, target_shape, device="cpu", channel=None,
                 nearest=False):
    """Trilinear resample *img* onto the grid given by affine and shape.

    ``get_atlas`` resizes the template array onto the target dimensions and so
    silently assumes both cover the same field of view.  That holds for the
    label atlases it is used with, but ``cat_bloodvessels.nii.gz`` lives on the
    SPM TPM grid (origin -90/-126/-72) while T1Prep works on the shooting
    template grid (origin -84/-120/-72).  A plain resize would misplace the
    prior by several millimetres, which matters most exactly where it is used
    -- around the insula.  So the real affines are honoured here.

    The sampling grid is built at roughly the template resolution and only
    then resized to the target shape, so the explicit coordinate array stays
    small even when the target is a 0.5 mm volume.

    Set ``nearest`` for label atlases.  Interpolating label *ids* linearly
    invents labels that lie between two unrelated regions, which silently
    turns a protection mask into nonsense.
    """
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if channel is not None:
        data = data[..., channel]

    target_affine = np.asarray(target_affine, dtype=float)
    target_shape = np.asarray(target_shape, dtype=int)[:3]
    tgt_zoom = np.sqrt((target_affine[:3, :3] ** 2).sum(axis=0))
    src_zoom = np.asarray(img.header.get_zooms()[:3], dtype=float)

    # Intermediate grid: same field of view, roughly the template resolution.
    n_int = np.maximum(2, np.round(target_shape * tgt_zoom / src_zoom)).astype(int)
    inter = target_affine.copy()
    inter[:3, :3] = target_affine[:3, :3] * (target_shape / n_int)
    # Keep the outer field of view identical by shifting the first voxel centre.
    inter[:3, 3] = target_affine[:3, 3] + 0.5 * (
        inter[:3, :3] - target_affine[:3, :3]
    ) @ np.ones(3)

    # Intermediate voxel -> source voxel.
    to_src = np.linalg.inv(img.affine) @ inter
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in n_int], indexing="ij")
    coords = (
        to_src[:3, :3].astype(np.float32) @ np.stack([g.ravel() for g in grids])
        + to_src[:3, 3, None].astype(np.float32)
    )

    # grid_sample expects normalised coordinates in reversed axis order.
    shape_src = np.asarray(data.shape[:3], dtype=np.float32)
    norm = 2.0 * coords / np.maximum(shape_src - 1.0, 1.0)[:, None] - 1.0
    grid = torch.as_tensor(
        norm[::-1].T.reshape(*n_int, 3).copy(), device=device
    )[None]

    src = torch.as_tensor(
        np.ascontiguousarray(data, dtype=np.float32), device=device
    )[None, None]
    out = F.grid_sample(
        src,
        grid,
        mode="nearest" if nearest else "bilinear",
        align_corners=True,
        padding_mode="border",
    )
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    out = F.interpolate(
        out, size=tuple(int(v) for v in target_shape), mode=mode, **kwargs
    )
    return out[0, 0].cpu().numpy()


def get_atlas(
    t1,
    affine,
    target_header,
    target_affine,
    atlas_name,
    warp_yx=None,
    device="cpu",
    is_label_atlas: bool = True,
):
    """Generate an atlas-aligned image in the target space.

    Parameters
    ----------
    t1 : nib.Nifti1Image
        Reference image in target space. Only the shape is used here
        when applying the deformation field.
    affine : np.ndarray
        Affine of the target image used for atlas registration.
    target_header : nib.Nifti1Header
        Header of the target image; copied to the returned atlas image.
    target_affine : np.ndarray
        Affine of the target image; used as transform for the returned atlas.
    atlas_name : str
        Base file name of the atlas (``<atlas_name>.nii.gz`` located in
        ``TEMPLATE_PATH_T1PREP``).
    warp_yx : nib.Nifti1Image, optional
        Optional deformation field from atlas space to target space. If
        provided, the atlas is first warped using this field before
        resampling to the requested output grid.
    device : str or torch.device, optional
        Device on which interpolation is performed (default: ``"cpu"``).
    is_label_atlas : bool, optional
        If ``True`` (default), the atlas is assumed to contain discrete
        labels. Nearest-neighbour interpolation is used and the result is
        stored as an integer type (``uint8`` if the maximum label is
        smaller than 256, otherwise ``int16``).

        If ``False``, the atlas is assumed to contain continuous values
        (e.g., tissue probability maps). Linear interpolation is used and
        the output is stored as floating point (``float32``).

    Returns
    -------
    nib.Nifti1Image
        Atlas image resampled into the target space.

    """
    header = target_header
    dim_hdr = target_header["dim"][1:4]
    dim = tuple(int(x) for x in dim_hdr)
    transform = target_affine

    atlas = nib.as_closest_canonical(
        nib.load(_resolve_template_file(atlas_name, ".nii.gz"))
    )
    atlas_register = AtlasRegistration()

    if warp_yx is not None:
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(device)
        shape = tuple(shape_from_to(atlas, warp_yx))
        scaled_yx = F.interpolate(
            yx.permute(0, 4, 1, 2, 3), shape, mode="trilinear", align_corners=False
        )
        warps = {shape: scaled_yx.permute(0, 2, 3, 4, 1)}
        # AtlasRegistration internally pins its tensors to its own (CPU)
        # device.  When ``device`` here is MPS/CUDA, the warp would be on a
        # different device than the atlas tensor, triggering grid_sample's
        # "input and grid to be on same device" error.  Align them.
        atlas = atlas_register(
            affine, warps[shape].to(atlas_register.device), atlas, t1.shape
        )

    # Resizing onto the target dimensions silently assumes both grids cover
    # the same field of view.  That holds after the warp above -- deepmriprep
    # returns the atlas on WARP_TEMPLATE, which shares the working field of
    # view -- and for templates already on the working grid such as cat_wmh.
    # It does not hold for the 1 mm atlases (IBSR, Neuromorphometrics), whose
    # field of view is 161x197x161 mm against the working 169.5x205.5x169.5:
    # a plain resize stretches them by 5% and displaces structures by up to
    # 4.2 mm at the edges.  So the affines decide, and the fast path is taken
    # only when they genuinely agree.
    src_fov = np.asarray(atlas.shape[:3]) * np.sqrt(
        (np.asarray(atlas.affine)[:3, :3] ** 2).sum(axis=0)
    )
    tgt_affine = np.asarray(transform, dtype=float)
    tgt_fov = np.asarray(dim) * np.sqrt((tgt_affine[:3, :3] ** 2).sum(axis=0))
    src_origin = np.asarray(atlas.affine)[:3, 3]
    aligned = np.allclose(src_fov, tgt_fov, atol=1e-3) and np.allclose(
        src_origin, tgt_affine[:3, 3], atol=1e-3
    )

    if aligned:
        atlas_tensor = nifti_to_tensor(atlas)[None, None].to(device)
        if is_label_atlas:
            atlas_np = F.interpolate(atlas_tensor, dim, mode="nearest")[0, 0]
        else:
            atlas_np = F.interpolate(
                atlas_tensor, dim, mode="trilinear", align_corners=False
            )[0, 0]
        atlas_np = atlas_np.cpu().numpy()
    else:
        atlas_np = _resample_to(
            atlas, tgt_affine, dim, device=device, nearest=is_label_atlas
        )

    if is_label_atlas:
        atlas_np = np.round(atlas_np)
        atlas_np = atlas_np.astype(
            np.uint8 if atlas_np.max() < 256 else np.int16
        )
    else:
        atlas_np = atlas_np.astype(np.float32)

    return nib.Nifti1Image(atlas_np, transform, header)


def get_regions_mask(
    atlas: nib.Nifti1Image,
    atlas_name: str,
    region_name: list[str],
) -> np.ndarray:
    """Return a binary mask for a set of regions in a label atlas.

    This helper reads the ROI definition CSV associated with ``atlas_name``
    (``<atlas_name>.csv`` in ``TEMPLATE_PATH_T1PREP``), maps region
    name to their numeric IDs, and returns a boolean mask where
    voxels belonging to any of the requested regions are ``True``.

    Parameters
    ----------
    atlas : nib.Nifti1Image
        Label atlas image in the same space as the desired mask.
    atlas_name : str
        Base name of the atlas (e.g. ``"ibsr"``). The corresponding CSV
        file is expected at ``TEMPLATE_PATH_T1PREP/<atlas_name>.csv`` and
        must contain at least the columns ``ROIid`` and ``ROIabbr``.
    region_name : list of str
        List of ROI name (``"ROIname"``) to include in the mask (e.g.
        ``["Left Cerebellum White Matter", "Right Cerebellum White Matter"]``).

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as ``atlas.get_fdata()``, where
        ``True`` indicates voxels belonging to any of the requested
        regions.

    """
    rois = pd.read_csv(_resolve_template_file(atlas_name, ".csv"), sep=";")[
        ["ROIid", "ROIname"]
    ]
    regions = dict(zip(rois.ROIname, rois.ROIid))
    atlas_data = np.round(atlas.get_fdata())
    region_ids = [regions[r] for r in region_name if r in regions]
    return np.isin(atlas_data, region_ids)


def _octagon_dilation(mask, iterations, mask_region=None):
    """Dilate with an alternating 6/26 neighbourhood.

    Repeatedly dilating with ``generate_binary_structure(3, 3)`` grows a *cube*
    (Chebyshev ball): a 10-iteration dilation reaches 10 voxels along the axes
    but 17 along the body diagonal, so the resulting boundary is faceted.
    Alternating the 26- and 6-connected structuring elements grows an octagon
    instead, which is within ~8% of a sphere, at the same cost per iteration.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array to dilate.
    iterations : int
        Number of dilation steps.
    mask_region : np.ndarray, optional
        If given, the front may only expand into voxels that are ``True`` here,
        which turns the dilation into a geodesic propagation.

    Returns
    -------
    np.ndarray
        Boolean array holding the dilated mask.
    """
    struct26 = generate_binary_structure(3, 3)
    struct6 = generate_binary_structure(3, 1)
    out = mask
    for i in range(iterations):
        prev = out
        out = binary_dilation(
            out, struct26 if i % 2 else struct6, 1, mask=mask_region
        )
        if mask_region is not None and out.sum() == prev.sum():
            break  # front has converged, further steps cannot change it
    return out


def ventricle_fill(p0_data, atlas_data, regions, vx=0.5, reach_mm=5.0):
    """Grow the atlas ventricle labels through the subject's own ventricle.

    The fill that makes the hemisphere maps usable for surface extraction has
    to cover the whole ventricular system, and dilating the warped atlas label
    by a fixed margin does not: on a brain with enlarged ventricles the label
    sits inside the real cavity and the margin runs out before the roof, which
    leaves a CSF band under the cingulate that PBT then tracks as a sulcus.
    Worse, the same misregistration puts the atlas *cortical* label over that
    band, so the ``gm_mask`` veto blocks it a second time.

    Growing the label geodesically instead -- through non-WM voxels only --
    follows the cavity rather than a distance: the front flows along the
    ventricle to its true extent whatever the size, and stops at the wall
    because white matter blocks it.  ``reach_mm`` is therefore not the size of
    the ventricle but the margin of error allowed on the atlas: it bounds how
    far the fill can stray if the registration is off, and 5 mm is far too
    short to cross the corpus callosum into the interhemispheric fissure.

    Parameters
    ----------
    p0_data : np.ndarray
        PVE label map (0 = background, 1 = CSF, 2 = GM, 3 = WM).
    atlas_data : np.ndarray
        IBSR label volume resampled onto the same grid.
    regions : dict
        Mapping from IBSR ``ROIabbr`` to ``ROIid``.
    vx : float, optional
        Voxel size in mm, used to convert ``reach_mm`` into steps.
    reach_mm : float, optional
        Geodesic budget for the growth.  Raising it past ~7 mm lets the front
        reach the callosal sulcus and the medial cortex, so it is deliberately
        tight.

    Returns
    -------
    np.ndarray
        Boolean array, ``True`` on the ventricular system as segmented.
    """
    ventricles = ["lLatVen", "rLatVen", "lInfLatVen", "rInfLatVen"]
    seed = np.isin(atlas_data, [regions[r] for r in ventricles])
    # Grow through everything that is not white matter.  The ventricle is
    # CSF, but on brains like this the segmentation calls parts of its
    # interior GM, so keying on CSF alone would stop at the first mislabelled
    # voxel.
    return _octagon_dilation(seed, int(round(reach_mm / vx)), p0_data < 2.5)


def get_partition(p0_large, atlas):
    """Partition a segmentation into left and right hemispheres."""
    rois = pd.read_csv(_resolve_template_file("IBSR", ".csv"), sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))

    bin_struct3 = generate_binary_structure(3, 3)
    atlas_data = atlas.get_fdata().copy()
    atlas_mask = atlas_data > 0
    atlas_mask = binary_dilation(atlas_mask, bin_struct3, 3)

    p0_data = p0_large.get_fdata().copy()
    gm_regions = ["lCbrGM", "rCbrGM", "lAmy", "lHip", "rAmy", "rHip"]
    gm_mask = np.isin(atlas_data, [regions[r] for r in gm_regions])
    gm_mask = binary_dilation(gm_mask, bin_struct3, 2)

    left_regions = [
        "lCbrWM",
        "lCbrGM",
        "lLatVen",
        "lInfLatVen",
        "lThaPro",
        "lCau",
        "lPut",
        "lPal",
        "lHip",
        "lAmy",
        "lAcc",
        "lVenDC",
    ]
    right_regions = [r.replace("l", "r", 1) for r in left_regions]

    left = np.isin(atlas_data, [regions[r] for r in left_regions])
    right = np.isin(atlas_data, [regions[r] for r in right_regions])

    bin_struct3 = generate_binary_structure(3, 3)
    left = binary_opening(left, bin_struct3, 3)
    left = binary_closing(left, bin_struct3, 3)

    lh = binary_dilation(left, bin_struct3, 5) & ~right
    rh = binary_dilation(right, bin_struct3, 5) & ~left

    left = binary_closing(lh, bin_struct3, 2) & ~rh
    right = binary_closing(rh, bin_struct3, 2) & ~left

    excl_regions = ["lCbeWM", "lCbeGM", "rCbeWM", "rCbeGM", "b3thVen", "b4thVen"]
    exclude = np.isin(atlas_data, [regions[r] for r in excl_regions])
    exclude = binary_dilation(exclude, bin_struct3, 1)
    exclude = exclude | binary_dilation(
        np.isin(atlas_data, regions["bBst"]), bin_struct3, 5
    )
    exclude = exclude | ~atlas_mask

    wm_regions = [
        "lThaPro",
        "lCau",
        "lPut",
        "lPal",
        "lAcc",
        "lLatVen",
        "lInfLatVen",
        "rThaPro",
        "rCau",
        "rPut",
        "rPal",
        "rAcc",
        "rLatVen",
        "rInfLatVen",
    ]

    # Blind dilation of the atlas structures, vetoed by the dilated atlas
    # cortical label so it cannot eat into the ribbon where the warp is off.
    # This is what covers the deep grey nuclei.
    wm_fill = np.isin(atlas_data, [regions[r] for r in wm_regions])
    wm_fill = binary_dilation(wm_fill, bin_struct3, 10) & ~gm_mask

    # The ventricles get a subject-driven fill on top.  The veto above is
    # exactly what used to block their roof: on a brain with enlarged
    # ventricles the warp puts the atlas cortical label over the roof, so the
    # one place the blind dilation still had to reach was the one place it was
    # forbidden from.
    vx = float(np.mean(p0_large.header.get_zooms()[:3])) or 0.5
    wm_fill = wm_fill | ventricle_fill(p0_data, atlas_data, regions, vx=vx)

    lh = np.copy(p0_data)
    lh[lh < 1] = 1
    lh[wm_fill] = 3
    lh[exclude | right] = 1

    rh = np.copy(p0_data)
    rh[rh < 1] = 1
    rh[wm_fill] = 3
    rh[exclude | left] = 1

    mask = (lh > 1) | (rh > 1)
    mask = binary_closing(mask, bin_struct3, 1)
    mask = binary_opening(mask, bin_struct3, 3)
    mask = find_largest_cluster(mask)
    mask = binary_dilation(mask, bin_struct3, 1)
    lh[~mask] = 1
    rh[~mask] = 1

    return lh, rh


def compute_euler_number(vol, threshold=2.5):
    """Compute the Euler number of a 3D volume at a given threshold.

    Thresholds the volume at the given level (default 2.5, the GM/WM
    boundary) and computes the Euler characteristic of the resulting
    binary cubical cell complex using the formula:

        chi = V - E + F - C

    where V = foreground voxels, E = foreground edges (6-connected
    adjacent pairs), F = foreground faces (2x2 blocks), and C =
    foreground cubes (2x2x2 blocks).

    For a topologically perfect hemisphere (single connected component,
    no handles), chi = 1.  Each topological defect (handle/tunnel)
    decreases chi by 1.

    Parameters
    ----------
    vol : np.ndarray
        3-D label or probability array (e.g. from ``get_partition``).
    threshold : float, optional
        Iso-level that separates foreground from background (default 2.5).

    Returns
    -------
    int
        Euler characteristic of the binary volume.
    """
    b = vol >= threshold

    # Vertices (foreground voxels)
    V = int(np.count_nonzero(b))

    # Edges along each axis
    ex = int(np.count_nonzero(b[:-1] & b[1:]))
    ey = int(np.count_nonzero(b[:, :-1] & b[:, 1:]))
    ez = int(np.count_nonzero(b[:, :, :-1] & b[:, :, 1:]))
    E = ex + ey + ez

    # Faces (2x2 blocks in each plane)
    fxy = int(np.count_nonzero(
        b[:-1, :-1] & b[1:, :-1] & b[:-1, 1:] & b[1:, 1:]
    ))
    fxz = int(np.count_nonzero(
        b[:-1, :, :-1] & b[1:, :, :-1] & b[:-1, :, 1:] & b[1:, :, 1:]
    ))
    fyz = int(np.count_nonzero(
        b[:, :-1, :-1] & b[:, 1:, :-1] & b[:, :-1, 1:] & b[:, 1:, 1:]
    ))
    F = fxy + fxz + fyz

    # Cubes (2x2x2 blocks)
    C = int(np.count_nonzero(
        b[:-1, :-1, :-1] & b[1:, :-1, :-1]
        & b[:-1, 1:, :-1] & b[1:, 1:, :-1]
        & b[:-1, :-1, 1:] & b[1:, :-1, 1:]
        & b[:-1, 1:, 1:] & b[1:, 1:, 1:]
    ))

    return V - E + F - C
