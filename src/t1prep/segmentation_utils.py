import math
import torch
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
    generate_binary_structure,
    grey_opening,
    median_filter,
)
from scipy.ndimage import label as label_image
from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from torchreg.utils import smooth_kernel
from deepmriprep.utils import DEVICE, nifti_to_tensor
from deepmriprep.atlas import shape_from_to, AtlasRegistration
from utils import DATA_PATH_T1PREP, TEMPLATE_PATH_T1PREP, find_largest_cluster
from typing import Union, Tuple


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

def cleanup_vessels(gm0, wm0, csf0, threshold_wm=0.4, cerebellum=None, 
      csf_TPM=None, vessel_TPM=None):
    """
    Clean/regularize CSF, GM, and WM probability maps by removing vessel-like
    misclassifications in CSF spaces *outside the cerebellum*, while preserving
    cortical/cerebellar parenchyma.

    Pipeline summary
    ----------------
    1) WM core: Threshold WM and keep only the largest connected component, then
       dilate once (26-connectivity) to obtain a robust parenchyma core.
    2) Vessel-targeting mask: Voxels that are (a) outside the WM core,
       (b) outside the cerebellum (if provided), and (c) likely CSF per a TPM
       (> 2.5% on a 0–255 scale, if provided).
    3) WM→GM inside that mask: Collapse thin/high-intensity vessel fragments
       that were misclassified as WM in CSF spaces.
    4) Identify vessels by assuming that they are surrounded by CSF or GM (checked by
       filling) and are estimated as WM. Vessels in WM are moved into CSF.
    5) Parenchyma support: Grayscale opening of (GM+WM) to remove narrow bridges
       (e.g., vessels), then keep the largest component and union with the
       cerebellum mask (if provided).
    6) GM→CSF outside the parenchyma support: This removes remaining GM vessel
       fragments in CSF spaces; prior WM→GM reassignment ensures WM vessels are
       also absorbed.
    7) Renormalize probabilities to sum to 1 per voxel and return images.

    Parameters
    ----------
    gm0, wm0, csf0 : nib.Nifti1Image
        Input gray matter, white matter, and CSF *probability* maps (same shape,
        affine, and header). Values are expected in [0, 1].
    threshold_wm : float, optional
        WM threshold used to build the robust WM core (default: 0.4).
        Higher values make the core more conservative.
    cerebellum : np.ndarray or None, optional
        Binary (0/1) array in the same space, where >0 denotes cerebellum voxels.
        When provided, cleanup avoids acting *inside* the cerebellum and later
        explicitly keeps cerebellar voxels as parenchyma.
    csf_TPM : np.ndarray or None, optional
        CSF tissue probability map in the same space, *scaled 0–255* (e.g., SPM).
        If provided, vessel cleanup is restricted to voxels with CSF probability
        ≥ 2.5% (i.e., >= 0.025 * 255). Omit to apply outside-WM-core everywhere
        outside the cerebellum.
    vessel_TPM : np.ndarray or None, optional
        Blood vessel tissue probability map in the same space, *scaled 0–255* (e.g., SPM).
        If provided, vessel cleanup is restricted to voxels with vessel probability
        ≥ 10% (i.e., >= 0.1 * 255). 

    Returns
    -------
    label : nib.Nifti1Image
        Soft-encoded label image (csf + 2*gm + 3*wm). This is NOT a hard label;
        use argmax over (csf, gm, wm) if a discrete map is required.
    gm, wm, csf : nib.Nifti1Image
        Cleaned GM, WM, and CSF probability maps, renormalized to sum to 1
        per voxel.

    Notes
    -----
    - Connectivity: `find_largest_cluster` is assumed to use 26-connectivity in 3D.
    - Morphology: `binary_dilation` uses a full 3x3x3 structuring element
      (`generate_binary_structure(3, 3)`) for one iteration.
    - Grayscale opening: `grey_opening(gm + wm, size=[3,3,3])` suppresses thin
      vessel-like bridges/spurs prior to component selection.
    - Assumes gm0/wm0/csf0 share shape/affine/header. No resampling is performed.
    - If you need a discrete label map, compute `np.argmax([csf, gm, wm], axis=0)`.

    """
    gm = gm0.get_fdata().copy()
    wm = wm0.get_fdata().copy()
    csf = csf0.get_fdata().copy()

    # 1) Robust WM core from the largest component, slightly dilated (26-connectivity).
    wm_morph = find_largest_cluster(wm > threshold_wm)
    wm_morph = binary_dilation(wm_morph, generate_binary_structure(3, 3), 1)

    # 2) Vessel-targeting mask: outside WM core, outside cerebellum, and (optionally) CSF-like.
    mask = ~wm_morph
    if cerebellum is not None:
        mask = mask & (cerebellum == 0)
    if csf_TPM is not None:
        mask = mask & (csf_TPM >= 0.025 * 255)
    if vessel_TPM is not None:
        mask = mask & (vessel_TPM >= 0.1 * 255)

    # 3) Collapse WM vessels in CSF spaces to GM (to be pushed to CSF in step 6 if outside parenchyma).
    gm[mask] += wm[mask]
    wm[mask] = 0

    # 4) Identify vessels by assuming that they are surrounded by CSF or GM (checked by
    # filling) and are estimated as WM. Move vessels in WM into CSF.
    lbl = np.argmax(np.stack([csf, gm, wm], axis=0), axis=0)
    csf_label = lbl == 0
    gm_label = lbl == 1
    wm_label = lbl == 2
    csf_filled = binary_closing(csf_label, generate_binary_structure(3, 3), 2)
    gm_filled = binary_closing(gm_label, generate_binary_structure(3, 3), 2)
    vessels = wm_label & binary_dilation(
        gm_filled & csf_filled, generate_binary_structure(3, 3), 1
    )
    csf[vessels] += wm[vessels]
    wm[vessels] = 0    
    
    # 5) Build parenchyma support via grayscale opening and largest component, keep cerebellum.
    gm_wm = grey_opening(gm + wm, size=[3, 3, 3])
    gm_wm = find_largest_cluster(gm_wm > 0.5)
    if cerebellum is not None:
        gm_wm = gm_wm | (cerebellum > 0)

    # 6) Move GM outside parenchyma support into CSF (removes residual vessel fragments).
    csf[~gm_wm] += gm[~gm_wm]
    gm[~gm_wm] = 0

    # 7) Renormalize and package outputs.
    gm, wm, csf = normalize_to_sum1(gm, wm, csf)
    label = csf + 2 * gm + 3 * wm

    gm = nib.Nifti1Image(gm, gm0.affine, gm0.header)
    wm = nib.Nifti1Image(wm, wm0.affine, wm0.header)
    csf = nib.Nifti1Image(csf, csf0.affine, csf0.header)
    label = nib.Nifti1Image(label, gm0.affine, gm0.header)

    return label, gm, wm, csf


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


def fit_intensity_field(
    brain, seg, limit=None, steps=1000, spacing=1.0, stopthr=5e-4, use_prctile=3
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


def apply_LAS(t1, label):
    """Apply Local Adaptive Segmentation to T1 images."""
    eps = np.finfo(float).eps
    stopthr = 5e-4
    spacing = 1.0

    Ysrc = t1.get_fdata().copy()
    minYsrc = np.min(Ysrc)

    fit_csf = fit_intensity_field(
        t1, label, limit=[1, 1.25], spacing=spacing, stopthr=stopthr, use_prctile=3
    )
    fit_gm = fit_intensity_field(
        t1, label, limit=[1.5, 2.85], spacing=spacing, stopthr=stopthr, use_prctile=3
    )
    fit_wm = fit_intensity_field(
        t1, label, limit=[2.5, 3], spacing=spacing, stopthr=stopthr, use_prctile=3
    )

    Yml = np.zeros_like(Ysrc, dtype=np.float32)
    if not (fit_csf.shape == fit_gm.shape == fit_wm.shape == Ysrc.shape):
        raise ValueError("All fitted fields and source image must have the same shape.")

    mask_wm = Ysrc >= fit_wm
    Yml += mask_wm * (3 + (Ysrc - fit_wm) / np.maximum(eps, fit_wm - fit_csf))

    mask_gm = (Ysrc >= fit_gm) & (Ysrc < fit_wm)
    Yml += mask_gm * (2 + (Ysrc - fit_gm) / np.maximum(eps, fit_wm - fit_gm))

    mask_csf = (Ysrc >= fit_csf) & (Ysrc < fit_gm)
    Yml += mask_csf * (1 + (Ysrc - fit_csf) / np.maximum(eps, fit_gm - fit_csf))

    mask_bg = Ysrc < fit_csf
    Yml += mask_bg * ((Ysrc - minYsrc) / np.maximum(eps, fit_csf - minYsrc))

    Yml[Yml < 0.25] = 0
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

        p1_large_uncorr = p1_large
        p2_large_uncorr = p2_large
        p3_large_uncorr = p3_large

        p0_value = p0_large_orig.get_fdata().copy()
        p0_value[csf | wm] = 1.5
        p0_value -= 1.5
        p1_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )

        p0_value = p0_large_orig.get_fdata().copy()
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

    label_map, _ = label_image(ind_wmh)
    sizes = np.bincount(label_map.ravel())
    min_lesion_size = 500
    remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
    ind_wmh[remove] = 0

    wmh_value[~ind_wmh] = 0

    if use_amap:
        csf_discrep_large = (
            p3_large_uncorr.get_fdata().copy() - p3_large.get_fdata().copy()
        )
        csf_discrep_large = median_filter(csf_discrep_large, size=3)
        ind_csf_discrep = csf_discrep_large < 0

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
        nib.load(f"{TEMPLATE_PATH_T1PREP}/{atlas_name}.nii.gz")
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
        atlas = atlas_register(affine, warps[shape], atlas, t1.shape)

    atlas_tensor = nifti_to_tensor(atlas)[None, None].to(device)

    # Choose interpolation mode and output dtype depending on whether
    # the atlas contains discrete labels or continuous values.
    if is_label_atlas:
        atlas_tensor = F.interpolate(atlas_tensor, dim, mode="nearest")[0, 0]
        atlas_tensor = atlas_tensor.type(
            torch.uint8 if atlas_tensor.max() < 256 else torch.int16
        )
    else:
        atlas_tensor = F.interpolate(
            atlas_tensor, dim, mode="trilinear", align_corners=False
        )[0, 0]
        atlas_tensor = atlas_tensor.to(torch.float32)

    atlas_np = atlas_tensor.cpu().numpy()
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
    rois = pd.read_csv(f"{TEMPLATE_PATH_T1PREP}/{atlas_name}.csv", sep=";")[
        ["ROIid", "ROIname"]
    ]
    regions = dict(zip(rois.ROIname, rois.ROIid))
    atlas_data = np.round(atlas.get_fdata())
    region_ids = [regions[r] for r in region_name if r in regions]
    return np.isin(atlas_data, region_ids)


def get_partition(p0_large, atlas):
    """Partition a segmentation into left and right hemispheres."""
    rois = pd.read_csv(f"{TEMPLATE_PATH_T1PREP}/ibsr.csv", sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))

    bin_struct3 = generate_binary_structure(3, 3)
    atlas_data = atlas.get_fdata().copy()
    atlas_mask = atlas_data > 0
    atlas_mask = binary_dilation(atlas_mask, bin_struct3, 3)

    p0_data = p0_large.get_fdata().copy()
    gm = (p0_data > 1.5) & (p0_data < 2.5)
    gm_regions = ["lCbrGM", "rCbrGM", "lAmy", "lHip", "rAmy", "rHip"]
    gm_mask = np.isin(atlas_data, [regions[r] for r in gm_regions])
    gm_mask = binary_dilation(gm_mask, bin_struct3, 2)
    gm = gm & gm_mask

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

    wm_fill = np.isin(atlas_data, [regions[r] for r in wm_regions])
    wm_fill = binary_dilation(wm_fill, bin_struct3, 10)

    lh = np.copy(p0_data)
    lh[lh < 1] = 1
    lh[wm_fill & ~gm_mask] = 3
    lh[exclude | right] = 1

    rh = np.copy(p0_data)
    rh[rh < 1] = 1
    rh[wm_fill & ~gm_mask] = 3
    rh[exclude | left] = 1

    mask = (lh > 1) | (rh > 1)
    mask = binary_closing(mask, bin_struct3, 1)
    mask = binary_opening(mask, bin_struct3, 3)
    mask = find_largest_cluster(mask)
    mask = binary_dilation(mask, bin_struct3, 1)
    lh[~mask] = 1
    rh[~mask] = 1

    return lh, rh
