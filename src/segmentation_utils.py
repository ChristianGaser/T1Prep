import math
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.ndimage import (
    binary_opening,
    binary_dilation,
    grey_opening,
    binary_closing,
    generate_binary_structure,
    median_filter,
)
from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from torchreg.utils import smooth_kernel
from deepmriprep.utils import DEVICE, nifti_to_tensor
from deepmriprep.atlas import shape_from_to, AtlasRegistration

from utils import DATA_PATH, find_largest_cluster

import numpy as np
import nibabel as nib
from typing import Union, Tuple

def normalize_to_sum1(
    data1: Union[np.ndarray, nib.Nifti1Image],
    data2: Union[np.ndarray, nib.Nifti1Image],
    data3: Union[np.ndarray, nib.Nifti1Image]
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

    return (wrap_nifti(norm1, nifti1),
            wrap_nifti(norm2, nifti2),
            wrap_nifti(norm3, nifti3))


def cleanup(gm0, wm0, csf0, threshold_wm=0.4, cerebellum=None, csf_TPM=None):
    """Clean CSF, GM and WM probability maps."""
    gm = gm0.get_fdata().copy()
    wm = wm0.get_fdata().copy()
    csf = csf0.get_fdata().copy()

    wm_morph = find_largest_cluster(wm > threshold_wm)
    wm_morph = binary_dilation(wm_morph, generate_binary_structure(3, 3), 1)
    
    mask = ~wm_morph
    if cerebellum is not None:
        mask = mask & (cerebellum == 0)
    if csf_TPM is not None:
        mask = mask & (csf_TPM >= 0.025 * 255)

    gm[mask] += wm[mask]
    wm[mask] = 0
    gm_wm = grey_opening(gm + wm, size=[3, 3, 3])
    gm_wm = find_largest_cluster(gm_wm > 0.5)
    if cerebellum is not None:
        gm_wm = gm_wm | (cerebellum > 0)
    csf[~gm_wm] += gm[~gm_wm]
    gm[~gm_wm] = 0

    gm, wm, csf = normalize_to_sum1(gm, wm, csf)

    label = csf + 2 * gm + 3 * wm

    gm = nib.Nifti1Image(gm, gm0.affine, gm0.header)
    wm = nib.Nifti1Image(wm, wm0.affine, wm0.header)
    csf = nib.Nifti1Image(csf, csf0.affine, csf0.header)
    label = nib.Nifti1Image(label, gm0.affine, gm0.header)

    return label, gm, wm, csf


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


def correct_bias_field(brain, seg, steps=1000, spacing=1.0, get_discrepancy=False):
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
    seg0 = seg.get_fdata().copy()
    mask = seg0 >= 2.75
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
    datalog[np.logical_not(wm_mask)] = 0
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
    datalog[np.logical_not(maskSub)] = 0
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


def get_atlas(
    t1, affine, target_header, target_affine, atlas_name, warp_yx=None, device="cpu"
):
    """Generate an atlas-aligned image for segmentation."""
    header = target_header
    dim_hdr = target_header["dim"][1:4]
    dim = tuple(int(x) for x in dim_hdr)
    transform = target_affine

    atlas = nib.as_closest_canonical(
        nib.load(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/{atlas_name}.nii.gz")
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
    atlas_tensor = F.interpolate(atlas_tensor, dim, mode="nearest")[0, 0]
    atlas_tensor = atlas_tensor.type(
        torch.uint8 if atlas_tensor.max() < 256 else torch.int16
    )
    atlas_np = atlas_tensor.cpu().numpy()
    return nib.Nifti1Image(atlas_np, transform, header)


def get_cerebellum(atlas):
    """Return a binary cerebellum mask from the IBSR atlas."""
    rois = pd.read_csv(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv", sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))
    atlas = atlas.get_fdata().copy()
    cerebellum = (
        (atlas == regions["lCbeWM"])
        | (atlas == regions["lCbeGM"])
        | (atlas == regions["rCbeWM"])
        | (atlas == regions["rCbeGM"])
    )
    return cerebellum


def get_partition(p0_large, atlas):
    """Partition a segmentation into left and right hemispheres."""
    rois = pd.read_csv(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv", sep=";")[
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
