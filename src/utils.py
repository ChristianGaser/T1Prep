import os
import sys
import platform
import torch
import argparse
import warnings
import math
import shutil
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from spline_resize import resize
from deepbet.utils import reoriented_nifti
from deepmriprep.utils import DEVICE, nifti_to_tensor
from deepmriprep.atlas import shape_from_to, AtlasRegistration
from torchreg.utils import INTERP_KWARGS, smooth_kernel
from scipy.ndimage import (
    binary_opening,
    binary_dilation,
    grey_opening,
    binary_closing,
    binary_erosion,
    generate_binary_structure,
    median_filter,
    label,
)
from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from pathlib import Path
from skimage import filters

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
name_file = ROOT_PATH / "Names.tsv"

codes = [
    "Hemi_volume",
    "mT1_volume",
    "GM_volume",
    "WM_volume",
    "CSF_volume",
    "WMH_volume",
    "Label_volume",
    "Affine_space",
    "Warp_space",
    "Warp_modulated_space",
    "Def_volume",
    "invDef_volume",
]


def load_namefile(filename):
    """Parse a two-column TSV name file into a dictionary."""

    name_dict = {}
    with open(filename) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split(None, 2)  # split on whitespace, max 3 fields
            while len(parts) < 3:
                parts.append("")
            code, col2, col3 = parts
            name_dict[code] = (col2, col3)
    return name_dict


def substitute_pattern(pattern, bname, side, desc, space, nii_ext):
    """Replace placeholders in *pattern* using the provided values."""

    if not pattern:
        return ""
    result = pattern
    replacements = {
        "bname": bname,
        "side": side,
        "desc": desc,
        "space": space,
        "nii_ext": nii_ext,
    }
    for key, val in replacements.items():
        result = result.replace("{" + key + "}", val if val is not None else "")
    return result


def get_filenames(use_bids_naming, bname, side, desc, space, nii_ext):
    """Return a mapping of output filenames based on the name table."""

    name_dict = load_namefile(name_file)

    # BIDS/naming logic
    if use_bids_naming:
        name_columns = 1  # 0=old, 1=new
        hemi = "L" if side == "left" else "R"
    else:
        name_columns = 0
        hemi = "lh" if side == "left" else "rh"

    code_vars = {}
    for code in codes:
        patterns = name_dict.get(code)
        if not patterns:
            continue
        if name_columns >= len(patterns):
            continue
        pattern = patterns[name_columns]
        if not pattern:
            continue
        value = substitute_pattern(pattern, bname, hemi, desc, space, nii_ext)
        code_vars[code] = value

    return code_vars


def progress_bar(elapsed, total, name):
    """
    Displays a progress bar.

    Args:
        elapsed (int): Elapsed progress count.
        total (int): Total count.
        name (str): Name of the process.

    Usage:
        progress_bar(1, 100, "Name")

    Returns:
        int: Elapsed progress count increased by 1.
    """
    # Calculate percentage completion
    it = elapsed * 100 // total

    # Create the progress bar
    prog = "█" * elapsed
    remaining = " " * (total - elapsed)

    # Format the name with padding
    name = name.ljust(50)

    # Print the progress bar with percentage and name
    print(f"{prog}{remaining} {elapsed}/{total} {name}\r", end="")

    if elapsed == total:
        spaces = " " * 100
        print(f"{spaces}\r", end="")

    elapsed += 1
    return elapsed


def remove_file(name):
    """
    Remove file if exists.

    Args:
        name (str): Name of file.

    Usage:
        remove_file("Filename")
    """
    if os.path.exists(name):
        os.remove(name)
    else:
        print(f"The file '{name}' does not exist.")


def get_ras(aff, dim):
    """
    Determines the RAS axes order and directions for an affine matrix.

    Parameters:
    aff (numpy.ndarray): The affine transformation matrix.
    dim (int): The number of dimensions (e.g., 3 for 3D space).

    Returns:
    aff_ras: Index of the dominant axis.
    directions: Directon of the dominant axis (+1 or -1).
    """
    aff_inv = np.linalg.inv(aff)
    aff_ras = np.argmax(np.abs(aff_inv[:dim, :dim]), axis=1)
    directions = np.sign(aff_inv[np.arange(dim), aff_ras])

    return aff_ras, directions


def find_largest_cluster(binary_volume, min_size=None, max_n_cluster=1):
    """
    Finds up to max_n_cluster largest connected clusters in a binary 3D volume, optionally above min_size.

    Parameters:
        binary_volume (numpy.ndarray): A 3D binary numpy array (0 and 1).
        min_size (int, optional): Minimum cluster size to include (in voxels). If None, no threshold.
        max_n_cluster (int, optional): Maximum number of clusters to include. If None, all clusters above min_size.

    Returns:
        cluster_mask (numpy.ndarray): Binary 3D numpy array containing selected clusters.
    """
    # Label connected components
    labeled_volume, num_features = label(binary_volume)
    if num_features == 0:
        raise ValueError("No clusters found in the binary volume.")

    # Get sizes of all components
    component_sizes = np.bincount(labeled_volume.ravel())
    component_sizes[0] = 0  # background

    # Filter by min_size if specified
    valid_labels = np.arange(len(component_sizes))
    if min_size is not None:
        keep = component_sizes >= min_size
        keep[0] = False  # never keep background
        valid_labels = valid_labels[keep]
    else:
        valid_labels = valid_labels[component_sizes > 0]

    # Sort clusters by size, descending
    sorted_labels = valid_labels[np.argsort(component_sizes[valid_labels])[::-1]]

    # Apply max_n_cluster if set
    if max_n_cluster is not None:
        sorted_labels = sorted_labels[:max_n_cluster]
    if sorted_labels.size == 0:
        raise ValueError("No clusters meet the min_size/max_n_cluster criteria.")

    # Build output mask
    cluster_mask = np.isin(labeled_volume, sorted_labels)
    return cluster_mask


def cleanup(gm0, wm0, csf0, threshold_wm=0.4, cerebellum=None, csf_TPM=None):
    """
    Perform cleanup operations on CSF (Cerebrospinal Fluid), GM (Gray Matter),
    and WM (White Matter) maps to refine their segmentation by isolating clusters
    (e.g. removing vessels).

    Parameters:
        gm0 (nibabel.Nifti1Image): Nifti map representing the GM probability map.
        wm0 (nibabel.Nifti1Image): Nifti map representing the WM probability map.
        csf0 (nibabel.Nifti1Image): Nifti map representing the CSF probability map.
        threshold_wm (float): Initial threshold for isolating WM
        cerebellum (tuple): 3D array defining cerebellum
        csf_TPM (tuple): 3D array defining tissue probability map (TPM)
        of CSF

    Returns:
        tuple: Updated CSF, GM, and WM Nifti maps and label after cleanup.

    Steps:
        1. Identify the largest WM cluster to isolate the main WM structure.
        2. Dilate and close the CSF mask to refine its boundaries.
        3. Create a mask of largest WM cluster.
        4. Adjust the CSF and WM maps using the mask to correct the overlaps.
        5. Apply morphological opening to the GM-WM map to smooth boundaries.
        6. Retain only the largest cluster in the GM-WM map and correct the GM map.
    """

    gm = gm0.get_fdata().copy()
    wm = wm0.get_fdata().copy()
    csf = csf0.get_fdata().copy()

    # Identify the largest WM cluster to isolate the main WM structure
    wm_morph = find_largest_cluster(wm > threshold_wm)
    wm_morph = binary_dilation(wm_morph, generate_binary_structure(3, 3), iterations=1)

    # Create a mask that isolates WM clusters
    mask = ~wm_morph

    # Additionally restrict mask to areas outside cerebellum
    if cerebellum is not None:
        mask = mask & (cerebellum == 0)

    # Additionally restrict mask to CSF areas defined in tissue probability map
    if csf_TPM is not None:
        mask = mask & (csf_TPM >= 0.025 * 255)

    # Adjust CSF and WM maps using the mask
    # Add the WM contribution to the GM and set WM to zero in masked regions
    gm[mask] += wm[mask]
    wm[mask] = 0

    # Perform cleanup of the combined GM-WM map using morphological opening
    gm_wm = grey_opening(gm + wm, size=[3, 3, 3])

    # Retain only the largest GM-WM cluster
    gm_wm = find_largest_cluster(gm_wm > 0.5)

    # Additionally restrict mask to areas outside cerebellum
    if cerebellum is not None:
        gm_wm = gm_wm | (cerebellum > 0)

    # Add the GM contribution to the CSF and set GM to zero in masked regions
    csf[~gm_wm] += gm[~gm_wm]
    gm[~gm_wm] = 0

    # Compute label
    label = csf + 2 * gm + 3 * wm

    gm = nib.Nifti1Image(gm, gm0.affine, gm0.header)
    wm = nib.Nifti1Image(wm, wm0.affine, wm0.header)
    csf = nib.Nifti1Image(csf, csf0.affine, csf0.header)
    label = nib.Nifti1Image(label, gm0.affine, gm0.header)

    return label, gm, wm, csf


def piecewise_linear_scaling(input_img, label_img):
    """
    Piecewise linear scaling of an intensity image based on reference label regions.

    This function performs a piecewise linear transformation of the input image,
    using median intensity values from regions defined by a label image as breakpoints.
    Each segment between breakpoints is mapped linearly to a corresponding interval
    in the `target_values` array. The final result is normalized by dividing by 3.

    Parameters
    ----------
    input_img : np.ndarray
        Input image (e.g., intensity or probability map), 1D or ND.
    label_img : np.ndarray
        Label image of same shape as input_img, with integer class labels
        (0=background, 1=CSF, 2=GM, 3=WM).

    Returns
    -------
    Ym : np.ndarray
        Piecewise linearly scaled image, normalized to the range [0, 1] (if input covers full range).

    Notes
    -----
    - Median intensities of input_img are computed within regions defined by the labels:
        - label==0 and intensity < 0.9*median(CSF): Background (BG)
        - label==1: CSF
        - label==2: GM
        - label==3: WM
        - label==4: WM+ (extrapolated as median_WM + (median_WM - median_GM))
    - The breakpoints for piecewise scaling are set to these median values.
    - Each interval [median_i-1, median_i] is mapped linearly to [target_i-1, target_i].
    - For values above the last median (WM+), the mapping continues linearly.
    - Output is normalized by dividing by 3, so that the range maps to [0, 1.33].
    """

    # Define output target values for each class interval (BG, CSF, GM, WM, WM+)
    target_values = np.arange(0, 5)  # [0, 1, 2, 3, 4]
    Ym = input_img.copy().astype(float)
    N = len(target_values)

    # Compute medians for class-specific reference regions
    median_input = {}
    # CSF, GM, WM peaks
    for k in [1, 2, 3]:
        mask = np.abs(label_img - k) < 0.01
        median_input[k] = np.median(input_img[mask])
    # BG peak: label==0 and much lower than CSF
    mask = (label_img == 0) & (input_img < 0.9 * median_input[1])  # BG mask
    median_input[0] = np.median(input_img[mask])
    # Extrapolated "WM+" for values beyond WM (e.g., fat, vessels)
    median_input[4] = median_input[3] + (median_input[3] - median_input[2])

    # Piecewise linear mapping for each interval [median_input[i-1], median_input[i]]
    for i in range(1, N):
        mask = (input_img > median_input[i - 1]) & (input_img <= median_input[i])
        Ym[mask] = target_values[i - 1] + (input_img[mask] - median_input[i - 1]) / (
            median_input[i] - median_input[i - 1]
        ) * (target_values[i] - target_values[i - 1])
    # For outliers above the last median, extrapolate linearly using last segment's slope
    mask = input_img >= median_input[4]
    slope = (target_values[4] - target_values[3]) / (median_input[4] - median_input[3])
    Ym[mask] = target_values[4] + (input_img[mask] - median_input[4]) * slope

    # Normalize so WM (target 3) maps to 1.0
    Ym = Ym / 3
    return Ym


def correct_bias_field(brain, seg, steps=1000, spacing=1.0, get_discrepancy=False):
    """
    Applies bias field correction to an input brain image.

    Args:
        brain (nibabel.Nifti1Image): Input brain image.
        seg (nibabel.Nifti1Image): Segmentation mask.

    Returns:
        tuple: Bias field-corrected brain image and bias field.
    """

    # Defaults
    subdivide = True
    bcl = True
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.2
    subsamp = 5
    stopthr = 5e-4

    # Process voxel sizes and mask for brain segmentation
    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata().copy()
    seg0 = seg.get_fdata().copy()

    # Generate mask based on segmentation or brain data
    mask = seg0 >= 2.75

    # Subsampling for efficiency
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

    # Prepare data and parameters for bias correction
    datalog = dataSub.astype(np.float32)
    datalog[wm_mask] = np.log(datalog[wm_mask])
    datalog[np.logical_not(wm_mask)] = 0
    datalogmasked = datalog[wm_mask]
    fit_data = np.zeros_like(datalog)
    datalogmaskedcur = np.copy(datalogmasked)

    # Descending FWHM scheme
    levels = [lvl for lvl in range(maxlevel) for _ in range(steps)]

    # At some point will have to generalise into fwhm and subdivision
    # level scheme, at the moment it's either or:
    levelfwhm = (
        fwhm / (np.arange(maxlevel) + 1) if not subdivide else fwhm * np.ones(maxlevel)
    )

    splsm3d = SplineSmooth3DUnregularized(
        datalog, dataSubVoxSize, spacing, domainMethod="minc", mask=wm_mask
    )

    # Prediction interpolator
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
        thisFWHM = levelfwhm[levels[N]]  # * math.sqrt(8*math.log(2))
        thisSD = thisFWHM / math.sqrt(8 * math.log(2))
        mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)

        histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
        histfiltclip = np.clip(histfilt, 0, None)

        uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
        datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
        logbc = datalogmaskedcur - datalogmaskedupd
        meanadj = True
        if meanadj:
            logbc = logbc - np.mean(logbc)
        usegausspde = True

        # Need masking!
        fit_data[wm_mask] = logbc
        splsm3d.fit(fit_data, reportingLevel=0)
        log_bias_field = splsm3d.predict()
        log_bias_masked = log_bias_field[wm_mask]

        correction = log_bias_masked

        bcratio = np.exp(correction)
        ratiomean = bcratio.mean()
        ratiosd = bcratio.std()
        conv = ratiosd / ratiomean

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
            # Applies to both cumulative and normal iterative
            # mode, in normal iterative mode we're just upgrading
            # to a finer mesh for the following updates.
            # In cumulative mode we first get the current cumulative
            # estimate before refining.
            splsm3d.P = controlField
            splsm3d = splsm3d.promote()
            predictor = predictor.promote()
            controlField = splsm3d.P

    splsm3d.P = controlField

    predictor.P = splsm3d.P

    # Apply computed bias field correction
    bias0 = np.exp(predictor.predict())

    # apply nu-correction
    tissue_idx = bias0 != 0
    brain0[tissue_idx] /= bias0[tissue_idx]

    brain0 = piecewise_linear_scaling(brain0, seg0)

    brain_normalized = nib.Nifti1Image(brain0, brain.affine, brain.header)

    return brain_normalized


def fit_intensity_field(
    brain, seg, limit=None, steps=1000, spacing=1.0, stopthr=5e-4, use_prctile=3
):
    """
    Estimate a smooth, bias-like intensity field in a specified tissue mask using log-domain smoothing
    and optional robust percentile-based outlier exclusion.

    Parameters
    ----------
    brain : nibabel.Nifti1Image
        Input brain image.
    seg : nibabel.Nifti1Image
        Segmentation mask. Voxels with values in the interval ``(limit[0], limit[1]]``
        are considered part of the mask (e.g., white matter by default).
    limit : list or tuple of float, optional
        Lower and upper thresholds for the segmentation mask (default: [2.75, 3]).
    steps : int, optional
        Number of iterations per mesh level (default: 1000).
    spacing : float, optional
        Knot spacing for the spline representation (default: 0.75).
    stopthr : float, optional
        Stopping threshold for convergence, defined as ratio of std/mean (default: 1e-4).
    use_prctile : int, optional
        How to restrict the mask based on data percentiles:
        0 = no restriction,
        1 = remove values below 5th percentile,
        2 = remove values above 95th percentile,
        3 = restrict to 5th-95th percentile (default: 3).

    Returns
    -------
    field : np.ndarray
        3D array of the smooth intensity field (same shape as input image).
        The field is on the same scale as the median intensity in the original mask.
    """

    if limit is None:
        limit = [2.75, 3]
    if not (isinstance(limit, (list, tuple)) and len(limit) == 2):
        raise ValueError("limit must be a 2-element list or tuple")

    # Parameters
    subdivide = True
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.2
    subsamp = 5

    # Prepare data and mask
    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata().copy()
    seg0 = seg.get_fdata().copy()
    mask = (seg0 > limit[0]) & (seg0 <= limit[1])

    # Subsampling for efficiency
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

    # Percentile-based mask refinement (optional, robust outlier rejection)
    if use_prctile == 3:
        p5, p95 = np.percentile(dataSub[maskSub], [5, 95])
        maskSub = maskSub & (dataSub > p5) & (dataSub < p95)
    elif use_prctile == 2:
        p95 = np.percentile(dataSub[maskSub], 95)
        maskSub = maskSub & (dataSub < p95)
    elif use_prctile == 1:
        p5 = np.percentile(dataSub[maskSub], 5)
        maskSub = maskSub & (dataSub > p5)
    # If use_prctile==0, use maskSub as is.

    # save temporary maks images
    if False:
        masked_data_to_save = np.zeros_like(dataSub, dtype=np.float32)
        masked_data_to_save[maskSub] = dataSub[maskSub]
        limit_str = "_".join([f"{l:.2f}" for l in limit])
        tmp_filename = f"tmp_maskedsrc_limit_{limit_str}.nii"

        if subsamp:
            # Adjust the affine for subsampled data
            orig_affine = brain.affine.copy()
            new_affine = orig_affine.copy()
            for d in range(3):
                new_affine[:3, d] *= subsamp  # scale voxel sizes
            # The header shape and zooms will be auto-derived from the array/affine.
        else:
            new_affine = brain.affine

        nib.save(
            nib.Nifti1Image(masked_data_to_save, new_affine, brain.header),
            tmp_filename,
        )
        print(f"Saved masked source image for fitting: {tmp_filename}")

    datalog = dataSub.astype(np.float32)

    # Sanity check: Ensure no nonpositive values inside mask before log
    if np.any(datalog[maskSub] <= 0):
        raise ValueError(
            "Non-positive values found in the masked data. Adjust mask or preprocess the image."
        )

    datalog[maskSub] = np.log(datalog[maskSub])
    datalog[np.logical_not(maskSub)] = 0
    datalogmasked = datalog[maskSub]
    fit_data = np.zeros_like(datalog)
    datalogmaskedcur = np.copy(datalogmasked)

    # Level and smoothing schedule
    levels = [lvl for lvl in range(maxlevel) for _ in range(steps)]
    levelfwhm = (
        fwhm / (np.arange(maxlevel) + 1) if not subdivide else fwhm * np.ones(maxlevel)
    )

    # Initialize spline field
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

        # Convergence: as in bias field correction
        correction = diff_masked
        bcratio = np.exp(correction)
        ratiomean = bcratio.mean()
        ratiosd = bcratio.std()
        conv = ratiosd / ratiomean

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

    # Final field in original intensity space
    splsm3d.P = controlField
    predictor.P = splsm3d.P
    field = np.exp(predictor.predict())

    # Rescale field to match the original mask's median intensity
    mean_raw = np.median(brain0[mask])
    mean_field = np.median(field[mask])
    field = field * (mean_raw / mean_field)

    return field


def apply_LAS(t1, label):
    """
    Apply Local Adaptive Segmentation (LAS) to a brain MRI image.

    This method estimates local intensity fields for CSF, GM, and WM,
    then assigns each voxel a normalized, continuous tissue class value
    based on the local intensities. Output values are in the range [0, 1].

    Parameters
    ----------
    t1 : nibabel.Nifti1Image
        Input T1-weighted MRI image.
    label : nibabel.Nifti1Image
        Segmentation map for defining tissue-specific masks.

    Returns
    -------
    brain_normalized : nibabel.Nifti1Image
        LAS-normalized image, with values in [0, 1].
        - 0 = background, 1 = white matter, intermediate values = CSF/GM.
    """

    eps = np.finfo(float).eps  # machine epsilon for float
    stopthr = 5e-4
    spacing = 1.0

    Ysrc = t1.get_fdata().copy()
    minYsrc = np.min(Ysrc)

    # Fit smooth intensity fields for each tissue
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

    # Ensure shapes match
    if not (fit_csf.shape == fit_gm.shape == fit_wm.shape == Ysrc.shape):
        raise ValueError("All fitted fields and source image must have the same shape.")

    # WM: scale highest tissue (WM in T1w)
    mask_wm = Ysrc >= fit_wm
    Yml += mask_wm * (3 + (Ysrc - fit_wm) / np.maximum(eps, fit_wm - fit_csf))

    # GM: scale second highest tissue (GM in T1w)
    mask_gm = (Ysrc >= fit_gm) & (Ysrc < fit_wm)
    Yml += mask_gm * (2 + (Ysrc - fit_gm) / np.maximum(eps, fit_wm - fit_gm))

    # CSF: scale third highest tissue (CSF in T1w)
    mask_csf = (Ysrc >= fit_csf) & (Ysrc < fit_gm)
    Yml += mask_csf * (1 + (Ysrc - fit_csf) / np.maximum(eps, fit_gm - fit_csf))

    # Background: scale below CSF
    mask_bg = Ysrc < fit_csf
    Yml += mask_bg * ((Ysrc - minYsrc) / np.maximum(eps, fit_csf - minYsrc))

    Yml[Yml < 0.25] = 0

    # Normalize to [0, 1]
    brain_normalized = nib.Nifti1Image(Yml / 3, t1.affine, t1.header)

    return brain_normalized


def correct_label_map(brain, seg):
    """
    Apply local corrections to a label map and its associated brain image
    based on local intensity/label discrepancies.

    This function performs two main corrections:
      1. For white matter voxels (seg > 2.5) where the intensity is unexpectedly low,
         the label value is downscaled to match intensity.
      2. For rim (boundary) voxels labeled as CSF (seg < 1.5) but with intensity higher than expected,
         the brain intensity is downscaled in these rim areas.

    Parameters
    ----------
    brain : nibabel.Nifti1Image
        Input brain image (3D).
    seg : nibabel.Nifti1Image
        Segmentation label map (3D, integer or float).

    Returns
    -------
    seg_corrected : nibabel.Nifti1Image
        Corrected segmentation image.
    brain_corrected : nibabel.Nifti1Image
        Corrected brain image.
    """

    # Defensive copies to avoid mutating input data
    brain0 = brain.get_fdata().copy()
    seg0 = seg.get_fdata().copy()

    # Compute a local discrepancy map between intensity and label value
    discrepancy0 = (1 + brain0 * 3) / (1 + seg0)
    discrepancy0 = median_filter(discrepancy0, size=3)

    # 1. For WM: If intensity is low for a high label, reduce the label value
    wm_mask = (seg0 > 2.5) & (discrepancy0 < 1)
    # consistent with original (*discrepancy*discrepancy)
    seg0[wm_mask] *= discrepancy0[wm_mask] ** 2

    # For voxels labeled CSF but intensity is high, scale down brain intensity
    csf_mask = (seg0 < 1.5) & (discrepancy0 > 1)
    brain0[csf_mask] /= discrepancy0[csf_mask] ** 2

    # For voxels labeled GM but intensity is slightly below or above threshold,
    # scale brain intensity
    gm_mask1 = (seg0 > 1.5) & (seg0 <= 2)
    brain0[gm_mask1 & (brain0 > 1.4 / 3) & (brain0 <= 1.6 / 3)] = 1.6 / 3
    gm_mask2 = (seg0 > 2) & (seg0 <= 2.5)
    brain0[gm_mask2 & (brain0 > 2.4 / 3) & (brain0 <= 2.6 / 3)] = 2.4 / 3

    # Wrap outputs in Nifti images
    seg_corrected = nib.Nifti1Image(seg0, seg.affine, seg.header)
    brain_corrected = nib.Nifti1Image(brain0, brain.affine, brain.header)

    return seg_corrected, brain_corrected


def unsmooth_kernel(factor=3.0, sigma=0.6, device="cpu"):
    # Hand-optimized factor and sigma for compensation of smoothing caused by affine transformation (inspired by CAT12)
    kernel = -factor * smooth_kernel(
        kernel_size=3 * [3], sigma=torch.tensor(3 * [sigma], device=device)
    )
    kernel[1, 1, 1] = 0
    kernel[1, 1, 1] = 1 - kernel.sum()
    return kernel


def get_atlas(
    t1, affine, target_header, target_affine, atlas_name, warp_yx=None, device="cpu"
):
    """
    Generates an atlas-aligned image for brain segmentation.

    Args:
        t1 (nibabel.Nifti1Image): Input T1-weighted image.
        affine (numpy.ndarray): Affine transformation matrix.
        warp_yx (nibabel.Nifti1Image, optional): Warp field for alignment. Defaults to None.
        target_header (nibabel.Nifti1Image.header): Header for target image.
        target_affine (numpy.ndarray): Affine transformation matrix for target image.
        atlas_name (str): Name of the atlas template.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        nibabel.Nifti1Image: Aligned atlas image.
    """

    # Extract headers and affine transformations
    header = target_header
    dim_hdr = target_header["dim"][1:4]
    dim = tuple(int(x) for x in dim_hdr)
    transform = target_affine

    # Load the atlas template
    atlas = nib.as_closest_canonical(
        nib.load(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/{atlas_name}.nii.gz")
    )

    atlas_register = AtlasRegistration()

    if warp_yx is not None:
        # Perform affine and warp-based registration
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(device)

        # Compute the shape for interpolation
        shape = tuple(shape_from_to(atlas, warp_yx))
        scaled_yx = F.interpolate(
            yx.permute(0, 4, 1, 2, 3), shape, mode="trilinear", align_corners=False
        )
        warps = {shape: scaled_yx.permute(0, 2, 3, 4, 1)}

        # Apply registration
        atlas = atlas_register(affine, warps[shape], atlas, t1.shape)

    # Convert atlas to tensor and interpolate to match segmentation shape
    atlas_tensor = nifti_to_tensor(atlas)[None, None].to(device)
    atlas_tensor = F.interpolate(atlas_tensor, dim, mode="nearest")[0, 0]

    # Choose dtype based on max value
    atlas_tensor = atlas_tensor.type(
        torch.uint8 if atlas_tensor.max() < 256 else torch.int16
    )

    # Convert back to CPU numpy array and return as Nifti
    atlas_np = atlas_tensor.cpu().numpy()
    return nib.Nifti1Image(atlas_np, transform, header)


def crop_nifti_image_with_border(img, border=5, threshold=0):
    """
    Crop a NIfTI image to the smallest bounding box containing non-zero values,
    add a border, ensure the resulting dimensions are even, and update the affine
    and header to preserve the spatial origin.

    Parameters:
    img (nib.Nifti1Image): Input NIfTI image
    border (int): Number of voxels to add as a border (default=5)

    Returns:
    nib.Nifti1Image: Cropped and padded NIfTI image with updated affine and header
    """
    # Load image data, affine, and header
    data = img.get_fdata().copy()
    affine = img.affine
    header = img.header

    # Find the bounding box of non-zero values
    mask = data > threshold
    coords = np.array(np.where(mask))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1)

    # Add border
    min_coords = np.maximum(min_coords - border, 0)
    max_coords = np.minimum(
        max_coords + border + 1, data.shape
    )  # +1 for inclusive index

    # Crop the data
    cropped_data = data[
        min_coords[0] : max_coords[0],
        min_coords[1] : max_coords[1],
        min_coords[2] : max_coords[2],
    ]

    # Ensure even dimensions
    pad_x = (0, (cropped_data.shape[0] % 2))  # Pad 1 voxel if odd
    pad_y = (0, (cropped_data.shape[1] % 2))
    pad_z = (0, (cropped_data.shape[2] % 2))
    cropped_data = np.pad(cropped_data, (pad_x, pad_y, pad_z))

    # Update affine matrix to keep the origin
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] += np.dot(affine[:3, :3], min_coords)

    # Create a new NIfTI image
    cropped_img = nib.Nifti1Image(cropped_data, affine=cropped_affine, header=header)

    # Update header dimensions
    cropped_img.header.set_data_shape(cropped_data.shape)

    return cropped_img


def resample_and_save_nifti(
    nifti_obj, grid, affine, header, out_name, align=None, crop=None, clip=None
):
    """
    Saves a NIfTI object with resampling and reorientation.

    Args:
        nifti_obj: The input NIfTI object to process.
        affine: affine matrix for saved file.
        header: header for saved file.
        reference: Reference containing affine and header for reorientation.
        out_name: Output filename.

    Returns:
        None
    """

    # Step 1: Convert NIfTI to tensor and add batch/channel dimensions
    tensor = nifti_to_tensor(nifti_obj)[None, None]

    # Step 2: Resample using grid
    tensor = F.grid_sample(tensor, grid, align_corners=INTERP_KWARGS["align_corners"])[
        0, 0
    ]

    if clip is not None:
        if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
            raise ValueError("limit must be a 2-element list or tuple")
        tensor = torch.clamp(tensor, min=clip[0], max=clip[1])

    # Step 3: Reorient and save as NIfTI
    if align:
        tensor, tmp1, tmp2, tmp3 = align_brain(
            tensor.cpu().numpy(), affine, header, np.eye(4), do_flip=1
        )
        nii_data = nib.Nifti1Image(tensor, affine, header)
    else:
        nii_data = reoriented_nifti(tensor, affine, header)

    if crop:
        nii_data = crop_nifti_image_with_border(nii_data, threshold=1.1)

    nib.save(nii_data, out_name)


def get_resampled_header(header, aff, new_vox_size, ras_aff, reorder_method=1):
    """
    Adjust the NIfTI header and affine matrix for a new voxel size.

    Parameters:
    header : nibabel header object
        Header information of the input NIfTI image.
    aff : numpy.ndarray
        Affine transformation matrix of the input image.
    new_vox_size : numpy.ndarray
        Desired voxel size as a 3-element array [x, y, z] in mm.

    Returns:
    tuple:
        Updated header and affine transformation matrix.
    """

    header2 = header.copy()

    # Update dimensions and pixel sizes
    dim = header2["dim"]
    pixdim = header2["pixdim"]

    ras_ref, dirs_ref = get_ras(aff, 3)
    factor = pixdim[1:4] / new_vox_size
    reordered_factor = np.zeros_like(pixdim[1:4])
    for i, axis in enumerate(ras_ref):
        if reorder_method == 1:
            reordered_factor[i] = factor[np.where(ras_aff == axis)[0][0]]
        else:
            reordered_factor[axis] = (
                dirs_ref[i] * factor[i]
            )  # Adjust for axis direction
    factor = reordered_factor

    dim[1:4] = np.abs(np.round(dim[1:4] * factor))

    header2["dim"] = dim

    pixdim[1:4] = new_vox_size
    header2["pixdim"] = pixdim

    # Update affine matrix to match new voxel size
    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    # Update header transformation fields
    header2["srow_x"] = aff2[0, :]
    header2["srow_y"] = aff2[1, :]
    header2["srow_z"] = aff2[2, :]
    header2["qoffset_x"] = aff2[0, 3]
    header2["qoffset_y"] = aff2[1, 3]
    header2["qoffset_z"] = aff2[2, 3]

    return header2, aff2


def get_cerebellum(atlas):
    """
    Get labeled volume using IBSR atlas with Ones in cerebellum

    Parameters:
    atlas : Nifti1Image
        Atlas template for anatomical regions.

    Returns:
    tuple:
        Labeled volume with Ones in cerebellum.
    """

    rois = pd.read_csv(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv", sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))

    atlas = atlas.get_fdata().copy()

    # get ceerebellum
    cerebellum = (
        (atlas == regions["lCbeWM"])
        | (atlas == regions["lCbeGM"])
        | (atlas == regions["rCbeWM"])
        | (atlas == regions["rCbeGM"])
    )

    return cerebellum


def get_partition(p0_large, atlas):
    """
    Partition the input volume into left and right hemispheres using IBSR atlas
    with labels for CSF, GM, and WM.

    Parameters:
    p0_large : Nifti1Image
        Input probability map from segmentation.
    atlas : Nifti1Image
        Atlas template for anatomical regions.

    Returns:
    tuple:
        Left hemisphere (lh) and right hemisphere (rh) labeled volumes.
    """

    rois = pd.read_csv(f"{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv", sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))

    atlas_data = atlas.get_fdata().copy()
    atlas_mask = atlas_data > 0
    atlas_mask = binary_dilation(atlas_mask, generate_binary_structure(3, 3), 3)

    p0_data = p0_large.get_fdata().copy()
    gm = (p0_data > 1.5) & (p0_data < 2.5)
    gm_regions = ["lCbrGM", "rCbrGM", "lAmy", "lHip", "rAmy", "rHip"]

    # Create cerebral GM mask
    gm_mask = np.isin(atlas_data, [regions[r] for r in gm_regions])
    gm_mask = binary_dilation(gm_mask, generate_binary_structure(3, 3), 2)
    gm = gm & gm_mask

    # Define left and right hemisphere regions
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
    right_regions = [
        r.replace("l", "r", 1) for r in left_regions
    ]  # Replace only first 'l'

    # Create left/right masks
    left = np.isin(atlas_data, [regions[r] for r in left_regions])
    right = np.isin(atlas_data, [regions[r] for r in right_regions])

    # Clean left hemisphere by opening and closing
    left = binary_opening(left, generate_binary_structure(3, 3), 3)
    left = binary_closing(left, generate_binary_structure(3, 3), 3)

    # Process hemispheres: dilation and closing to refine boundaries
    lh = binary_dilation(left, generate_binary_structure(3, 3), 5) & ~right
    rh = binary_dilation(right, generate_binary_structure(3, 3), 5) & ~left

    left = binary_closing(lh, generate_binary_structure(3, 3), 2) & ~rh
    right = binary_closing(rh, generate_binary_structure(3, 3), 2) & ~left

    # Define regions to exclude
    excl_regions = ["lCbeWM", "lCbeGM", "rCbeWM", "rCbeGM", "b3thVen", "b4thVen"]

    # Create masks
    exclude = np.isin(atlas_data, [regions[r] for r in excl_regions])
    exclude = binary_dilation(exclude, generate_binary_structure(3, 3), 1)
    exclude = exclude | binary_dilation(
        np.isin(atlas_data, regions["bBst"]), generate_binary_structure(3, 3), 5
    )
    exclude = exclude | ~atlas_mask

    # Define regions that should be filled with WM
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
    wm_fill = binary_dilation(wm_fill, generate_binary_structure(3, 3), 10)

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    lh = np.copy(p0_data)
    lh[lh < 1] = 1
    lh[wm_fill & ~gm_mask] = 3
    lh[exclude | right] = 1

    rh = np.copy(p0_data)
    rh[rh < 1] = 1
    rh[wm_fill & ~gm_mask] = 3
    rh[exclude | left] = 1

    # Finally remove small non-connected parts from hemi maps
    mask = (lh > 1) | (rh > 1)
    mask = binary_closing(mask, generate_binary_structure(3, 3), 1)
    mask = binary_opening(mask, generate_binary_structure(3, 3), 3)
    mask = find_largest_cluster(mask)
    mask = binary_dilation(mask, generate_binary_structure(3, 3), 1)
    lh[~mask] = 1
    rh[~mask] = 1

    return lh, rh


def align_brain(data, aff, header, aff_ref, do_flip=1):
    """Align ``data`` to the orientation defined by ``aff_ref``.

    Parameters
    ----------
    data : ndarray
        Input image volume.
    aff : ndarray
        Affine matrix describing ``data``.
    header : nibabel.Nifti1Header
        NIfTI header to update.
    aff_ref : ndarray
        Reference affine defining the desired orientation.
    do_flip : int, optional
        If ``True`` (default) flip axes to match the reference orientation.

    Returns
    -------
    tuple
        ``(aligned_data, aff, header, ras_aff)`` where ``ras_aff`` describes the
        RAS orientation of the input volume.
    """

    dim = 3
    ras_aff, dirs_aff = get_ras(aff, dim)
    ras_ref, dirs_ref = get_ras(aff_ref, dim)

    # Step 1: Reorder the rotation-scaling part (3x3) to match reference axes
    reordered_aff = np.zeros_like(aff)

    # Reorder the rotation-scaling part (3x3) to match reference axes and directions
    if False:
        for i, axis_ref in enumerate(ras_ref):
            # Find the corresponding axis in the input affine matrix
            matching_axis_idx = np.where(ras_aff == axis_ref)[0][0]
            reordered_aff[:dim, i] = (
                dirs_ref[i] * dirs_aff[matching_axis_idx] * aff[:dim, matching_axis_idx]
            )

    else:
        for i, axis in enumerate(ras_ref):
            reordered_aff[:dim, i] = aff[:dim, np.where(ras_aff == axis)[0][0]]

    # Copy the translation vector
    reordered_aff[:dim, 3] = aff[:dim, 3]
    reordered_aff[3, :] = [0, 0, 0, 1]  # Ensure the bottom row remains [0, 0, 0, 1]

    header["srow_x"] = reordered_aff[0, :]
    header["srow_y"] = reordered_aff[1, :]
    header["srow_z"] = reordered_aff[2, :]
    header["qoffset_x"] = reordered_aff[0, 3]
    header["qoffset_y"] = reordered_aff[1, 3]
    header["qoffset_z"] = reordered_aff[2, 3]

    # Update the affine matrix after reordering
    aff = reordered_aff

    # Step 2: Transpose the data axes to match the reference
    align_ax = [np.where(ras_aff == axis)[0][0] for axis in ras_ref]
    aligned_data = np.transpose(data, axes=align_ax)

    # Step 5: Flip image axes if necessary
    if do_flip:
        dot_products = np.sum(aff[:dim, :dim] * aff_ref[:dim, :dim], axis=0)
        for i in range(dim):
            if dot_products[i] < 0:
                aligned_data = np.flip(aligned_data, axis=i)

    return aligned_data, aff, header, ras_aff
