import os
import torch
import warnings
import math
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
    "Atlas_ROI",
    "Report_file",
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


def get_volume_native_space(vol_nifti, affine_values):
    """
    Estimate GM volume in native/original space (cm³) from a registered probability
    NIfTI and the 4x4 affine matrix that maps native to registered space.

    Args:
        vol_nifti: nibabel.Nifti1Image, probability map in registered space
        affine_values: (4, 4) numpy array, affine matrix (native->registered)

    Returns:
        volume in native/original space in cm³ (float)
    """

    if vol_nifti is None:
        return 0

    vol_prob = vol_nifti.get_fdata()
    vol_sum = np.sum(vol_prob)
    # Voxel volume in target/registered space (mm³)
    voxel_volume_target = abs(np.linalg.det(vol_nifti.affine[:3, :3]))
    # Volume in registered space (mm³)
    volume__target = vol_sum * voxel_volume_target
    # Affine scaling (native->target)
    scaling = abs(np.linalg.det(affine_values[:3, :3]))
    # Volume in native space (mm³)
    volume__native_mm3 = volume__target * scaling
    # Convert mm³ to cm³
    volume__native_cm3 = volume__native_mm3 / 1000.0
    return volume__native_cm3
