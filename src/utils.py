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
from deepbet.utils import reoriented_nifti
from deepmriprep.utils import DEVICE, nifti_to_tensor
from deepmriprep.atlas import shape_from_to, AtlasRegistration
from torchreg.utils import INTERP_KWARGS
from scipy.ndimage import binary_opening, binary_dilation, grey_opening, binary_closing, generate_binary_structure, label
from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from pathlib import Path
from skimage import filters

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

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
    prog = '█' * elapsed
    remaining = ' ' * (total - elapsed)
    
    # Format the name with padding
    name = name.ljust(50)
    
    # Print the progress bar with percentage and name
    print(f'{prog}{remaining} {elapsed}/{total} {name}\r', end='')
    
    if (elapsed == total):
        spaces = ' ' * 100
        print(f'{spaces}\r', end='')
    
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

def find_largest_cluster(binary_volume):
    """
    Finds the largest connected cluster in a binary 3D volume.

    Parameters:
        binary_volume (numpy.ndarray): A 3D binary numpy array (0 and 1).

    Returns:
        largest_cluster (numpy.ndarray): A binary 3D numpy array containing only the largest cluster.
    """
    # Label connected components
    labeled_volume, num_features = label(binary_volume)
    
    if num_features == 0:
        raise ValueError("No clusters found in the binary volume.")
    
    # Find the sizes of each connected component
    component_sizes = np.bincount(labeled_volume.ravel())
    
    # The background (label 0) is not a cluster, so ignore it
    component_sizes[0] = 0
    
    # Identify the largest component by size
    largest_component_label = component_sizes.argmax()
    
    # Create a binary mask for the largest component
    largest_cluster = labeled_volume == largest_component_label
    
    return largest_cluster

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

    gm  = gm0.get_fdata()
    wm  = wm0.get_fdata()
    csf = csf0.get_fdata()

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
        mask = mask & (csf_TPM >= 0.025*255)

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
    label = csf + 2*gm + 3*wm

    gm  = nib.Nifti1Image(gm, gm0.affine, gm0.header)
    wm  = nib.Nifti1Image(wm, wm0.affine, wm0.header)
    csf = nib.Nifti1Image(csf, csf0.affine, csf0.header)
    label  = nib.Nifti1Image(label, gm0.affine, gm0.header)

    return label, gm, wm, csf

def correct_bias_field(brain, seg, steps=1000):
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
    accumulate = True
    bcl = True
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.05
    subsamp = 5
    stopthr = 1e-4
    spacing = 1

    # Process voxel sizes and mask for brain segmentation
    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata()
    
    # Generate mask based on segmentation or brain data
    if (seg is None) :
        print("No mask")
        mask = np.ones(brain0.shape) > 0
        _thresh = filters.threshold_otsu(brain0[mask])
        print(_thresh)
        mask = np.logical_and(brain0 > _thresh, mask)
    else:
        mask = (seg.get_fdata() > 2.5)

    # Subsampling for efficiency
    if subsamp:
        offset = 0
        dataSub = brain0[offset::subsamp,offset::subsamp,offset::subsamp]
        mask = mask[offset::subsamp,offset::subsamp,offset::subsamp]
        dataSubVoxSize = dataVoxSize * subsamp
    
    dataSubVoxSize = 1 / (np.array(dataSub.shape) -1)
    dataVoxSize = dataSubVoxSize / subsamp

    # Prepare data and parameters for bias correction
    datalog = dataSub.astype(np.float32)
    datalog[mask] = np.log(datalog[mask])
    datalog[np.logical_not(mask)] = 0
    datalogmasked = datalog[mask]
    datafill = np.zeros_like(datalog)    
    datalogmaskedcur = np.copy(datalogmasked)

    # Descending FWHM scheme
    levels=[ lvl for lvl in range(maxlevel) for _ in range(steps) ]
    # At some point will have to generalise into fwhm and subdivision
    # level scheme, at the moment it's either or:
    levelfwhm = fwhm / (np.arange(maxlevel) + 1) if not subdivide else fwhm * np.ones(maxlevel)
    
    splsm3d = SplineSmooth3DUnregularized(
        datalog, dataSubVoxSize, spacing, domainMethod="minc", mask=mask)
    
    # Prediction interpolator
    predictor = SplineSmooth3D(
        brain0, dataVoxSize, spacing, knts=splsm3d.kntsArr, dofit=False)
    lastinterpbc = np.zeros(datalogmasked.shape[0])
    datalogcur = np.copy(datalog)
    nextlevel = 0
    
    controlField=None
    chosenkernelfn = kernelfntri
    
    for N in range(len(levels)):
        if levels[N] < nextlevel:
          continue
        
        hist, histvaledge, histval, histbinwidth = (
          distrib_kde(datalogmaskedcur, Nbins, kernfn=chosenkernelfn,
                      binCentreLimits=bcl))
        thisFWHM = levelfwhm[levels[N]] # * math.sqrt(8*math.log(2))
        thisSD = thisFWHM / math.sqrt(8*math.log(2))
        mfilt, mfiltx, mfiltmid, mfiltbins = symGaussFilt(thisSD, histbinwidth)
    
        histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
        histfiltclip = np.clip(histfilt,0,None)
    
        uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
        datalogmaskedupd = map_Eu_v(histval, uest, datalogmaskedcur)
        if accumulate:
          logbc = datalogmaskedcur - datalogmaskedupd
        else:
          logbc = datalogmasked - datalogmaskedupd
        meanadj=True
        if meanadj:
          logbc = logbc - np.mean(logbc)
        usegausspde=True
    
        # Need masking!
        datafill[mask] = logbc
        splsm3d.fit(datafill, reportingLevel=0)
        logbcsmfull = splsm3d.predict()
        logbcsm = logbcsmfull[mask]
    
        if accumulate:
            logbcratio = logbcsm
        else:
            logbcratio = logbcsm - lastinterpbc
            lastinterpbc = logbcsm
            
        bcratio = np.exp(logbcratio)
        ratiomean = bcratio.mean()
        ratiosd = bcratio.std()
        conv = ratiosd / ratiomean
    
        if accumulate:
            datalogmaskedcur = datalogmaskedcur - logbcsm
            if controlField is None:
                controlField  = splsm3d.P.copy()
            else:
                controlField += splsm3d.P
        else:
            datalogmaskedcur = datalogmasked - logbcsm
            
        datalogcur[mask] = datalogmaskedcur
        if (conv < stopthr):
            nextlevel = levels[N] + 1
            
        if (subdivide and (N+1)<len(levels) and (nextlevel>levels[N] or 
          levels[N+1] != levels[N])):
            # Applies to both cumulative and normal iterative
            # mode, in normal iterative mode we're just upgrading
            # to a finer mesh for the following updates.
            # In cumulative mode we first get the current cumulative
            # estimate before refining.
            if accumulate:
                splsm3d.P = controlField
            splsm3d = splsm3d.promote()
            predictor = predictor.promote()
            controlField = splsm3d.P
    
    if accumulate:
        splsm3d.P = controlField
    
    predictor.P = splsm3d.P

    # Apply computed bias field correction
    bias0 = np.exp(predictor.predict())

    # apply nu-correction
    tissue_idx = bias0 != 0
    brain0[tissue_idx] /= bias0[tissue_idx]

    brain = nib.Nifti1Image(brain0, brain.affine, brain.header)
    bias  = nib.Nifti1Image(bias0,  brain.affine, brain.header)
    
    return bias, brain

def get_atlas(t1, affine, p1_large, p2_large, p3_large, atlas_name, warp_yx=None, device='cpu'):
    """
    Generates an atlas-aligned image for brain segmentation.

    Args:
        t1 (nibabel.Nifti1Image): Input T1-weighted image.
        affine (numpy.ndarray): Affine transformation matrix.
        warp_yx (nibabel.Nifti1Image, optional): Warp field for alignment. Defaults to None.
        p1_large, p2_large, p3_large (nibabel.Nifti1Image): Probability maps for tissue segmentation.
        atlas_name (str): Name of the atlas template.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        nibabel.Nifti1Image: Aligned atlas image.
    """

    # Extract headers and affine transformations
    header = p1_large.header
    transform = p1_large.affine

    # Convert inputs to tensors
    p1_large, p2_large, p3_large = [nifti_to_tensor(p).to(device) for p in [p1_large, p2_large, p3_large]]

    # Load the atlas template
    atlas = nib.as_closest_canonical(
        nib.load(f'{DATA_PATH}/templates_MNI152NLin2009cAsym/{atlas_name}.nii.gz'))
        
    atlas_register = AtlasRegistration()

    if warp_yx is not None:
        # Perform affine and warp-based registration
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(device)

        # Compute the shape for interpolation
        shape = tuple(shape_from_to(atlas, warp_yx))
        scaled_yx = F.interpolate(
            yx.permute(0, 4, 1, 2, 3), shape, mode='trilinear', align_corners=False)
        warps = {shape: scaled_yx.permute(0, 2, 3, 4, 1)}
        
        # Apply registration
        atlas = atlas_register(affine, warps[shape], atlas, t1.shape)

    # Convert atlas to tensor and interpolate to match segmentation shape
    atlas_tensor = nifti_to_tensor(atlas)[None, None].to(device)
    atlas_tensor = F.interpolate(atlas_tensor, p1_large.shape, mode='nearest')[0, 0]

    # Choose dtype based on max value
    atlas_tensor = atlas_tensor.type(torch.uint8 if atlas_tensor.max() < 256 else torch.int16)

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
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Find the bounding box of non-zero values
    mask = data > threshold
    coords = np.array(np.where(mask))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1)

    # Add border
    min_coords = np.maximum(min_coords - border, 0)
    max_coords = np.minimum(max_coords + border + 1, data.shape)  # +1 for inclusive index

    # Crop the data
    cropped_data = data[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2]]

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
    
def resample_and_save_nifti(nifti_obj, grid, affine, header, out_name, align=None, crop=None):
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
    tensor = F.grid_sample(
        tensor, grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]

    # Step 3: Reorient and save as NIfTI
    if (align):
        tensor, tmp1, tmp2, tmp3  = align_brain(
            tensor.cpu().numpy(), affine, header, np.eye(4), 1)
        nii_data = nib.Nifti1Image(tensor, affine, header)
    else:
        nii_data = reoriented_nifti(tensor, affine, header)

    if (crop):
        nii_data = crop_nifti_image_with_border(nii_data, threshold = 1.1)

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
    dim = header2['dim']
    pixdim = header2['pixdim']

    ras_ref, dirs_ref = get_ras(aff, 3)
    factor = pixdim[1:4] / new_vox_size
    reordered_factor = np.zeros_like(pixdim[1:4])
    for i, axis in enumerate(ras_ref):
        if (reorder_method == 1):
            reordered_factor[i] = factor[np.where(ras_aff == axis)[0][0]]
        else:
            reordered_factor[axis] = dirs_ref[i] * factor[i]  # Adjust for axis direction
    factor = reordered_factor

    dim[1:4] = np.abs(np.round(dim[1:4]*factor))
    
    header2['dim'] = dim

    pixdim[1:4] = new_vox_size
    header2['pixdim'] = pixdim

    # Update affine matrix to match new voxel size
    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))
         
    # Update header transformation fields
    header2['srow_x'] = aff2[0,:]
    header2['srow_y'] = aff2[1,:]
    header2['srow_z'] = aff2[2,:]
    header2['qoffset_x'] = aff2[0,3]
    header2['qoffset_y'] = aff2[1,3]
    header2['qoffset_z'] = aff2[2,3]
    
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

    rois = pd.read_csv(f'{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv', sep=';')[['ROIid', 'ROIabbr']]
    regions = dict(zip(rois.ROIabbr,rois.ROIid))

    atlas = atlas.get_fdata()

    # get ceerebellum
    cerebellum = ((atlas == regions["lCbeWM"]) | (atlas == regions["lCbeGM"]) |
                  (atlas == regions["rCbeWM"]) | (atlas == regions["rCbeGM"]))
    
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

    rois = pd.read_csv(f'{DATA_PATH}/templates_MNI152NLin2009cAsym/ibsr.csv', sep=';')[['ROIid', 'ROIabbr']]
    regions = dict(zip(rois.ROIabbr,rois.ROIid))

    atlas_data = atlas.get_fdata()
    p0_data = p0_large.get_fdata()
    gm = (p0_data > 1.5) & (p0_data < 2.5)
    gm_regions = ["lCbrGM","rCbrGM","lAmy", "lHip", "rAmy", "rHip"]

    # Create cerebral GM mask
    gm_mask = np.isin(atlas_data, [regions[r] for r in gm_regions])
    gm_mask = binary_dilation(gm_mask, generate_binary_structure(3, 3), 2)
    gm = gm & gm_mask

    # Define left and right hemisphere regions
    left_regions = ["lCbrWM", "lCbrGM", "lLatVen", "lInfLatVen", "lThaPro",
                    "lCau", "lPut", "lPal", "lHip", "lAmy", "lAcc", "lVenDC"]
    right_regions = [r.replace("l", "r", 1) for r in left_regions]  # Replace only first 'l'

    # Create left/right masks
    left  = np.isin(atlas_data, [regions[r] for r in left_regions])
    right = np.isin(atlas_data, [regions[r] for r in right_regions])

    # Process hemispheres: dilation and closing to refine boundaries
    lh = binary_dilation(left, generate_binary_structure(3, 3), 5) & ~right
    rh = binary_dilation(right, generate_binary_structure(3, 3), 5) & ~left

    left  = binary_closing(lh, generate_binary_structure(3, 3), 2) & ~rh
    right = binary_closing(rh, generate_binary_structure(3, 3), 2) & ~left

    # Define regions to exclude
    excl_regions = ["lCbeWM", "lCbeGM", "rCbeWM", "rCbeGM", "b3thVen", "b4thVen"]

    # Create masks
    exclude = np.isin(atlas_data, [regions[r] for r in excl_regions])
    exclude = binary_dilation(exclude, generate_binary_structure(3, 3), 1)
    exclude = exclude | binary_dilation(np.isin(atlas_data, regions["bBst"]), generate_binary_structure(3, 3), 5)

    # Define regions that should be filled with WM 
    wm_regions = ["lThaPro", "lCau", "lPut", "lPal", "lAcc", "lLatVen", "lInfLatVen",
                  "rThaPro", "rCau", "rPut", "rPal", "rAcc", "rLatVen", "rInfLatVen"]

    wm_fill  = np.isin(atlas_data, [regions[r] for r in wm_regions])
    wm_fill  = binary_dilation(wm_fill, generate_binary_structure(3, 3), 10) 

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
    mask = (lh > 1)| (rh > 1)
    mask = binary_closing(mask, generate_binary_structure(3, 3), 1)
    mask = binary_opening(mask, generate_binary_structure(3, 3), 2)
    mask = find_largest_cluster(mask)
    mask = binary_dilation(mask, generate_binary_structure(3, 3), 1)
    lh[~mask] = 1
    rh[~mask] = 1

    return lh, rh

def align_brain(data, aff, header, aff_ref, do_flip):
    """
    Aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.

    Parameters:
        dim (ndarray): dimension of input data.
        aff (ndarray): Affine matrix of the volume.
        aff_ref (ndarray): Reference affine matrix.

    Returns:
        ndarray: Aligned image data.
        ndarray: Aligned affine matrix.
        ndarray: Aligned nifti header.
    """

    dim = 3
    ras_aff, dirs_aff = get_ras(aff, dim)
    ras_ref, dirs_ref = get_ras(aff_ref, dim)    

    # Step 1: Reorder the rotation-scaling part (3x3) to match reference axes
    reordered_aff = np.zeros_like(aff)

    # Reorder the rotation-scaling part (3x3) to match reference axes and directions
    if (False):
        for i, axis_ref in enumerate(ras_ref):
            # Find the corresponding axis in the input affine matrix
            matching_axis_idx = np.where(ras_aff == axis_ref)[0][0]
            reordered_aff[:dim, i] = (dirs_ref[i] * dirs_aff[matching_axis_idx] * 
                aff[:dim, matching_axis_idx])
    
    else:
        for i, axis in enumerate(ras_ref):
            reordered_aff[:dim, i] = aff[:dim, np.where(ras_aff == axis)[0][0]]

    # Copy the translation vector
    reordered_aff[:dim, 3] = aff[:dim, 3]
    reordered_aff[3, :] = [0, 0, 0, 1]     # Ensure the bottom row remains [0, 0, 0, 1]

    header['srow_x'] = reordered_aff[0,:]
    header['srow_y'] = reordered_aff[1,:]
    header['srow_z'] = reordered_aff[2,:]
    header['qoffset_x'] = reordered_aff[0,3]
    header['qoffset_y'] = reordered_aff[1,3]
    header['qoffset_z'] = reordered_aff[2,3]

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
