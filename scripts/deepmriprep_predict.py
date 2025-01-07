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
import xml.etree.cElementTree as ET

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from deepbet import BrainExtraction
from deepbet.utils import reoriented_nifti
from deepmriprep.preprocess import Preprocess, save_output
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.utils import DEVICE, DATA_PATH, nifti_to_tensor, unsmooth_kernel, nifti_volume
from deepmriprep.atlas import ATLASES, get_volumes, shape_from_to, AtlasRegistration
from torchreg.utils import INTERP_KWARGS
from scipy.ndimage import binary_dilation, generate_binary_structure
from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from pathlib import Path

DATA_PATH0 = Path(__file__).resolve().parent.parent / 'data/'
MODEL_FILES = (['brain_extraction_bbox_model.pt', 'brain_extraction_model.pt', 'segmentation_nogm_model.pt'] +
               [f'segmentation_patch_{i}_model.pt' for i in range(18)] + ['segmentation_model.pt', 'warp_model.pt'])

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
    prog = '■' * elapsed
    remaining = ' ' * (total - elapsed)
    
    # Format the name with padding
    name = name.ljust(50)
    
    # Print the progress bar with percentage and name
    #print(f'{prog}{remaining} {it}% {name}\r', end='')
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

def correct_bias_field(brain, seg):
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
    steps = 100
    subsamp = 5
    stopthr = 1e-4
    spacing = 1

    # Process voxel sizes and mask for brain segmentation
    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata()
    
    # Generate mask based on segmentation or brain data
    mask = (seg.get_fdata() > 2.5) if seg is not None else (brain0 > 0.0)

    # Subsampling for efficiency
    if subsamp :
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
    
    splsm3d = SplineSmooth3DUnregularized(datalog, dataSubVoxSize,
                                            spacing, domainMethod="minc",
                                            mask=mask)
    
    # Prediction interpolator
    predictor = SplineSmooth3D(brain0, dataVoxSize,
                               spacing, knts=splsm3d.kntsArr, dofit=False)
    lastinterpbc = np.zeros(datalogmasked.shape[0])
    datalogcur = np.copy(datalog)
    nextlevel = 0
    
    controlField=None
    chosenkernelfn = kernelfntri
    
    for N in range(len(levels)):
        if levels[N] < nextlevel:
          continue
        
        hist,histvaledge,histval,histbinwidth = \
          distrib_kde(datalogmaskedcur, Nbins, kernfn=chosenkernelfn,
                      binCentreLimits=bcl)
        thisFWHM = levelfwhm[levels[N]] # * math.sqrt(8*math.log(2))
        thisSD = thisFWHM /  math.sqrt(8*math.log(2))
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
            
        if subdivide and (N+1)<len(levels) and (nextlevel>levels[N] or levels[N+1] != levels[N]):
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

def get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_name, device='cpu'):
    """
    Generates an atlas-aligned image for brain segmentation.

    Args:
        t1 (nibabel.Nifti1Image): Input T1-weighted image.
        affine (numpy.ndarray): Affine transformation matrix.
        warp_yx (nibabel.Nifti1Image): Warp field for alignment.
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
    warp_yx = nib.as_closest_canonical(warp_yx)
    yx = nifti_to_tensor(warp_yx)[None].to(device)

    # Load and resample the atlas template
    atlas = nib.as_closest_canonical(nib.load(f'{DATA_PATH}/templates/{atlas_name}.nii.gz'))
    shape = tuple(shape_from_to(atlas, warp_yx))
    scaled_yx = F.interpolate(yx.permute(0, 4, 1, 2, 3), shape, mode='trilinear', align_corners=False)

    # Perform atlas registration
    warps = {shape: scaled_yx.permute(0, 2, 3, 4, 1)}
    atlas_register = AtlasRegistration()
    atlas = atlas_register(affine, warps[shape], atlas, t1.shape)

    # Interpolate and finalize atlas alignment
    atlas = nifti_to_tensor(atlas)[None, None].to(device)
    atlas = F.interpolate(atlas, p1_large.shape, mode='nearest')[0, 0]
    atlas = atlas.type(torch.uint8 if atlas.max() < 256 else torch.int16)

    # Return the aligned atlas image
    return nib.Nifti1Image(atlas, transform, header)
    
def resample_and_save_nifti(nifti_obj, grid, affine, header, out_name, align = None):
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
    tensor = F.grid_sample(tensor, grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]

    # Step 3: Align to reference orientation
    #if (align):

    # Step 4: Reorient and save as NIfTI
    if (align):
        tensor, tmp1, tmp2  = align_brain(tensor.cpu().numpy(), affine, header, np.eye(4), 1)
        nib.save(nib.Nifti1Image(tensor, affine, header), out_name)
    else:
        nib.save(reoriented_nifti(tensor, affine, header), out_name)

def get_resampled_header(header, aff, new_vox_size):
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

    factor = pixdim[1:4] / new_vox_size
    dim[1:4] = np.round(dim[1:4]*factor)
    
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
    
def get_partition(p0_large, atlas, atlas_name):
    """
    Partition the input volume into left and right hemispheres with labels for CSF, GM, and WM.

    Parameters:
    p0_large : Nifti1Image
        Input probability map from segmentation.
    atlas : Nifti1Image
        Atlas template for anatomical regions.
    atlas_name : str
        Name of the atlas used to define regions.

    Returns:
    tuple:
        Left hemisphere (lh) and right hemisphere (rh) labeled volumes.
    """

    rois = pd.read_csv(f'{DATA_PATH}/templates/{atlas_name}.csv', sep=';')[['ROIid', 'ROIabbr']]
    regions = dict(zip(rois.ROIabbr,rois.ROIid))

    atlas = atlas.get_fdata()

    # left hemisphere    
    # first we have to dilate the ventricles because otherwise after filling there remains
    # a rim around it
    lateral_ventricle = (atlas == regions["lLatVen"]) | (atlas == regions["lInfLatVen"])
    lateral_ventricle = binary_dilation(lateral_ventricle, generate_binary_structure(3, 3), 2)
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["rLatVen"]) & \
                       ~(atlas == regions["rCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["lAmy"]) & ~(atlas == regions["lHip"])
    #WM 
    wm = ((atlas >= regions["lThaPro"])  &  (atlas <= regions["lPal"])) | \
           (atlas == regions["lAcc"])    |  (atlas == regions["lVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, generate_binary_structure(3, 3), 2) | lateral_ventricle

    # CSF + BKG
    csf = (atlas == 0)                   |  (atlas == regions["lCbeWM"]) | \
           (atlas == regions["lCbeGM"])  |  (atlas == regions["b3thVen"]) | \
           (atlas == regions["b4thVen"]) |  (atlas == regions["bBst"]) | \
           (atlas >= regions["rCbrWM"])

    lesion_mask = atlas == regions["lCbrWM"]

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    lh = p0_large.get_fdata() + 0
    lh[lh < 1] = 1
    lh[csf] = 1
    lh[wm]  = 3
        
    # right hemisphere    
    # first we have to dilate the ventricles because otherwise after filling there remains
    # a rim around it
    lateral_ventricle = (atlas == regions["rLatVen"]) | (atlas == regions["rInfLatVen"])
    lateral_ventricle = binary_dilation(lateral_ventricle, generate_binary_structure(3, 3), 2)
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["lLatVen"]) & \
                       ~(atlas == regions["lCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["rAmy"]) & ~(atlas == regions["rHip"])
    # WM 
    wm =  ((atlas >= regions["rThaPro"]) &  (atlas <= regions["rPal"])) | \
            (atlas == regions["rAcc"])   |  (atlas == regions["rVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, generate_binary_structure(3, 3), 2) | lateral_ventricle

    # CSF + BKG
    csf = ((atlas <= regions["lVenDC"])  & ~(atlas == regions["bCSF"])) | \
            (atlas == regions["rCbeWM"]) |  (atlas == regions["rCbeGM"])

    csf = (atlas == 0)                   |  (atlas == regions["rCbeWM"]) | \
           (atlas == regions["rCbeGM"])  |  (atlas == regions["b3thVen"]) | \
           (atlas == regions["b4thVen"]) |  (atlas == regions["bBst"]) | \
           (atlas <= regions["lAmy"])    | (atlas == regions["lVenDC"]) | (atlas == regions["lAcc"]) 

    lesion_mask = atlas == regions["rCbrWM"]

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    rh = p0_large.get_fdata() + 0
    rh[rh < 1] = 1
    rh[csf] = 1
    rh[wm]  = 3
    
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
    def get_ras(aff, dim):
        """
        Determines the RAS axes order for an affine matrix.
        """
        aff_inv = np.linalg.inv(aff)
        aff_ras = np.argmax(np.abs(aff_inv[:dim, :dim]), axis=1)
        return aff_ras

    dim = 3  # Assume 3D volume
    ras_aff = get_ras(aff, dim)
    ras_ref = get_ras(aff_ref, dim)

    # Step 1: Reorder the rotation-scaling part (3x3) to match reference axes
    reordered_aff = np.zeros_like(aff)
    for i, axis in enumerate(ras_ref):
        reordered_aff[:dim, i] = aff[:dim, np.where(ras_aff == axis)[0][0]]
    reordered_aff[:dim, 3] = aff[:dim, 3]  # Copy the translation vector
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
                                                              
    return aligned_data, aff, header

def run_segment():

    """
    Perform brain segmentation on input medical image data using preprocessing, affine registration, and segmentation techniques.

    Command-line Arguments:
    -i, --input : str (required)
        Input file or folder containing the MRI data (.nii format).
    -o, --outdir : str (required)
        Output directory to save the processed results.
    -a, --amap : flag (optional)
        Enable AMAP segmentation if specified. Default is False.
    -d, --amapdir : str (optional)
        Path to the AMAP binary folder if AMAP segmentation is enabled.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file', required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output folder', required=True, type=str)
    parser.add_argument("-s", '--surf', action="store_true", help="(optional) Save partioned segmentation map for surface estimation.", default=None)
    parser.add_argument("-m", '--mwp', action="store_true", help="(optional) Save modulated and warped segmentations.", default=None)
    parser.add_argument("-w", '--wp', action="store_true", help="(optional) Save warped segmentations.", default=None)
    parser.add_argument("-p", '--p', action="store_true", help="(optional) Save native segmentations.", default=None)
    parser.add_argument("-r", '--rp', action="store_true", help="(optional) Save affine registered segmentations.", default=None)
    parser.add_argument("-b", '--bids', action="store_true", help="(optional) Use bids naming convention.", default=None)
    parser.add_argument("-a", '--amap', action="store_true", help="(optional) Use AMAP segmentation.", default=None)
    parser.add_argument('-d', '--amapdir', help='Amap binary folder', type=str, default=None)
    args = parser.parse_args()

    # Input/output parameters
    t1_name  = args.input
    out_dir  = args.outdir
    amap_dir = args.amapdir
    
    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    
    # Save options
    save_mwp = args.mwp
    save_wp  = args.wp
    save_rp  = args.rp
    save_p   = args.p
    do_surf  = args.surf

    # Check for GPU support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        no_gpu = False
    elif torch.backends.mps.is_available() and False: # not yet fully supported
        device = torch.device("mps")
        no_gpu = False
    else:
        device = torch.device("cpu")
        no_gpu = True
            
    # Set processing parameters
    target_res = np.array([0.5]*3) # Target resolution for resampling
    count = 1
    end_count = 4
    if (save_mwp):
        end_count = 5
    if (do_surf):
        end_count = 7
    
    # Prepare filenames and load input MRI data
    out_name = os.path.basename(os.path.basename(t1_name).replace('_desc-sanlm', '')).replace('.nii', '').replace('.gz','')

    t1 = nib.load(t1_name)

    # copy necessary model files from local folder to install it, since often the API rate limit is exceeded
    Path(f'{DATA_PATH}/models').mkdir(exist_ok=True)
    for file in MODEL_FILES:
        if not Path(f'{DATA_PATH}/models/{file}').exists():
            shutil.copy(f'{DATA_PATH0}/models/{file}', f'{DATA_PATH}/models/{file}') 

    # Preprocess the input volume
    vol = t1.get_fdata()
    vol, affine2, header2 = align_brain(vol, t1.affine, t1.header, np.eye(4), 0)
    t1 = nib.Nifti1Image(vol, affine2, header2)
    
    # Step 1: Skull-stripping
    count = progress_bar(count, end_count, 'Skull-stripping               ')
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    # Step 2: Affine registration
    count = progress_bar(count, end_count, 'Affine registration           ')
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']
    brain_large = output_aff['brain_large']
    mask_large = output_aff['mask_large']
    
    # Step 3: Segmentation
    count = progress_bar(count, end_count, 'Deepmriprep segmentation                  ')    
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg['p0_large']

    # Prepare for esampling
    header2, affine2 = get_resampled_header(brain.header, brain.affine, target_res)
    dim_target_res = header2['dim']
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())        
    grid_target_res = F.affine_grid(inv_affine[None, :3], [1, 3, *dim_target_res[1:4]], align_corners=INTERP_KWARGS['align_corners'])
    shape = nib.as_closest_canonical(mask).shape
    grid_native = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
        
    # Conditional processing based on AMAP flag
    if (use_amap):
        # AMAP segmentation pipeline
        amapdir = args.amapdir
        count = progress_bar(count, end_count, 'Fine Amap segmentation')
        bias, brain_large = correct_bias_field(brain_large, p0_large)
        nib.save(brain_large, f'{out_dir}/{out_name}_brain_large.nii')
        nib.save(p0_large, f'{out_dir}/{out_name}_seg_large.nii')
        cmd = os.path.join(amapdir, 'CAT_VolAmap') + ' -nowrite-corr -bias-fwhm 0 -cleanup 2 -mrf 0 -write-seg 1 1 1 -label ' + f'{out_dir}/{out_name}_seg_large.nii' + ' ' + f'{out_dir}/{out_name}_brain_large.nii'
        os.system(cmd)

        # Load probability maps for GM, WM, CSF
        p0_large = nib.load(f'{out_dir}/{out_name}_brain_large_seg.nii')
        p1_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.nii')
        p2_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.nii')
        p3_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-CSF_probseg.nii')
        
        warp_template = nib.load(f'{DATA_PATH}/templates/Template_4_GS.nii.gz')
        p1_affine = F.interpolate(nifti_to_tensor(p1_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p2_affine = F.interpolate(nifti_to_tensor(p2_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p1_affine = reoriented_nifti(p1_affine, warp_template.affine, warp_template.header)
        p2_affine = reoriented_nifti(p2_affine, warp_template.affine, warp_template.header)
        
        wj_affine = np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
        wj_affine = pd.Series([wj_affine])
    else:
        # DeepMRI prep segmentation pipeline
        count = progress_bar(count, end_count, 'Fine Deepmriprep segmentation')
        output_nogm = prep.run_segment_nogm(p0_large, affine, t1)
        p1_large = output_nogm['p1_large']
        p2_large = output_nogm['p2_large']
        p3_large = output_nogm['p3_large']
        p1_affine = output_nogm['p1_affine']
        p2_affine = output_nogm['p2_affine']
        wj_affine = output_nogm['wj_affine']

        gmv = output_nogm['gmv']
        tiv = output_nogm['tiv']

    # Save affine registration
    if (save_rp):
        nib.save(p1_affine, f'{out_dir}/rp1{out_name}_affine.nii')
        nib.save(p2_affine, f'{out_dir}/rp2{out_name}_affine.nii')

    # Save native registration
    resample_and_save_nifti(p0_large, grid_native, mask.affine, mask.header, f'{out_dir}/p0{out_name}.nii', True)
    if (save_p):
        resample_and_save_nifti(brain_large, grid_native, mask.affine, mask.header, f'{out_dir}/m{out_name}.nii')
        resample_and_save_nifti(p1_large, grid_native, mask.affine, mask.header, f'{out_dir}/p1{out_name}.nii')
        resample_and_save_nifti(p2_large, grid_native, mask.affine, mask.header, f'{out_dir}/p2{out_name}.nii')
        resample_and_save_nifti(p3_large, grid_native, mask.affine, mask.header, f'{out_dir}/p3{out_name}.nii')

    # Warping is necessary for surface creation and saving warped segmentations
    if ((do_surf) | (save_mwp) | (save_wp)):
        # Step 5: Warping
        count = progress_bar(count, end_count, 'Warping                          ')
        output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        warp_yx = output_reg['warp_yx']
        warp_xy = output_reg['warp_xy']
        
        if (save_mwp):
            mwp1 = output_reg['mwp1']
            mwp2 = output_reg['mwp2']
            nib.save(mwp1, f'{out_dir}/mwp1{out_name}.nii')
            nib.save(mwp2, f'{out_dir}/mwp2{out_name}.nii')
            
        if (save_wp):
            wp1 = output_reg['mwp1']
            wp2 = output_reg['mwp2']
            nib.save(wp1, f'{out_dir}/wp1{out_name}.nii')
            nib.save(wp2, f'{out_dir}/wp2{out_name}.nii')

        nib.save(warp_xy, f'{out_dir}/y_{out_name}.nii')
        #nib.save(warp_yx, f'{out_dir}/iy_{out_name}.nii')

        """
        # write atlas ROI volumes to csv files
        atlas_list = tuple([f'{atlas}_volumes' for atlas in ATLASES])
        atlas_list = list(atlas_list)
        output_paths = tuple([f'{out_dir}/../label/{out_name}_{atlas}.csv' for atlas in ATLASES])
        output_paths = list(output_paths)

        output_atlas = prep.run_atlas_register(t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list)
        for k, output in output_atlas.items():
            print("k")
            print(k)
            print("output")
            print(output)
         
        for i, atl in enumerate(output_atlas):
            print("atl")
            atl
            print("output_paths")
            output_paths[i]
            print("output_atlas")
            output_atlas[i]
        """    
                
    # Atlas is necessary for surface creation
    if (do_surf):
        # Step 6: Atlas creation
        count = progress_bar(count, end_count, 'Atlas creation                 ')
        atlas = get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large, 'ibsr', device)
        lh, rh = get_partition(p0_large, atlas, 'ibsr')

        # Step 7: Save hemisphere outputs
        count = progress_bar(count, end_count, 'Resampling                     ')
        resample_and_save_nifti(nib.Nifti1Image(lh, p0_large.affine, p0_large.header), grid_target_res, affine2, header2, f'{out_dir}/{out_name}_seg_hemi-L.nii', True)
        resample_and_save_nifti(nib.Nifti1Image(rh, p0_large.affine, p0_large.header), grid_target_res, affine2, header2, f'{out_dir}/{out_name}_seg_hemi-R.nii', True)

    # remove temporary AMAP files
    if (use_amap):
        remove_file(f'{out_dir}/{out_name}_brain_large.nii')
        remove_file(f'{out_dir}/{out_name}_brain_large_seg.nii')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.nii')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.nii')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-CSF_probseg.nii')

if __name__ == '__main__':
    run_segment()
