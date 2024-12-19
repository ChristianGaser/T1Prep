import os
import sys
import torch
import argparse
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from deepbet import BrainExtraction
from deepmriprep.preprocess import Preprocess
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.utils import (DEVICE, DATA_PATH, nifti_to_tensor, unsmooth_kernel, nifti_volume)
from deepmriprep.atlas import ATLASES, get_volumes, shape_from_to, AtlasRegistration
from torchreg.utils import INTERP_KWARGS
from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening, binary_dilation, binary_erosion, grey_opening, grey_closing, gaussian_filter

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

from nxbc.filter import *
from SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from lab2im import utils as tools
from lab2im import edit_volumes

def get_bias_field(brain, mask):

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

    dataVoxSize = nib.as_closest_canonical(brain).header.get_zooms()[:3]
    brain0 = brain.get_fdata()
      
    if subsamp :
        # Can't use offset != 0 yet, as the spline smoother takes voxel positions
        # to start from 0, meaning some small interface changes to:
        # 1. control initial voxel offsets
        # 2. keep domain consistent allowing same parameters to be used to
        #    supersample from the spline model.
        offset = 0 # subsamp // 2
        dataSub = brain0[offset::subsamp,offset::subsamp,offset::subsamp]
        mask = mask[offset::subsamp,offset::subsamp,offset::subsamp]
        dataSubVoxSize = dataVoxSize * subsamp
    
    dataSubVoxSize = 1 / (np.array(dataSub.shape) -1)
    dataVoxSize = dataSubVoxSize / subsamp

    # Since assigning into it we need to make sure float
    # beforehand, otherwise assigning into int array will
    # cause a cast
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
    if not subdivide:
        levelfwhm = fwhm / (np.arange(maxlevel) + 1)
    else:
        levelfwhm = fwhm * np.ones(maxlevel)
    
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
    
    for N in tqdm(range(len(levels))):
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
    # Back from subsampled space to full size:
    
    predictor.P = splsm3d.P
    bfieldlog = predictor.predict()
    
    bias = np.exp(bfieldlog)

    # apply nu-correction
    tissue_idx = bias != 0 
    brain0[tissue_idx] /= bias[tissue_idx]
    brain = nib.Nifti1Image(brain0, brain.affine, brain.header)
    
    return bias, brain

def get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_name, device='cpu'):
    header = p1_large.header
    transform = p1_large.affine
    p1_large, p2_large, p3_large = [nifti_to_tensor(p).to(device) for p in [p1_large, p2_large, p3_large]]
    warp_yx = nib.as_closest_canonical(warp_yx)
    yx = nifti_to_tensor(warp_yx)[None].to(device)
    atlas = nib.as_closest_canonical(nib.load(f'{DATA_PATH}/templates/{atlas_name}.nii.gz'))
    shape = tuple(shape_from_to(atlas, warp_yx))
    scaled_yx = F.interpolate(yx.permute(0, 4, 1, 2, 3), shape, mode='trilinear', align_corners=False)
    warps = {}
    warps.update({shape: scaled_yx.permute(0, 2, 3, 4, 1)})
    atlas_register = AtlasRegistration()
    atlas = atlas_register(affine, warps[shape], atlas, t1.shape)
    atlas = nifti_to_tensor(atlas)[None, None].to(device)
    atlas = F.interpolate(atlas, p1_large.shape, mode='nearest')[0, 0]
    atlas = atlas.type(torch.uint8 if atlas.max() < 256 else torch.int16)
    return nib.Nifti1Image(atlas, transform, header)
    
def run_segment():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=True, type=str, default=None)
    parser.add_argument('-o', '--outdir', help='Output folder', required=True, type=str, default=None)
    parser.add_argument("-a", '--amap', action="store_true", help="(optional) Use AMAP segmentation.")
    parser.add_argument('-d', '--amapdir', help='Amap binary folder', required=True, type=str, default=None)
    args = parser.parse_args()
    t1_name = args.input
    out_dir = args.outdir
    use_amap = args.amap
    amapdir = args.amapdir

    out_name = os.path.basename(t1_name).replace('.nii', '')
    t1 = nib.load(t1_name)
    no_gpu = True
    target_res = np.array([0.5]*3)
    
    vol = t1.get_fdata()
    vol, affine2, header2 = align_brain(vol, t1.affine, t1.header, np.eye(4), 0)
    t1 = nib.Nifti1Image(vol, affine2, header2)
    
    print('Skull-stripping')
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    print('Affine registration')
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']
    brain_large = output_aff['brain_large']
    mask_large = output_aff['mask_large']
    
    print('Segmentation')    
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg['p0_large']

    print('Resampling')
    header2, affine2 = get_resampled_header(brain.header, brain.affine, target_res)
    dim = header2['dim']
    shape = dim[1:4]
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())        
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
    
    vol = F.grid_sample(nifti_to_tensor(p0_large)[None, None], grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    vol, tmp1, tmp2   = align_brain(vol.cpu().numpy(),affine2, header2, np.eye(4), 1)
    nib.save(nib.Nifti1Image(vol, affine2, header2), f'{out_dir}/{out_name}_seg.nii')
    
    if (use_amap):
        print('Fine Amap segmentation')
        wm_large = (p0_large.get_fdata() > 2.5)
        bias, brain_large = get_bias_field(brain_large, wm_large)
        nib.save(brain_large, f'{out_dir}/{out_name}_brain_large.nii')
        nib.save(p0_large, f'{out_dir}/{out_name}_seg_large.nii')
        cmd = os.path.join(amapdir, 'CAT_VolAmap') + ' -cleanup 2 -mrf 0 -write-seg 1 1 1 -label ' + f'{out_dir}/{out_name}_seg_large.nii' + ' ' + f'{out_dir}/{out_name}_brain_large.nii'
        os.system(cmd)
        p1_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.nii')
        p2_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.nii')
        p3_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-CSF_probseg.nii')
        p1_affine = F.interpolate(nifti_to_tensor(p1_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p2_affine = F.interpolate(nifti_to_tensor(p2_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p3_affine = F.interpolate(nifti_to_tensor(p3_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        warp_template = nib.load(f'{DATA_PATH}/templates/Template_4_GS.nii.gz')

        p1_affine = nib.Nifti1Image(p1_affine, warp_template.affine, warp_template.header)
        p2_affine = nib.Nifti1Image(p2_affine, warp_template.affine, warp_template.header)
        p3_affine = nib.Nifti1Image(p3_affine, warp_template.affine, warp_template.header)

        wj_affine = np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
        wj_affine = pd.Series([wj_affine])
    else:
        print('Fine Deepmriprep segmentation')
        output_nogm = prep.run_segment_nogm(p0_large, affine, t1)
        p1_large = output_nogm['p1_large']
        p2_large = output_nogm['p2_large']
        p3_large = output_nogm['p3_large']
        p1_affine = output_nogm['p1_affine']
        p2_affine = output_nogm['p2_affine']
        wj_affine = output_nogm['wj_affine']

    print('Warping')
    output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
    warp_yx = output_reg['warp_yx']
    mwp1 = output_reg['mwp1']
    mwp2 = output_reg['mwp2']

    print('Atlas creation')
    atlas = get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large,'ibsr')
    lh, rh = get_partition(p0_large, atlas, 'ibsr')

    print('Resampling')
    vol = F.grid_sample(nifti_to_tensor(nib.Nifti1Image(lh, p0_large.affine, p0_large.header))[None, None], grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    vol, tmp1, tmp2   = align_brain(vol.cpu().numpy(),affine2, header2, np.eye(4), 1)
    nib.save(nib.Nifti1Image(vol, affine2, header2), f'{out_dir}/{out_name}_seg_hemi-L.nii')

    vol = F.grid_sample(nifti_to_tensor(nib.Nifti1Image(rh, p0_large.affine, p0_large.header))[None, None], grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    vol, tmp1, tmp2   = align_brain(vol.cpu().numpy(),affine2, header2, np.eye(4), 1)
    nib.save(nib.Nifti1Image(vol, affine2, header2), f'{out_dir}/{out_name}_seg_hemi-R.nii')

def get_resampled_header(header, aff, new_vox_size):
    """
    This function changes nifti-header and affine matrix to the new voxelsize
    :param header: a nifti structure with the header info
    :param aff: affine matrix of the input volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new header and affine matrix
    """

    header2 = header.copy()
    
    dim = header2['dim']
    pixdim = header2['pixdim']

    factor = pixdim[1:4] / new_vox_size
    dim[1:4] = np.round(dim[1:4]*factor)
    
    header2['dim'] = dim

    pixdim[1:4] = new_vox_size
    header2['pixdim'] = pixdim
    
    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))
    
    header2['srow_x'] = aff2[0,:]
    header2['srow_y'] = aff2[1,:]
    header2['srow_z'] = aff2[2,:]
    header2['qoffset_x'] = aff2[0,3]
    header2['qoffset_y'] = aff2[1,3]
    header2['qoffset_z'] = aff2[2,3]

    return header2, aff2
    
def get_partition(p0_large, atlas, atlas_name):
    rois = pd.read_csv(f'{DATA_PATH}/templates/{atlas_name}.csv', sep=';')[['ROIid', 'ROIabbr']]
    regions = dict(zip(rois.ROIabbr,rois.ROIid))

    atlas = atlas.get_fdata()

    # left hemisphere    
    # first we have to dilate the ventricles because otherwise after filling there remains
    # a rim around it
    lateral_ventricle = (atlas == regions["lLatVen"]) | (atlas == regions["lInfLatVen"])
    lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(4, 3))
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["rLatVen"]) & \
                       ~(atlas == regions["rCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["lAmy"]) & ~(atlas == regions["lHip"])
    #WM 
    wm = ((atlas >= regions["lThaPro"])  &  (atlas <= regions["lPal"])) | \
           (atlas == regions["lAcc"])    |  (atlas == regions["lVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, tools.build_binary_structure(4, 3)) | lateral_ventricle

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
    lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(4, 3))
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["lLatVen"]) & \
                       ~(atlas == regions["lCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["rAmy"]) & ~(atlas == regions["rHip"])
    # WM 
    wm =  ((atlas >= regions["rThaPro"]) &  (atlas <= regions["rPal"])) | \
            (atlas == regions["rAcc"])   |  (atlas == regions["rVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, tools.build_binary_structure(4, 3)) | lateral_ventricle

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

if __name__ == '__main__':
    run_segment()
