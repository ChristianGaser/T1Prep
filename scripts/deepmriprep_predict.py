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
from deepbet.utils import reoriented_nifti
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
    brain = brain.get_fdata()
      
    if subsamp :
        # Can't use offset != 0 yet, as the spline smoother takes voxel positions
        # to start from 0, meaning some small interface changes to:
        # 1. control initial voxel offsets
        # 2. keep domain consistent allowing same parameters to be used to
        #    supersample from the spline model.
        offset = 0 # subsamp // 2
        dataSub = brain[offset::subsamp,offset::subsamp,offset::subsamp]
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
    predictor = SplineSmooth3D(brain, dataVoxSize,
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
    brain[tissue_idx] /= bias[tissue_idx]

    return bias, brain

def segment_brain(no_gpu, brain, brain_large, affine, device='cpu'):
    segment = BrainSegmentation(no_gpu)
    x = nifti_to_tensor(brain_large)[None, None]
    x = x[:, :, 1:-2, 15:-12, :-3]
    x = scale_intensity(x)
    p0_large = segment.run_model(x)
    p0_large = F.pad(p0_large, (0, 3, 15, 12, 1, 2))[0, 0]
    
    p0_kernel = unsmooth_kernel(device=device)[None, None]
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float().to(device))
    p0 = F.conv3d(p0_large[None, None].to(device), p0_kernel, padding=1)
    shape = nib.as_closest_canonical(brain).shape
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
    p0 = F.grid_sample(p0, grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    return reoriented_nifti(p0.cpu().numpy(), brain.affine, brain.header), reoriented_nifti(p0_large.cpu().numpy(), brain_large.affine, brain_large.header)

def get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_name, device='cpu'):
    header = p1_large.header
    transform = p1_large.affine
    p1_large, p2_large, p3_large = [nifti_to_tensor(p).to(device) for p in [p1_large, p2_large, p3_large]]
    warp_yx = nib.as_closest_canonical(warp_yx)
    yx = nifti_to_tensor(warp_yx)[None].to(device)
    atlas = nib.as_closest_canonical(nib.load(f'{DATA_PATH}/templates/{atlas_name}.nii.gz'))
    shape = tuple(shape_from_to(atlas, warp_yx))
    print(shape)
    scaled_yx = F.interpolate(yx.permute(0, 4, 1, 2, 3), shape, mode='trilinear', align_corners=False)
    warps = {}
    warps.update({shape: scaled_yx.permute(0, 2, 3, 4, 1)})
    atlas_register = AtlasRegistration()
    atlas = atlas_register(affine, warps[shape], atlas, t1.shape)
    atlas = nifti_to_tensor(atlas)[None, None].to(device)
    atlas = F.interpolate(atlas, p1_large.shape, mode='nearest')[0, 0]
    atlas = atlas.type(torch.uint8 if atlas.max() < 256 else torch.int16)
    return nib.Nifti1Image(atlas, transform, header)
    
def run_segment_fast():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--t1', help='Input file or folder', required=True, type=str, default=None)
    parser.add_argument('-b', '--biascorr', help='Output bias corrected file', required=False, type=str, default=None)
    parser.add_argument('-l', '--label', help='Output label file', required=True, type=str, default=None)
    args = parser.parse_args()
    t1 = args.t1
    no_gpu = True
    
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']
    brain_large = output_aff['brain_large']
    mask_large = output_aff['brain_large']

    zoom = output_aff['zoom']
    matrix = np.zeros((4, 4))

    # Fill the diagonal with the first 3 entries of the vector
    matrix[0, 0] = zoom[0]
    matrix[1, 1] = zoom[1]
    matrix[2, 2] = zoom[2]
    matrix[3, 3] = 1
    
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())
    print(np.matmul(inv_affine, matrix).float())

    shape = nib.as_closest_canonical(brain).shape
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
    p0 = F.grid_sample(nifti_to_tensor(brain_large)[None, None], grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    
    nib.save(nib.Nifti1Image(p0, brain.affine, brain.header), 'test.nii')
    
    """
    p0, p0_large = segment_brain(no_gpu, brain, brain_large, affine)
    nib.save(p0_large, args.label)
    
    wm = (p0_large.get_fdata() > 2.5)
    bias, brain_large = get_bias_field(brain_large, wm)
    
    if args.biascorr is not None:
        nib.save(nib.Nifti1Image(brain_large,  mask_large.affine, mask_large.header), args.biascorr)
    """
def run_segment_resample():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=True, type=str, default=None)
    parser.add_argument('-o', '--outdir', help='Output folder', required=True, type=str, default=None)
    args = parser.parse_args()
    t1_name = args.input
    out_dir = args.outdir
    out_name = os.path.basename(t1_name).replace('.nii', '')
    t1 = nib.load(t1_name)
    
    print('Resampling')
    target_res = np.array([0.5]*3)
    t1_data = t1.get_fdata()
    aff = t1.affine
    header = t1.header
    t1_large, aff_t1 = edit_volumes.resample_volume(t1_data, aff, target_res)
    t1_name = f'{out_dir}/{out_name}_resampled.nii'
    tools.save_volume(t1_large, aff_t1, header, t1_name, dtype='float32')
    t1 = nib.load(t1_name)

    no_gpu = True
    
    print('Skull-stripping')
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1_name)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    print('Affine registration')
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']    
    brain_large = output_aff['brain_large']
    mask_large = output_aff['mask_large']
    #nib.save(brain_large, f'{out_dir}/{out_name}_resampled_affine.nii')
    
    print('Segmentation')
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0 = output_seg['p0']
    p0_large = output_seg['p0_large']
    nib.save(p0, f'{out_dir}/{out_name}_seg.nii')
    
    print('Fine segmentation')
    output_nogm = prep.run_segment_nogm(p0_large, affine, t1)
    p1 = output_nogm['p1']
    p2 = output_nogm['p2']
    p1_large = output_nogm['p1_large']
    p2_large = output_nogm['p2_large']
    p3_large = output_nogm['p3_large']
    p1_affine = output_nogm['p1_affine']
    p2_affine = output_nogm['p2_affine']
    wj_affine = output_nogm['wj_affine']
    nib.save(p1, f'{out_dir}/{out_name}_GM.nii')
    nib.save(p2, f'{out_dir}/{out_name}_WM.nii')

    print('Warping')
    output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
    warp_yx = output_reg['warp_yx']
    mwp1 = output_reg['mwp1']
    mwp2 = output_reg['mwp2']

    print('Partitioning')
    affine2 = pd.DataFrame({
        0: [0.5, 0.0, 0.0, 0.0],
        1: [0.0, 0.5, 0.0, 0.0],
        2: [0.0, 0.0, 0.5, 0.0],
        3: [0.0, 0.0, 0.0, 1.0]
    })
    atlas = get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large,'ibsr')
    #nib.save(nib.Nifti1Image(atlas_large,  p1.affine, p1.header), f'{out_dir}/{out_name}_ibsr.nii')
    #atlas = atlas_large#.cpu().numpy()
    """
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())
    
    atlas_large = torch.from_numpy(atlas_large.get_fdata()[None, None])
    shape = torch.from_numpy(np.array(mask.shape)).double().to('cpu')
    
    size = [1, 3, *shape.int().tolist()]
    grid = F.affine_grid(inv_affine[None, :3], size, align_corners=INTERP_KWARGS['align_corners'])
    atlas = F.grid_sample(atlas_large, grid.double(), align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    atlas = atlas.cpu().numpy()
    """

    #nib.save(nib.Nifti1Image(atlas,  p1.affine, p1.header), f'{out_dir}/{out_name}_ibsr.nii')
    print(atlas.shape)
    print(p0.shape)
    print(t1.shape)
    lh, rh = get_partition(p0, atlas, 'ibsr')
    
    nib.save(nib.Nifti1Image(lh,  p1.affine, p1.header), f'{out_dir}/{out_name}_seg_hemi-L.nii')
    nib.save(nib.Nifti1Image(rh,  p1.affine, p1.header), f'{out_dir}/{out_name}_seg_hemi-R.nii')

def run_segment_orig():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=True, type=str, default=None)
    parser.add_argument('-o', '--outdir', help='Output folder', required=True, type=str, default=None)
    args = parser.parse_args()
    t1_name = args.input
    out_dir = args.outdir
    out_name = os.path.basename(t1_name).replace('.nii', '')
    t1 = nib.load(t1_name)
    no_gpu = True
    print(t1.header)
    
    print('Skull-stripping')
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1_name)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    print('Affine registration')
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']
    brain_large = output_aff['brain_large']
    mask_large = output_aff['mask_large']
    nib.save(brain_large, f'{out_dir}/{out_name}_resampled.nii')
    
    
    
    print('Resampling')
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())

    #print(np.matmul(inv_affine, matrix).float())
        
        
    target_res = np.array([0.5]*3)
    t1_data = t1.get_fdata()
    aff = t1.affine
    header = t1.header
    t1_large, aff_t1 = edit_volumes.resample_volume(t1_data, aff, target_res)
    t1_name = f'{out_dir}/{out_name}_resampled.nii'
    tools.save_volume(t1_large, aff_t1, header, t1_name, dtype='float32')
    t1_large = nib.load(t1_name)

    #print(brain_large.affine)
    #print(brain_large.header)
    pixdim0 = t1.header['pixdim']
    pixdim = pixdim0 + 0;
    pixdim[1:4] = target_res
    t1.header['pixdim'] = pixdim
    print(pixdim)
    
    dim = t1.header['dim']
    print(dim)
    print(pixdim)
    print(pixdim0)
    dim[1:4] = np.round(dim[1:4]*(pixdim[1:4]/pixdim0[1:4]))
    print(dim)
    t1.header['dim'] = dim
    
    srow_x = t1.header['srow_x']
    print(srow_x[0])
    srow_y = t1.header['srow_y']
    print(srow_y[1])
    srow_z = t1.header['srow_z']
    print(srow_z[2])
    qoffset_x = t1.header['qoffset_x']
    qoffset_y = t1.header['qoffset_y']
    qoffset_z = t1.header['qoffset_z']

    print(brain_large.header['quatern_c'])
    
    return
    print(t1_large.affine)
    print(t1_large.header)

    shape = nib.as_closest_canonical(t1_large).shape
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
    p0 = F.grid_sample(nifti_to_tensor(brain_large)[None, None], grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]    
    nib.save(nib.Nifti1Image(p0, t1_large.affine, t1_large.header), 'test.nii')
    test = nib.load('test.nii')
    print(test.affine)
    print(test.header)
    
    return
    
    print('Segmentation')
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg['p0_large']
    nib.save(p0_large, f'{out_dir}/{out_name}_seg.nii')
    
    print('Fine segmentation')
    output_nogm = prep.run_segment_nogm(p0_large, affine, t1)
    p1_large = output_nogm['p1_large']
    p2_large = output_nogm['p2_large']
    p3_large = output_nogm['p3_large']
    p1_affine = output_nogm['p1_affine']
    p2_affine = output_nogm['p2_affine']
    wj_affine = output_nogm['wj_affine']
    nib.save(p1_large, f'{out_dir}/{out_name}_GM.nii')
    nib.save(p2_large, f'{out_dir}/{out_name}_WM.nii')


    print('Warping')
    output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
    warp_yx = output_reg['warp_yx']
    mwp1 = output_reg['mwp1']
    mwp2 = output_reg['mwp2']


    atlas = get_atlas(t1, affine, warp_yx, p1_large, p2_large, p3_large,'ibsr')
    lh, rh = get_partition(p0_large, atlas, 'ibsr')
    
    nib.save(nib.Nifti1Image(lh,  p1_large.affine, p1_large.header), f'{out_dir}/{out_name}_seg_hemi-L.nii')
    nib.save(nib.Nifti1Image(rh,  p1_large.affine, p1_large.header), f'{out_dir}/{out_name}_seg_hemi-R.nii')
    
def get_partition(p0_large, atlas, atlas_name):
    rois = pd.read_csv(f'{DATA_PATH}/templates/{atlas_name}.csv', sep=';')[['ROIid', 'ROIabbr']]
    regions = dict(zip(rois.ROIabbr,rois.ROIid))

    atlas = atlas.get_fdata()

    # left hemisphere    
    # first we have to dilate the ventricles because otherwise after filling there remains
    # a rim around it
    lateral_ventricle = (atlas == regions["lLatVen"]) | (atlas == regions["lInfLatVen"])
    lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(3, 3))
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["rLatVen"]) & \
                       ~(atlas == regions["rCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["lAmy"]) & ~(atlas == regions["lHip"])
    #WM 
    wm = ((atlas >= regions["lThaPro"])         &  (atlas <= regions["lPal"])) | \
           (atlas == regions["lAcc"])    |  (atlas == regions["lVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, tools.build_binary_structure(4, 3)) | lateral_ventricle

    # CSF + BKG
    csf = (atlas == 0)               |  (atlas == regions["lCbeWM"]) | \
           (atlas == regions["lCbeGM"]) |  (atlas == regions["b3thVen"]) | \
           (atlas == regions["b4thVen"])      |  (atlas == regions["bBst"]) | \
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
    lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(3, 3))
    # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
    lateral_ventricle = lateral_ventricle & ~(atlas == regions["lLatVen"]) & \
                       ~(atlas == regions["lCbrWM"]) & ~(atlas == regions["bCSF"]) & \
                       ~(atlas == regions["rAmy"]) & ~(atlas == regions["rHip"])
    # WM 
    wm =  ((atlas >= regions["rThaPro"])         &  (atlas <= regions["rPal"])) | \
            (atlas == regions["rAcc"])    |  (atlas == regions["rVenDC"])
    # we also have to dilate whole WM to close the remaining rims
    wm = binary_dilation(wm, tools.build_binary_structure(4, 3)) | lateral_ventricle

    # CSF + BKG
    csf = ((atlas <= regions["lVenDC"])        & ~(atlas == regions["bCSF"])) | \
            (atlas == regions["rCbeWM"])     |  (atlas == regions["rCbeGM"])

    csf = (atlas == 0)               |  (atlas == regions["rCbeWM"]) | \
           (atlas == regions["rCbeGM"]) |  (atlas == regions["b3thVen"]) | \
           (atlas == regions["b4thVen"])      |  (atlas == regions["bBst"]) | \
           (atlas <= regions["lAmy"]) | (atlas == regions["lVenDC"]) | (atlas == regions["lAcc"]) 

    lesion_mask = atlas == regions["rCbrWM"]

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    rh = p0_large.get_fdata() + 0
    rh[rh < 1] = 1
    rh[csf] = 1
    rh[wm]  = 3
    
    return lh, rh
    

if __name__ == '__main__':
    run_segment_orig()
