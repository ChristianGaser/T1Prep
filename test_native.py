import torch
import argparse
import nibabel as nib
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from deepbet import BrainExtraction
from deepbet.utils import reoriented_nifti
from deepmriprep.preprocess import Preprocess
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.utils import (nifti_to_tensor, unsmooth_kernel)
from torchreg.utils import INTERP_KWARGS
from ext.nxbc.filter import *
from ext.SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized

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
    subsamp = 3
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

def segment_brain(no_gpu, brain, brain_large, affine):
    segment = BrainSegmentation(no_gpu)
    x = nifti_to_tensor(brain_large)[None, None]
    x = x[:, :, 1:-2, 15:-12, :-3]
    x = scale_intensity(x)
    p0 = segment.run_model(x)
    p0 = F.pad(p0, (0, 3, 15, 12, 1, 2))[0, 0]
    p0_kernel = unsmooth_kernel()[None, None]
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())
    p0 = F.conv3d(p0[None, None], p0_kernel, padding=1)
    shape = nib.as_closest_canonical(brain).shape
    grid = F.affine_grid(inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
    p0 = F.grid_sample(p0, grid, align_corners=INTERP_KWARGS['align_corners'])[0, 0]
    return reoriented_nifti(p0.cpu().numpy(), brain.affine, brain.header)

def run_segment():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=True, type=str, default=None)
    args = parser.parse_args()
    input = args.input
    no_gpu = True
    
    prep = Preprocess(no_gpu)
    output1 = prep.run_bet(input)
    brain = output1['brain']
    mask = output1['mask']
    
    output2 = prep.run_affine_register(brain, mask)
    affine = output2['affine']
    brain_large = output2['brain_large']
    
    p0 = segment_brain(no_gpu, brain, brain_large, affine)
    
    wm = (p0.get_fdata() > 2.5)
    bias, brain = get_bias_field(brain, wm)
    
    nib.save(p0, args.input.replace('.nii', f'_seg.nii'))
    nib.save(nib.Nifti1Image(bias,  mask.affine, mask.header), args.input.replace('.nii', f'_bias.nii'))
    nib.save(nib.Nifti1Image(brain, mask.affine, mask.header), args.input.replace('.nii', f'_brain.nii'))
        
if __name__ == '__main__':
    run_segment()
