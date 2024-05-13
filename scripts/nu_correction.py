# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache 2.0 license
# terms, and this file has been changed.
#
# The original nxbc code where some ideas are based on is found at:
# https://github.com/imalone/nxbc/blob/33c1097d2e4927f4dc50df1be8364d014bb6e5cf/bin/nxbc
#
# [May 2024] CHANGES:
#    * use ideas from nxbc to create a function for nu-correction that can be calles

# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

from ext.SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from ext.nxbc.filter import *
#from ext.nxbc.plotsupport import *
from scipy.ndimage import gaussian_filter, distance_transform_edt, grey_opening, binary_opening
from T1Prep import utils
from skimage import filters

def get_bias_nx(im, mask, im_res, aff):

    # Defaults
    subdivide = True
    unregularized = True
    accumulate = True
    bcl = True
    withotsu = False
    Z = 0.01
    Nbins = 256
    maxlevel = 4
    fwhm = 0.05
    steps = 100
    subsamp = 3
    stopthr = 1e-4
    spacing = 1
  
    dataSub = im
    dataVoxSize = im_res
    affineSub = np.copy(aff)
    dataSubVoxSize = dataVoxSize
    
    if (mask is None) :
        withotsu = True
        mask = np.ones(im.shape) > 0
    
    if subsamp :
        # Can't use offset != 0 yet, as the spline smoother takes voxel positions
        # to start from 0, meaning some small interface changes to:
        # 1. control initial voxel offsets
        # 2. keep domain consistent allowing same parameters to be used to
        #    supersample from the spline model.
        offset = 0 # subsamp // 2
        dataSub = dataSub[offset::subsamp,offset::subsamp,offset::subsamp]
        mask = mask[offset::subsamp,offset::subsamp,offset::subsamp]
        affineSub[0:3,3] = affineSub[0:3,0:3].sum(1) * offset + affineSub[0:3,3]
        affineSub[0:3,0:3] *= subsamp
        dataSubVoxSize = dataVoxSize * subsamp
    
    dataSubVoxSize = 1 / (np.array(dataSub.shape) -1)
    dataVoxSize = dataSubVoxSize / subsamp
    
    if withotsu :
        _thresh = filters.threshold_otsu(dataSub[mask])
        mask = np.logical_and(dataSub > _thresh, mask)
    
    datamasked = dataSub[mask]
    # Since assigning into it we need to make sure float
    # beforehand, otherwise assigning into int array will
    # cause a cast
    datalog = dataSub.astype(np.float32)
    datalog[mask] = np.log(datalog[mask])
    datalog[np.logical_not(mask)] = 0
    datalogmasked = datalog[mask]
    datafill = np.zeros_like(datalog)
    
    datalogmaskedcur = np.copy(datalogmasked)
    eps=0.01
    min_fill=0.5
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
    predictor = SplineSmooth3D(im, dataVoxSize,
                               spacing, knts=splsm3d.kntsArr, dofit=False)
    lastinterpbc = np.zeros(datalogmasked.shape[0])
    datalogcur = np.copy(datalog)
    nextlevel = 0
    
    controlField=None
    chosenkernelfn = kernelfntri
    
    for N in range(len(levels)):
        if N%100 == 0 :
            print("{}/{}".format(N,len(levels)))
        if levels[N] < nextlevel:
          continue
        nextlevel = levels[N]
        hist,histvaledge,histval,histbinwidth = \
          distrib_kde(datalogmaskedcur, Nbins, kernfn=chosenkernelfn,
                      binCentreLimits=bcl)
        #thisFWHM = optFWHM(hist,histbinwidth)
        #thisFWHM = optEntropyFWHM(hist, histbinwidth, histval, datalogmaskedcur, distrib="kde")
        thisFWHM = levelfwhm[levels[N]] # * math.sqrt(8*math.log(2))
        thisSD = thisFWHM /  math.sqrt(8*math.log(2))
    #    print ("reduced sigma {} fwhm {}".format(thisSD, thisFWHM))
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
        splsm3d.fit(datafill, reportingLevel=1)
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
    #    print(conv,ratiosd,ratiomean)
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
        if subdivide and (N+1)<len(levels) and N%steps == 0:
            print ("subdividing")
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
    return bias


# parse arguments
parser = ArgumentParser()
parser.add_argument("--i", metavar="file", required=True,
    help="Input image for bias correction.")
parser.add_argument("--o", metavar="file", required=False,
    help="Bias corrected output.")
parser.add_argument("--label", metavar="file", required=False,
    help="Optional label image.")
parser.add_argument("--target-res", type=float, default=-1, 
    help="(optional) Target voxel size in mm for resampled and hemispheric label data that will be used for cortical surface extraction. Default is 0.5. Use a negative value to save outputs with original voxel size.")

# check for no arguments
if len(sys.argv) < 1:
    print("\nMust provide at least -i flags.")
    parser.print_help()
    sys.exit(1)

args = vars(parser.parse_args())

# Prepare list of images to process
name_input = os.path.abspath(args['i'])
basename = os.path.basename(name_input)

# output name
if args['o'] is not None:
    name_corrected = os.path.abspath(args['o'])
else:
    name_corrected = name_input.replace('.nii', '_nxbc.nii')

# get target resolution
target_res = args['target_res']

assert os.path.isfile(name_input), "file does not exist: %s " \
                                    "\nplease make sure the path and the extension is correct" % name_input

# Do the actual bias correction
print('Bias correction of ' + name_input)

# load input image and reorient it
im, aff, hdr = tools.load_volume(name_input, im_only=False, dtype='float')
im, _, aff, _, _, hdr, im_res = tools.get_volume_info(name_input, return_volume=True)

# get original resolution
if (target_res < 0):
    target_res = np.array(hdr['pixdim'][1:4])

# we can use the label image as mask for WM
if args['label'] is not None:
    label, aff_label, hdr_label = tools.load_volume(os.path.abspath(args['label']), im_only=False, dtype='float')
    
    #mask = label > 2.5
    mask = label > 0
else:
    mask = None

bias = get_bias_nx(im, mask, im_res, aff)

print(im.shape)
# resample if necessary
#bias, aff = edit_volumes.align_volume_to_ref(bias, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
#bias, aff_resamp = edit_volumes.resample_volume(bias, aff, target_res)
#im, aff_resamp  = edit_volumes.resample_volume(im, aff, target_res)
print(bias.shape)
im = im / bias

# save output
tools.save_volume(im, aff, None, name_corrected)
