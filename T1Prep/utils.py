# python imports
import os
import sys
import traceback
import numpy as np

from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening, binary_dilation, binary_erosion, grey_opening, grey_closing, gaussian_filter
from ext.lab2im import edit_volumes
from ext.lab2im import utils as tools
from ext.nxbc.filter import *
from ext.SplineSmooth3D.SplineSmooth3D import SplineSmooth3D, SplineSmooth3DUnregularized
from skimage import filters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# globally define tissue labels or better understanding the applied thresholds 
# inside functions
tissue_labels = {
  "BKG": 0.0,
  "CSF": 1.0,
  "CGM": 1.5,
  "GM":  2.0,
  "GWM": 2.5,
  "WM":  3.0
}

# regions of SynthSeg segmentation
regions = {
  "Bkg":               0,
  "lCerebralWM":       2,
  "lCerebralCortex":   3,
  "lLateralVentricle": 4,
  "lInfLatVent":       5,
  "lCerebellumWM":     7,
  "lCerebellumCortex": 8,
  "lThalamus":         10,
  "lCaudate":          11,
  "lPutamen":          12,
  "lPallidum":         13,
  "3rdVentricle":      14,
  "4thVentricle":      15,
  "BrainStem":         16,
  "lHippocampus":      17,
  "lAmygdala":         18,
  "CSF":               24,
  "lAccumbensArea":    26,
  "lVentralDC":        28,
  "rCerebralWM":       41,
  "rCerebralCortex":   42,
  "rLateralVentricle": 43,
  "rInfLatVent":       44,
  "rCerebellumWM":     46,
  "rCerebellumCortex": 47,
  "rThalamus":         49,
  "rCaudate":          50,
  "rPutamen":          51,
  "rPallidum":         52,
  "rHippocampus":      53,
  "rAmygdala":         54,
  "rAccumbensArea":    58,
  "rVentralDC":        60
}

# numbered regions of SynthSeg segmentation as list for the posterior
post_regions = {
  "Bkg":               0,
  "lCerebralWM":       1,
  "lCerebralCortex":   2,
  "lLateralVentricle": 3,
  "lInfLatVent":       4,
  "lCerebellumWM":     5,
  "lCerebellumCortex": 6,
  "lThalamus":         7,
  "lCaudate":          8,
  "lPutamen":          9,
  "lPallidum":         10,
  "3rdVentricle":      11,
  "4thVentricle":      12,
  "BrainStem":         13,
  "lHippocampus":      14,
  "lAmygdala":         15,
  "CSF":               16,
  "lAccumbensArea":    17,
  "lVentralDC":        18,
  "rCerebralWM":       19,
  "rCerebralCortex":   20,
  "rLateralVentricle": 21,
  "rInfLatVent":       22,
  "rCerebellumWM":     23,
  "rCerebellumCortex": 24,
  "rThalamus":         25,
  "rCaudate":          26,
  "rPutamen":          27,
  "rPallidum":         28,
  "rHippocampus":      29,
  "rAmygdala":         30,
  "rAccumbensArea":    31,
  "rVentralDC":        32
}

def gradient3(F, axis):
    """
    Compute the gradient of a 3D array along a specified axis.

    This function calculates the gradient of a three-dimensional array `F` along
    one of the axes ('x', 'y', 'z', or 'xyz'). The gradient is computed using central
    differences in the interior points and first differences at the boundaries.
    This approach is similar to MATLAB's gradient function but is optimized to
    operate optionally along a single direction at a time, which can be more efficient
    in terms of time and memory usage.

    Parameters:
    F (numpy.ndarray): A 3D array of numerical values. The gradient will be
                       computed on this array.
    axis (str): The axis along which to compute the gradient. Valid options are
                'x', 'y', 'z', or 'xyz'. These correspond to the first, second, and
                third dimensions of the array, respectively or all dimensions.

    Returns:
    numpy.ndarray: A 3D array of the same shape as `F`. Each element of the
                   array represents the gradient of `F` at that point along the
                   specified axis.

    Example:
    >>> import numpy as np
    >>> F = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                      [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    >>> gradient_x = gradient3(F, 'x')
    >>> gradient_y = gradient3(F, 'y')
    >>> gradient_z = gradient3(F, 'z')

    """

    D = np.zeros_like(F)

    if axis == 'x':
        D[0,:,:]    =  F[1,:,:]  - F[0,:,:]
        D[-1,:,:]   =  F[-1,:,:] - F[-2,:,:]
        D[1:-1,:,:] = (F[2:,:,:] - F[:-2,:,:]) / 2
        return D
    elif axis == 'y':
        D[:,0,:]    =  F[:,1,:]  - F[:,0,:]
        D[:,-1,:]   =  F[:,-1,:] - F[:,-2,:]
        D[:,1:-1,:] = (F[:,2:,:] - F[:,:-2,:]) / 2
        return D
    elif axis == 'z':
        D[:,:,0]    =  F[:,:,1]  - F[:,:,0]
        D[:,:,-1]   =  F[:,:,-1] - F[:,:,-2]
        D[:,:,1:-1] = (F[:,:,2:] - F[:,:,:-2]) / 2
        return D
    elif axis == 'xyz':
        return gradient3(F,'x'),gradient3(F,'y'),gradient3(F,'z')

def divergence3(F):
    """
    Calculate the divergence of a 3D vector field.

    This function computes the divergence of a three-dimensional vector field `F`.
    The divergence is a scalar field representing the volume density of the outward
    flux of a vector field from an infinitesimal volume around a given point. In
    this case, `F` is interpreted as a vector field, and its divergence is computed
    as the sum of the second-order gradients along each axis (x, y, z).

    The computation involves calculating the gradient of the input array along each
    axis twice (second-order gradient) and summing these gradients to obtain the
    divergence. This approach provides an insight into the rate at which the quantity
    represented by `F` spreads out or converges at each point in space.

    Parameters:
    F (numpy.ndarray): A 3D array representing a vector field. The divergence will
                       be computed for this field.

    Returns:
    numpy.ndarray: A 3D array of the same shape as `F`. Each element of the array
                   represents the divergence of `F` at that point.

    Example:
    >>> import numpy as np
    >>> F = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                      [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    >>> div_F = divergence(F)
    """


    Dz  = gradient3(F,  'z')
    Dzz = gradient3(Dz, 'z')

    Dy  = gradient3(F,  'y')
    Dyy = gradient3(Dy, 'y')

    Dx  = gradient3(F,  'x')
    Dxx = gradient3(Dx, 'x')

    Div  = Dxx + Dyy + Dzz

    return Div

def get_bias_field(im, mask, im_res, aff):

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


def suppress_vessels_and_skull_strip(volume, label, vessel_strength, res, vessel_mask=None, debug=None):
    """
    Use label image to correct vessels in input volume and skull-strip
    the image by removing non-brain parts of the brain.
    """
    # we need 3 labels without partial volume effects
    label_csf = np.round(label) == tissue_labels["CSF"]
    label_gm  = np.round(label) == tissue_labels["GM"]
    label_wm  = np.round(label) == tissue_labels["WM"]    
    
    # for automatic estimation of vessel-strength we obtain the ratio between 99 and 
    # 50 percentile inside CSF where large values indicate 
    # the presence of high intensities in CSF areas (=vessels)
    if vessel_strength < 0: 
        # for automatic estimation of vessel-strength we use squared values to 
        # emphasize higher intensities in the percentiles
        values = volume*volume
        percentile_csf2 = np.percentile(np.array(values[label_csf]),[50,99])
        ratio_high = percentile_csf2[1] / percentile_csf2[0]
        
        # check wehther we have to perform strong correction or not
        if ratio_high >= 15:
            vessel_strength = 2
        else:
            vessel_strength = 1

    # create a mask with ones everywhere to not limit vessel correction spatially if 
    # vessel_mask is not defined
    if vessel_mask is None:
        vessel_mask = np.ones(shape=np.shape(volume), dtype='int8') > 0

    # we need the 1% and 50% percentiles for CSF
    percentile_csf = np.percentile(np.array(volume[label_csf]),[1,50])    

    # medians for CSF and GM
    median_csf = percentile_csf[1] # percentile for 50%
    median_gm  = np.median(np.array(volume[label_gm]))
    
    # obtain a threshold between median for GM and CSF
    th_csf = (median_csf + median_gm)/2

    # use maximum WM value to detect high-intensity areas that have larger intensities than WM
    max_wm = np.max(np.array(volume[label_wm]))    
    
    if debug is not None:
        values = volume*volume
        percentile_csf2 = np.percentile(np.array(values[label_csf]),[50,99])
        ratio_high = percentile_csf2[1] / percentile_csf2[0]
        print('Estimated CSF ratio is %g' % ratio_high)
        
    del label_wm, label_csf
    
    # get areas where label values are CSF, but also consider areas closer
    # to background
    label_csf_loose = (label < (tissue_labels["CSF"] + tissue_labels["GM"])/2)

    # check for T2w-contrast and invert mask in that case
    is_t1w = median_csf < median_gm
    if not is_t1w:
        print('Image contrast is not T1-weighted')
        # we have to invert volume mask for low CSF values
        volume_gt_csf = ~volume_gt_csf
        
    # we can use divergence which is a quite sensitive for finding vessels but also
    # meninges and other non-brain parts that are characterized by small structures
    # that only have small connections to the brain or have different intensity
    div_mask = divergence3(volume)
    
    #invert map
    div_mask = -div_mask

    # ignore negative values in the inverted map because we are now only interested 
    # in the pos. divergence
    div_mask[div_mask<0] = 0
    div_mask0 = div_mask + 0

    # obtain divergence mask using 99.9% percentile threshold and slightly
    # dilate mask
    percentile_div_mask = np.percentile(np.array(div_mask),[99,99.9])
    div_mask = (div_mask > percentile_div_mask[1])
    div_mask = binary_dilation(div_mask, tools.build_binary_structure(1, 3))
        
    
    # apply medium vessel correction either if automatical vessel correction is enabled
    # or if vessel correction is set to > 0
    if vessel_strength > 0:
        if vessel_strength < 2:
            print('Apply vessel-correction')

        volume_gt_wm = (volume > max_wm)
    
        # create mask where volume data are ~CSF
        volume_gt_csf = volume > th_csf

        # limit correction to areas inside CSF with high divergence values abobe threshold
        # and inside vessel mask
        mask_vessels = label_csf_loose & div_mask & vessel_mask

        # set high-intensity areas above threshold inside CSF to a CSF-like
        # intensity which works quite reasonable to remove large vessels
        volume[mask_vessels] = percentile_csf[0]

        # use a dilated mask which is smoothed to allow a weighted average
        mask_vessels = binary_dilation(mask_vessels, tools.build_binary_structure(4, 3))
        mask_vessels = gaussian_filter(1000*mask_vessels, sigma=0.1/res)/1000

        # smooth original data and use a weighted average to smoothly fill these
        # smoothed values inside the mask
        volume_smoothed = gaussian_filter(volume, sigma=1/res)
        volume = mask_vessels*volume_smoothed + (1-mask_vessels)*volume

    # optionally suppress structures with higher intensity in non-WM areas by replacing values
    # inside mask with grey-opening filtered output
    if vessel_strength == 2:
        print('Apply strong vessel-correction')
        volume_open = grey_opening(volume, size=(3,3,3))
        mask = (label_gm | label_csf_loose | volume_gt_wm) & volume_gt_csf & vessel_mask
        volume[mask] = volume_open[mask]

    del label_gm, vessel_mask, volume_gt_csf, volume_gt_wm, div_mask
    
    # obtain some outer shell of the label map to remove areas where SynthSeg label is CSF, but
    # we have higher intensity (e.g. due to dura or meninges)
    mask = label > 0.5
    bkg  = binary_opening(~mask, tools.build_binary_structure(1, 3))
    bkg  = edit_volumes.get_largest_connected_component(bkg)

    # fill isolated holes (estimated from background) in mask and finally apply closing 
    mask[~bkg] = 1
    mask = binary_closing(mask, tools.build_binary_structure(3, 3))
    
    del bkg

    # thickness of shell is resolution-dependent and will be controlled by a gaussian filter
    thickness_shell = 0.75/res 
    eroded_mask = gaussian_filter(mask, sigma=thickness_shell) > 0.5

    div_mask = (div_mask0 > percentile_div_mask[0])
    #div_mask = binary_dilation(div_mask, tools.build_binary_structure(1, 3))

    # only change values inside the shell where SynthSeg label is CSF and set values a bit
    # smaller than CSF because it's the outer shell and brighter spots would be more
    # difficult to deal with
    # use higher filling intensities for non-T1w images because CSF is much brighter
    if is_t1w:
        volume_fill_value = percentile_csf[0]
        label_fill_value  = 0.5*tissue_labels["CSF"]
    else:
        volume_fill_value = percentile_csf[1]
        label_fill_value  = tissue_labels["CSF"]

    div_mask = mask & ~eroded_mask & label_csf_loose
    volume[div_mask] = volume_fill_value

    div_mask = binary_dilation(div_mask, tools.build_binary_structure(1, 3))
    div_mask = gaussian_filter(1000*div_mask, sigma=2/res)/1000

    volume_smoothed = gaussian_filter(volume, sigma=1/res)
    volume = div_mask*volume_smoothed + (1-div_mask)*volume
    
    #label[div_mask>0.75 & mask] = label_fill_value

    # remove remaining background
    volume[~mask] = 0

    return volume, div_mask

def suppress_vessels_and_skull_strip_old(volume, label, vessel_strength, res, vessel_mask=None):
    """
    Use label image to correct vessels in input volume and skull-strip
    the image by removing non-brain parts of the brain.
    """

    # create a mask with ones everywhere to not limit vessel correction spatially
    if vessel_mask is None:
        vessel_mask = np.ones(shape=np.shape(volume), dtype='int8') > 0

    label_csf = np.round(label) == tissue_labels["CSF"]
    label_gm  = np.round(label) == tissue_labels["GM"]

    # for automatic estimation of vessel-strength we obtain the ratio between 99 and 
    # 50 percentile inside CSF where large values indicate 
    # the presence of high intensities in CSF areas (=vessels)
    if vessel_strength < 0: 
        # for automatic estimation of vessel-strength we use squared values to 
        # emphasize higher intensities in the percentiles
        values = volume*volume
        percentile_csf2 = np.percentile(np.array(values[label_csf]),[50,99])
        ratio_high = percentile_csf2[1] / percentile_csf2[0]
    elif vessel_strength == 0: # no correction
        ratio_high = 0
    elif vessel_strength == 1: # medium correction
        ratio_high = 10
    elif vessel_strength == 2: # strong correction
        ratio_high = 15

    # get areas where label values are CSF, but also consider areas closer
    # to background
    label_csf_loose = (label < (tissue_labels["CSF"]+tissue_labels["GM"])/2)

    # we need the 1% and 50% percentiles for CSF
    percentile_csf = np.percentile(np.array(volume[label_csf]),[1,50])

    # obtain a threshold based on median for GM and CSF
    median_csf = percentile_csf[1] # percentile for 50%
    median_gm  = np.median(np.array(volume[label_gm]))
    th_csf = (median_csf + median_gm)/2

    # create mask where volume data are ~CSF
    volume_gt_csf = volume > th_csf

    # check for T2w-contrast and invert mask
    is_t1w = median_csf < median_gm
    if not is_t1w:
        print('Image contrast is not T1-weighted')
        # we have to invert volume mask for low CSF values
        volume_gt_csf = ~volume_gt_csf

    if ratio_high >= 10:
        if ratio_high < 15:
            print('Apply medium vessel-correction')

        # set high-intensity areas above threshold inside CSF to a CSF-like
        # intensity which works quite reasonable to remove large vessels
        mask_csf = label_csf_loose & volume_gt_csf & vessel_mask
        # initially replace mask with median CSF
        volume[mask_csf] = percentile_csf[0]
        # and again replace mask with smoothed values to obtain a smoother
        # correction
        volume_smoothed = gaussian_filter(volume, sigma=1.0/res)
        volume[mask_csf] = volume_smoothed[mask_csf]


    # additionally suppress structures with higher intensity in non-WM areas by replacing values
    # inside mask with grey-opening filtered output
    if ratio_high >= 15:
        print('Apply strong vessel-correction')
        volume_open = grey_opening(volume, size=(3,3,3))
        mask = (label_gm | label_csf_loose) & volume_gt_csf & vessel_mask
        volume[mask] = volume_open[mask]

    # obtain some outer shell of the label map to remove areas where SynthSeg label is CSF, but
    # we have higher intensity (e.g. due to dura or meninges)
    mask = label > 0.5
    bkg  = binary_opening(~mask, tools.build_binary_structure(1, 3))
    bkg  = edit_volumes.get_largest_connected_component(bkg)

    # fill isolated holes (estimated from background) in mask and finally apply closing 
    mask[~bkg] = 1
    mask = binary_closing(mask, tools.build_binary_structure(3, 3))

    # only apply the dura-correction for T1w images, because surrounding CSF is quite sparse in
    # T2 or Flair and no dura is usually visible

    # thickness of shell is resolution-dependent and will be controlled by a gaussian filter
    thickness_shell = 0.75/res 
    eroded_mask = gaussian_filter(mask, sigma=thickness_shell) > 0.5

    # is the intensity closer to CSF or GM
    deviation_mask = np.abs(volume - median_csf) > np.abs(volume - median_gm)

    # only change values inside the shell where SynthSeg label is CSF and set values a bit
    # smaller than CSF because it's the outer shell and brighter spots would be more
    # difficult to deal with
    # use higher filling intensities for non-T1w images because CSF is much brighter
    if is_t1w:
        volume_fill_value = percentile_csf[0]
        label_fill_value  = 0.5*tissue_labels["CSF"]
    else:
        volume_fill_value = percentile_csf[1]
        label_fill_value  = tissue_labels["CSF"]

    #volume[mask & ~eroded_mask & label_csf_loose & deviation_mask] = volume_fill_value
    label[mask & ~eroded_mask & label_csf_loose & deviation_mask] = label_fill_value

    # remove remaining background
    volume[~mask] = 0

    return volume, mask

def posteriors2label(posterior):
    """
    Use posteriors with 33 classes to create a label image with CSF=1, GM=2, and WM=3
    while smoothing non-cortical GM to obtain a smoother border

    """

    # CSF
    csf =  posterior[..., post_regions["lLateralVentricle"]] + \
           posterior[..., post_regions["lInfLatVent"]] + \
           posterior[..., post_regions["3rdVentricle"]] + \
           posterior[..., post_regions["4thVentricle"]] + \
           posterior[..., post_regions["CSF"]] + \
           posterior[..., post_regions["rLateralVentricle"]] + \
           posterior[..., post_regions["rInfLatVent"]]


    # non-cortical GM should be smoothed before adding to GM
    im =  posterior[..., post_regions["lThalamus"]] + \
          posterior[..., post_regions["lCaudate"]] + \
          posterior[..., post_regions["lPutamen"]] + \
          posterior[..., post_regions["lPallidum"]] + \
          posterior[..., post_regions["lHippocampus"]] + \
          posterior[..., post_regions["lAmygdala"]] + \
          posterior[..., post_regions["lAccumbensArea"]] + \
          posterior[..., post_regions["lVentralDC"]] + \
          posterior[..., post_regions["rThalamus"]] + \
          posterior[..., post_regions["rCaudate"]] + \
          posterior[..., post_regions["rPutamen"]] + \
          posterior[..., post_regions["rPallidum"]] + \
          posterior[..., post_regions["rHippocampus"]] + \
          posterior[..., post_regions["rAmygdala"]] + \
          posterior[..., post_regions["rAccumbensArea"]] + \
          posterior[..., post_regions["rVentralDC"]]

    # GM
    gm =  posterior[..., post_regions["lCerebralCortex"]] + \
          posterior[..., post_regions["lCerebellumCortex"]] + \
          posterior[..., post_regions["rCerebralCortex"]] + \
          posterior[..., post_regions["rCerebellumCortex"]] + im

    # WM
    wm =  posterior[..., post_regions["lCerebralWM"]] + \
          posterior[..., post_regions["lCerebellumWM"]] + \
          posterior[..., post_regions["BrainStem"]] + \
          posterior[..., post_regions["rCerebralWM"]] + \
          posterior[..., post_regions["rCerebellumWM"]]
  
    label = tissue_labels["CSF"]*csf + tissue_labels["GM"]*gm + tissue_labels["WM"]*wm

    # we have to smooth the label in areas of non-cortical GM
    label2 = gaussian_filter(label, sigma=0.5)
    im2 = gaussian_filter(im, sigma=0.5)

    im_mask = im2 > 0
    label[im_mask] = label2[im_mask] 

    return label


def amap2hemiseg(amap, aff_amap, seg, aff_seg, hemi=1):
    """
    Use Amap label segmentation to create a hemispheric label image with CSF=1, GM=2, 
    and WM=3 where any subcortical structures or ventricles close to the midline are 
    filled with WM.
    Information about which regions should be filled is used from SynthSeg segmentation.
    Use hemi=1 for estimating the left and hemi=2 for the right hemisphere.
    
    Parameters:
    - amap (ndarray): The Amap label segmentation.
    - aff_amap (ndarray): The affine transformation matrix for amap.
    - seg (ndarray): The SynthSeg segmentation.
    - aff_seg (ndarray): The affine transformation matrix for seg.
    - hemi (int): The hemisphere to estimate. 1 for left hemisphere, 2 for right hemisphere.
    
    Returns:
    - label (ndarray): The hemispheric label image.
    """
    
    # just to ensure that some corrections are not made for the cerebellum in future
    is_cerebellum = 0

    # what to do with lesion
    lesion_filling = 1

    # we have to round seg because we need the integer labels
    seg = np.round(seg)

    if hemi==1:
        # first we have to dilate the ventricles because otherwise after filling there remains
        # a rim around it
        lateral_ventricle = (seg == regions["lLateralVentricle"]) | (seg == regions["lInfLatVent"])
        lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(3, 3))
        # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
        lateral_ventricle = lateral_ventricle & ~(seg == regions["rLateralVentricle"]) & \
                           ~(seg == regions["rCerebralWM"]) & ~(seg == regions["CSF"]) & \
                           ~(seg == regions["lAmygdala"]) & ~(seg == regions["lHippocampus"])
        #WM 
        wm0 = ((seg >= regions["lThalamus"])         &  (seg < regions["3rdVentricle"])) | \
               (seg == regions["lAccumbensArea"])    |  (seg == regions["lVentralDC"])
        # we also have to dilate whole WM to close the remaining rims
        wm0 = binary_dilation(wm0, tools.build_binary_structure(2, 3)) | lateral_ventricle

        # CSF + BKG
        csf0 = (seg == regions["Bkg"])               |  (seg == regions["lCerebellumWM"]) | \
               (seg == regions["lCerebellumCortex"]) |  (seg == regions["3rdVentricle"]) | \
               (seg == regions["4thVentricle"])      |  (seg == regions["BrainStem"]) | \
               (seg >= regions["rCerebralWM"])

        lesion_mask0 = seg == regions["lCerebralWM"]

    else:
        # first we have to dilate the ventricles because otherwise after filling there remains
        # a rim around it
        lateral_ventricle = (seg == regions["rLateralVentricle"]) | (seg == regions["rInfLatVent"])
        lateral_ventricle = binary_dilation(lateral_ventricle, tools.build_binary_structure(3, 3))
        # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
        lateral_ventricle = lateral_ventricle & ~(seg == regions["lLateralVentricle"]) & \
                           ~(seg == regions["lCerebralWM"]) & ~(seg == regions["CSF"]) & \
                           ~(seg == regions["rAmygdala"]) & ~(seg == regions["rHippocampus"])
        # WM 
        wm0 =  ((seg >= regions["rThalamus"])         &  (seg <= regions["rPallidum"])) | \
                (seg == regions["rAccumbensArea"])    |  (seg == regions["rVentralDC"])
        # we also have to dilate whole WM to close the remaining rims
        wm0 = binary_dilation(wm0, tools.build_binary_structure(2, 3)) | lateral_ventricle

        # CSF + BKG
        csf0 = ((seg <= regions["lVentralDC"])        & ~(seg == regions["CSF"])) | \
                (seg == regions["rCerebellumWM"])     |  (seg == regions["rCerebellumCortex"])

        lesion_mask0 = seg == regions["rCerebralWM"]

    wm  = edit_volumes.resample_volume_like(amap, aff_amap, wm0,  aff_seg, interpolation='nearest')
    csf = edit_volumes.resample_volume_like(amap, aff_amap, csf0, aff_seg, interpolation='nearest')

    # finally round and convert wm and csf masks to boolean type because of interpolation during resampling
    wm  = np.round(wm)  > 0.5
    csf = np.round(csf) > 0.5

    if lesion_filling:
        lesion_mask = edit_volumes.resample_volume_like(amap, aff_amap, lesion_mask0, aff_seg, interpolation='nearest')
        lesion_mask = np.round(lesion_mask) > 0.5


    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    label = amap + 0 
    label[csf] = tissue_labels["CSF"]
    label[wm]  = tissue_labels["WM"]

    # fill remaining holes in WM (e.g. due to WMHs)
    if not is_cerebellum & lesion_filling:
        # lesions are always inside deeper WM, thus we erode first
        lesion_mask = binary_erosion(lesion_mask, tools.build_binary_structure(2, 3))
        lesion_mask = binary_closing(lesion_mask, tools.build_binary_structure(2, 3))

        # set areas inside lesion_mask to WM if not yet WM
        label[lesion_mask & (label < tissue_labels["WM"])] = tissue_labels["WM"]
    else:
        label_mask = label > tissue_labels["GWM"]
        label_mask = binary_erosion(label_mask, tools.build_binary_structure(5, 3))
        label_mask = binary_closing(label_mask, tools.build_binary_structure(5, 3))
        label[label_mask] = tissue_labels["WM"]

    # grey opening is applied to non-WM areas only, because in WM it could lead to decreased 
    # values close to the GM border
    label_open = grey_opening(label, size=(3,3,3))
    label_mask = label < tissue_labels["GWM"] # limit to non-WM areas
    label[label_mask] = label_open[label_mask]

    # grey closing fills remaining thin lines and spots in WM
    label_closing = grey_closing(label, size=(3,3,3))
    label_mask = label > tissue_labels["GM"] # limit to WM
    label[label_mask] = label_closing[label_mask]

    # remove non-connected structures and keep largest component that is finally dilated
    # to ensure that no brain is removed in the final step where areas outside label_mask
    # are set to CSF
    if not is_cerebellum:
        label_mask = label > tissue_labels["CSF"]
        label_mask = binary_opening(label_mask, tools.build_binary_structure(3, 3))
        label_mask = edit_volumes.get_largest_connected_component(label_mask)
        label_mask = binary_closing(label_mask, tools.build_binary_structure(3, 3))

        # set areas outside label_mask to CSF
        label[~label_mask | (label == tissue_labels["BKG"])] = tissue_labels["CSF"]

    return label

def posteriors2hemiseg(posterior, hemi=1):
    """
    Use posteriors with 33 classes to create a hemispheric label image with CSF=1, GM=2, 
    and WM=3 where any subcortical structures or ventricles close to the midline are 
    filled with WM.
    Use hemi=1 for left and hemi=2 for the right hemisphere.
    """

    # CSF + BKG
    csf =  posterior[..., post_regions["Bkg"]] + \
           posterior[..., post_regions["CSF"]]

    # GM
    if hemi==1:
        gm =  posterior[..., post_regions["lCerebralCortex"]] + \
              posterior[..., post_regions["lHippocampus"]] + \
              posterior[..., post_regions["lAmygdala"]]
    else:
        gm =  posterior[..., post_regions["rCerebralCortex"]] + \
              posterior[..., post_regions["rHippocampus"]] + \
              posterior[..., post_regions["rAmygdala"]]

    # WM
    if hemi==1:
        wm =  posterior[..., post_regions["lCerebralWM"]] + \
              posterior[..., post_regions["lLateralVentricle"]] + \
              posterior[..., post_regions["lInfLatVent"]] + \
              posterior[..., post_regions["lThalamus"]] + \
              posterior[..., post_regions["lCaudate"]] + \
              posterior[..., post_regions["lPutamen"]] + \
              posterior[..., post_regions["lPallidum"]] + \
              posterior[..., post_regions["3rdVentricle"]] + \
              posterior[..., post_regions["lAccumbensArea"]] + \
              posterior[..., post_regions["lVentralDC"]]
    else:
        wm =  posterior[..., post_regions["rCerebralWM"]] + \
              posterior[..., post_regions["rLateralVentricle"]] + \
              posterior[..., post_regions["rInfLatVent"]] + \
              posterior[..., post_regions["rThalamus"]] + \
              posterior[..., post_regions["rCaudate"]] + \
              posterior[..., post_regions["rPutamen"]] + \
              posterior[..., post_regions["rPallidum"]] + \
              posterior[..., post_regions["3rdVentricle"]] + \
              posterior[..., post_regions["rAccumbensArea"]] + \
              posterior[..., post_regions["rVentralDC"]]

    # build label with CSF=1, GM=2, and WM=3
    label = tissue_labels["CSF"]*csf + tissue_labels["GM"]*gm + tissue_labels["WM"]*wm

    # remove remaining small parts after thresholding
    label_mask = label > tissue_labels["CSF"]
    label_mask = edit_volumes.get_largest_connected_component(label_mask)
    label[~label_mask] = tissue_labels["CSF"]

    return label


def bbox_volume(volume, pad=0):
    """
    Obtain bounding box for values > 0 and optionally add some padding
    """

    r = np.any(volume, axis=(1, 2))
    c = np.any(volume, axis=(0, 2))
    z = np.any(volume, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # pad defined voxel around bounding box
    if pad > 0:
      rmin -= pad
      cmin -= pad
      zmin -= pad
      rmax += pad
      cmax += pad
      zmax += pad

    return rmin, cmin, zmin, rmax, cmax, zmax

def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondence between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]

    return labels_segmentation, flip_indices, unique_idx


def write_csv(path_csv, data, unique_file, labels, names, skip_first=True, last_first=False):

    # initialisation
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names = names[unique_idx].tolist()
        if skip_first:
            names = names[1:]
        header = names
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

