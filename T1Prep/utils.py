# python imports
import os
import sys
import traceback
import numpy as np

from ext.lab2im import edit_volumes
from ext.lab2im import utils as tools
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion, grey_opening, grey_closing, gaussian_filter

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

def suppress_vessels_and_skull_strip(volume, label, vessel_strength, res, vessel_mask=None):
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
        
    correct_dura = 0
    
    # obtain a threshold based on median for GM and CSF
    median_csf = percentile_csf[1] # percentile for 50%
    median_gm  = np.median(np.array(volume[label_gm]))
    th_csf = (median_csf + median_gm)/2
    
    # create mask where volume data are ~CSF
    volume_gt_csf = volume > th_csf
    
    # check for T2w-contrast and invert mask
    is_T1w = median_csf < median_gm
    if not is_T1w:
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
    if is_T1w:
        volume_fill_value = percentile_csf[0]
        label_fill_value  = 0.5*tissue_labels["CSF"]
    else:
        volume_fill_value = percentile_csf[1]
        label_fill_value  = tissue_labels["CSF"]
    
    volume[mask & ~eroded_mask & label_csf_loose & deviation_mask] = volume_fill_value

    """
    volume[mask & ~eroded_mask] += 400
    volume[label_csf_loose] += 800
    volume[deviation_mask] += 1600
    """
    label[ mask & ~eroded_mask & label_csf_loose & deviation_mask] = label_fill_value
    
    # remove remaining background
    volume[~mask] = 0

    return volume


def get_bias_field(volume, label, target_size, nu_strength=2, bias_weight=None):
    """
    Use label image to correct input volume for non-uniformities
    We estimate bias correction by smoothing the residuals that remain after
    using the mean vavlues inside the label values and repeat that with decreasing
    smoothing kernels.
    
    Args:
        volume (numpy.ndarray): The input volume to be corrected for non-uniformities.
        label (numpy.ndarray): The label image used to correct the input volume.
        target_size (tuple): The target voxel size of the output.
        nu_strength (int): The strength of the non-uniformity correction. Default is 2.
        bias_weight (float): The weight of the bias field. Default is None.

    Returns:
        numpy.ndarray: The bias field used to correct the input volume for non-uniformities.
    """

    # size of smoothing kernel in sigma w.r.t. target size and weighting
    if (nu_strength == 0):
        return np.zeros(shape=np.shape(volume), dtype='float32')
    elif (nu_strength == 1):
        sigma = [16, 12, 8, 6]
    elif (nu_strength == 2):
        sigma = [12, 9, 6, 3]
    elif (nu_strength == 3):
        sigma = [11, 8, 4, 2]
    elif (nu_strength == 4):
        sigma = [9, 6, 3, 1]
      
    # correct for target size
    sigma = sigma/np.array([np.mean(target_size)]*len(sigma))
    
    # we have to create a new volume to not overwrite the original one
    corrected_volume = volume + 0
    
    # we need the final bias field later to apply it to the resampled data
    bias = np.zeros(shape=np.shape(corrected_volume), dtype='float32')

    # we use decreasing smoothing sizes
    for sigma in sigma:
        bias_step = np.zeros(shape=np.shape(volume), dtype='float32')

        for i in range(0, 3):
            tissue_idx = np.round(label) == i + 1
            mean_tissue = np.mean(np.array(corrected_volume[tissue_idx]))
            bias_step[tissue_idx]  += (corrected_volume[tissue_idx] - mean_tissue);
        
        # weight bias field
        if bias_weight is not None:
            bias_step *= bias_weight
             
        bias_step = gaussian_filter(bias_step, sigma=sigma)
        corrected_volume -= bias_step;
        bias += bias_step
        
    # we have to set bias field outside label mask to 0
    bias[label == 0] = 0.0
    
    return bias

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
        LateralVentricle = (seg == regions["lLateralVentricle"]) | (seg == regions["lInfLatVent"])
        LateralVentricle = binary_dilation(LateralVentricle, tools.build_binary_structure(3, 3))
        # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
        LateralVentricle = LateralVentricle & ~(seg == regions["rLateralVentricle"]) & \
                           ~(seg == regions["rCerebralWM"]) & ~(seg == regions["CSF"]) & \
                           ~(seg == regions["lAmygdala"]) & ~(seg == regions["lHippocampus"])
        #WM             
        wm0 = ((seg >= regions["lThalamus"])         &  (seg < regions["3rdVentricle"])) | \
               (seg == regions["lAccumbensArea"])    |  (seg == regions["lVentralDC"])
        # we also have to dilate whole WM to close the remaining rims
        wm0 = binary_dilation(wm0, tools.build_binary_structure(2, 3)) | LateralVentricle

        # CSF + BKG
        csf0 = (seg == regions["Bkg"])               |  (seg == regions["lCerebellumWM"]) | \
               (seg == regions["lCerebellumCortex"]) |  (seg == regions["3rdVentricle"]) | \
               (seg == regions["4thVentricle"])      |  (seg == regions["BrainStem"]) | \
               (seg >= regions["rCerebralWM"])
        
        lesion_mask0 = seg == regions["lCerebralWM"]
        
    else:
        # first we have to dilate the ventricles because otherwise after filling there remains
        # a rim around it
        LateralVentricle = (seg == regions["rLateralVentricle"]) | (seg == regions["rInfLatVent"])
        LateralVentricle = binary_dilation(LateralVentricle, tools.build_binary_structure(3, 3))
        # don't use dilated ventricles in the opposite hemisphere or Amygdala/Hippocampus
        LateralVentricle = LateralVentricle & ~(seg == regions["lLateralVentricle"]) & \
                           ~(seg == regions["lCerebralWM"]) & ~(seg == regions["CSF"]) & \
                           ~(seg == regions["rAmygdala"]) & ~(seg == regions["rHippocampus"])
        # WM         
        wm0 =  ((seg >= regions["rThalamus"])         &  (seg <= regions["rPallidum"])) | \
                (seg == regions["rAccumbensArea"])    |  (seg == regions["rVentralDC"])
        # we also have to dilate whole WM to close the remaining rims
        wm0 = binary_dilation(wm0, tools.build_binary_structure(2, 3)) | LateralVentricle

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

