# python imports
import os
import sys
import traceback
import numpy as np

from ext.lab2im import edit_volumes
from ext.lab2im import utils as tools
from scipy.ndimage import binary_closing, binary_opening, grey_opening, grey_closing, gaussian_filter

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

    # obtain a threshold based on median for GM and CSF
    median_csf = percentile_csf[1] # percentile for 50%
    median_gm  = np.median(np.array(volume[label_gm]))
    th_csf = (median_csf + median_gm)/2
    
    # create mask where volume data are ~CSF
    volume_gt_csf = volume > th_csf
    
    # check for T2w-contrast and invert mask
    is_T2w = median_csf > median_gm
    if is_T2w:
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
    mask = label > 0
    
    # thickness of shell is resolution-dependent and will be controlled by a gaussian filter
    thickness_shell = 0.75/res 
    eroded_mask = gaussian_filter(mask, sigma=thickness_shell) > 0.5
    
    # only change values inside the shell where SynthSeg label is CSF and set values a bit
    # smaller than CSF because it's the outer shell and brighter spots would be more
    # difficult to deal with
    volume[mask & ~eroded_mask & label_csf_loose] = percentile_csf[0]
    label[ mask & ~eroded_mask & label_csf_loose] = 0.5*tissue_labels["CSF"]
    
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
    csf =  posterior[..., 3]  + posterior[..., 4]  + posterior[..., 11] + posterior[..., 12]
    csf += posterior[..., 16] + posterior[..., 21] + posterior[..., 22]
    
    # non-cortical GM should be smoothed before adding to GM
    im =  posterior[..., 7]  + posterior[..., 8]  + posterior[..., 9]  + posterior[..., 10]
    im += posterior[..., 14] + posterior[..., 15] + posterior[..., 17] + posterior[..., 18]
    im += posterior[..., 25] + posterior[..., 26] + posterior[..., 27] + posterior[..., 28]
    im += posterior[..., 29] + posterior[..., 30] + posterior[..., 31] + posterior[..., 32]
        
    # GM
    gm =  posterior[..., 2] + posterior[..., 6] + posterior[..., 20] + posterior[..., 24] + im

    # WM
    wm =  posterior[..., 1] + posterior[..., 5] + posterior[..., 13] + posterior[..., 19]
    wm += posterior[..., 23]

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
    
    # we have to round seg because we need the integer labels
    seg = np.round(seg)
    
    # CSF + BKG
    if hemi==1:
        csf = (seg == 0) | (seg == 7)  | (seg == 8)  | (seg == 14) | (seg == 15) | (seg == 16) | (seg > 40)
    else:
        csf = ((seg < 40) & ~(seg == 24)) | (seg == 46) | (seg == 47)
                
    # WM
    if hemi==1:
        wm = (seg == 4)  | (seg == 5)  | ((seg > 9) & (seg < 14))  | (seg == 26) | (seg == 28)
    else:
        wm = (seg == 43) | (seg == 44) | ((seg > 48) & (seg < 53)) | (seg == 58) | (seg == 60)

    # resample wm and csf to voxel size of Amap label 
    wm  = edit_volumes.resample_volume_like(amap, aff_amap, wm,  aff_seg, interpolation='linear')
    csf = edit_volumes.resample_volume_like(amap, aff_amap, csf, aff_seg, interpolation='linear')
    
    # finally round and convert wm and csf masks to boolean type because of interpolation during resampling
    wm  = np.round(wm)  > 0.5
    csf = np.round(csf) > 0.5

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is neccessary to create a new variable otherwise amap is also modified
    label = amap + 0 
    label[csf] = tissue_labels["CSF"]
    label[wm]  = tissue_labels["WM"]
    
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
    label_mask = label > tissue_labels["CSF"]
    label_mask = binary_opening(label_mask, tools.build_binary_structure(1, 3))
    label_mask = edit_volumes.get_largest_connected_component(label_mask)
    label_mask = binary_closing(label_mask, tools.build_binary_structure(5, 3))
    
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
    csf =  posterior[..., 0]  + posterior[..., 16]
            
    # GM
    if hemi==1:
        gm =  posterior[..., 2]  + posterior[..., 14] + posterior[..., 15]
    else:
        gm =  posterior[..., 20] + posterior[..., 29] + posterior[..., 30]
    
    # WM
    if hemi==1:
        wm =  posterior[..., 1]  + posterior[..., 3]  + posterior[..., 4]  + posterior[..., 7]
        wm += posterior[..., 8]  + posterior[..., 9]  + posterior[..., 10] + posterior[..., 11]
        wm += posterior[..., 17] + posterior[..., 18]
    else:
        wm =  posterior[..., 19] + posterior[..., 21] + posterior[..., 22] + posterior[..., 25]
        wm += posterior[..., 26] + posterior[..., 27] + posterior[..., 28] + posterior[..., 11]
        wm += posterior[..., 31] + posterior[..., 32]

    # build label with CSF=1, GM=2, and WM=3
    label = tissue_labels["CSF"]*csf + tissue_labels["GM"]*gm + tissue_labels["WM"]*wm

    # remove remaining small parts after thresholding
    label_mask = label > tissue_labels["CSF"]
    label_mask = edit_volumes.get_largest_connected_component(label_mask)
    label[~label_mask] = tissue_labels["CSF"]

    return label
    
def bbox_volume(volume, pad=0):

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

