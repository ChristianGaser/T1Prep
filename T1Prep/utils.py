# python imports
import os
import sys
import traceback
import numpy as np

from ext.lab2im import edit_volumes
from ext.lab2im import utils as tools
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening, grey_opening, grey_closing, gaussian_filter, distance_transform_edt, binary_fill_holes

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

def correct_and_skull_strip(volume, label, vessel_mask=None):
    """
    Use label image to correct input volume for non-uniformities and skull-strip
    the image by removing non-brain parts of the brain.
    """
    
    print('Vessel correction an skull-stripping')
    # create a mask with ones everywhere to not limit vessel correction spatially
    if vessel_mask is None:
        vessel_mask = np.ones(shape=np.shape(volume), dtype='int8') > 0

    # obtain a threshold based on median for GM and CSF
    median_csf = np.median(np.array(volume[np.round(label) == 1]))
    median_gm  = np.median(np.array(volume[np.round(label) == 2]))
    th_csf = (median_csf + median_gm)/2
    
    # set high-intensity areas above threshold inside CSF to a CSF-like
    # intensity which works quite reasonable to remove vessels
    mask = (label < (tissue_labels["CSF"]+tissue_labels["GM"])/2) & (volume > th_csf) & vessel_mask
    volume[mask] = 0.5*median_csf

    # additionally suppress structures with higher intensity in non-WM areas
    volume_open = grey_opening(volume, size=(3,3,3))
    mask = (label < (tissue_labels["GM"]+0.2)) & vessel_mask
    volume[mask] = volume_open[mask]

    # remove remaining background
    volume[label < 0.1] = 0

    # bias correction works better if it's called after vessel correction
    # we use decreasing smoothing sizes
    print('NU correction')
    for sigma in [24, 18, 12, 9, 6]:
        bias = np.zeros(shape=np.shape(volume), dtype='float32')

        for i in range(0, 3):
            tissue_idx = np.round(label) == i + 1
            mean_tissue = np.mean(np.array(volume[tissue_idx]))
            bias[tissue_idx]  += volume[tissue_idx] - mean_tissue;
                
        # sigma = fwhm / 2.354 / voxelsize
        # here we use fwhm of 20mm and have 0.5mm voxel size
        bias = gaussian_filter(bias, sigma=sigma)
        volume -= bias;

    return volume

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


def amap2hemiseg(amap, seg, hemi=1):
    """
    Use Amap label segmentation to create a hemispheric label image with CSF=1, GM=2, 
    and WM=3 where any subcortical structures or ventricles close to the midline are 
    filled with WM.
    Information about which regions should be filled is used from synthSeg segmentation.
    Use hemi=1 for estimating the left and hemi=2 for the right hemisphere.
    """
    
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

    # build hemispheric label with CSF=1, GM=2, and WM=3
    # adding 0 is necessary to create a new variable otherwise amap is also modified
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
