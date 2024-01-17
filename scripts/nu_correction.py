# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from T1Prep import utils
from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening, binary_dilation, binary_erosion, grey_opening, grey_closing, gaussian_filter

def get_bias_field(volume, label, target_size, nu_strength=2):
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

    Returns:
        numpy.ndarray: The bias field used to correct the input volume for non-uniformities.
    """

    # size of smoothing kernel in sigma w.r.t. target size and weighting
    if (nu_strength == 0):
        return np.ones(shape=np.shape(volume), dtype='float32')
    elif (nu_strength == 1):
        sigma = [10]
    elif (nu_strength == 2):
        sigma = [8]
    elif (nu_strength == 3):
        sigma = [6]
    elif (nu_strength == 4):
        sigma = [4]

    # correct for target size
    sigma = sigma/np.array([np.mean(target_size)]*len(sigma))

    # we have to create a new volume to not overwrite the original one
    corrected_volume = volume + 0

    # we need the final bias field later to apply it to the resampled data
    bias = np.zeros(shape=np.shape(corrected_volume), dtype='float32')

    # we need a tight brainmask without remaining small parts that are only
    # connected by a few voxels
    brain_idx = label > 1.5
    brain_idx = binary_opening(brain_idx, tools.build_binary_structure(3, 3))
    brain_idx = binary_erosion(brain_idx, tools.build_binary_structure(3, 3))

    # use GM and WM only to estimate bias
    used_labels = [1, 2]
    
    # we use decreasing smoothing sizes if defined
    for sigma in sigma:

        bias_tissue = np.zeros(shape=np.shape(volume), dtype='float32')

        for i in used_labels:
            tissue_idx = np.round(label) == i + 1
            mean_tissue = np.mean(np.array(corrected_volume[tissue_idx]))

            bias_tissue[tissue_idx] += (corrected_volume[tissue_idx] / mean_tissue)

        _, dist_idx = distance_transform_edt(~brain_idx, return_indices=True)
        bias_tissue = bias_tissue[dist_idx[0], dist_idx[1], dist_idx[2]]

        bias_tissue = gaussian_filter(bias_tissue, sigma=sigma)

        corrected_volume[brain_idx] /= bias_tissue[brain_idx];
        bias += bias_tissue

    return bias

# parse arguments
parser = ArgumentParser(description="nu_correction", epilog='\n')

# input/outputs
parser.add_argument("--orig", 
    help="Original image to correct")
parser.add_argument("--atlas", 
    help="Atlas from SynthSeg")
parser.add_argument("--label", 
    help="Label segmentation")


# check for no arguments
if len(sys.argv) < 3:
    print("\nMust provide at least --orig --label --atlas flags.")
    parser.print_help()
    sys.exit(1)

# parse commandline
args = vars(parser.parse_args())

target_res = np.array([1.0]*3)
target_res = np.array([0.7]*3)
nu_strength = 3
vessel_strength = -1

im_res = np.array([1.0]*3)
path_images = args['orig']
seg, aff, h_seg = tools.load_volume(args['atlas'], im_only=False, dtype='float32')
label, aff_label, h_label = tools.load_volume(args['label'], im_only=False, dtype='float32')

#############################
# copied from predict.py
#############################
# use fast nu-correction with lower resolution of original preprocessed images
use_fast_nu_correction = False

resamp, aff_resamp, h_resamp = tools.load_volume(path_images, im_only=False, dtype='float32')

# resample original input to 1mm voxel size for fast nu-correction
if use_fast_nu_correction:
    im, _ = edit_volumes.resample_volume(resamp, aff_resamp, im_res)

# resample original input to target voxel size
resamp = edit_volumes.resample_volume_like(label, aff_label, resamp, aff_resamp, interpolation='linear')
aff_resamp = aff_label   
resamp, aff_resamp = edit_volumes.resample_volume(resamp, aff_resamp, target_res)

# limit vessel correction to cerebral cortex (+hippocampus+amygdala+CSF) only
cortex_mask = (seg == 2)  | (seg == 3)  | (seg == 41) | (seg == 42) | (seg == 24) | \
              (seg == 17) | (seg == 18) | (seg == 53) | (seg == 54) | (seg == 0)

# resample cortex_mask to target voxel size
cortex_mask, _ = edit_volumes.resample_volume(cortex_mask, aff, target_res)

# finally convert to boolean type for the mask and round because of resampling
cortex_mask = np.round(cortex_mask) > 0.5

# create mask for weighting nu-correction
cortex_weight = np.ones(shape=np.shape(seg), dtype='float32')
# weight subcortical structures and cerebellum with 50%
cortex_weight[((seg > 9)  & (seg < 14)) | ((seg > 48) & (seg < 53)) | (seg== 8) | (seg < 47)] = 0.5
# ignore Amygdala + Hippocampus 
cortex_weight[(seg == 17) | (seg == 18) | (seg == 53) | (seg == 54)] = 0.0

# resample cortex_mask to target voxel size except for fast nu-correction
if not use_fast_nu_correction:
    cortex_weight, _ = edit_volumes.resample_volume(cortex_weight, aff, target_res)

# correct vessels and skull-strip image
print('Vessel-correction and skull-stripping')
resamp, mask = utils.suppress_vessels_and_skull_strip(resamp, label, vessel_strength, target_res, vessel_mask=cortex_mask)

# nu-correction works better if it's called after vessel correction
print('NU-correction')

# Using the 1mm data from SynthSeg is a bit faster
if use_fast_nu_correction:
    bias = get_bias_field(im, label, im_res, nu_strength)
    
    # resample bias field to the target voxel size of the resampled input volume
    bias, _ = edit_volumes.resample_volume(bias, aff, target_res)
else:
    im_res_resampled = target_res
    bias = get_bias_field(resamp, label, im_res_resampled, nu_strength)

# apply nu-correction
tissue_idx = bias != 0 
resamp[tissue_idx] /= bias[tissue_idx]

# after nu-correction we might have negative values that should be prevented
min_resamp = np.min(np.array(resamp))
if (min_resamp < 0):
    resamp -= min_resamp
resamp[~mask] = 0

#############################
# end: copied from predict.py
#############################
"""
print('load images')
resamp, aff_resamp, h_resamp = tools.load_volume(args['orig'], im_only=False, dtype='float32')
seg, aff, h_seg = tools.load_volume(args['atlas'], im_only=False, dtype='float32')
label, aff_label, h_label = tools.load_volume(args['label'], im_only=False, dtype='float32')

print('resample')
# resample to target voxel size
resamp, aff_resamp = edit_volumes.resample_volume(resamp, aff_resamp, target_res)
label  = edit_volumes.resample_volume_like(resamp, aff_resamp, label,  aff_label, interpolation='linear')

print('create masks')
# limit vessel correction to cerebral cortex (+hippocampus+amygdala+CSF) only
cortex_mask = (seg == 2)  | (seg == 3)  | (seg == 41) | (seg == 42) | (seg == 24) | \
              (seg == 17) | (seg == 18) | (seg == 53) | (seg == 54) | (seg == 0)

# resample cortex_mask to target voxel size
cortex_mask, aff_cortex = edit_volumes.resample_volume(cortex_mask, aff, target_res)

# finally convert to boolean type for the mask and round because of resampling
cortex_mask = np.round(cortex_mask) > 0.5
        
# create mask for weighting nu-correction
cortex_weight = np.ones(shape=np.shape(seg), dtype='float32')
# weight subcortical structures and cerebellum with 50%
cortex_weight[((seg > 9)  & (seg < 14)) | ((seg > 48) & (seg < 53)) | (seg== 8) | (seg < 47)] = 0.5
# ignore Amygdala + Hippocampus 
cortex_weight[(seg == 17) | (seg == 18) | (seg == 53) | (seg == 54)] = 0.0

# resample cortex_mask to target voxel size
cortex_weight, aff_cortex = edit_volumes.resample_volume(cortex_weight, aff, target_res)

# nu-correction works better if it's called after vessel correction
print('NU-correction')

im_res_resampled = target_res
bias = utils.get_bias_field(resamp, label, im_res_resampled, nu_strength, bias_weight=cortex_weight)

# apply nu-correction
resamp -= bias

# after nu-correction we might have negative values that should be prevented
min_resamp = np.min(np.array(resamp))
if (min_resamp < 0):
    resamp -= min_resamp
resamp[label < 0.1] = 0

# correct vessels and skull-strip image
print('Vessel-correction and skull-stripping')
resamp = utils.suppress_vessels_and_skull_strip(resamp, label, -1, target_res, vessel_mask=cortex_mask)
"""

name = args['orig'].replace('.nii', '_corrected.nii')
tools.save_volume(resamp, aff_resamp, h_resamp, name, dtype='float32')

name = args['orig'].replace('.nii', '_bias.nii')
tools.save_volume(bias, aff_resamp, h_resamp, name, dtype='float32')

"""
name = args['orig'].replace('.nii', '_weight.nii')
tools.save_volume(cortex_weight, aff_resamp, h_resamp, name, dtype='int16')
"""
 
