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

from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from T1Prep import utils

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
target_res = np.array([0.5]*3)
nu_strength = 3
vessel_strength = -1

im_res = np.array([1.0]*3)
path_images = args['orig']
seg, aff, h_seg = tools.load_volume(args['atlas'], im_only=False, dtype='float32')
label, aff_label, h_label = tools.load_volume(args['label'], im_only=False, dtype='float32')

#############################
# copied from utils.py
#############################
# use fast nu-correction with lower resolution of original preprocessed images
use_fast_nu_correction = False

resamp, aff_resamp, h_resamp = tools.load_volume(path_images, im_only=False, dtype='float32')

# resample original input to 1mm voxel size for fast nu-correction
if use_fast_nu_correction:
    im, aff_im = edit_volumes.resample_volume(resamp, aff_resamp, im_res)

# resample original input to target voxel size
resamp = edit_volumes.resample_volume_like(label, aff_label, resamp, aff_resamp, interpolation='linear')
aff_resamp = aff_label   
resamp, aff_resamp = edit_volumes.resample_volume(resamp, aff_resamp, target_res)

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

# resample cortex_mask to target voxel size except for fast nu-correction
if not use_fast_nu_correction:
    cortex_weight, aff_cortex = edit_volumes.resample_volume(cortex_weight, aff, target_res)

# correct vessels and skull-strip image
print('Vessel-correction and skull-stripping')
resamp = utils.suppress_vessels_and_skull_strip(resamp, label, vessel_strength, target_res, vessel_mask=cortex_mask)

# nu-correction works better if it's called after vessel correction
print('NU-correction')

# Using the 1mm data from SynthSeg is a bit faster
if use_fast_nu_correction:
    bias = utils.get_bias_field(im, label, im_res, nu_strength, bias_weight=cortex_weight)
    
    # resample bias field to the target voxel size of the resampled input volume
    bias, aff_bias = edit_volumes.resample_volume(bias, aff, target_res)
else:
    im_res_resampled = target_res
    bias = utils.get_bias_field(resamp, label, im_res_resampled, nu_strength, bias_weight=cortex_weight)

# apply nu-correction
resamp -= bias

# after nu-correction we might have negative values that should be prevented
min_resamp = np.min(np.array(resamp))
if (min_resamp < 0):
    resamp -= min_resamp
resamp[label < 0.1] = 0

#############################
# end: copied from utils.py
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
tools.save_volume(resamp, aff_resamp, h_resamp, name, dtype='int16')

name = args['orig'].replace('.nii', '_bias.nii')
tools.save_volume(bias, aff_resamp, h_resamp, name, dtype='int16')

"""
name = args['orig'].replace('.nii', '_weight.nii')
tools.save_volume(cortex_weight, aff_resamp, h_resamp, name, dtype='int16')
"""
 
