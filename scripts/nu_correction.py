# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache 2.0 license
# terms, and this file has been changed.
#
# The original SynthSR prediction where some ideas are based on is found at:
# https://github.com/BBillot/SynthSR/blob/7fc9cf7afb8875f3e21dfb0ff09bcbaf88c7cc99/scripts/predict_command_line.py
#
# [November 2023] CHANGES:
#    * changed upper limit for image intensity from 128 to 256
#    * updated arguments and argument handling 

# python imports
import os
import sys
import numpy as np
from argparse import ArgumentParser

# limit the number of threads to be used if running on CPU
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add main folder to python path and import SynthSR packages
synthSR_home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(synthSR_home)

from ext.neuron import models as nrn_models
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from scipy.ndimage import gaussian_filter, distance_transform_edt

# parse arguments
parser = ArgumentParser()
parser.add_argument("--i", metavar="file", required=True,
    help="Input image for bias correction.")
parser.add_argument("--o", metavar="file", required=False,
    help="Bias corrected output.")
parser.add_argument("--target-res", type=float, default=-1, 
    help="(optional) Target voxel size in mm for resampled and hemispheric label data that will be used for cortical surface extraction. Default is 0.5. Use a negative value to save outputs with original voxel size.")
parser.add_argument("--bias-sigma", type=float, default=3, 
    help="(optional) Kernel size (in sigma) for gaussian filtering of resulting bias field.")
parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
parser.add_argument("--threads", type=int, default=-1, dest="threads",
    help="number of threads to be used by tensorflow when running on CPU.")
parser.add_argument("--model", default=None, 
    help="(optional) Use a different model file.")
parser.add_argument("--enable_flipping", action="store_true", 
    help="(optional) Use this flag to enable flipping augmentation at test time.")

# check for no arguments
if len(sys.argv) < 1:
    print("\nMust provide at least -i flags.")
    parser.print_help()
    sys.exit(1)

args = vars(parser.parse_args())

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if (args['threads'] > 0):
    tf.config.threading.set_intra_op_parallelism_threads(args['threads'])

# Build Unet and load weights
unet_model = nrn_models.unet(nb_features=24,
                             input_shape=[None, None, None, 1],
                             nb_levels=5,
                             conv_size=3,
                             nb_labels=1,
                             feat_mult=2,
                             nb_conv_per_level=2,
                             conv_dropout=0,
                             final_pred_activation='linear',
                             batch_norm=-1,
                             activation='elu',
                             input_model=None)
                             
# load model file
if args['model'] is None:
    unet_model.load_weights(os.path.join(synthSR_home, 'models/SynthSR_v10_210712.h5'), by_name=True)
else:
    print('Using user-specified model: ' + args)
    unet_model.load_weights(args, by_name=True)

# Prepare list of images to process
name_input = os.path.abspath(args['i'])
basename = os.path.basename(name_input)

# output name
if args['o'] is not None:
    name_corrected = os.path.abspath(args['o'])
else:
    name_corrected = name_input.replace('.nii', '_nu-corrected.nii')

# get target resolution
target_res = args['target_res']

assert os.path.isfile(name_input), "file does not exist: %s " \
                                    "\nplease make sure the path and the extension is correct" % name_input

# Do the actual bias correction
print('  Bias correction of ' + name_input)

# load input image and reorient it
im, aff, hdr = utils.load_volume(name_input, im_only=False, dtype='float')
im, aff2 = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)

# normalize it to a range 0..1
mn = np.min(im)
im = im - mn
mx = np.max(im)
im = im / mx

# prepare input for unet
I = im[np.newaxis, ..., np.newaxis]

if args['enable_flipping']:
    output = 0.5 * unet_model.predict(I) + 0.5 * np.flip(unet_model.predict(np.flip(I, axis=1)), axis=1)
else:
    output = unet_model.predict(I)

# remove not needed (empty) dimensions and rescue old max value
pred = np.squeeze(output)
pred = mx * pred

# ensue that no zeros are present and clip max
pred[pred < 1] = 1
pred[pred > mx] = mx

# get bias field
bias = np.zeros(np.shape(im))
ind = (im > 0)
bias[ind] = im[ind] / (pred[ind]/mx)

# only keep percentiles 15..85
percentile_bias = np.percentile(np.array(bias),[15,85])
bias[bias < percentile_bias[0]] = 0
bias[bias > percentile_bias[1]] = 0

# fill the wholes using distance function
bias_idx = bias > 0
_, dist_idx = distance_transform_edt(~bias_idx, return_indices=True)
bias = bias[dist_idx[0], dist_idx[1], dist_idx[2]]

# finally apply Gaussian smoothing
bias = gaussian_filter(bias, sigma=args['bias_sigma'])
im = im / bias

# resample if necessary
if (target_res > 0):
    bias, aff_resamp = edit_volumes.resample_volume(bias, aff2, target_res)
    im, aff_resamp = edit_volumes.resample_volume(im, aff2, target_res)
else:
    aff_resamp = aff2

# save output
bias_name = name_input.replace('.nii', '_nu.nii')
utils.save_volume(im, aff_resamp, None, name_corrected)
utils.save_volume(bias, aff_resamp, None, bias_name)
