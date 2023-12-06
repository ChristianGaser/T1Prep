# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache 2.0 license
# terms, and this file has been changed.
#
# The original file this work derives from is found at:
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# add main folder to python path and import SynthSR packages
synthSR_home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(synthSR_home)

from ext.neuron import models as nrn_models
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from scipy.ndimage import gaussian_filter

# parse arguments
parser = ArgumentParser()
parser.add_argument("--i", metavar="file", required=True,
    help="Input image for bias correction.")
parser.add_argument("--o", metavar="file", required=True,
    help="Bias corrected output.")
parser.add_argument("--target-res", type=float, default=0.5, 
    help="(optional) Target voxel size in mm for resampled and hemispheric label data that will be used for cortical surface extraction. Default is 0.5. Use a negative value to save outputs with original voxel size.")
parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
parser.add_argument("--threads", type=int, default=1, dest="threads",
    help="number of threads to be used by tensorflow when running on CPU.")
parser.add_argument("--model", default=None, 
    help="(optional) Use a different model file.")
parser.add_argument("--disable_flipping", action="store_true", 
    help="(optional) Use this flag to disable flipping augmentation at test time.")

# check for no arguments
if len(sys.argv) < 2:
    print("\nMust provide at least -i or -o output flags.")
    parser.print_help()
    sys.exit(1)

args = vars(parser.parse_args())

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
if args['model'] is None:
    unet_model.load_weights(os.path.join(synthSR_home, 'models/SynthSR_v10_210712.h5'), by_name=True)
else:
    print('Using user-specified model: ' + args)
    unet_model.load_weights(args, by_name=True)

# Prepare list of images to process
path_images = os.path.abspath(args['i'])
basename = os.path.basename(path_images)
path_predictions = os.path.abspath(args['o'])

target_res = args['target_res']

assert os.path.isfile(path_images), "file does not exist: %s " \
                                    "\nplease make sure the path and the extension is correct" % path_images
images_to_segment = [path_images]
path_predictions = [path_predictions]

# Do the actual work
for idx, (path_image, path_prediction) in enumerate(zip(images_to_segment, path_predictions)):
    print('  Bias correction of ' + path_image)

    im, aff, hdr = utils.load_volume(path_image, im_only=False, dtype='float')
    #im, aff = edit_volumes.resample_volume(im, aff, [1.0, 1.0, 1.0])

    im, aff2 = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    im = im - np.min(im)
    im = im / np.max(im)
    I = im[np.newaxis, ..., np.newaxis]
    W = (np.ceil(np.array(I.shape[1:-1]) / 32.0) * 32).astype('int')
    idx = np.floor((W - I.shape[1:-1]) / 2).astype('int')
    S = np.zeros([1, *W, 1])
    S[0, idx[0]:idx[0] + I.shape[1], idx[1]:idx[1] + I.shape[2], idx[2]:idx[2] + I.shape[3], :] = I
    
    if args['disable_flipping']:
        output = unet_model.predict(S)
    else:
        output = 0.5 * unet_model.predict(S) + 0.5 * np.flip(unet_model.predict(np.flip(S, axis=1)), axis=1)
          
    pred = np.squeeze(output)
    pred = 255 * pred
    pred[pred < 0] = 0
    pred[pred > 255] = 255
    pred = pred[idx[0]:idx[0] + I.shape[1], idx[1]:idx[1] + I.shape[2], idx[2]:idx[2] + I.shape[3]]
    
    resamp, aff_resamp = edit_volumes.resample_volume(255*im, aff2, target_res)
    im = 255 * im - pred

    #im = gaussian_filter(im, sigma=2)
    bias, aff_resamp = edit_volumes.resample_volume(im, aff2, target_res)
    resamp = resamp - bias
    
    utils.save_volume(resamp, aff_resamp, None, path_prediction)
    utils.save_volume(bias, aff_resamp, None, 'bias.nii')