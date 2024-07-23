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
from ext.lab2im import utils as tools
from ext.lab2im import edit_volumes
from scipy.ndimage import gaussian_filter, distance_transform_edt, grey_opening, binary_opening
from T1Prep import utils

# only for debugging
import ipdb
from matplotlib import pyplot as plt

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

# parse arguments
parser = ArgumentParser()
parser.add_argument("--i", metavar="file", required=True,
    help="Input image for bias correction.")
parser.add_argument("--o", metavar="file", required=False,
    help="Bias corrected output.")
parser.add_argument("--s", metavar="file", required=False,
    help="SynthSR output.")
parser.add_argument("--label", metavar="file", required=False,
    help="Optional label image to handle WMHs and vessels.")
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
    print('Using user-specified model: ' + args['model'])
    unet_model.load_weights(args['model'], by_name=True)

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
print('Bias correction of ' + name_input)

# load input image and reorient it
im0, aff, hdr = tools.load_volume(name_input, im_only=False, dtype='float')
im0, aff = edit_volumes.align_volume_to_ref(im0, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
im0, aff2 = edit_volumes.resample_volume(im0, aff, [1.0, 1.0, 1.0])

# get original resolution
if (target_res < 0):
    target_res = np.array(hdr['pixdim'][1:4])

# pad image to shape dividible by 32
n_levels = 5
padding_shape = (np.ceil(np.array(im0.shape[:3]) / 2**n_levels) * 2**n_levels).astype('int')
im, pad_idx = edit_volumes.pad_volume(im0, padding_shape, return_pad_idx=True)

# normalize input image to a range 0..1
mn = np.min(im)
im = im - mn
mx = np.max(im)
im = im / mx

# add batch and channel axes
I = im[np.newaxis, ..., np.newaxis]

if args['enable_flipping']:
    output = 0.5 * unet_model.predict(I) + 0.5 * np.flip(unet_model.predict(np.flip(I, axis=1)), axis=1)
else:
    output = unet_model.predict(I)

# remove not needed (empty) dimensions and rescue old max value
pred = np.squeeze(output)
pred = mx * pred

# get image without padding
pred = edit_volumes.crop_volume_with_idx(pred, pad_idx, n_dims=3)

# ensue that no zeros are present and clip max
pred[pred < 1] = 1
pred[pred > mx] = mx

# get bias field
bias = np.zeros(np.shape(im0))
ind = (im0 > 0)
bias[ind] = im0[ind] / (pred[ind])
bias0 = bias + 0

# we can use the label image to identify WMHs and vessels
if args['label'] is not None:
    label, aff_label, hdr_label = tools.load_volume(os.path.abspath(args['label']), im_only=False, dtype='float')
    label, aff_label = edit_volumes.align_volume_to_ref(label, aff_label, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    label, aff_label = edit_volumes.resample_volume(label, aff_label, [1.0, 1.0, 1.0])
    
    # we need 3 labels without partial volume effects
    label_csf = np.round(label) == tissue_labels["CSF"]
    label_gm  = np.round(label) == tissue_labels["GM"]
    label_wm  = np.round(label) == tissue_labels["WM"]
    label_mask = label > 0
    
    # we need some percentiles for tissues
    percentile_wm  = np.percentile(np.array(bias[label_wm]), [10,90])    
    
    # medians for tissues
    median_csf = np.median(np.array(im0[label_csf]))
    median_gm  = np.median(np.array(im0[label_gm]))

    # check for T1w-contrast
    is_t1w = median_csf < median_gm
    
    WMHs = np.zeros(np.shape(label),dtype=bool)
    if is_t1w:
        WMHs[label_wm] = bias[label_wm] < percentile_wm[0]
    else:
        WMHs[label_wm] = bias[label_wm] > percentile_wm[0]
    
    # create mask where volume data are ~CSF
    vessels = np.zeros(np.shape(label),dtype=bool)

    if is_t1w:
        vessels[label_csf] = bias[label_csf] > percentile_wm[1]
    else:
        # we have to invert volume mask for low CSF values
        vessels[label_csf] = bias[label_csf] > percentile_wm[1]

    # obtain a mask of areas where percentile of bias values inside WM is outside 10..90
    mask_percentile = np.zeros(np.shape(label),dtype=bool)
    mask_percentile[label_wm] = (bias[label_wm] < percentile_wm[0]) | (bias[label_wm] > percentile_wm[1])
    
    bias[mask_percentile] = 0
    bias[vessels] = 0
    bias[WMHs] = 0

    # only keep percentiles 5..95
    percentile_bias = np.percentile(np.array(bias),[5,95])
    bias[bias < percentile_bias[0]] = 0
    bias[bias > percentile_bias[1]] = 0
    
    """
    # fill the holes using distance function
    bias_idx = bias > 0
    _, dist_idx = distance_transform_edt(~bias_idx, return_indices=True)
    bias = bias[dist_idx[0], dist_idx[1], dist_idx[2]]
    
    # initially apply Gaussian smoothing with doubled sigma
    bias_filtered = gaussian_filter(bias, sigma=2*args['bias_sigma'])
    im_corrected = im0 / bias_filtered
    # get residual bias field
    bias_residual = im_corrected - im0

    # median of residual bias for each tissue
    median_csf = np.median(np.array(bias_residual[label_csf]))
    median_gm  = np.median(np.array(bias_residual[label_gm]))
    median_wm  = np.median(np.array(bias_residual[label_wm]))
    
    # normalize the residual bias
    bias_residual[label_gm]  = bias_residual[label_gm] *median_wm/median_gm
    bias_residual[label_csf] = bias_residual[label_csf]*median_wm/median_csf

    percentile_brain  = np.percentile(np.array(bias_residual[label > 0]), [10,90])    
    
    # obtain a mask of areas where percentile of bias values inside brain is not in 5..95%
    mask_percentile = np.zeros(np.shape(label),dtype=bool)
    mask_percentile[label_mask] = (bias_residual[label_mask] < percentile_brain[0]) | (bias_residual[label_mask] > percentile_brain[1])

    bias = bias0
    bias = grey_opening(bias, size=(2,2,2))
    bias[mask_percentile] = 0
    bias[label < 0.1] = 0
    #ipdb.set_trace()


    # fill the holes using distance function
    bias_idx = bias > 0
    _, dist_idx = distance_transform_edt(~bias_idx, return_indices=True)
    bias = bias[dist_idx[0], dist_idx[1], dist_idx[2]]
    
    # finally apply Gaussian smoothing with defined sigma
    bias_filtered = gaussian_filter(bias, sigma=args['bias_sigma'])
    im_corrected = im0 / bias_filtered
        
    bias_residual_res, aff_resamp = edit_volumes.resample_volume(bias_residual, aff2, target_res)
    bias_name = name_input.replace('.nii', '_nu_residual.nii')
    tools.save_volume(bias_residual_res, aff_resamp, None, bias_name)
    """

#ipdb.set_trace()

# only keep percentiles 5..95
percentile_bias = np.percentile(np.array(bias),[5,95])
bias[bias < percentile_bias[0]] = 0
bias[bias > percentile_bias[1]] = 0

# fill the holes using distance function
bias_idx = bias > 0
_, dist_idx = distance_transform_edt(~bias_idx, return_indices=True)
bias = bias[dist_idx[0], dist_idx[1], dist_idx[2]]

# finally apply Gaussian smoothing
bias = gaussian_filter(bias, sigma=args['bias_sigma'])
im0 = im0 / bias

# resample if necessary
bias, aff_resamp = edit_volumes.resample_volume(bias, aff2, target_res)
im0, aff_resamp  = edit_volumes.resample_volume(im0, aff2, target_res)

# save output
bias_name = name_input.replace('.nii', '_nu.nii')
tools.save_volume(im0, aff_resamp, None, name_corrected)
tools.save_volume(bias, aff_resamp, None, bias_name)

# SynthSR name
if args['s'] is not None:
    name_synthsr = os.path.abspath(args['s'])
    tools.save_volume(pred, aff_resamp, None, name_synthsr)

