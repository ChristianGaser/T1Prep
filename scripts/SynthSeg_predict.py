# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache 2.0 license
# terms, and this file has been changed.
#
# The original file this work derives from is found at:
# https://github.com/BBillot/SynthSeg/blob/0369118b9a0dbd410b35d1abde2529f0f46f9341/scripts/commands/SynthSeg_predict.py
#
# [September 2023] CHANGES:
#    * added a few new options


# python imports
import os
import sys
from argparse import ArgumentParser

# add main folder to python path and import ./ext/SynthSeg/predict_synthseg.py
home = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(home)
sys.path.append(os.path.join(home, 'ext'))

model_dir = os.path.join(home, 'models')
labels_dir = os.path.join(home, 'data/labels_classes_priors')

# set tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from T1Prep.predict import predict

# parse arguments
parser = ArgumentParser(description="SynthSeg_predict", epilog='\n')

# input/outputs


parser.add_argument("--i", metavar="file", required=True,
    help="Input image to segment.")
parser.add_argument("--o", metavar="file", required=True,
    help="Segmentation output.")
parser.add_argument("--robust", action="store_true", 
    help="(optional) Whether to use robust predictions (slower).")
parser.add_argument("--fast", action="store_true", 
    help="(optional) Bypass some processing for faster prediction.")
parser.add_argument("--vol", 
    help="(optional) Output CSV file with volumes for all structures and subjects.")
parser.add_argument("--qc", 
    help="(optional) Output CSV file with qc scores for all subjects.")
parser.add_argument("--hemi", 
    help="(optional) Hemispheric label output in target voxel size.")
parser.add_argument("--resample", 
    help="(optional) Image resampled to target voxl size.")
parser.add_argument("--label", 
    help="(optional) Label output.")
parser.add_argument("--target-res", type=float, default=0.5, 
    help="(optional) Target voxel size in mm for resampled and hemispheric label data that will be used for cortical surface extraction. Default is 0.5.")
parser.add_argument("--nu-strength", type=float, default=2, 
    help="(optional) Strength of nu-correction (0 - none, 1 - light, 2 - medium, 3 - strong, 4 - heavy). Default is 2.")
parser.add_argument("--threads", type=int, default=1, 
    help="(optional) Number of cores to be used. Default is 1.")
parser.add_argument("--cpu", action="store_true", 
    help="(optional) Enforce running with CPU rather than GPU.")


# check for no arguments
if len(sys.argv) < 2:
    print("\nMust provide at least -i or -o output flags.")
    parser.print_help()
    sys.exit(1)

# parse commandline
args = vars(parser.parse_args())

# check rage of nu_strength
if ((args['nu_strength']<0) | (args['nu_strength']>4)):
    print("\nParameter nu-strength must be in the range 0..4")
    parser.print_help()
    sys.exit(1)

# print SynthSeg version and checks boolean params for SynthSeg-robust
if args['robust']:
    args['fast'] = True
    version = 'SynthSeg-robust 2.0'
else:
    version = 'SynthSeg 2.0'
    if args['fast']:
        version += ' (fast)'
print(version)

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# limit the number of threads to be used if running on CPU
import tensorflow as tf
if args['threads'] == 1:
    print('using 1 thread')
else:
    print('using %s threads' % args['threads'])
tf.config.threading.set_inter_op_parallelism_threads(args['threads'])
tf.config.threading.set_intra_op_parallelism_threads(args['threads'])

# path models
if args['robust']:
    args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_robust_2.0.h5')
else:
    args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_2.0.h5')
args['path_model_parcellation'] = os.path.join(model_dir, 'synthseg_parc_2.0.h5')
args['path_model_qc'] = os.path.join(model_dir, 'synthseg_qc_2.0.h5')

# path labels
args['labels_segmentation'] = os.path.join(labels_dir, 'synthseg_segmentation_labels_2.0.npy')
args['labels_denoiser'] = os.path.join(labels_dir, 'synthseg_denoiser_labels_2.0.npy')
args['labels_parcellation'] = os.path.join(labels_dir, 'synthseg_parcellation_labels.npy')
args['labels_qc'] = os.path.join(labels_dir, 'synthseg_qc_labels_2.0.npy')
args['names_segmentation_labels'] = os.path.join(labels_dir, 'synthseg_segmentation_names_2.0.npy')
args['names_parcellation_labels'] = os.path.join(labels_dir, 'synthseg_parcellation_names.npy')
args['names_qc_labels'] = os.path.join(labels_dir, 'synthseg_qc_names_2.0.npy')
args['topology_classes'] = os.path.join(labels_dir, 'synthseg_topological_classes_2.0.npy')
args['n_neutral_labels'] = 19

# run prediction
predict(path_images=args['i'],
        path_segmentations=args['o'],
        path_model_segmentation=args['path_model_segmentation'],
        labels_segmentation=args['labels_segmentation'],
        robust=args['robust'],
        fast=args['fast'],
        v1=False, # we cannot use this option because we need the 33-class label
        do_parcellation=False, # we cannot use this option because we need the 33-calss label
        n_neutral_labels=args['n_neutral_labels'],
        names_segmentation=args['names_segmentation_labels'],
        labels_denoiser=args['labels_denoiser'],
        path_posteriors=None, # not necessary
        path_label=args['label'],
        path_hemi=args['hemi'],
        path_resampled=args['resample'],
        path_volumes=args['vol'],
        path_model_parcellation=args['path_model_parcellation'],
        labels_parcellation=args['labels_parcellation'],
        names_parcellation=args['names_parcellation_labels'],
        path_qc_scores=args['qc'],
        path_model_qc=args['path_model_qc'],
        labels_qc=args['labels_qc'],
        names_qc=args['names_qc_labels'],
        cropping=None, # not necessary
        topology_classes=args['topology_classes'],
        target_res=args['target_res'],
        nu_strength=args['nu_strength'])
