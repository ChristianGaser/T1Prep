#!/usr/bin/env python3

import os
import sys
import csv
import glob
import time
import argparse
import traceback
import surfa as sf
import numpy as np
import nibabel as nib
from datetime import timedelta
from itertools import combinations
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening, grey_opening, grey_closing, gaussian_filter, distance_transform_edt, binary_fill_holes

# set tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

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

region_labels = {
    2:  "Left Cerebral WM",
    3:  "Left Cerebral Cortex",
    4:  "Left Lateral Ventricle",
    5:  "Left Inf Lat Vent",
    7:  "Left Cerebellum WM",
    8:  "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brain Stem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    24: "CSF",
    26: "Left Accumbens area",
    28: "Left VentralDC",
    41: "Right Cerebral WM",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inf Lat Vent",
    46: "Right Cerebellum WM",
    47: "Right Cerebellum Cortex", 
    49: "Right Thalamus",
    50: "Right Caudate",
    51: "Right Putamen",
    52: "Right Pallidum",
    53: "Right Hippocampus",
    54: "Right Amygdala",
    58: "Right Accumbens area",
    60: "Right VentralDC"
}

# ================================================================================================
#                                         Main Entrypoint
# ================================================================================================


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="SynthSeg", epilog='\n')

    # input/outputs
    parser.add_argument("--i", help="Image(s) to segment.")
    parser.add_argument("--o", help="Segmentation output(s).")
    parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
    parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
    parser.add_argument("--fast", action="store_true", help="(optional) Bypass some processing for faster prediction.")
    parser.add_argument("--vol", help="(optional) Output CSV file with volumes for all structures and subjects.")
    parser.add_argument("--qc", help="(optional) Output CSV file with qc scores for all subjects.")
    parser.add_argument("--post", help="(optional) Posteriors output(s).")
    parser.add_argument("--label", help="(optional) Label output(s).")
    parser.add_argument("--hemi", help="(optional) Hemispheric label output(s).")
    parser.add_argument("--resample", help="(optional) Resampled image(s).")
    parser.add_argument("--crop", nargs='+', type=int, help="(optional) Only analyse an image patch of the given size.")
    parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
    parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")

    # check for no arguments
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    # parse commandline
    args = parser.parse_args()

    # print SynthSeg version and checks boolean params for SynthSeg-robust
    if args.robust:
        args.fast = True
        if args.v1:
            sf.system.fatal('The flag --v1 cannot be used with --robust since SynthSeg-robust only came out with 2.0.')
        version = 'SynthSeg-robust 2.0'
    else:
        version = 'SynthSeg 1.0' if args.v1 else 'SynthSeg 2.0'
        if args.fast:
            version += ' (fast)'
    print(version)

    # enforce CPU processing if necessary
    if args.cpu:
        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # limit the number of threads to be used if running on CPU
    if args.threads == 1:
        print('using 1 thread')
    else:
        print('using %s threads' % args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # check that freesurfer has been sourced
    synthseg_home = './'

    # path models
    if args.robust:
        path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_robust_2.0.h5')
    else:
        path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_2.0.h5')
    path_model_parcellation = os.path.join(synthseg_home, 'models', 'synthseg_parc_2.0.h5')
    path_model_qc = os.path.join(synthseg_home, 'models', 'synthseg_qc_2.0.h5')

    # path labels
    labels_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_segmentation_labels_2.0.npy')
    labels_denoiser = os.path.join(synthseg_home, 'models', 'synthseg_denoiser_labels_2.0.npy')
    labels_parcellation = os.path.join(synthseg_home, 'models', 'synthseg_parcellation_labels.npy')
    labels_qc = os.path.join(synthseg_home, 'models', 'synthseg_qc_labels_2.0.npy')
    names_segmentation_labels = os.path.join(synthseg_home, 'models', 'synthseg_segmentation_names_2.0.npy')
    names_parcellation_labels = os.path.join(synthseg_home, 'models', 'synthseg_parcellation_names.npy')
    names_qc_labels = os.path.join(synthseg_home, 'models', 'synthseg_qc_names_2.0.npy')
    topology_classes = os.path.join(synthseg_home, 'models', 'synthseg_topological_classes_2.0.npy')
    n_neutral_labels = 19

    # use v1 model if needed
    if args.v1:
        path_model_segmentation = os.path.join(synthseg_home, 'models', 'synthseg_1.0.h5')  # to modify
        labels_segmentation = labels_segmentation.replace('_2.0.npy', '.npy')
        labels_qc = labels_qc.replace('_2.0.npy', '.npy')
        names_segmentation_labels = names_segmentation_labels.replace('_2.0.npy', '.npy')
        names_qc_labels = names_qc_labels.replace('_2.0.npy', '.npy')
        topology_classes = topology_classes.replace('_2.0.npy', '.npy')
        n_neutral_labels = 18

    # run prediction
    predict(
        path_images=args.i,
        path_segmentations=args.o,
        path_model_segmentation=path_model_segmentation,
        labels_segmentation=labels_segmentation,
        robust=args.robust,
        fast=args.fast,
        v1=args.v1,
        do_parcellation=args.parc,
        n_neutral_labels=n_neutral_labels,
        names_segmentation=names_segmentation_labels,
        labels_denoiser=labels_denoiser,
        path_posteriors=args.post,
        path_label=args.label,
        path_hemi=args.hemi,
        path_resampled=args.resample,
        path_volumes=args.vol,
        path_model_parcellation=path_model_parcellation,
        labels_parcellation=labels_parcellation,
        names_parcellation=names_parcellation_labels,
        path_qc_scores=args.qc,
        path_model_qc=path_model_qc,
        labels_qc=labels_qc,
        names_qc=names_qc_labels,
        cropping=args.crop,
        topology_classes=topology_classes,
    )


# ================================================================================================
#                                 Prediction and Processing Utilities
# ================================================================================================


def predict(path_images,
            path_segmentations,
            path_model_segmentation,
            labels_segmentation,
            robust,
            fast,
            v1,
            n_neutral_labels,
            names_segmentation,
            labels_denoiser,
            path_posteriors,
            path_label,
            path_hemi,
            path_resampled,
            path_volumes,
            do_parcellation,
            path_model_parcellation,
            labels_parcellation,
            names_parcellation,
            path_qc_scores,
            path_model_qc,
            labels_qc,
            names_qc,
            cropping,
            topology_classes):
    '''
    Prediction pipeline.
    '''

    # prepare input/output filepaths
    outputs = prepare_output_files(path_images, path_segmentations, path_posteriors, path_label, path_hemi, \
                                   path_resampled, path_volumes, path_qc_scores)
    path_images = outputs[0]
    path_segmentations = outputs[1]
    path_posteriors = outputs[2]
    path_label = outputs[3]
    path_hemi = outputs[4]
    path_resampled = outputs[5]
    path_volumes = outputs[6]
    unique_vol_file = outputs[7]
    path_qc_scores = outputs[8]
    unique_qc_file = outputs[9]

    # get label lists
    labels_segmentation, _ = get_list_labels(label_list=labels_segmentation)
    if (n_neutral_labels is not None) & (not fast) & (not robust):
        labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
    else:
        labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
        flip_indices = None

    # prepare other labels list
    names_segmentation = load_array_if_path(names_segmentation)[unique_idx]
    topology_classes = load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]
    labels_denoiser = np.unique(get_list_labels(labels_denoiser)[0])
    if do_parcellation:
        labels_parcellation, unique_i_parc = np.unique(get_list_labels(labels_parcellation)[0], return_index=True)
        labels_volumes = np.concatenate([labels_segmentation, labels_parcellation[1:]])
        names_parcellation = load_array_if_path(names_parcellation)[unique_i_parc][1:]
        names_volumes = np.concatenate([names_segmentation, names_parcellation])
    else:
        labels_volumes = labels_segmentation
        names_volumes = names_segmentation
    if not v1:
        labels_volumes = np.concatenate([labels_volumes, np.array([np.max(labels_volumes + 1)])])
        names_volumes = np.concatenate([names_volumes, np.array(['total intracranial'])])
    do_qc = True if path_qc_scores[0] is not None else False
    if do_qc:
        labels_qc = get_list_labels(labels_qc)[0][unique_idx]
        names_qc = load_array_if_path(names_qc)[unique_idx]

    # set cropping/padding
    if cropping is not None:
        cropping = reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128

    # prepare volume/QC files if necessary
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, labels_volumes, names_volumes, last_first=(not v1))
    if unique_qc_file & do_qc:
        write_csv(path_qc_scores[0], None, True, labels_qc, names_qc)

    # build network
    net = build_model(model_file_segmentation=path_model_segmentation,
                      model_file_parcellation=path_model_parcellation,
                      model_file_qc=path_model_qc,
                      labels_segmentation=labels_segmentation,
                      labels_denoiser=labels_denoiser,
                      labels_parcellation=labels_parcellation,
                      labels_qc=labels_qc,
                      flip_indices=flip_indices,
                      robust=robust,
                      do_parcellation=do_parcellation,
                      do_qc=do_qc)

    # perform segmentation
    if len(path_images) <= 10:
        loop_info = LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = LoopInfo(len(path_images), 10, 'predicting', True)
    list_errors = list()
    for i in range(len(path_images)):
        loop_info.update(i)

        try:

            # preprocessing
            image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image=path_images[i],
                                                                         crop=cropping,
                                                                         min_pad=min_pad,
                                                                         path_resample=path_resampled[i])
                                                                         

            # prediction
            shape_input = add_axis(np.array(image.shape[1:-1]))
            if do_parcellation & do_qc:
                post_patch_segmentation, post_patch_parcellation, qc_score = net.predict([image, shape_input])
            elif do_parcellation & (not do_qc):
                post_patch_segmentation, post_patch_parcellation = net.predict(image)
                qc_score = None
            elif (not do_parcellation) & do_qc:
                post_patch_segmentation, qc_score = net.predict([image, shape_input])
                post_patch_parcellation = None
            else:
                post_patch_segmentation = net.predict(image)
                post_patch_parcellation = qc_score = None

            # postprocessing
            seg, posteriors, volumes = postprocess(post_patch_seg=post_patch_segmentation,
                                                   post_patch_parc=post_patch_parcellation,
                                                   shape=shape,
                                                   pad_idx=pad_idx,
                                                   crop_idx=crop_idx,
                                                   labels_segmentation=labels_segmentation,
                                                   labels_parcellation=labels_parcellation,
                                                   aff=aff,
                                                   im_res=im_res,
                                                   fast=fast,
                                                   topology_classes=topology_classes,
                                                   v1=v1)
                                                   
            # write predictions to disc
            save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
            if path_posteriors[i] is not None:
                save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')
            
            # write label image
            if path_label[i] is not None:
                print('Estimate label')
                label0 = posteriors2label(posteriors)
                
                # resample to 0.5mm voxel size
                im_res = np.array([.5]*3)
                label, aff_label = resample_volume(label0, aff, im_res)

                save_volume(label, aff_label, h, path_label[i], dtype='uint8')

            # write hemi images
            if path_hemi[i] is not None:
                
                hemi_str = ['left', 'right']
                
                for j in [0]:
                    print('Estimate hemispheric label for %s hemisphere' % hemi_str[j])
                    hemi_name = path_hemi[i].replace('.nii', '_%s.nii' % hemi_str[j])
                    hemi = posteriors2hemiseg(posteriors, hemi=j+1)
                    
                    # resample to 0.5mm voxel size
                    im_res = np.array([.5]*3)
                    hemi, aff_hemi = resample_volume(hemi, aff, im_res)

                    # crop hemi image and add 2 voxels
                    crop_idx = bbox_volume(hemi > 1, pad=2)
                    hemi, aff_hemi = crop_volume_with_idx(hemi, crop_idx, aff=aff_hemi, n_dims=3, return_copy=False)
                    
                    save_volume(hemi, aff_hemi, h, hemi_name, dtype='uint8')

            if path_resampled[i] is not None:
                print('Apply nu-correction and skull-stripping' )
                resamp, aff_resamp, h_resamp = load_volume(path_images[i], im_only=False, dtype='float32')

                # resample to 0.5mm voxel size
                im_res = np.array([.5]*3)
                resamp, aff_resamp = resample_volume(resamp, aff_resamp, im_res)

                # resample seg to 0.5mm voxel size using nn-interpolation because its a label map
                im_res = np.array([.5]*3)
                seg, aff_seg = resample_volume(seg, aff, im_res, interpolation='nearest')
                
                # finally we need integer values for the labels
                seg = np.round(seg)
                
                # limit vessel correction to cerebral cortex (+hippocampus+amygdala+CSF) only
                cortex_mask = (seg == 3)  | (seg == 4)  | (seg == 41) | (seg == 42) | (seg == 24) | \
                              (seg == 17) | (seg == 18) | (seg == 53) | (seg == 54) | (seg == 0)

                save_volume(resamp, aff_resamp, h, './resamp.nii', dtype='float32')
                save_volume(label, aff_resamp, h, './seg.nii', dtype='float32')

                # correct intensity non-uniformities and vessels, and skull-strip image
                resamp = correct_and_skull_strip(resamp, label, vessel_mask=cortex_mask)
                
                save_volume(resamp, aff_resamp, h, path_resampled[i])
                
                # call Amap segmentation
                print('Segment image using Amap')
                cmd = 'CAT_VolAmap -mrf 0 -sub 128 -write_seg 1 1 1 -iters_icm 0 -label ' + path_label[i]
                cmd += ' ' + path_resampled[i]

                if os.system(cmd) > 0:
                    print('CAT_VolAmap could not be called')
                              
                path_amap = path_resampled[i].replace('.nii', '_seg.nii')

                amap, aff_amap, h_amap = load_volume(path_amap, im_only=False, dtype='float32')

                hemi_str = ['left', 'right']
                
                for j in [0,1]:
                    print('Estimate hemispheric amap label for %s hemisphere' % hemi_str[j])
                    hemi_name = path_amap.replace('_seg.nii', '_%s_amap.nii' % hemi_str[j])
                    hemi = amap2hemiseg(amap, seg, hemi=j+1)

                    # crop hemi image and add 2 voxels
                    crop_idx = bbox_volume(hemi > 1, pad=5)
                    hemi, aff_hemi = crop_volume_with_idx(hemi, crop_idx, aff=aff_amap, n_dims=3, return_copy=False)

                    #save_volume(hemi, aff_amap, h_amap, hemi_name, dtype='uint8')
                    save_volume(hemi, aff_hemi, h_amap, hemi_name, dtype='uint8')
                
                
            # write volumes to disc if necessary
            if path_volumes[i] is not None:
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, labels_volumes, names_volumes, last_first=(not v1))

            # write QC scores to disc if necessary
            if path_qc_scores[i] is not None:
                qc_score = np.around(np.clip(np.squeeze(qc_score)[1:], 0, 1), 4)
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + ['%.4f' % q for q in qc_score]
                write_csv(path_qc_scores[i], row, unique_qc_file, labels_qc, names_qc)

        except Exception as e:
            list_errors.append(path_images[i])
            print('\nthe following problem occured with image %s :' % path_images[i])
            print(traceback.format_exc())
            print('resuming program execution\n')
            continue

    # print output info
    if len(path_segmentations) == 1:  # only one image is processed
        print('\nsegmentation  saved in:    ' + path_segmentations[0])
        if path_posteriors[0] is not None:
            print('posteriors saved in:       ' + path_posteriors[0])
        if path_resampled[0] is not None:
            print('resampled image saved in:  ' + path_resampled[0])
        if path_label[0] is not None:
            print('label image saved in:  ' + path_label[0])
        if path_hemi[0] is not None:
            print('label image saved in:  ' + path_hemi[0])
        if path_volumes[0] is not None:
            print('volumes saved in:          ' + path_volumes[0])
        if path_qc_scores[0] is not None:
            print('QC scores saved in:        ' + path_qc_scores[0])
    else:  # all segmentations are in the same folder, and we have unique vol/QC files
        if len(set([os.path.dirname(path_segmentations[i]) for i in range(len(path_segmentations))])) <= 1:
            print('\nsegmentations saved in:    ' + os.path.dirname(path_segmentations[0]))
            if path_posteriors[0] is not None:
                print('posteriors saved in:       ' + os.path.dirname(path_posteriors[0]))
            if path_resampled[0] is not None:
                print('resampled images saved in: ' + os.path.dirname(path_resampled[0]))
            if path_volumes[0] is not None:
                print('volumes saved in:          ' + path_volumes[0])
            if path_qc_scores[0] is not None:
                print('QC scores saved in:        ' + path_qc_scores[0])
    """
    if robust:
        print('\nIf you use the new robust version of SynthSeg in a publication, please cite:')
        print('Robust Segmentation of Brain MRI in the Wild with Hierarchical CNNs and no Retraining')
        print('B. Billot, M. Collin, S.E. Arnold, S. Das, J.E. Iglesias')
    else:
        print('\nIf you use this tool in a publication, please cite:')
        print('SynthSeg: domain randomisation for segmentation of brain MRI scans of any contrast and resolution')
        print('B. Billot, D.N. Greeve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias')
    """
    if len(list_errors) > 0:
        print('\nERROR: some problems occured for the following inputs (see corresponding errors above):')
        for path_error_image in list_errors:
            print(path_error_image)
        sys.exit(1)


def prepare_output_files(path_images, out_seg, out_posteriors, out_label, out_hemi, out_resampled, out_volumes, out_qc):
    '''
    Prepare output files.
    '''

    # check inputs
    if path_images is None:
        sf.system.fatal('please specify an input file/folder (--i)')
    if out_seg is None:
        sf.system.fatal('please specify an output file/folder (--o)')

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_label = os.path.abspath(out_label) if (out_label is not None) else out_label
    out_hemi = os.path.abspath(out_hemi) if (out_hemi is not None) else out_hemi
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes
    out_qc = os.path.abspath(out_qc) if (out_qc is not None) else out_qc


    # input images
    if not os.path.isfile(path_images):
        sf.system.fatal('file does not exist: %s \n'
                        'please make sure the path and the extension are correct' % path_images)
    path_images = [path_images]

    # define helper to deal with outputs
    def helper_im(path, name, file_type, suffix):
        unique_file = False
        if path is not None:
            if file_type == 'csv':
                if path[-4:] != '.csv':
                    print('%s provided without csv extension. Adding csv extension.' % name)
                    path += '.csv'
                recompute_files = [True]
                unique_file = True
            else:
                recompute_files = [not os.path.isfile(path)]
            mkdir(os.path.dirname(path))
        else:
            recompute_files = [False]
        path = [path]
        return path, recompute_files, unique_file

    # use helper on all outputs
    out_seg, recompute_seg, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
    out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
    out_label, recompute_label, _ = helper_im(out_label, 'path_label', '', 'label')
    out_hemi, recompute_hemi, _ = helper_im(out_hemi, 'path_hemi', '', 'hemi')
    out_resampled, recompute_resampled, _ = helper_im(out_resampled, 'path_resampled', '', 'resampled')
    out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')
    out_qc, recompute_qc, unique_qc_file = helper_im(out_qc, 'path_qc_scores', 'csv', '')

    return path_images, out_seg, out_posteriors, out_label, out_hemi, out_resampled, out_volumes, unique_volume_file, \
           out_qc, unique_qc_file


def preprocess(path_image, n_levels=5, crop=None, min_pad=None, path_resample=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = get_volume_info(path_image, True)
    if n_dims < 3:
        sf.system.fatal('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        sf.system.fatal('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample image if necessary
    if np.any((im_res > 1.05) | (im_res < 0.95)):
        im_res = np.array([1.]*3)
        im, aff = resample_volume(im, aff, im_res)
        """
        if path_resample is not None:
            save_volume(im, aff, h, path_resample)
        """

    # align image
    im = align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    im = rescale_volume(im, new_min=0, new_max=1, min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    min_pad = reformat_to_list(min_pad, length=n_dims, dtype='int')
    min_pad = [find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
    pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx


def build_model(model_file_segmentation,
                model_file_parcellation,
                model_file_qc,
                labels_segmentation,
                labels_denoiser,
                labels_parcellation,
                labels_qc,
                flip_indices,
                robust,
                do_parcellation,
                do_qc):

    if not os.path.isfile(model_file_segmentation):
        sf.system.fatal("The provided model path does not exist.")

    # get labels
    n_labels_seg = len(labels_segmentation)

    if robust:
        n_groups = len(labels_denoiser)

        # build first UNet
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_groups,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet')

        # transition between the two networks: one_hot -> argmax -> one_hot (it simulates how the network was trained)
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build denoiser
        net = unet(nb_features=16,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=5,
                   nb_labels=n_groups,
                   feat_mult=2,
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   activation='elu',
                   skip_n_concatenations=2,
                   input_model=net,
                   name='l2l')

        # transition between the two networks: one_hot -> argmax -> one_hot, and concatenate input image and labels
        input_image = net.inputs[0]
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build 2nd network
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 2],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_seg,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   input_model=net,
                   name='unet2')
        net.load_weights(model_file_segmentation, by_name=True)
        name_segm_prediction_layer = 'unet2_prediction'

    else:

        # build UNet
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 1],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_seg,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet')
        net.load_weights(model_file_segmentation, by_name=True)
        input_image = net.inputs[0]
        name_segm_prediction_layer = 'unet_prediction'

        # smooth posteriors
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        if flip_indices is not None:

            # segment flipped image
            seg = net.output
            image_flipped = RandomFlip(flip_axis=0, prob=1)(input_image)
            last_tensor = net(image_flipped)

            # flip back and re-order channels
            last_tensor = RandomFlip(flip_axis=0, prob=1)(last_tensor)
            last_tensor = KL.Lambda(lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
            reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
            # we need the [0, ...] below because in this vresion of TF, tf.split adds a dim at the beginning...
            last_tensor = KL.Lambda(lambda x: tf.concat(x, -1)[0, ...], name='concat')(reordered_channels)

            # average two segmentations and build model
            name_segm_prediction_layer = 'average_lr'
            last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
            net = keras.Model(inputs=net.inputs, outputs=last_tensor)

    # add aparc segmenter if needed
    if do_parcellation:
        n_labels_parcellation = len(labels_parcellation)

        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
        last_tensor = ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        parcellation_masking_values = np.array([1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
        last_tensor = ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build UNet
        net = unet(nb_features=24,
                   input_shape=[None, None, None, 3],
                   nb_levels=5,
                   conv_size=3,
                   nb_labels=n_labels_parcellation,
                   feat_mult=2,
                   activation='elu',
                   nb_conv_per_level=2,
                   batch_norm=-1,
                   name='unet_parc',
                   input_model=net)
        net.load_weights(model_file_parcellation, by_name=True)

        # smooth predictions
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=[net.get_layer(name_segm_prediction_layer).output, last_tensor])

    # add CNN regressor for automated QC if needed
    if do_qc:
        n_labels_qc = len(np.unique(labels_qc))

        # transition between the two networks: one_hot -> argmax -> qc_labels -> one_hot
        shape_prediction = KL.Input([3], dtype='int32')
        if do_parcellation:
            last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x[0], axis=-1), 'int32'))(net.outputs)
        else:
            last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(net.output)
        last_tensor = MakeShape(224)([last_tensor, shape_prediction])
        last_tensor = ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        last_tensor = ConvertLabels(labels_segmentation, labels_qc)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_labels_qc, axis=-1))(last_tensor)
        net = keras.Model(inputs=[*net.inputs, shape_prediction], outputs=last_tensor)

        # build QC regressor network
        net = conv_enc(nb_features=24,
                       input_shape=[None, None, None, n_labels_qc],
                       nb_levels=4,
                       conv_size=5,
                       name='qc',
                       feat_mult=2,
                       activation='relu',
                       use_residuals=True,
                       nb_conv_per_level=2,
                       batch_norm=-1,
                       input_model=net)
        last_tensor = net.outputs[0]
        conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
        last_tensor = KL.MaxPool3D(pool_size=(2, 2, 2), name='qc_maxpool_3', padding='same')(last_tensor)
        last_tensor = KL.Conv3D(n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_0')(last_tensor)
        last_tensor = KL.Conv3D(n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_1')(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='qc_final_pred')(last_tensor)

        # build model
        if do_parcellation:
            outputs = [net.get_layer(name_segm_prediction_layer).output,
                       net.get_layer('unet_parc_prediction').output,
                       last_tensor]
        else:
            outputs = [net.get_layer(name_segm_prediction_layer).output, last_tensor]
        net = keras.Model(inputs=net.inputs, outputs=outputs)
        net.load_weights(model_file_qc, by_name=True)

    return net


def correct_and_skull_strip(volume, label, vessel_mask=None):
    """
    Use label image to correct input volume for non-uniformities and skull-strip
    the image by removing non-brain parts of the brain.
    """
    
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
    label_mask = binary_opening(label_mask, build_binary_structure(1, 3))
    label_mask = get_largest_connected_component(label_mask)
    label_mask = binary_closing(label_mask, build_binary_structure(5, 3))
    
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
    label_mask = get_largest_connected_component(label_mask)
    label[~label_mask] = tissue_labels["CSF"]

    return label
    
def postprocess(post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):

    # get posteriors
    post_patch_seg = np.squeeze(post_patch_seg)
    if fast:
        post_patch_seg = crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)

    # keep biggest connected component
    tmp_post_patch_seg = post_patch_seg[..., 1:]
    post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
    post_patch_seg_mask = get_largest_connected_component(post_patch_seg_mask)
    post_patch_seg_mask = np.stack([post_patch_seg_mask]*tmp_post_patch_seg.shape[-1], axis=-1)
    tmp_post_patch_seg = mask_volume(tmp_post_patch_seg, mask=post_patch_seg_mask, return_copy=False)
    post_patch_seg[..., 1:] = tmp_post_patch_seg

    # reset posteriors to zero outside the largest connected component of each topological class
    if not fast:
        post_patch_seg_mask = post_patch_seg > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch_seg[..., idx] *= tmp_mask
        post_patch_seg = crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)
    else:
        post_patch_seg_mask = post_patch_seg > 0.2
        post_patch_seg[..., 1:] *= post_patch_seg_mask[..., 1:]

    # get hard segmentation
    post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch_seg.argmax(-1).astype('int32')].astype('int32')

    # postprocess parcellation
    if post_patch_parc is not None:
        post_patch_parc = np.squeeze(post_patch_parc)
        post_patch_parc = crop_volume_with_idx(post_patch_parc, pad_idx, n_dims=3, return_copy=False)
        mask = (seg_patch == 3) | (seg_patch == 42)
        post_patch_parc[..., 0] = np.ones_like(post_patch_parc[..., 0])
        post_patch_parc[..., 0] = mask_volume(post_patch_parc[..., 0], mask=mask < 0.1, return_copy=False)
        post_patch_parc /= np.sum(post_patch_parc, axis=-1)[..., np.newaxis]
        parc_patch = labels_parcellation[post_patch_parc.argmax(-1).astype('int32')].astype('int32')
        seg_patch[mask] = parc_patch[mask]

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        posteriors[..., 0] = np.ones(shape)  # place background around patch
        seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
        posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch_seg
    else:
        seg = seg_patch
        posteriors = post_patch_seg

    # align prediction back to first orientation
    seg = align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    posteriors = align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

    # compute volumes
    volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
    if not v1:
        volumes = np.concatenate([np.array([np.sum(volumes)]), volumes])
    if post_patch_parc is not None:
        volumes_parc = np.sum(post_patch_parc[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
        total_volume_cortex = np.sum(volumes[np.where((labels_segmentation == 3) | (labels_segmentation == 42))[0] - 1])
        volumes_parc = volumes_parc / np.sum(volumes_parc) * total_volume_cortex
        volumes = np.concatenate([volumes, volumes_parc])
    volumes = np.around(volumes * np.prod(im_res), 3)

    return seg, posteriors, volumes


class MakeShape(KL.Layer):
    """Expects one-hot encoding of the two input label maps."""

    def __init__(self, target_shape, **kwargs):
        self.n_dims = None
        self.target_shape = target_shape
        self.cropping_shape = None
        super(MakeShape, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["target_shape"] = self.target_shape
        return config

    def build(self, input_shape):
        self.n_dims = input_shape[1][1]
        self.cropping_shape = np.array(reformat_to_list(self.target_shape, length=self.n_dims))
        self.built = True
        super(MakeShape, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self._single_process, inputs, dtype=tf.int32)

    def _single_process(self, inputs):

        x = inputs[0]
        shape = inputs[1]

        # find cropping indices
        mask = tf.logical_and(tf.not_equal(x, 0), tf.not_equal(x, 24))
        indices = tf.cast(tf.where(mask), 'int32')

        min_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.zeros(self.n_dims, dtype='int32'),
                           tf.maximum(tf.reduce_min(indices, axis=0), 0))
        max_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.minimum(shape, self.cropping_shape),
                           tf.minimum(tf.reduce_max(indices, axis=0) + 1, shape))

        # expand/retract (depending on the desired shape) the cropping region around the centre
        intermediate_vol_shape = max_idx - min_idx
        min_idx = min_idx - tf.cast(tf.math.ceil((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        max_idx = max_idx + tf.cast(tf.math.floor((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        tmp_min_idx = tf.maximum(min_idx, 0)
        tmp_max_idx = tf.minimum(max_idx, shape)
        x = tf.slice(x, begin=tmp_min_idx, size=tf.minimum(tmp_max_idx - tmp_min_idx, shape))

        # pad if necessary
        min_padding = tf.abs(tf.minimum(min_idx, 0))
        max_padding = tf.maximum(max_idx - shape, 0)
        x = K.switch(tf.reduce_any(tf.logical_or(tf.greater(min_padding, 0), tf.greater(max_padding, 0))),
                     tf.pad(x, tf.stack([min_padding, max_padding], axis=1)),
                     x)

        return x


def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondance between labels
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
    mkdir(os.path.dirname(path_csv))
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


# ================================================================================================
#                       Neurite Utilities - See github.com/adalca/neurite
# ================================================================================================


def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         skip_n_concatenations=0,
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None,
         input_model=None):
    """
    Unet-style keras model with an overdose of parametrization. Copied with permission
    from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         input_model=input_model)

    # get decoder
    # use_skip_connections=True makes it a u-net
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = conv_dec(nb_features,
                         None,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=True,
                         skip_n_concatenations=skip_n_concatenations,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         layer_nb_feats=lnf,
                         conv_dropout=conv_dropout,
                         input_model=enc_model)
    final_model = dec_model

    return final_model


def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             input_model=None):
    """
    Fully Convolutional Encoder. Copied with permission from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # first layer: input
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation, 'data_format': 'channels_last'}
    maxpool = getattr(keras.layers, 'MaxPooling%dD' % ndims)

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0  # level feature index
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features * feat_mult ** level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** level

        for conv in range(nb_conv_per_level):  # does several conv per level, max pooling applied at the end
            if layer_nb_feats is not None:  # None or List of all the feature numbers
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:  # no activation
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                # conv dropout along feature space only
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        if use_residuals:
            convarm_layer = last_tensor

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            nb_feats_in = lvl_first_tensor.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(lvl_first_tensor)
                add_layer = last_tensor

                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]

            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([add_layer, convarm_layer], name=name)

            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = keras.Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             skip_n_concatenations=0,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             input_model=None):
    """
    Fully Convolutional Decoder. Copied with permission from github.com/adalca/neurite.

    Parameters:
        ...
        use_skip_connections (bool): if true, turns an Enc-Dec to a U-Net.
            If true, input_tensor and tensors are required.
            It assumes a particular naming of layers. conv_enc...
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # if using skip connections, make sure need to use them.
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"

    # first layer: input
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]

    # vol size info
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(keras.layers, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(nb_features * feat_mult ** (nb_levels - 2 - level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** (nb_levels - 2 - level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor

        # merge layers combining previous layer
        if use_skip_connections & (level < (nb_levels - skip_n_concatenations - 1)):
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = keras.layers.concatenate([cat_tensor, last_tensor], axis=ndims + 1, name=name)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        # residual block
        if use_residuals:

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            add_layer = up_tensor
            nb_feats_in = add_layer.get_shape()[-1]
            nb_feats_out = last_tensor.get_shape()[-1]
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)

                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    last_tensor = keras.layers.Dropout(conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([last_tensor, add_layer], name=name)

            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    if final_pred_activation == 'softmax':
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=ndims + 1)
        pred_tensor = keras.layers.Lambda(softmax_lambda_fcn, name=name)(last_tensor)

    # otherwise create a layer that does nothing.
    else:
        name = '%s_prediction' % prefix
        pred_tensor = keras.layers.Activation('linear', name=name)(like_tensor)

    # create the model and retun
    model = keras.Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model


# ================================================================================================
#                                        Lab2Im Utilities
# ================================================================================================


# ---------------------------------------------- loading/saving functions ----------------------------------------------


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    #mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def get_list_labels(label_list=None, labels_dir=None, save_label_list=None, FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list. Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: (optional) if path_label_list is None, the label list is computed by reading all the label maps
    in the given folder. Can also be the path to a single label map.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    if label_list is not None:
        label_list = np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int'))

    # compute label list from all label files
    elif labels_dir is not None:
        print('Compiling list of unique labels')
        # go through all labels files and compute unique list of labels
        labels_paths = list_images_in_folder(labels_dir)
        label_list = np.empty(0)
        loop_info = LoopInfo(len(labels_paths), 10, 'processing', print_time=True)
        for lab_idx, path in enumerate(labels_paths):
            loop_info.update(lab_idx)
            y = load_volume(path, dtype='int32')
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate((label_list, y_unique))).astype('int')

    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                             109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                             251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                             502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                             531, 532, 533, 534, 535, 536, 537]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                    (la == 865) | (20100 < la < 20110):
                if la not in left:
                    left.append(la)
            elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                    (la == 866):
                if la not in right:
                    right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
        return np.int32(label_list), None


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


# ----------------------------------------------- reformatting functions -----------------------------------------------


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int32, np.int32, np.int64, np.float32, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var


# ----------------------------------------------- path-related functions -----------------------------------------------


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz whithin a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)


# ---------------------------------------------- shape-related functions -----------------------------------------------


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


# --------------------------------------------------- miscellaneous ----------------------------------------------------


def infer(x):
    ''' Try to parse input to float. If it fails, tries boolean, and otherwise keep it as string '''
    try:
        x = float(x)
    except ValueError:
        if x == 'False':
            x = False
        elif x == 'True':
            x = True
        elif not isinstance(x, str):
            raise TypeError('input should be an int/float/boolean/str, had {}'.format(type(x)))
    return x


class LoopInfo:
    """
    Class to print the current iteration in a for loop, and optionally the estimated remaining time.
    Instantiate just before the loop, and call the update method at the start of the loop.
    The printed text has the following format:
    processing i/total    remaining time: hh:mm:ss
    """

    def __init__(self, n_iterations, spacing=10, text='processing', print_time=False):
        """
        :param n_iterations: total number of iterations of the for loop.
        :param spacing: frequency at which the update info will be printed on screen.
        :param text: text to print. Default is processing.
        :param print_time: whether to print the estimated remaining time. Default is False.
        """

        # loop parameters
        self.n_iterations = n_iterations
        self.spacing = spacing

        # text parameters
        self.text = text
        self.print_time = print_time
        self.print_previous_time = False
        self.align = len(str(self.n_iterations)) * 2 + 1 + 3

        # timing parameters
        self.iteration_durations = np.zeros((n_iterations,))
        self.start = time.time()
        self.previous = time.time()

    def update(self, idx):

        # time iteration
        now = time.time()
        self.iteration_durations[idx] = now - self.previous
        self.previous = now

        # print text
        if idx == 0:
            print(self.text + ' 1/{}'.format(self.n_iterations))
        elif idx % self.spacing == self.spacing - 1:
            iteration = str(idx + 1) + '/' + str(self.n_iterations)
            if self.print_time:
                # estimate remaining time
                max_duration = np.max(self.iteration_durations)
                average_duration = np.mean(self.iteration_durations[self.iteration_durations > .01 * max_duration])
                remaining_time = int(average_duration * (self.n_iterations - idx))
                # print total remaining time only if it is greater than 1s or if it was previously printed
                if (remaining_time > 1) | self.print_previous_time:
                    eta = str(timedelta(seconds=remaining_time))
                    print(self.text + ' {:<{x}} remaining time: {}'.format(iteration, eta, x=self.align))
                    self.print_previous_time = True
                else:
                    print(self.text + ' {}'.format(iteration))
            else:
                print(self.text + ' {}'.format(iteration))


def get_mapping_lut(source, dest=None):
    """This functions returns the look-up table to map a list of N values (source) to another list (dest).
    If the second list is not given, we assume it is equal to [0, ..., N-1]."""

    # initialise
    source = np.array(reformat_to_list(source), dtype='int32')
    n_labels = source.shape[0]

    # build new label list if neccessary
    if dest is None:
        dest = np.arange(n_labels, dtype='int32')
    else:
        assert len(source) == len(dest), 'label_list and new_label_list should have the same length'
        dest = np.array(reformat_to_list(dest, dtype='int'))

    # build look-up table
    lut = np.zeros(np.max(source) + 1, dtype='int32')
    for source, dest in zip(source, dest):
        lut[source] = dest

    return lut


def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)


def build_binary_structure(connectivity, n_dims, shape=None):
    """Return a dilation/erosion element with provided connectivity"""
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


# ---------------------------------------------------- edit volume -----------------------------------------------------

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

def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False, masking_value=0,
                return_mask=False, return_copy=True):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be uni- or multi-channel.
     In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = new_volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], 'mask should have shape {0}, or {1}, had {2}'.format(
            vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape))
        mask = mask > 0
    if dilate > 0:
        dilate_struct = build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume


def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2., max_percentile=98., use_positive_only=False):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None, return_crop_idx=False, mode='center'):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff_crop = aff + 0
        aff_crop[0:3, -1] = aff_crop[0:3, -1] + aff_crop[:3, :3] @ crop_idx[:3]
        return new_volume, aff_crop
    else:
        return new_volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :return: flipped volume
    """

    new_volume = volume.copy() if return_copy else volume
    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(new_volume, axis=axis)


def resample_volume(volume, aff, new_vox_size, interpolation='linear'):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.round(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1
    
    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


# --------------------------------------------------- edit label map ---------------------------------------------------


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()


# ---------------------------------------------------- keras layers- ---------------------------------------------------

class RandomFlip(keras.layers.Layer):

    """This function flips the input tensors along the specified axes with a probability of 0.5.
    The input tensors are expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    If specified, this layer can also swap corresponding values, such that the flip tensors stay consistent with the
    native spatial orientation (especially when flipping in the righ/left dimension).
    :param flip_axis: integer, or list of integers specifying the dimensions along which to flip. The values exclude the
    batch dimension (e.g. 0 will flip the tensor along the first axis after the batch dimension). Default is None, where
    the tensors can be flipped along any of the axes (except batch and channel axes).
    :param swap_labels: list of booleans to specify wether to swap the values of each input. All the inputs for which
    the values need to be swapped must have a int32 ot int64 dtype.
    :param label_list: if swap_labels is True, list of all labels contained in labels. Must be ordered as follows, first
     the neutral labels (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: if swap_labels is True, number of non-sided labels

    example 1:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip()(input) will randomly flip input along one of the 1st, 2nd, or 3rd axis (i.e. those with shape
    10, 100, 200).

    example 2:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip(flip_axis=1)(input) will randomly flip input along the 3rd axis (with shape 100), i.e. the axis
    with index 1 if we don't count the batch axis.

    example 3:
    input = tf.convert_to_tensor(np.array([[1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 0, 0, 0]]))
    label_list = np.array([0, 1, 2])
    n_neutral_labels = 1
    output = RandomFlip(flip_axis=1, swap_labels=True, label_list=label_list, n_neutral_labels=n_neutral_labels)(input)
    where output will either be equal to input (bear in mind the flipping occurs with a 0.5 probability), or:
    output = [[0, 0, 0, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 2]]
    Note that the input must have a dtype int32 or int64 for its values to be swapped, otherwise an error will be raised

    example 4:
    if labels is the same as in the input of example 3, and image is a float32 image, then we can swap consistently both
    the labels and the image with:
    labels, image = RandomFlip(flip_axis=1, swap_labels=[True, False], label_list=label_list,
                               n_neutral_labels=n_neutral_labels)([labels, image]])
    Note that the labels must have a dtype int32 or int64 to be swapped, otherwise an error will be raised.
    This doesn't concern the image input, as its values are not swapped.
    """

    def __init__(self, flip_axis=None, swap_labels=False, label_list=None, n_neutral_labels=None, prob=0.5, **kwargs):

        # shape attributes
        self.several_inputs = True
        self.n_dims = None
        self.list_n_channels = None

        # axis along which to flip
        self.flip_axis = reformat_to_list(flip_axis)

        # wether to swap labels, and corresponding label list
        self.swap_labels = reformat_to_list(swap_labels)
        self.label_list = label_list
        self.n_neutral_labels = n_neutral_labels
        self.swap_lut = None

        self.prob = prob

        super(RandomFlip, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["flip_axis"] = self.flip_axis
        config["swap_labels"] = self.swap_labels
        config["label_list"] = self.label_list
        config["n_neutral_labels"] = self.n_neutral_labels
        config["prob"] = self.prob
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            self.several_inputs = False
            inputshape = [input_shape]
        else:
            inputshape = input_shape
        self.n_dims = len(inputshape[0][1:-1])
        self.list_n_channels = [i[-1] for i in inputshape]
        self.swap_labels = reformat_to_list(self.swap_labels, length=len(inputshape))

        # create label list with swapped labels
        if any(self.swap_labels):
            assert (self.label_list is not None) & (self.n_neutral_labels is not None), \
                'please provide a label_list, and n_neutral_labels when swapping the values of at least one input'
            n_labels = len(self.label_list)
            if self.n_neutral_labels == n_labels:
                self.swap_labels = [False] * len(self.swap_labels)
            else:
                rl_split = np.split(self.label_list, [self.n_neutral_labels,
                                                      self.n_neutral_labels + int((n_labels-self.n_neutral_labels)/2)])
                label_list_swap = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
                swap_lut = get_mapping_lut(self.label_list, label_list_swap)
                self.swap_lut = tf.convert_to_tensor(swap_lut, dtype='int32')

        self.built = True
        super(RandomFlip, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # convert inputs to list, and get each input type
        if not self.several_inputs:
            inputs = [inputs]
        types = [v.dtype for v in inputs]

        # sample boolean for each element of the batch
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]
        rand_flip = K.less(tf.random.uniform(tf.concat([batchsize, tf.ones(1, dtype='int32')], axis=0), 0, 1),
                           self.prob)

        # swap r/l labels if necessary
        swapped_inputs = list()
        for i in range(len(inputs)):
            if self.swap_labels[i]:
                swapped_inputs.append(tf.map_fn(self._single_swap, [inputs[i], rand_flip], dtype=types[i]))
            else:
                swapped_inputs.append(inputs[i])

        # flip inputs and convert them back to their original type
        inputs = tf.concat([tf.cast(v, 'float32') for v in swapped_inputs], axis=-1)
        inputs = tf.map_fn(self._single_flip, [inputs, rand_flip], dtype=tf.float32)
        inputs = tf.split(inputs, self.list_n_channels, axis=-1)

        return [tf.cast(v, t) for (t, v) in zip(types, inputs)]

    def _single_swap(self, inputs):
        return K.switch(inputs[1], tf.gather(self.swap_lut, inputs[0]), inputs[0])

    def _single_flip(self, inputs):
        if self.flip_axis is None:
            flip_axis = tf.random.uniform([1], 0, self.n_dims, dtype='int32')
        else:
            idx = tf.squeeze(tf.random.uniform([1], 0, len(self.flip_axis), dtype='int32'))
            flip_axis = tf.expand_dims(tf.convert_to_tensor(self.flip_axis, dtype='int32')[idx], axis=0)
        return K.switch(inputs[1], K.reverse(inputs[0], axes=flip_axis), inputs[0])


class GaussianBlur(keras.layers.Layer):
    """Applies gaussian blur to an input image."""

    def __init__(self, sigma, random_blur_range=None, use_mask=False, **kwargs):
        self.sigma = reformat_to_list(sigma)
        assert np.all(np.array(self.sigma) >= 0), 'sigma should be superior or equal to 0'
        self.use_mask = use_mask

        self.n_dims = None
        self.n_channels = None
        self.blur_range = random_blur_range
        self.stride = None
        self.separable = None
        self.kernels = None
        self.convnd = None
        super(GaussianBlur, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["random_blur_range"] = self.blur_range
        config["use_mask"] = self.use_mask
        return config

    def build(self, input_shape):

        # get shapes
        if self.use_mask:
            assert len(input_shape) == 2, 'please provide a mask as second layer input when use_mask=True'
            self.n_dims = len(input_shape[0]) - 2
            self.n_channels = input_shape[0][-1]
        else:
            self.n_dims = len(input_shape) - 2
            self.n_channels = input_shape[-1]
            
        # prepare blurring kernel
        self.stride = [1]*(self.n_dims+2)
        self.sigma = reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:  # fixed kernels
            self.kernels = gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

        self.built = True
        super(GaussianBlur, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None

        # redefine the kernels at each new step when blur_range is activated
        if self.blur_range is not None:
            self.kernels = gaussian_kernel(self.sigma, blur_range=self.blur_range, separable=self.separable)

        if self.separable:
            for k in self.kernels:
                if k is not None:
                    image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), k, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    if self.use_mask:
                        maskb = tf.cast(mask, 'float32')
                        maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), k, self.stride, 'SAME')
                                           for n in range(self.n_channels)], -1)
                        image = image / (maskb + keras.backend.epsilon())
                        image = tf.where(mask, image, tf.zeros_like(image))
        else:
            if any(self.sigma):
                image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), self.kernels, self.stride, 'SAME')
                                   for n in range(self.n_channels)], -1)
                if self.use_mask:
                    maskb = tf.cast(mask, 'float32')
                    maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), self.kernels, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    image = image / (maskb + keras.backend.epsilon())
                    image = tf.where(mask, image, tf.zeros_like(image))

        return image


class ConvertLabels(keras.layers.Layer):
    """Convert all labels in a tensor by the corresponding given set of values.
    labels_converted = ConvertLabels(source_values, dest_values)(labels).
    labels must be an int32 tensor, and labels_converted will also be int32.

    :param source_values: list of all the possible values in labels. Must be a list or a 1D numpy array.
    :param dest_values: list of all the target label values. Must be ordered the same as source values:
    labels[labels == source_values[i]] = dest_values[i].
    If None (default), dest_values is equal to [0, ..., N-1], where N is the total number of values in source_values,
    which enables to remap label maps to [0, ..., N-1].
    """

    def __init__(self, source_values, dest_values=None, **kwargs):
        self.source_values = source_values
        self.dest_values = dest_values
        self.lut = None
        super(ConvertLabels, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["source_values"] = self.source_values
        config["dest_values"] = self.dest_values
        return config

    def build(self, input_shape):
        self.lut = tf.convert_to_tensor(get_mapping_lut(self.source_values, dest=self.dest_values), dtype='int32')
        self.built = True
        super(ConvertLabels, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.gather(self.lut, tf.cast(inputs, dtype='int32'))


# ---------------------------------------------------- edit tensors ----------------------------------------------------


def gaussian_kernel(sigma, max_sigma=None, blur_range=None, separable=True):
    """Build gaussian kernels of the specified standard deviation. The outputs are given as tensorflow tensors.
    :param sigma: standard deviation of the tensors. Can be given as a list/numpy array or as tensors. In each case,
    sigma must have the same length as the number of dimensions of the volume that will be blurred with the output
    tensors (e.g. sigma must have 3 values for 3D volumes).
    :param max_sigma:
    :param blur_range:
    :param separable:
    :return:
    """
    # convert sigma into a tensor
    if not tf.is_tensor(sigma):
        sigma_tens = tf.convert_to_tensor(reformat_to_list(sigma), dtype='float32')
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    shape = sigma_tens.get_shape().as_list()

    # get n_dims and batchsize
    if shape[0] is not None:
        n_dims = shape[0]
        batchsize = None
    else:
        n_dims = shape[1]
        batchsize = tf.split(tf.shape(sigma_tens), [1, -1])[0]

    # reformat max_sigma
    if max_sigma is not None:  # dynamic blurring
        max_sigma = np.array(reformat_to_list(max_sigma, length=n_dims))
    else:  # sigma is fixed
        max_sigma = np.array(reformat_to_list(sigma, length=n_dims))

    # randomise the burring std dev and/or split it between dimensions
    if blur_range is not None:
        if blur_range != 1:
            sigma_tens = sigma_tens * tf.random.uniform(tf.shape(sigma_tens), minval=1 / blur_range, maxval=blur_range)

    # get size of blurring kernels
    windowsize = np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1

    if separable:

        split_sigma = tf.split(sigma_tens, [1] * n_dims, axis=-1)

        kernels = list()
        comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])
        for (i, wsize) in enumerate(windowsize):

            if wsize > 1:

                # build meshgrid and replicate it along batch dim if dynamic blurring
                locations = tf.cast(tf.range(0, wsize), 'float32') - (wsize - 1) / 2
                if batchsize is not None:
                    locations = tf.tile(tf.expand_dims(locations, axis=0),
                                        tf.concat([batchsize, tf.ones(tf.shape(tf.shape(locations)), dtype='int32')],
                                                  axis=0))
                    comb[i] += 1

                # compute gaussians
                exp_term = -K.square(locations) / (2 * split_sigma[i] ** 2)
                g = tf.exp(exp_term - tf.math.log(np.sqrt(2 * np.pi) * split_sigma[i]))
                g = g / tf.reduce_sum(g)

                for axis in comb[i]:
                    g = tf.expand_dims(g, axis=axis)
                kernels.append(tf.expand_dims(tf.expand_dims(g, -1), -1))

            else:
                kernels.append(None)

    else:

        # build meshgrid
        mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
        diff = tf.stack([mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)

        # replicate meshgrid to batch size and reshape sigma_tens
        if batchsize is not None:
            diff = tf.tile(tf.expand_dims(diff, axis=0),
                           tf.concat([batchsize, tf.ones(tf.shape(tf.shape(diff)), dtype='int32')], axis=0))
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=1)
        else:
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=0)

        # compute gaussians
        sigma_is_0 = tf.equal(sigma_tens, 0)
        exp_term = -K.square(diff) / (2 * tf.where(sigma_is_0, tf.ones_like(sigma_tens), sigma_tens)**2)
        norms = exp_term - tf.math.log(tf.where(sigma_is_0, tf.ones_like(sigma_tens), np.sqrt(2 * np.pi) * sigma_tens))
        kernels = K.sum(norms, -1)
        kernels = tf.exp(kernels)
        kernels /= tf.reduce_sum(kernels)
        kernels = tf.expand_dims(tf.expand_dims(kernels, -1), -1)

    return kernels


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def meshgrid(*args, **kwargs):
    """
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    """

    indexing = kwargs.pop("indexing", "xy")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output


# execute script
if __name__ == '__main__':
    main()
