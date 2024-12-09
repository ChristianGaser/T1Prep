# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache 2.0 license
# terms, and this file has been changed.
#

# python imports
import os
import sys
import traceback
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# third-party imports
from ext.lab2im import utils as tools
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from T1Prep import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def segment(label,
            brain,
            target_res = 0.5,
            vessel_strength = -1):
    '''
    Prediction pipeline.

    Args:
        target_res (float): Target resolution.
        vessel_strength (int): Strength of the vessel correction.

    Returns:
        None
    '''

    # use resolution of input image if target_res <= 0
    if target_res <= 0:
        target_res = im_res_orig
        print('Use original image resolution for resampling: %g %g %g mm' % (target_res[0],target_res[1],target_res[2]))
    else: # otherwise extend to array
        target_res = np.array([target_res]*3)

    # write predictions to disc
    tools.save_volume(seg, aff, h, path_segmentations, dtype='int32')
    if path_posteriors is not None:
        tools.save_volume(posteriors, aff, h, path_posteriors, dtype='float32')

    # obtain and save label image
    if path_label is not None or path_resampled is not None:
        print('Estimate label')
        label_orig = utils.posteriors2label(posteriors)

        # resample to target voxel size
        label, aff_label = edit_volumes.resample_volume(label_orig, aff, target_res)
        tools.save_volume(label, aff_label, h, path_label, dtype='uint8')

    # write hemi images
    if path_hemi is not None:

        hemi_str  = ['-L_seg', '-R_seg'] # name for output file
        hemi_str2 = ['left', 'right']     # name for print

        for j in [0, 1]:
            print('Estimate hemispheric label for %s hemisphere' % hemi_str2[j])
            hemi_name = path_hemi.replace('.nii', '_%s.nii' % hemi_str[j])
            hemi = utils.posteriors2hemiseg(posteriors, hemi=j+1)

            # resample to target voxel size
            hemi, aff_hemi = edit_volumes.resample_volume(hemi, aff, target_res)

            # crop hemi image and add 5 voxels
            crop_idx = utils.bbox_volume(hemi > 1, pad=5)
            hemi, aff_hemi = edit_volumes.crop_volume_with_idx(hemi, crop_idx, aff=aff_hemi, n_dims=3, return_copy=False)

            tools.save_volume(hemi, aff_hemi, h, hemi_name, dtype='uint8')

    if path_resampled is not None:
        resamp, aff_resamp, _ = tools.load_volume(path_images, im_only=False, dtype='float32')

       # use fast nu-correction with lower resolution of original preprocessed images
        use_fast_nu_correction = False # not working yet?!
        
        # resample original input to 1mm voxel size for fast nu-correction
        if use_fast_nu_correction:
            im, _ = edit_volumes.resample_volume(resamp, aff_resamp, im_res)

        # resample original input to target voxel size
        resamp = edit_volumes.resample_volume_like(label, aff_label, resamp, aff_resamp, interpolation='linear')
        aff_resamp = aff_label
        resamp, aff_resamp = edit_volumes.resample_volume(resamp, aff_resamp, target_res)
        
        # limit vessel correction to cerebral cortex (+hippocampus+amygdala+CSF) only
        cortex_mask = (seg == 2)  | (seg == 3)  | (seg == 41) | (seg == 42) | (seg == 24) | \
                      (seg == 17) | (seg == 18) | (seg == 53) | (seg == 54)

        # resample cortex_mask to target voxel size
        cortex_mask, _ = edit_volumes.resample_volume(cortex_mask, aff, target_res)
        
        # finally convert to boolean type for the mask and round because of resampling
        cortex_mask = np.round(cortex_mask) > 0.5
        
        # Using the 1mm data from SynthSeg is a bit faster
        if use_fast_nu_correction:
            mask = (label_orig > 2.5) & (im != 0)
            bias = utils.get_bias_field(im, mask, im_res, aff)

            # resample bias field to the target voxel size of the resampled input volume
            bias, _ = edit_volumes.resample_volume(bias, aff, target_res)
        else:
            im_res_resampled = target_res
            mask = (label > 2.5) & (resamp != 0)
            bias = utils.get_bias_field(resamp, mask, im_res_resampled, aff_resamp)

       # apply nu-correction
        tissue_idx = bias != 0 
        resamp[tissue_idx] /= bias[tissue_idx]

        # after nu-correction we might have negative values that should be prevented
        min_resamp = np.min(np.array(resamp))
        if min_resamp < 0:
            resamp -= min_resamp
                
        # correct vessels and skull-strip image
        print('Vessel-correction and skull-stripping')
        #resamp, _ = utils.suppress_vessels_and_skull_strip(resamp, label, vessel_strength, target_res, vessel_mask=cortex_mask, debug=1)
        resamp, div_mask = utils.suppress_vessels_and_skull_strip(resamp, label, vessel_strength, target_res, vessel_mask=cortex_mask, debug=1)
        
        tools.save_volume(resamp, aff_resamp, h, path_resampled, dtype='float32')
        name = os.path.basename(path_images).replace('.nii', '_bias.nii')
        #tools.save_volume(bias, aff_resamp, h, name, dtype='float32')
       
    # write volumes to disc if necessary
    if path_volumes is not None:
        row = [os.path.basename(path_images).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
        write_csv(path_volumes, row, unique_vol_file, labels_volumes, names_volumes, last_first=not v1)

    # write QC scores to disc if necessary
    if path_qc_scores is not None:
        qc_score = np.around(np.clip(np.squeeze(qc_score)[1:], 0, 1), 4)
        row = [os.path.basename(path_images).replace('.nii.gz', '')] + ['%.4f' % q for q in qc_score]
        write_csv(path_qc_scores, row, unique_qc_file, labels_qc, names_qc)

def preprocess(path_image, target_res_synthseg=1., n_levels=5, crop=None, min_pad=None, path_resample=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = tools.get_volume_info(path_image, True)
    
    # save original resolution
    im_res_orig = im_res
    
    if n_dims == 2 and 1 < n_channels < 4:
        raise Exception('either the input is 2D with several channels, or is 3D with at most 3 slices. '
                        'Either way, results are going to be poor...')
    elif n_dims == 2 and 3 < n_channels < 11:
        print('warning: input with very few slices')
        n_dims = 3
    elif n_dims < 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample image if necessary
    target_res_synthseg = np.squeeze(tools.reformat_to_n_channels_array(target_res_synthseg, n_dims))
    if np.any((im_res > target_res_synthseg + 0.05) | (im_res < target_res_synthseg - 0.05)):
        im_res = target_res_synthseg
        im, aff = edit_volumes.resample_volume(im, aff, im_res)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = tools.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [tools.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [tools.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    min_pad = tools.reformat_to_list(min_pad, length=n_dims, dtype='int')
    min_pad = [tools.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
    pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = tools.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx, im_res_orig


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
        net = nrn_models.unet(nb_features=24,
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
        net = Model(inputs=net.inputs, outputs=last_tensor)

        # build denoiser
        net = nrn_models.unet(nb_features=16,
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
        net = Model(inputs=net.inputs, outputs=last_tensor)

        # build 2nd network
        net = nrn_models.unet(nb_features=24,
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
        net = nrn_models.unet(nb_features=24,
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
        last_tensor = layers.GaussianBlur(sigma=0.5)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

        if flip_indices is not None:

            # segment flipped image
            seg = net.output
            image_flipped = layers.RandomFlip(flip_axis=0, prob=1)(input_image)
            last_tensor = net(image_flipped)

            # flip back and re-order channels
            last_tensor = layers.RandomFlip(flip_axis=0, prob=1)(last_tensor)
            last_tensor = KL.Lambda(lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
            reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
            # we need the [0, ...] below because in this vresion of TF, tf.split adds a dim at the beginning...
            last_tensor = KL.Lambda(lambda x: tf.concat(x, -1)[0, ...], name='concat')(reordered_channels)

            # average two segmentations and build model
            name_segm_prediction_layer = 'average_lr'
            last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
            net = Model(inputs=net.inputs, outputs=last_tensor)

    # add aparc segmenter if needed
    if do_parcellation:
        n_labels_parcellation = len(labels_parcellation)

        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
        last_tensor = layers.ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        parcellation_masking_values = np.array([1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
        last_tensor = layers.ConvertLabels(labels_segmentation, parcellation_masking_values)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = Model(inputs=net.inputs, outputs=last_tensor)

        # build UNet
        net = nrn_models.unet(nb_features=24,
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
        last_tensor = layers.GaussianBlur(sigma=0.5)(last_tensor)
        net = Model(inputs=net.inputs, outputs=[net.get_layer(name_segm_prediction_layer).output, last_tensor])

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
        last_tensor = layers.ConvertLabels(np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        last_tensor = layers.ConvertLabels(labels_segmentation, labels_qc)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_labels_qc, axis=-1))(last_tensor)
        net = Model(inputs=[*net.inputs, shape_prediction], outputs=last_tensor)

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
        net = Model(inputs=net.inputs, outputs=outputs)
        net.load_weights(model_file_qc, by_name=True)

    return net



def postprocess(post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):

    # get posteriors
    post_patch_seg = np.squeeze(post_patch_seg)
    
    if fast | (topology_classes is None):
        post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)

    # keep biggest connected component
    tmp_post_patch_seg = post_patch_seg[..., 1:]
    post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
    post_patch_seg_mask = edit_volumes.get_largest_connected_component(post_patch_seg_mask)
    post_patch_seg_mask = np.stack([post_patch_seg_mask]*tmp_post_patch_seg.shape[-1], axis=-1)
    tmp_post_patch_seg = edit_volumes.mask_volume(tmp_post_patch_seg, mask=post_patch_seg_mask, return_copy=False)
    post_patch_seg[..., 1:] = tmp_post_patch_seg

    # reset posteriors to zero outside the largest connected component of each topological class
    if (not fast) & (topology_classes is not None):
        post_patch_seg_mask = post_patch_seg > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = edit_volumes.get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch_seg[..., idx] *= tmp_mask
        post_patch_seg = edit_volumes.crop_volume_with_idx(post_patch_seg, pad_idx, n_dims=3, return_copy=False)
    else:
        post_patch_seg_mask = post_patch_seg > 0.2
        post_patch_seg[..., 1:] *= post_patch_seg_mask[..., 1:]

    # get hard segmentation
    post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch_seg.argmax(-1).astype('int32')].astype('int32')

    # postprocess parcellation
    if post_patch_parc is not None:
        post_patch_parc = np.squeeze(post_patch_parc)
        post_patch_parc = edit_volumes.crop_volume_with_idx(post_patch_parc, pad_idx, n_dims=3, return_copy=False)
        mask = (seg_patch == 3) | (seg_patch == 42)
        post_patch_parc[..., 0] = np.ones_like(post_patch_parc[..., 0])
        post_patch_parc[..., 0] = edit_volumes.mask_volume(post_patch_parc[..., 0], mask=mask < 0.1, return_copy=False)
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
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    posteriors = edit_volumes.align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

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
        self.cropping_shape = np.array(tools.reformat_to_list(self.target_shape, length=self.n_dims))
        self.built = True
        super(MakeShape, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self._single_process, inputs, fn_output_signature=tf.int32)

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
