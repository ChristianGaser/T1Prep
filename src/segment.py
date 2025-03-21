#!/usr/bin/env python3

import os
import sys
import platform
import torch
import argparse
import warnings
import math
import shutil
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from deepbet.utils import reoriented_nifti
from deepmriprep.preprocess import Preprocess
from deepmriprep.utils import DATA_PATH, nifti_to_tensor, nifti_volume
from deepmriprep.atlas import ATLASES, get_volumes
from torchreg.utils import INTERP_KWARGS
from pathlib import Path
from scipy.ndimage import grey_opening
from utils import progress_bar, remove_file, correct_bias_field, get_atlas, resample_and_save_nifti, get_resampled_header, get_partition, align_brain, cleanup, get_cerebellum
DATA_PATH0 = Path(__file__).resolve().parent.parent / 'data/'
MODEL_FILES = (['brain_extraction_bbox_model.pt', 'brain_extraction_model.pt', 
                'segmentation_nogm_model.pt'] +
               [f'segmentation_patch_{i}_model.pt' for i in range(18)] + ['segmentation_model.pt', 
               'warp_model.pt'])

def run_segment():

    """
    Perform brain segmentation (either using deepmriprep or AMAP) on input T1w brain data using 
    preprocessing, affine registration, and segmentation techniques.

    Command-line Arguments:
    -i, --input : str (required)
        Input file or folder containing the MRI data (.nii/.nii.gz format).
    -o, --outdir : str (required)
        Output directory to save the processed results.
    -a, --amap : flag (optional)
        Enable AMAP segmentation if specified. Default is False.
    -d, --amapdir : str (optional)
        Path to the AMAP binary folder if AMAP segmentation is enabled.
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', 
                        help='Input file', required=True, type=str)
    parser.add_argument('-o', '--outdir', 
                        help='Output folder', required=True, type=str)
    parser.add_argument("-s", '--surf', action="store_true", 
                        help="(optional) Save partioned segmentation map for surface estimation.")
    parser.add_argument("-c", '--csf', action="store_true", 
                        help="(optional) Save also CSF segmentations.")
    parser.add_argument("-m", '--mwp', action="store_true", 
                        help="(optional) Save modulated and warped segmentations.")
    parser.add_argument("-w", '--wp', action="store_true", 
                        help="(optional) Save warped segmentations.")
    parser.add_argument("-p", '--p', action="store_true", 
                        help="(optional) Save native segmentations.")
    parser.add_argument("-r", '--rp', action="store_true", 
                        help="(optional) Save affine registered segmentations.")
    parser.add_argument("-l", '--lesions', action="store_true", 
                        help="(optional) Save also WMH lesions.")
    parser.add_argument("-b", '--bids', action="store_true", 
                        help="(optional) Use bids naming convention.")
    parser.add_argument("-g", '--gz', action="store_true", 
                        help="(optional) Save nii.gz images.")
    parser.add_argument("-a", '--amap', action="store_true", 
                        help="(optional) Use AMAP segmentation.")
    parser.add_argument('-d', '--amapdir', 
                        help='Amap binary folder', type=str)
    parser.add_argument('-v', '--vessel', type=float, default=0.4, 
                        help="Initial threshold to isolate WM for vessel removal") 
    args = parser.parse_args()

    # Input/output parameters
    t1_name = args.input
    out_dir = args.outdir
    amap_dir = args.amapdir
    
    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    vessel = args.vessel
    
    # Save options
    save_mwp = args.mwp
    save_wp = args.wp
    save_rp = args.rp
    save_p = args.p
    save_csf = args.csf
    save_gz = args.gz
    save_lesions = args.lesions
    save_hemilabel = args.surf

    # Check for GPU support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        no_gpu = False
    elif torch.backends.mps.is_available() and False: # not yet fully supported
        device = torch.device("mps")
        no_gpu = False
    else:
        device = torch.device("cpu")
        no_gpu = True

    # Set processing parameters
    target_res = np.array([0.5]*3) # Target resolution for resampling
    count = 1
    end_count = 4
    if save_mwp:
        end_count += 1
    if save_hemilabel:
        end_count += 2
    if (save_lesions):
        end_count += 1
        
    if save_gz:
        ext = 'nii.gz'
    else:
        ext = 'nii'
    
    # Prepare filenames and load input MRI data
    out_name = os.path.basename(
        os.path.basename(t1_name).replace('_desc-sanlm', '')).replace('.nii', '').replace('.gz','')
    t1 = nib.load(t1_name)
    
    # copy necessary model files from local folder to install it, since often the API rate limit is exceeded
    Path(f'{DATA_PATH}/models').mkdir(exist_ok=True)
    for file in MODEL_FILES:
        if not Path(f'{DATA_PATH}/models/{file}').exists():
            shutil.copy(f'{DATA_PATH0}/models/{file}', f'{DATA_PATH}/models/{file}') 

    # Preprocess the input volume
    vol = t1.get_fdata()
    vol, affine2, header2, ras_affine = align_brain(vol, t1.affine, t1.header, np.eye(4), 0)
    t1 = nib.Nifti1Image(vol, affine2, header2)
    
    # Step 1: Skull-stripping
    count = progress_bar(count, end_count, 'Skull-stripping               ')
    prep = Preprocess(no_gpu)
    output_bet = prep.run_bet(t1)
    brain = output_bet['brain']
    mask = output_bet['mask']
    
    # Step 2: Affine registration
    count = progress_bar(count, end_count, 'Affine registration           ')
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff['affine']
    brain_large = output_aff['brain_large']
    mask_large = output_aff['mask_large']
        
    # Step 3: Segmentation
    count = progress_bar(count, end_count, 'Deepmriprep segmentation                  ')    
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg['p0_large']

    # Prepare for resampling
    header2, affine2 = get_resampled_header(brain.header, brain.affine, target_res, ras_affine)
    header3, affine3 = get_resampled_header(brain.header, brain.affine, target_res, ras_affine, 0)
    dim_target_res = header2['dim']
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())        
    grid_target_res = F.affine_grid(
        inv_affine[None, :3], [1, 3, *dim_target_res[1:4]], 
        align_corners=INTERP_KWARGS['align_corners'])
    shape = nib.as_closest_canonical(mask).shape
    grid_native = F.affine_grid(
        inv_affine[None, :3], [1, 3, *shape], align_corners=INTERP_KWARGS['align_corners'])
        
    warp_template = nib.load(f'{DATA_PATH}/templates/Template_4_GS.nii.gz')

    # Correct bias using label from deepmriprep
    bias, brain_large = correct_bias_field(brain_large, p0_large)

    # Conditional processing based on AMAP or lesion flag
    if (use_amap | save_lesions):
        # AMAP segmentation pipeline
        amapdir = args.amapdir
        
        count = progress_bar(count, end_count, 'Amap segmentation')
        nib.save(brain_large, f'{out_dir}/{out_name}_brain_large.{ext}')
        nib.save(p0_large, f'{out_dir}/{out_name}_seg_large.{ext}')
        
        # Call AMAP and write GM and label map
        cmd = (os.path.join(amapdir, 'CAT_VolAmap') +
            ' -nowrite-corr -bias-fwhm 20 -cleanup 1 -mrf 0 ' +
            ' -write-seg 1 1 1 -label ' +
            f'{out_dir}/{out_name}_seg_large.{ext}' + ' ' +
            f'{out_dir}/{out_name}_brain_large.{ext}')
        os.system(cmd)
      
    if (use_amap):
        # Load Amap label
        p1_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}')
        p2_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}')
        p3_large = nib.load(f'{out_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}')        
    else:
        # Call deepmriprep refinement of deepmriprep label
        count = progress_bar(count, end_count, 'Fine DeepMriPrep segmentation')
        output_nogm = prep.run_segment_nogm(p0_large, affine, t1)

        # Load probability maps for GM, WM, CSF
        p1_large = output_nogm['p1_large']
        p2_large = output_nogm['p2_large']
        p3_large = output_nogm['p3_large']

        gmv = output_nogm['gmv']
        tiv = output_nogm['tiv']
        
    if (use_amap | save_lesions):
        # Get original WM mask from deepmriprep label (without any lesions)
        wm = p0_large.get_fdata() > 2.5

        # Get uncorrected GM/WM maps from Amap
        if (use_amap):
            p1_large_uncorr = p1_large
            p2_large_uncorr = p2_large
            
            # We have to extract the corrected GM from deepmriprep p0 map to identify lesions
            tmp_p0 = p0_large.get_fdata()
            tmp_p0[(tmp_p0 < 1.5) | (tmp_p0 > 2.5)] = 1.5;
            tmp_p0 -= 1.5
            p1_large = nib.Nifti1Image(tmp_p0, affine3, header3)
            
        else:
            p1_large_uncorr = nib.load(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}')
            p2_large_uncorr = nib.load(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}')
            
        # Use difference between corrected and uncorrected GM after correction as lesion map and 
        # restrict it to WM areas
        wm_lesions_large = p1_large_uncorr.get_fdata() - p1_large.get_fdata()
        wm_lesions_large = grey_opening(wm_lesions_large, size=[3, 3, 3])
        ind_wm_lesions = (wm & (wm_lesions_large > 0))
        wm_lesions_large[~ind_wm_lesions] = 0
        
        # Correct GM/WM segmentation + Label
        if (use_amap):
            tmp_p1 = p1_large_uncorr.get_fdata()
            tmp_p1[ind_wm_lesions] -= wm_lesions_large[ind_wm_lesions]
            tmp_p1[tmp_p1 < 0] = 0
            p1_large = nib.Nifti1Image(tmp_p1, affine3, header3)
    
            tmp_p2 = p2_large_uncorr.get_fdata()
            tmp_p2[ind_wm_lesions] += wm_lesions_large[ind_wm_lesions]
            tmp_p2[tmp_p2 > 1] = 1
            p2_large = nib.Nifti1Image(tmp_p2, affine3, header3)
            
        # Convert back to nifti            
        wm_lesions_large = nib.Nifti1Image(wm_lesions_large, affine2, header2)
                
    else:
        p1_large = output_nogm['p1_large']
        p2_large = output_nogm['p2_large']
        p3_large = output_nogm['p3_large']

    wj_affine = np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
    wj_affine = pd.Series([wj_affine])

    # Cleanup (e.g. remove vessels outside cerebellum, but are surrounded by CSF) to refine segmentation
    if (vessel > 0):
        atlas = get_atlas(t1, affine, p1_large, p2_large, p3_large, 'ibsr', None, device)
        cerebellum = get_cerebellum(atlas)
        atlas = get_atlas(t1, affine, p1_large, p2_large, p3_large, 'csf_TPM', None, device)
        csf_TPM = atlas.get_fdata()
        p0_large, p1_large, p2_large, p3_large = cleanup(
            p1_large, p2_large, p3_large, vessel, cerebellum, csf_TPM)
    else:
        tmp = p3_large.get_fdata() + 2*p1_large.get_fdata() + 3*p2_large.get_fdata()
        p0_large = nib.Nifti1Image(tmp, affine2, header2)
    
    # Get affine segmentations
    if ((save_hemilabel) | (save_mwp) | (save_wp) | (save_rp)):
        p1_affine = F.interpolate(
            nifti_to_tensor(p1_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p1_affine = reoriented_nifti(p1_affine, warp_template.affine, warp_template.header)
        p2_affine = F.interpolate(
            nifti_to_tensor(p2_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
        p2_affine = reoriented_nifti(p2_affine, warp_template.affine, warp_template.header)
        if ((save_csf) and (save_rp)):
            p3_affine = F.interpolate(
                nifti_to_tensor(p3_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS)[0, 0]
            p3_affine = reoriented_nifti(p3_affine, warp_template.affine, warp_template.header)

    vol_t1 = 1e-3 * nifti_volume(t1)
    vol_p0 = 1e-3 * nifti_volume(p0_large)
    
    #print(vol_t1)
    #print(vol_p0)
            
    vol_abs_CGW = []
    mean_CGW = []
    for label in (1, 2, 3):
        mask_label = p0_large.get_fdata() == label
        mean_CGW.append(brain_large.get_fdata()[mask_label].mean())
        vol_abs_CGW.append(brain_large.get_fdata()[mask_label].mean())
        
    #print(vol_abs_CGW)
        
    # * p[None].mean((2, 3, 4)).cpu().numpy()
    #abs_vol = pd.DataFrame(vol, columns=['gmv_cm3', 'wmv_cm3', 'csfv_cm3'])
    #rel_vol = pd.DataFrame(vol / vol.sum(), columns=['gmv/tiv', 'wmv/tiv', 'csfv/tiv'])
    #tiv = pd.Series([vol.sum()], name='tiv_cm3')
    
    # Save affine registration
    if (save_rp):
        nib.save(p1_affine, f'{out_dir}/rp1{out_name}_affine.{ext}')
        nib.save(p2_affine, f'{out_dir}/rp2{out_name}_affine.{ext}')
        if (save_csf):
            nib.save(p3_affine, f'{out_dir}/rp3{out_name}_affine.{ext}')

    # Save native registration
    resample_and_save_nifti(p0_large, grid_native, mask.affine, mask.header, 
        f'{out_dir}/p0{out_name}.{ext}')
    if (save_p):
        resample_and_save_nifti(
            brain_large, grid_native, mask.affine, mask.header, 
            f'{out_dir}/m{out_name}.{ext}')
        resample_and_save_nifti(
            p1_large, grid_native, mask.affine, mask.header, 
            f'{out_dir}/p1{out_name}.{ext}')
        resample_and_save_nifti(
            p2_large, grid_native, mask.affine, mask.header, 
            f'{out_dir}/p2{out_name}.{ext}')
        if (save_csf):
            resample_and_save_nifti(
                p3_large, grid_native, mask.affine, mask.header, 
                f'{out_dir}/p3{out_name}.{ext}')
    if (save_lesions):
        resample_and_save_nifti(
            wm_lesions_large, grid_native, mask.affine, mask.header, 
            f'{out_dir}/p7{out_name}.{ext}')

    # Warping is necessary for surface creation and saving warped segmentations
    if ((save_hemilabel) | (save_mwp) | (save_wp)):
        # Step 5: Warping
        count = progress_bar(count, end_count, 'Warping                          ')
        output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        warp_yx = output_reg['warp_yx']
        warp_xy = output_reg['warp_xy']
        
        if (save_mwp):
            mwp1 = output_reg['mwp1']
            mwp2 = output_reg['mwp2']
            nib.save(mwp1, f'{out_dir}/mwp1{out_name}.{ext}')
            nib.save(mwp2, f'{out_dir}/mwp2{out_name}.{ext}')
            
        if (save_wp):
            wp1 = output_reg['mwp1']
            wp2 = output_reg['mwp2']
            nib.save(wp1, f'{out_dir}/wp1{out_name}.{ext}')
            nib.save(wp2, f'{out_dir}/wp2{out_name}.{ext}')

        nib.save(warp_xy, f'{out_dir}/y_{out_name}.{ext}')
        #nib.save(warp_yx, f'{out_dir}/iy_{out_name}.{ext}')

        """
        # write atlas ROI volumes to csv files
        atlas_list = tuple([f'{atlas}_volumes' for atlas in ATLASES])
        atlas_list = list(atlas_list)
        output_paths = tuple([f'{out_dir}/../label/{out_name}_{atlas}.csv' for atlas in ATLASES])
        output_paths = list(output_paths)

        output_atlas = prep.run_atlas_register(t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list)
        for k, output in output_atlas.items():
            print("k")
            print(k)
            print("output")
            print(output)
         
        for i, atl in enumerate(output_atlas):
            print("atl")
            atl
            print("output_paths")
            output_paths[i]
            print("output_atlas")
            output_atlas[i]
        """    
                
    # Atlas is necessary for surface creation
    if (save_hemilabel):
        # Step 6: Atlas creation
        count = progress_bar(count, end_count, 'Atlas creation                 ')
        atlas = get_atlas(t1, affine, p1_large, p2_large, p3_large, 'ibsr', warp_yx, device)
        lh, rh = get_partition(p0_large, atlas)

        # Step 7: Save hemisphere outputs
        count = progress_bar(count, end_count, 'Resampling                     ')
        resample_and_save_nifti(
            nib.Nifti1Image(lh, p0_large.affine, p0_large.header), grid_target_res, 
            affine2, header2, f'{out_dir}/{out_name}_seg_hemi-L.{ext}', True, True)
        resample_and_save_nifti(
            nib.Nifti1Image(rh, p0_large.affine, p0_large.header), grid_target_res, 
            affine2, header2, f'{out_dir}/{out_name}_seg_hemi-R.{ext}', True, True)

    # remove temporary AMAP files
#    if use_amap | save_lesions:
    if 0:
        remove_file(f'{out_dir}/{out_name}_brain_large.{ext}')
        remove_file(f'{out_dir}/{out_name}_brain_large_seg.{ext}')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}')
        remove_file(f'{out_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}')

if __name__ == '__main__':
    run_segment()
