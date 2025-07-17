import os
import sys
import platform
import torch
import argparse
import warnings
import math
import shutil
import zipfile
import urllib.request
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from spline_resize import resize
from deepbet.utils import reoriented_nifti
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.preprocess import Preprocess
from deepmriprep.utils import DATA_PATH, nifti_to_tensor, nifti_volume
from deepmriprep.atlas import ATLASES, get_volumes
from torchreg.utils import INTERP_KWARGS
from pathlib import Path
from scipy.ndimage import (
    grey_opening,
    median_filter,
    binary_dilation,
    binary_closing,
    binary_erosion,
    generate_binary_structure,
)
from utils import (
    progress_bar,
    remove_file,
    correct_bias_field,
    get_atlas,
    resample_and_save_nifti,
    get_resampled_header,
    get_partition,
    align_brain,
    cleanup,
    get_cerebellum,
    get_filenames,
    correct_label_map,
    apply_LAS,
)

from scipy.ndimage import label as label_image

ROOT_PATH = Path(__file__).resolve().parent.parent
TMP_PATH = ROOT_PATH / "tmp_models/"
DATA_PATH_T1PREP = ROOT_PATH / "data/"
MODEL_DIR = Path(DATA_PATH) / "models/"
MODEL_FILES = (
    [
        "brain_extraction_bbox_model.pt",
        "brain_extraction_model.pt",
        "segmentation_nogm_model.pt",
    ]
    + [f"segmentation_patch_{i}_model.pt" for i in range(18)]
    + ["segmentation_model.pt", "warp_model.pt"]
)
MODEL_ZIP_URL = "https://github.com/ChristianGaser/T1Prep/releases/download/v0.1.0-alpha/T1Prep_Models.zip"
MODEL_ZIP_LOCAL = ROOT_PATH / "T1Prep_Models.zip"


# Custom class to override BrainSegmentation
# Skip self.run_patch_models(x, p0) which takes a lot of time and is not needed
# for Amap segmentation
class CustomBrainSegmentation(BrainSegmentation):
    def __call__(self, x, mask):
        x = x[:, :, 1:-2, 15:-12, :-3]
        x = scale_intensity(x)
        p0 = self.run_model(x)  # Skip self.run_patch_models(x, p0)
        if self.fill_holes:
            mask = p0[0, 0].cpu().numpy() > 0.9
            import fill_voids

            mask_filled = fill_voids.fill(mask)
            filled = (mask == 0) & (mask_filled == 1)
            p0[0, 0][filled] = 1.0
        return F.pad(p0, (0, 3, 15, 12, 1, 2))


def all_models_present():
    """Return ``True`` if all required model files are available."""

    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def run_segment():
    """
    Perform brain segmentation (either using deepmriprep or AMAP) on input T1w brain data using
    preprocessing, affine registration, and segmentation techniques.

    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file", required=True, type=str)
    parser.add_argument("--outdir", help="Output folder", required=True, type=str)
    parser.add_argument(
        "--surf",
        action="store_true",
        help="(optional) Save partioned segmentation map for surface estimation.",
    )
    parser.add_argument(
        "--csf", action="store_true", help="(optional) Save also CSF segmentations."
    )
    parser.add_argument(
        "--mwp",
        action="store_true",
        help="(optional) Save modulated and warped segmentations.",
    )
    parser.add_argument(
        "--wp", action="store_true", help="(optional) Save warped segmentations."
    )
    parser.add_argument(
        "--p", action="store_true", help="(optional) Save native segmentations."
    )
    parser.add_argument(
        "--rp",
        action="store_true",
        help="(optional) Save affine registered segmentations.",
    )
    parser.add_argument(
        "--lesions", action="store_true", help="(optional) Save also WMH lesions."
    )
    parser.add_argument(
        "--bids", action="store_true", help="(optional) Use bids naming convention."
    )
    parser.add_argument(
        "--gz", action="store_true", help="(optional) Save nii.gz images."
    )
    parser.add_argument(
        "--amap", action="store_true", help="(optional) Use AMAP segmentation."
    )
    parser.add_argument(
        "--bias-fwhm",
        type=float,
        default=0,
        help="(optional)FWHM value for the bias correction in AMAP.",
    )
    parser.add_argument("--amapdir", help="Amap binary folder", type=str)
    parser.add_argument("--verbose", action="store_true", help="(optional) Be verbose.")
    parser.add_argument(
        "--debug", action="store_true", help="(optional) Do not delete temporary files."
    )
    parser.add_argument(
        "--vessel",
        type=float,
        default=0.4,
        help="Initial threshold to isolate WM for vessel removal",
    )
    args = parser.parse_args()

    # Input/output parameters
    t1_name = args.input
    out_dir = args.outdir
    amap_dir = args.amapdir

    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    vessel = args.vessel
    bias_fwhm = args.bias_fwhm
    verbose = args.verbose
    debug = args.debug

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
    elif torch.backends.mps.is_available() and False:  # not yet fully supported
        device = torch.device("mps")
        no_gpu = False
    else:
        device = torch.device("cpu")
        no_gpu = True

    # Set processing parameters
    target_res = np.array([0.5] * 3)  # Target resolution for resampling
    count = 1
    end_count = 4
    if save_mwp:
        end_count += 1
    if save_hemilabel:
        end_count += 2
    if save_lesions:
        end_count += 1

    if save_gz:
        ext = "nii.gz"
    else:
        ext = "nii"

    # Prepare filenames and load input MRI data
    out_name = (
        os.path.basename(os.path.basename(t1_name).replace("_desc-sanlm", ""))
        .replace(".nii", "")
        .replace(".gz", "")
    )
    t1 = nib.load(t1_name)

    # Copy necessary model files from local folder to install it, since often the API rate limit is exceeded
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download required model files from Github as zip-file and unzip it to DATA_PATH
    if not all_models_present():
        print("One or more model files are missing. Downloading zip archive...")
        # Download the zip file if not already present
        if not MODEL_ZIP_LOCAL.exists():
            print(f"Downloading {MODEL_ZIP_URL} ...")
            urllib.request.urlretrieve(MODEL_ZIP_URL, MODEL_ZIP_LOCAL)
            print("Download complete.")

        # Extract zip
        with zipfile.ZipFile(MODEL_ZIP_LOCAL, "r") as zip_ref:
            zip_ref.extractall(TMP_PATH)
        print(f"Unzipped models to {TMP_PATH}")
        for file in MODEL_FILES:
            shutil.copy(
                f"{TMP_PATH}/T1Prep/data/models/{file}", f"{DATA_PATH}/models/{file}"
            )
        shutil.rmtree(TMP_PATH)

        # Optionally, delete the zip to save space
        MODEL_ZIP_LOCAL.unlink()

    for file in MODEL_FILES:
        if not Path(f"{DATA_PATH}/models/{file}").exists():
            shutil.copy(
                f"{DATA_PATH_T1PREP}/models/{file}", f"{DATA_PATH}/models/{file}"
            )

    # Preprocess the input volume
    vol = t1.get_fdata().copy()
    vol = np.squeeze(vol)
    vol, affine_resamp, header_resamp, ras_affine = align_brain(
        vol, t1.affine, t1.header, np.eye(4), do_flip=0
    )
    t1 = nib.Nifti1Image(vol, affine_resamp, header_resamp)

    prep = Preprocess(no_gpu)

    # Use faster preprocessing and segmentation for Amap segmentation
    if use_amap:
        prep.brain_segment = CustomBrainSegmentation(no_gpu=no_gpu)

    # Step 1: Skull-stripping
    if verbose:
        count = progress_bar(count, end_count, "Skull-stripping               ")
    output_bet = prep.run_bet(t1)
    brain = output_bet["brain"]
    mask = output_bet["mask"]

    # Step 2: Affine registration
    if verbose:
        count = progress_bar(count, end_count, "Affine registration           ")
    output_aff = prep.run_affine_register(brain, mask)
    affine = output_aff["affine"]
    brain_large = output_aff["brain_large"]
    mask_large = output_aff["mask_large"]

    # Step 3: Segmentation
    if verbose:
        count = progress_bar(
            count, end_count, "DeepMriPrep segmentation                  "
        )
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg["p0_large"]

    # Prepare for resampling
    header_resamp, affine_resamp = get_resampled_header(
        brain.header, brain.affine, target_res, ras_affine
    )
    header_resamp_reordered, affine_resamp_reordered = get_resampled_header(
        brain.header, brain.affine, target_res, ras_affine, reorder_method=0
    )
    dim_target_res = header_resamp["dim"]
    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())
    grid_target_res = F.affine_grid(
        inv_affine[None, :3],
        [1, 3, *dim_target_res[1:4]],
        align_corners=INTERP_KWARGS["align_corners"],
    )
    shape = nib.as_closest_canonical(mask).shape
    grid_native = F.affine_grid(
        inv_affine[None, :3],
        [1, 3, *shape],
        align_corners=INTERP_KWARGS["align_corners"],
    )

    warp_template = nib.load(f"{DATA_PATH}/templates/Template_4_GS.nii.gz")

    # Get filenames for different spaces and sides w.r.t. BIDS flag
    code_vars = get_filenames(use_bids, out_name, "", "", "", ext)
    space_affine = code_vars.get("Affine_space", "")
    space_warp = code_vars.get("Warp_space", "")
    space_warp_modulated = code_vars.get("Warp_modulated_space", "")

    if use_bids:
        code_vars_affine = get_filenames(use_bids, out_name, "", "", space_affine, ext)
    else:
        code_vars_affine = get_filenames(
            use_bids, out_name, "", "_affine", space_affine, ext
        )
    code_vars_warped_modulated = get_filenames(
        use_bids, out_name, "", "", space_warp_modulated, ext
    )
    code_vars_warped = get_filenames(use_bids, out_name, "", "", space_warp, ext)
    code_vars_left = get_filenames(use_bids, out_name, "left", "", "", ext)
    code_vars_right = get_filenames(use_bids, out_name, "right", "", "", ext)

    # Correct bias using label from deepmriprep
    brain_large = correct_bias_field(brain_large, p0_large)

    # Conditional processing based on AMAP or lesion flag
    if use_amap or save_lesions:
        # AMAP segmentation pipeline
        amapdir = args.amapdir

        if verbose:
            count = progress_bar(count, end_count, "Amap segmentation        ")

        p0_large_orig = p0_large
        p0_large, brain_large = correct_label_map(brain_large, p0_large)
        brain_large = apply_LAS(brain_large, p0_large)

        nib.save(brain_large, f"{out_dir}/{out_name}_brain_large_tmp.{ext}")
        nib.save(p0_large, f"{out_dir}/{out_name}_seg_large.{ext}")

        # Call SANLM filter and rename output to original name
        cmd = (
            os.path.join(amapdir, "CAT_VolSanlm")
            + " "
            + f"{out_dir}/{out_name}_brain_large_tmp.{ext}"
            + " "
            + f"{out_dir}/{out_name}_brain_large.{ext}"
        )
        os.system(cmd)

        # Call AMAP and write GM and label map
        cmd = (
            os.path.join(amapdir, "CAT_VolAmap")
            + f" -use-bmap -nowrite-corr -bias-fwhm {bias_fwhm} -cleanup 1 -mrf 0 "
            + "-h-ornlm 0.025 -write-seg 1 1 1 -label "
            + f"{out_dir}/{out_name}_seg_large.{ext}"
            + " "
            + f"{out_dir}/{out_name}_brain_large.{ext}"
        )
        if verbose and debug:
            cmd += " -verbose"
        os.system(cmd)

    if use_amap:
        # Load Amap label
        p1_large = nib.load(f"{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        p2_large = nib.load(f"{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        p3_large = nib.load(f"{out_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")
    else:
        # Call deepmriprep refinement of deepmriprep label
        if verbose:
            count = progress_bar(
                count, end_count, "Fine DeepMriPrep segmentation         "
            )
        output_nogm = prep.run_segment_nogm(p0_large, affine, t1)

        # Load probability maps for GM, WM, CSF
        p1_large = output_nogm["p1_large"]
        p2_large = output_nogm["p2_large"]
        p3_large = output_nogm["p3_large"]

        gmv = output_nogm["gmv"]
        tiv = output_nogm["tiv"]

    if use_amap or save_lesions:

        if debug:
            p0_large_diff = (
                p3_large.get_fdata().copy()
                + 2 * p1_large.get_fdata().copy()
                + 3 * p2_large.get_fdata().copy()
                - p0_large_orig.get_fdata().copy()
            )
            p0_large_diff = median_filter(p0_large_diff, size=3)
            p0_large_diff = nib.Nifti1Image(
                p0_large_diff, p0_large.affine, p0_large.header
            )
            nib.save(
                p0_large_diff,
                f"{out_dir}/{out_name}_brain_large_label-discrepancy.{ext}",
            )

        # Get original WM and CSF mask from deepmriprep label (without any lesions)
        p0_tmp = p0_large_orig.get_fdata().copy()
        wm = p0_tmp >= 2.5
        wm = binary_closing(wm, generate_binary_structure(3, 3), 3)
        wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
        gm = (p0_tmp >= 1.5) & (p0_tmp < 2.5)
        csf = (p0_tmp < 1.5) & (p0_tmp > 0)

        # Get uncorrected GM/WM maps from Amap
        if use_amap:
            p1_large_uncorr = p1_large
            p2_large_uncorr = p2_large
            p3_large_uncorr = p3_large

            # We have to extract the corrected GM from deepmriprep p0 map to identify lesions
            tmp_p0 = p0_large_orig.get_fdata().copy()
            tmp_p0[csf | wm] = 1.5
            tmp_p0 -= 1.5
            p1_large = nib.Nifti1Image(
                tmp_p0, affine_resamp_reordered, header_resamp_reordered
            )

            tmp_p0 = p0_large_orig.get_fdata().copy()
            tmp_p0[~csf] = 0
            p3_large = nib.Nifti1Image(
                tmp_p0, affine_resamp_reordered, header_resamp_reordered
            )

        else:
            p1_large_uncorr = nib.load(
                f"{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}"
            )
            p2_large_uncorr = nib.load(
                f"{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}"
            )
            p3_large_uncorr = nib.load(
                f"{out_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}"
            )

        # Use difference between corrected and uncorrected GM after correction as lesion map and
        # restrict it to WM areas
        wm_lesions_large = (
            p1_large_uncorr.get_fdata().copy() - p1_large.get_fdata().copy()
        )
        wm_lesions_large = median_filter(wm_lesions_large, size=3)

        # Create deep WM mask
        deep_wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)

        # Exclude lesions near cortex (i.e. near GM)
        gm_border = binary_dilation(gm, generate_binary_structure(3, 3), 2)

        atlas = get_atlas(
            t1,
            affine,
            p0_large.header,
            p0_large.affine,
            "cat_wmh_miccai2017",
            None,
            device,
        )
        wmh_TPM = atlas.get_fdata().copy()
        wmh_TPM /= np.max(wmh_TPM)

        # Keep only lesions inside TPM for WMH and in deep WM, far from cortex
        ind_wm_lesions = ((wm_lesions_large * wmh_TPM) > 0.025) & deep_wm & (~gm_border)

        # Remove smaller clusters
        label_map, n_labels = label_image(ind_wm_lesions)
        sizes = np.bincount(label_map.ravel())
        min_lesion_size = 500  # is resolution-dependent and should be changed for other resolutiuons than 0.5mm
        remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
        ind_wm_lesions[remove] = 0

        wm_lesions_large[~ind_wm_lesions] = 0

        # Correct GM/WM segmentation + Label
        if use_amap:
            csf_discrep_large = (
                p3_large_uncorr.get_fdata().copy() - p3_large.get_fdata().copy()
            )
            csf_discrep_large = median_filter(csf_discrep_large, size=3)
            ind_csf_discrep = csf_discrep_large < 0

            if True:
                csf = binary_closing(csf, generate_binary_structure(3, 3), 2)
                nib.save(
                    nib.Nifti1Image(
                        csf_discrep_large, p0_large.affine, p0_large.header
                    ),
                    f"{out_dir}/csf_discrep.{ext}",
                )
                nib.save(
                    nib.Nifti1Image(csf, p0_large.affine, p0_large.header),
                    f"{out_dir}/csf.{ext}",
                )
                nib.save(
                    nib.Nifti1Image(ind_csf_discrep, p0_large.affine, p0_large.header),
                    f"{out_dir}/csf_ind.{ext}",
                )

            tmp_p3 = p3_large_uncorr.get_fdata().copy()
            tmp_p3[ind_csf_discrep] += csf_discrep_large[ind_csf_discrep]
            np.clip(tmp_p3, 0, 1)
            p3_large = nib.Nifti1Image(
                tmp_p3, affine_resamp_reordered, header_resamp_reordered
            )

            tmp_p1 = p1_large_uncorr.get_fdata().copy()
            tmp_p1[ind_wm_lesions] -= wm_lesions_large[ind_wm_lesions]
            tmp_p1[ind_csf_discrep] -= csf_discrep_large[ind_csf_discrep]
            np.clip(tmp_p1, 0, 1)
            p1_large = nib.Nifti1Image(
                tmp_p1, affine_resamp_reordered, header_resamp_reordered
            )

            tmp_p2 = p2_large_uncorr.get_fdata().copy()
            tmp_p2[ind_wm_lesions] += wm_lesions_large[ind_wm_lesions]
            np.clip(tmp_p2, 0, 1)
            p2_large = nib.Nifti1Image(
                tmp_p2, affine_resamp_reordered, header_resamp_reordered
            )

    # else:
    #    p1_large = output_nogm['p1_large']
    #    p2_large = output_nogm['p2_large']
    #    p3_large = output_nogm['p3_large']

    wj_affine = (
        np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
    )
    wj_affine = pd.Series([wj_affine])

    # Cleanup (e.g. remove vessels outside cerebellum, but are surrounded by CSF) to refine segmentation
    if vessel > 0:
        atlas = get_atlas(
            t1, affine, p0_large.header, p0_large.affine, "ibsr", None, device
        )
        cerebellum = get_cerebellum(atlas)
        atlas = get_atlas(
            t1, affine, p0_large.header, p0_large.affine, "csf_TPM", None, device
        )
        csf_TPM = atlas.get_fdata().copy()
        p0_large, p1_large, p2_large, p3_large = cleanup(
            p1_large, p2_large, p3_large, vessel, cerebellum, csf_TPM
        )
    else:
        tmp = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
        )
        p0_large = nib.Nifti1Image(tmp, affine_resamp, header_resamp)

    if use_amap or save_lesions:
        tmp_p0 = p0_large.get_fdata().copy()
        np.clip(tmp_p0, 0, 3)
        tmp_p0[ind_wm_lesions] += wm_lesions_large[ind_wm_lesions]
        np.clip(tmp_p0, 0, 4)
        p0_large = nib.Nifti1Image(
            tmp_p0, affine_resamp_reordered, header_resamp_reordered
        )

    # Get affine segmentations
    if save_hemilabel or save_mwp or save_wp or save_rp:
        p1_affine = resize(
            nifti_to_tensor(p1_large)[None, None],
            scale_factor=1 / 3,
            align_corners=INTERP_KWARGS["align_corners"],
            mask_value=0,
        )[0, 0]
        p1_affine = reoriented_nifti(
            p1_affine, warp_template.affine, warp_template.header
        )
        p2_affine = resize(
            nifti_to_tensor(p2_large)[None, None],
            scale_factor=1 / 3,
            align_corners=INTERP_KWARGS["align_corners"],
            mask_value=0,
        )[0, 0]
        p2_affine = reoriented_nifti(
            p2_affine, warp_template.affine, warp_template.header
        )
        if save_csf and save_rp:
            p3_affine = resize(
                nifti_to_tensor(p3_large)[None, None],
                scale_factor=1 / 3,
                align_corners=INTERP_KWARGS["align_corners"],
                mask_value=0,
            )[0, 0]
            p3_affine = reoriented_nifti(
                p3_affine, warp_template.affine, warp_template.header
            )

    vol_t1 = 1e-3 * nifti_volume(t1)
    vol_p0 = 1e-3 * nifti_volume(p0_large)

    # print(vol_t1)
    # print(vol_p0)

    vol_abs_CGW = []
    mean_CGW = []
    for label in (1, 2, 3):
        mask_label = p0_large.get_fdata().copy() == label
        mean_CGW.append(brain_large.get_fdata().copy()[mask_label].mean())
        vol_abs_CGW.append(brain_large.get_fdata().copy()[mask_label].mean())

    # print(vol_abs_CGW)

    # * p[None].mean((2, 3, 4)).cpu().numpy()
    # abs_vol = pd.DataFrame(vol, columns=['gmv_cm3', 'wmv_cm3', 'csfv_cm3'])
    # rel_vol = pd.DataFrame(vol / vol.sum(), columns=['gmv/tiv', 'wmv/tiv', 'csfv/tiv'])
    # tiv = pd.Series([vol.sum()], name='tiv_cm3')

    # Save affine registration
    if save_rp:
        gm_name = code_vars_affine.get("GM_volume", "")
        wm_name = code_vars_affine.get("WM_volume", "")
        nib.save(p1_affine, f"{out_dir}/{gm_name}")
        nib.save(p2_affine, f"{out_dir}/{wm_name}")
        if save_csf:
            csf_name = code_vars_affine.get("CSF_volume", "")
            nib.save(p3_affine, f"{out_dir}/{csf_name}")

    # Save native registration
    label_name = code_vars.get("Label_volume", "")
    mT1_name = code_vars.get("mT1_volume", "")
    resample_and_save_nifti(
        p0_large,
        grid_native,
        mask.affine,
        mask.header,
        f"{out_dir}/{label_name}",
        clip=[0, 4],
    )
    resample_and_save_nifti(
        brain_large, grid_native, mask.affine, mask.header, f"{out_dir}/{mT1_name}"
    )

    if save_p:
        gm_name = code_vars.get("GM_volume", "")
        wm_name = code_vars.get("WM_volume", "")
        resample_and_save_nifti(
            p1_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{out_dir}/{gm_name}",
            clip=[0, 1],
        )
        resample_and_save_nifti(
            p2_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{out_dir}/{wm_name}",
            clip=[0, 1],
        )
        if save_csf:
            csf_name = code_vars.get("CSF_volume", "")
            resample_and_save_nifti(
                p3_large,
                grid_native,
                mask.affine,
                mask.header,
                f"{out_dir}/{csf_name}",
                clip=[0, 1],
            )

    if save_lesions:
        # Convert back to nifti
        wm_lesions_large = nib.Nifti1Image(
            wm_lesions_large, p0_large.affine, p0_large.header
        )
        wmh_name = code_vars.get("WMH_volume", "")
        resample_and_save_nifti(
            wm_lesions_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{out_dir}/{wmh_name}",
        )

    # Warping is necessary for surface creation and saving warped segmentations
    if save_hemilabel or save_mwp or save_wp:
        # Step 5: Warping
        if verbose:
            count = progress_bar(count, end_count, "Warping                          ")
        output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        warp_yx = output_reg["warp_yx"]
        warp_xy = output_reg["warp_xy"]

        if save_mwp:
            gm_name = code_vars_warped_modulated.get("GM_volume", "")
            wm_name = code_vars_warped_modulated.get("WM_volume", "")
            mwp1 = output_reg["mwp1"]
            mwp2 = output_reg["mwp2"]
            nib.save(mwp1, f"{out_dir}/{gm_name}")
            nib.save(mwp2, f"{out_dir}/{wm_name}")
            if save_csf:
                csf_name = code_vars_warped_modulated.get("CSF_volume", "")
                mwp3 = output_reg["mwp3"]
                nib.save(mwp3, f"{out_dir}/{csf_name}")

        if save_wp:
            gm_name = code_vars_warped.get("GM_volume", "")
            wm_name = code_vars_warped.get("WM_volume", "")
            wp1 = output_reg["wp1"]
            wp2 = output_reg["wp2"]
            nib.save(wp1, f"{out_dir}/{gm_name}")
            nib.save(wp2, f"{out_dir}/{wm_name}")
            if save_csf:
                csf_name = code_vars_warped.get("CSF_volume", "")
                wp3 = output_reg["wp3"]
                nib.save(wp3, f"{out_dir}/{csf_name}")

        def_name = code_vars.get("Def_volume", "")
        nib.save(warp_xy, f"{out_dir}/{def_name}")
        invdef_name = code_vars.get("invDef_volume", "")
        # nib.save(warp_yx, f'{out_dir}/{invdef_name}')

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
    if save_hemilabel:
        # Step 6: Atlas creation
        if verbose:
            count = progress_bar(count, end_count, "Atlas creation                 ")
        atlas = get_atlas(
            t1, affine, p0_large.header, p0_large.affine, "ibsr", warp_yx, device
        )
        lh, rh = get_partition(p0_large, atlas)

        # Step 7: Save hemisphere outputs
        if verbose:
            count = progress_bar(count, end_count, "Resampling                     ")

        hemileft_name = code_vars_left.get("Hemi_volume", "")
        hemiright_name = code_vars_right.get("Hemi_volume", "")

        resample_and_save_nifti(
            nib.Nifti1Image(lh, p0_large.affine, p0_large.header),
            grid_target_res,
            affine_resamp,
            header_resamp,
            f"{out_dir}/{hemileft_name}",
            True,
            True,
        )
        resample_and_save_nifti(
            nib.Nifti1Image(rh, p0_large.affine, p0_large.header),
            grid_target_res,
            affine_resamp,
            header_resamp,
            f"{out_dir}/{hemiright_name}",
            True,
            True,
        )

    # remove temporary AMAP files
    if (use_amap or save_lesions) and not debug:
        remove_file(f"{out_dir}/{out_name}_brain_large_tmp.{ext}")
        remove_file(f"{out_dir}/{out_name}_brain_large_seg.{ext}")
        remove_file(f"{out_dir}/{out_name}_brain_large.{ext}")
        remove_file(f"{out_dir}/{out_name}_seg_large.{ext}")
        remove_file(f"{out_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        remove_file(f"{out_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        remove_file(f"{out_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")


if __name__ == "__main__":
    run_segment()
