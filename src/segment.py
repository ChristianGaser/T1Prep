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
import fill_voids
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from deepbet.utils import reoriented_nifti
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.preprocess import Preprocess
from deepmriprep.utils import DATA_PATH, nifti_to_tensor, nifti_volume
from deepmriprep.atlas import get_volumes
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
    unsmooth_kernel,
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
    get_volume_native_space,
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


class CustomBrainSegmentation(BrainSegmentation):
    """
    Custom class to override BrainSegmentation
    Skip self.run_patch_models(x, p0) which takes a lot of time and is not needed
    for Amap segmentation.
    Furthermore use run_model function with linear interpolation for p0, which
    prevents negative values due to sinc-interpolation
    """

    def __call__(self, x, mask):
        x = x[:, :, 1:-2, 15:-12, :-3]
        x = scale_intensity(x)
        p0 = self.run_model(x)  # Skip self.run_patch_models(x, p0)
        if self.fill_holes:
            mask = p0[0, 0].cpu().numpy() > 0.9
            mask_filled = fill_voids.fill(mask)
            filled = (mask == 0) & (mask_filled == 1)
            p0[0, 0][filled] = 1.0
        return F.pad(p0, (0, 3, 15, 12, 1, 2))

    def run_model(self, x, scale_factor=1.5):
        with torch.no_grad():
            p0 = self.model(
                F.interpolate(x, scale_factor=1 / scale_factor, **INTERP_KWARGS)
            )
        return F.interpolate(p0, scale_factor=scale_factor, **INTERP_KWARGS)


class CustomPreprocess(Preprocess):
    """
    Custom class to override Preprocess
    Use linear interpolation for p0, which prevents negative values due to
    sinc-interpolation
    """

    def run_segment_brain(self, brain_large, mask, affine, mask_large):
        brain_large = nifti_to_tensor(brain_large)
        mask_large = nifti_to_tensor(mask_large)
        p0_large = self.brain_segment(
            brain_large[None, None].to(self.device),
            mask_large[None, None].to(self.device),
        )[0, 0]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        p0_large[mask_large == 0.0] = 0.0
        inv_affine = torch.linalg.inv(
            torch.from_numpy(affine.values).float().to(self.device)
        )
        shape = nib.as_closest_canonical(mask).shape
        grid = F.affine_grid(
            inv_affine[None, :3],
            [1, 3, *shape],
            align_corners=INTERP_KWARGS["align_corners"],
        )
        p0 = F.conv3d(
            p0_large[None, None].to(self.device),
            unsmooth_kernel(device=self.device)[None, None],
            padding=1,
        )
        p0 = F.grid_sample(p0, grid, align_corners=INTERP_KWARGS["align_corners"])[0, 0]
        p0 = p0.clip(min=0, max=3).cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "p0_large": reoriented_nifti(
                p0_large.cpu().numpy(), **self.affine_template_metadata
            ),
            "p0": reoriented_nifti(p0.cpu().numpy(), mask.affine, mask.header),
        }


def all_models_present():
    """Return ``True`` if all required model files are available."""

    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file", required=True, type=str)
    parser.add_argument("--mri_dir", help="Output mri folder", required=True, type=str)
    parser.add_argument("--report_dir", help="Output report folder", type=str)
    parser.add_argument("--label_dir", help="Output label folder", type=str)
    parser.add_argument("--atlas", help="Atlases for ROI estimation", type=str)
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
    return parser.parse_args()


def setup_device() -> tuple[torch.device, bool]:
    """Return the torch device and ``no_gpu`` flag."""

    if torch.cuda.is_available():
        return torch.device("cuda"), False
    if torch.backends.mps.is_available() and False:  # not yet fully supported
        return torch.device("mps"), False
    return torch.device("cpu"), True


def prepare_model_files() -> None:
    """Ensure required model files are present, downloading if needed."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not all_models_present():
        print("One or more model files are missing. Downloading zip archive...")
        if not MODEL_ZIP_LOCAL.exists():
            print(f"Downloading {MODEL_ZIP_URL} ...")
            urllib.request.urlretrieve(MODEL_ZIP_URL, MODEL_ZIP_LOCAL)
            print("Download complete.")

        with zipfile.ZipFile(MODEL_ZIP_LOCAL, "r") as zip_ref:
            zip_ref.extractall(TMP_PATH)
        print(f"Unzipped models to {TMP_PATH}")
        for file in MODEL_FILES:
            shutil.copy(
                f"{TMP_PATH}/T1Prep/data/models/{file}", f"{DATA_PATH}/models/{file}"
            )
        shutil.rmtree(TMP_PATH)
        MODEL_ZIP_LOCAL.unlink()

    for file in MODEL_FILES:
        if not Path(f"{DATA_PATH}/models/{file}").exists():
            shutil.copy(
                f"{DATA_PATH_T1PREP}/models/{file}", f"{DATA_PATH}/models/{file}"
            )


def preprocess_input(t1: nib.Nifti1Image, no_gpu: bool, use_amap: bool):
    """Align the input volume and create the preprocessing object."""

    vol = t1.get_fdata().copy()
    vol = np.squeeze(vol)
    vol, affine_resamp, header_resamp, ras_affine = align_brain(
        vol, t1.affine, t1.header, np.eye(4), do_flip=0
    )
    t1 = nib.Nifti1Image(vol, affine_resamp, header_resamp)
    prep = CustomPreprocess(no_gpu)
    if use_amap:
        prep.brain_segment = CustomBrainSegmentation(no_gpu=no_gpu)
    return t1, prep, ras_affine


def skull_strip(
    prep: CustomPreprocess,
    t1: nib.Nifti1Image,
    verbose: bool,
    count: int,
    end_count: int,
):
    """Run skull stripping and return brain and mask images."""

    if verbose:
        count = progress_bar(count, end_count, "Skull-stripping               ")
    output = prep.run_bet(t1)
    return output["brain"], output["mask"], count


def affine_register(
    prep: CustomPreprocess,
    brain: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    verbose: bool,
    count: int,
    end_count: int,
):
    """Perform affine registration of the brain."""

    if verbose:
        count = progress_bar(count, end_count, "Affine registration           ")
    output = prep.run_affine_register(brain, mask)
    return (
        output["affine"], 
        output["brain_large"], 
        output["mask_large"], 
        output["affine_loss"], 
        count,
    )


def run_amap_segmentation(
    amapdir: str,
    p0_large: nib.Nifti1Image,
    brain_large: nib.Nifti1Image,
    mri_dir: str,
    out_name: str,
    ext: str,
    verbose: bool,
    debug: bool,
):
    """Execute the AMAP segmentation pipeline."""

    if verbose:
        print("Running AMAP segmentation")

    p0_large, brain_large = correct_label_map(brain_large, p0_large)
    brain_large = apply_LAS(brain_large, p0_large)

    nib.save(brain_large, f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
    nib.save(p0_large, f"{mri_dir}/{out_name}_seg_large.{ext}")

    # Call SANLM filter and rename output to original name
    cmd = (
        os.path.join(amapdir, "CAT_VolSanlm")
        + " "
        + f"{mri_dir}/{out_name}_brain_large_tmp.{ext}"
        + " "
        + f"{mri_dir}/{out_name}_brain_large.{ext}"
    )
    os.system(cmd)

    # Call AMAP and write tissue and label maps
    cmd = (
        os.path.join(amapdir, "CAT_VolAmap")
        + f" -nowrite-corr -bias-fwhm 0 -cleanup 1 -mrf 0 "
        + "-h-ornlm 0.05 -write-seg 1 1 1 -label "
        + f"{mri_dir}/{out_name}_seg_large.{ext}"
        + " "
        + f"{mri_dir}/{out_name}_brain_large.{ext}"
    )
    if verbose and debug:
        cmd += " -verbose"
    os.system(cmd)
    return brain_large, p0_large


def final_cleanup(
    mri_dir: str,
    out_name: str,
    ext: str,
    use_amap: bool,
    save_lesions: bool,
    debug: bool,
) -> None:
    """Remove temporary files generated during processing."""

    remove_file(f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
    if (use_amap or save_lesions) and not debug:
        remove_file(f"{mri_dir}/{out_name}_brain_large_seg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_seg_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")


def handle_lesions(
    t1: nib.Nifti1Image,
    affine,
    p0_large: nib.Nifti1Image,
    p0_large_orig: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    affine_resamp_reordered,
    header_resamp_reordered,
    mri_dir: str,
    out_name: str,
    ext: str,
    use_amap: bool,
    debug: bool,
    device: torch.device,
) -> tuple[
    nib.Nifti1Image,
    nib.Nifti1Image,
    nib.Nifti1Image,
    np.ndarray,
    np.ndarray,
]:
    """Detect lesions and correct tissue probability maps."""

    if debug:
        p0_large_diff = (
            p3_large.get_fdata().copy()
            + 2 * p1_large.get_fdata().copy()
            + 3 * p2_large.get_fdata().copy()
            - p0_large_orig.get_fdata().copy()
        )
        p0_large_diff = median_filter(p0_large_diff, size=3)
        p0_large_diff = nib.Nifti1Image(p0_large_diff, p0_large.affine, p0_large.header)
        nib.save(
            p0_large_diff,
            f"{mri_dir}/{out_name}_brain_large_label-discrepancy.{ext}",
        )

    p0_value = p0_large_orig.get_fdata().copy()
    wm = p0_value >= 2.5
    wm = binary_closing(wm, generate_binary_structure(3, 3), 3)
    wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
    gm = (p0_value >= 1.5) & (p0_value < 2.5)
    csf = (p0_value < 1.5) & (p0_value > 0)

    if use_amap:
        p1_large_uncorr = p1_large
        p2_large_uncorr = p2_large
        p3_large_uncorr = p3_large

        p0_value = p0_large_orig.get_fdata().copy()
        p0_value[csf | wm] = 1.5
        p0_value -= 1.5
        p1_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )

        p0_value = p0_large_orig.get_fdata().copy()
        p0_value[~csf] = 0
        p3_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )
    else:
        p1_large_uncorr = nib.load(
            f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}"
        )
        p2_large_uncorr = nib.load(
            f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}"
        )
        p3_large_uncorr = nib.load(
            f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}"
        )

    wmh_value = p1_large_uncorr.get_fdata().copy() - p1_large.get_fdata().copy()
    wmh_value = median_filter(wmh_value, size=3)

    deep_wm = binary_erosion(wm, generate_binary_structure(3, 3), 2)
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
    wmh_tpm = atlas.get_fdata().copy()
    wmh_tpm /= np.max(wmh_tpm)

    ind_wmh = ((wmh_value * wmh_tpm) > 0.025) & deep_wm & (~gm_border)

    label_map, _ = label_image(ind_wmh)
    sizes = np.bincount(label_map.ravel())
    min_lesion_size = 500
    remove = np.isin(label_map, np.where(sizes < min_lesion_size)[0])
    ind_wmh[remove] = 0

    wmh_value[~ind_wmh] = 0

    if use_amap:
        csf_discrep_large = (
            p3_large_uncorr.get_fdata().copy() - p3_large.get_fdata().copy()
        )
        csf_discrep_large = median_filter(csf_discrep_large, size=3)
        ind_csf_discrep = csf_discrep_large < 0

        tmp_p3 = p3_large_uncorr.get_fdata().copy()
        tmp_p3[ind_csf_discrep] += csf_discrep_large[ind_csf_discrep]
        np.clip(tmp_p3, 0, 1)
        p3_large = nib.Nifti1Image(
            tmp_p3, affine_resamp_reordered, header_resamp_reordered
        )

        tmp_p1 = p1_large_uncorr.get_fdata().copy()
        tmp_p1[ind_wmh] -= wmh_value[ind_wmh]
        tmp_p1[ind_csf_discrep] -= csf_discrep_large[ind_csf_discrep]
        np.clip(tmp_p1, 0, 1)
        p1_large = nib.Nifti1Image(
            tmp_p1, affine_resamp_reordered, header_resamp_reordered
        )

        tmp_p2 = p2_large_uncorr.get_fdata().copy()
        tmp_p2[ind_wmh] += wmh_value[ind_wmh]
        np.clip(tmp_p2, 0, 1)
        p2_large = nib.Nifti1Image(
            tmp_p2, affine_resamp_reordered, header_resamp_reordered
        )
    
    return p1_large, p2_large, p3_large, wmh_value, ind_wmh


def save_results(
    prep: CustomPreprocess,
    t1: nib.Nifti1Image,
    affine,
    p0_large: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wmh_large: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    brain_large: nib.Nifti1Image,
    grid_native,
    grid_target_res,
    warp_template: nib.Nifti1Image,
    wj_affine: pd.Series,
    save_p: bool,
    save_rp: bool,
    save_wp: bool,
    save_mwp: bool,
    save_hemilabel: bool,
    save_lesions: bool,
    save_csf: bool,
    verbose: bool,
    count: int,
    end_count: int,
    mri_dir: str,
    label_dir: str,
    report_dir: str,
    out_name: str,
    ext: str,
    use_bids: bool,
    device,
    affine_resamp,
    header_resamp,
    atlas_list,
) -> None:
    """Save segmentation and atlas results to disk."""

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

    # Get affine segmentations
    if save_hemilabel or save_mwp or save_wp or save_rp:
        p1_affine = F.interpolate(
            nifti_to_tensor(p1_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS
        )[0, 0]
        p1_affine = reoriented_nifti(
            p1_affine, warp_template.affine, warp_template.header
        )
        p2_affine = F.interpolate(
            nifti_to_tensor(p2_large)[None, None], scale_factor=1 / 3, **INTERP_KWARGS
        )[0, 0]
        p2_affine = reoriented_nifti(
            p2_affine, warp_template.affine, warp_template.header
        )
        if save_csf and save_rp:
            p3_affine = F.interpolate(
                nifti_to_tensor(p3_large)[None, None],
                scale_factor=1 / 3,
                **INTERP_KWARGS,
            )[0, 0]
            p3_affine = reoriented_nifti(
                p3_affine, warp_template.affine, warp_template.header
            )
    else:
        p1_affine = p2_affine = p3_affine = None

    # Save affine registered data
    if save_rp:
        gm_name = code_vars_affine.get("GM_volume", "")
        wm_name = code_vars_affine.get("WM_volume", "")
        nib.save(p1_affine, f"{mri_dir}/{gm_name}")
        nib.save(p2_affine, f"{mri_dir}/{wm_name}")
        if save_csf:
            csf_name = code_vars_affine.get("CSF_volume", "")
            nib.save(p3_affine, f"{mri_dir}/{csf_name}")

    # Save data in native space
    label_name = code_vars.get("Label_volume", "")
    mT1_name = code_vars.get("mT1_volume", "")
    resample_and_save_nifti(
        p0_large,
        grid_native,
        mask.affine,
        mask.header,
        f"{mri_dir}/{label_name}",
        clip=[0, 4],
    )
    resample_and_save_nifti(
        brain_large, grid_native, mask.affine, mask.header, f"{mri_dir}/{mT1_name}"
    )

    # Save remaining data in native space
    if save_p:
        gm_name = code_vars.get("GM_volume", "")
        wm_name = code_vars.get("WM_volume", "")
        resample_and_save_nifti(
            p1_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{gm_name}",
            clip=[0, 1],
        )
        resample_and_save_nifti(
            p2_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{wm_name}",
            clip=[0, 1],
        )
        if save_csf:
            csf_name = code_vars.get("CSF_volume", "")
            resample_and_save_nifti(
                p3_large,
                grid_native,
                mask.affine,
                mask.header,
                f"{mri_dir}/{csf_name}",
                clip=[0, 1],
            )

    # Save lesion maps
    if save_lesions and wmh_large is not None:
        wmh_name = code_vars.get("WMH_volume", "")
        resample_and_save_nifti(
            wmh_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{wmh_name}",
        )

    # Estimate raw volumes
    vol_abs_CGW = [get_volume_native_space(img, affine.values) for img in [p3_large, p1_large, p2_large, wmh_large]]
    tiv_volume = sum(vol_abs_CGW)

    # Compute relative volumes as fractions
    vol_rel_CGW = [v / tiv_volume for v in vol_abs_CGW]
    
    mean_CGW = []
    for label in (1, 2, 3):
        mask_label = np.round(p0_large.get_fdata()) == label
        mean_CGW.append(brain_large.get_fdata().copy()[mask_label].mean())

    # Prepare dictionary
    summary = {
        "vol_abs_CGW": vol_abs_CGW,            # list of floats
        "vol_rel_CGW": vol_rel_CGW,            # list of floats
        "mean_CGW": mean_CGW,                  # list of floats
        "tiv_volume": tiv_volume               # float
    }
    
    # Optional: ensure all values are Python floats (for JSON compatibility)
    #for key in ["vol_abs_CGW", "vol_rel_CGW", "mean_CGW"]:
    #    summary[key] = [float(x) for x in summary[key]]
    #summary["tiv_volume"] = float(summary["tiv_volume"])
    
    # Write to JSON file
    report_name = code_vars.get("Report_file", "")
    with open(f"{report_dir}/{report_name}", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Save non-linear registered data
    if save_hemilabel or save_mwp or save_wp:
        if verbose:
            count = progress_bar(count, end_count, "Warping           ")
        output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        warp_yx = output_reg["warp_yx"]
        warp_xy = output_reg["warp_xy"]
        warp_mse = output_reg["warp_mse"]

        output_atlas = prep.run_atlas_register(
            t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list
        )
        
        # Convert each DataFrame to a list of dicts:
        atlas_json = {
            key.removesuffix('_volumes'): df.to_dict(orient='records')
            for key, df in output_atlas.items()
        }

        # Write to a single JSON file:
        atlas_name = code_vars.get("Atlas_ROI", "")
        with open(f"{label_dir}/{atlas_name}", 'w') as f:
            json.dump(atlas_json, f, indent=2)
    
        if save_mwp:
            gm_name = code_vars_warped_modulated.get("GM_volume", "")
            wm_name = code_vars_warped_modulated.get("WM_volume", "")
            mwp1 = output_reg["mwp1"]
            mwp2 = output_reg["mwp2"]
            nib.save(mwp1, f"{mri_dir}/{gm_name}")
            nib.save(mwp2, f"{mri_dir}/{wm_name}")
            if save_csf:
                csf_name = code_vars_warped_modulated.get("CSF_volume", "")
                mwp3 = output_reg["mwp3"]
                nib.save(mwp3, f"{mri_dir}/{csf_name}")

        if save_wp:
            gm_name = code_vars_warped.get("GM_volume", "")
            wm_name = code_vars_warped.get("WM_volume", "")
            wp1 = output_reg["wp1"]
            wp2 = output_reg["wp2"]
            nib.save(wp1, f"{mri_dir}/{gm_name}")
            nib.save(wp2, f"{mri_dir}/{wm_name}")
            if save_csf:
                csf_name = code_vars_warped.get("CSF_volume", "")
                wp3 = output_reg["wp3"]
                nib.save(wp3, f"{mri_dir}/{csf_name}")

        def_name = code_vars.get("Def_volume", "")
        nib.save(warp_xy, f"{mri_dir}/{def_name}")
        invdef_name = code_vars.get("invDef_volume", "")
        # nib.save(warp_yx, f"{mri_dir}/{invdef_name}")

        # Save hemispheric partition for surface estimation
        if save_hemilabel:
            if verbose:
                count = progress_bar(count, end_count, "Atlas creation     ")
            atlas = get_atlas(
                t1, affine, p0_large.header, p0_large.affine, "ibsr", warp_yx, device
            )
            lh, rh = get_partition(p0_large, atlas)

            if verbose:
                count = progress_bar(count, end_count, "Resampling         ")

            hemileft_name = code_vars_left.get("Hemi_volume", "")
            hemiright_name = code_vars_right.get("Hemi_volume", "")

            resample_and_save_nifti(
                nib.Nifti1Image(lh, p0_large.affine, p0_large.header),
                grid_target_res,
                affine_resamp,
                header_resamp,
                f"{mri_dir}/{hemileft_name}",
                True,
                True,
            )
            resample_and_save_nifti(
                nib.Nifti1Image(rh, p0_large.affine, p0_large.header),
                grid_target_res,
                affine_resamp,
                header_resamp,
                f"{mri_dir}/{hemiright_name}",
                True,
                True,
            )


def run_segment():
    """Run the full segmentation workflow."""

    args = parse_arguments()

    # Input/output parameters
    t1_name = args.input
    mri_dir = args.mri_dir
    report_dir = args.report_dir
    label_dir = args.label_dir
    amap_dir = args.amapdir
    atlas = args.atlas

    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    vessel = args.vessel
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
    device, no_gpu = setup_device()

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

    # Get atlas list (currently restricted to ROI estimation)
    atlas = tuple(x.strip(" '") for x in atlas.split(','))
    atlas_list = tuple([f'{a}_volumes' for a in atlas])

    # Prepare filenames and load input MRI data
    out_name = os.path.basename(os.path.basename(t1_name).replace(".nii", "")).replace(
        ".gz", ""
    )
    
    t1 = nib.load(t1_name)

    # Ensure required model files are available
    prepare_model_files()

    # Preprocess volume and create preprocess object
    t1, prep, ras_affine = preprocess_input(t1, no_gpu, use_amap)

    # Step 1: Skull-stripping
    brain, mask, count = skull_strip(prep, t1, verbose, count, end_count)

    # Step 2: Affine registration
    affine, brain_large, mask_large, affine_loss, count = affine_register(
        prep, brain, mask, verbose, count, end_count
    )

    # Ensure that minimum of brain is not negative (which can happen after sinc-interpolation)
    brain_value = brain_large.get_fdata().copy()
    mask_value = brain_large.get_fdata().copy() > 0.5
    mask_value = binary_closing(mask_value, generate_binary_structure(3, 3), 7)
    min_brain = np.min(brain_value)
    if min_brain < 0:
        brain_value -= min_brain
    brain_value[~mask_value] = 0
    brain_large = nib.Nifti1Image(brain_value, brain_large.affine, brain_large.header)

    # Step 3: Segmentation
    if verbose:
        count = progress_bar(
            count, end_count, "DeepMriPrep segmentation                  "
        )
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg["p0_large"]

    # Due to sinc-interpolation we have to change values close to zero
    p0_value = p0_large.get_fdata()
    if np.min(brain_value) < 0:
        p0_value[~mask_value] = 0
        p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

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

    # Correct bias using label from deepmriprep
    brain_large = correct_bias_field(brain_large, p0_large)

    # Conditional processing based on AMAP or lesion flag
    if use_amap or save_lesions:
        # AMAP segmentation pipeline
        amapdir = args.amapdir

        if verbose:
            count = progress_bar(count, end_count, "Amap segmentation        ")

        p0_large_orig = p0_large
        brain_large, p0_large = run_amap_segmentation(
            amapdir,
            p0_large,
            brain_large,
            mri_dir,
            out_name,
            ext,
            verbose,
            debug,
        )

    else:
        if debug:
            nib.save(brain_large, f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
            nib.save(p0_large, f"{mri_dir}/{out_name}_seg_large.{ext}")

    if use_amap:
        # Load Amap label
        p1_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        p2_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        p3_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")
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
        (
            p1_large,
            p2_large,
            p3_large,
            wmh_value,
            ind_wmh,
        ) = handle_lesions(
            t1,
            affine,
            p0_large,
            p0_large_orig,
            p1_large,
            p2_large,
            p3_large,
            affine_resamp_reordered,
            header_resamp_reordered,
            mri_dir,
            out_name,
            ext,
            use_amap,
            debug,
            device,
        )

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
        p0_value = p0_large.get_fdata().copy()
        np.clip(p0_value, 0, 3)
        p0_value[ind_wmh] += wmh_value[ind_wmh]
        np.clip(p0_value, 0, 4)
        p0_large = nib.Nifti1Image(
            p0_value, affine_resamp_reordered, header_resamp_reordered
        )
        wmh_large = nib.Nifti1Image(
            wmh_value, affine_resamp_reordered, header_resamp_reordered
        )

    save_results(
        prep,
        t1,
        affine,
        p0_large,
        p1_large,
        p2_large,
        p3_large,
        wmh_large,
        mask,
        brain_large,
        grid_native,
        grid_target_res,
        warp_template,
        wj_affine,
        save_p,
        save_rp,
        save_wp,
        save_mwp,
        save_hemilabel,
        save_lesions,
        save_csf,
        verbose,
        count,
        end_count,
        mri_dir,
        label_dir,
        report_dir,
        out_name,
        ext,
        use_bids,
        device,
        affine_resamp,
        header_resamp,
        atlas_list,
    )

    final_cleanup(mri_dir, out_name, ext, use_amap, save_lesions, debug)


if __name__ == "__main__":
    run_segment()
