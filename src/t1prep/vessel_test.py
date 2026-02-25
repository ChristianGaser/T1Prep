"""T1Prep volume pipeline (segmentation) CLI.

This module is primarily invoked by the bash pipeline in scripts/T1Prep.
It can also be run directly for debugging:

    python src/t1prep/segment.py --help

Typical usage (as done by scripts/T1Prep) provides output directories and
optionally atlas names:

    python src/t1prep/segment.py \
        --input sub-01_T1w.nii.gz --mri_dir out/mri --report_dir out/report --label_dir out/label \
        --bin_dir /path/to/CAT/binaries --atlas "'neuromorphometrics','suit'" --bids
"""

import os
import sys
import platform

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import argparse
import warnings
import math
import shutil
import zipfile
import urllib.request
import fill_voids
import json
import random
import time
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from deepbet.utils import reoriented_nifti
from deepmriprep.segment import BrainSegmentation, scale_intensity
from deepmriprep.preprocess import Preprocess
from deepmriprep.utils import DATA_PATH, nifti_to_tensor, nifti_volume
from deepmriprep.atlas import get_volumes, shape_from_to
from torchreg.utils import INTERP_KWARGS
from pathlib import Path
from scipy.ndimage import (
    binary_closing,
    generate_binary_structure,
)
from utils import (
    smart_round,
    progress_bar,
    remove_file,
    resample_and_save_nifti,
    get_resampled_header,
    align_brain,
    get_filenames,
    get_volume_native_space,
    DATA_PATH_T1PREP,
    TEMPLATE_PATH_T1PREP,
)
from _segment_utils import (
    correct_bias_field,
    unsmooth_kernel,
    get_atlas,
    get_partition,
    cleanup_vessels,
    get_regions_mask,
    correct_label_map,
    apply_LAS,
    handle_lesions,
    normalize_to_sum1,
)

ROOT_PATH = Path(__file__).resolve().parents[2]
TMP_PATH = ROOT_PATH / "tmp_models/"
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
MODEL_ZIP_URL = "https://github.com/ChristianGaser/T1Prep/releases/download/v0.2.0-beta/T1Prep_Models.zip"
MODEL_ZIP_LOCAL = ROOT_PATH / "T1Prep_Models.zip"

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation pipeline."""

    parser = argparse.ArgumentParser(
        description="T1Prep volume pipeline: skull-stripping, segmentation and atlas ROI export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=str, help="Input NIfTI image")
    parser.add_argument("--mri_dir", required=True, type=str, help="Output folder for MRI volumes")
    parser.add_argument(
        "--atlas",
        type=str,
        default="",
        help=(
            "Atlases for ROI estimation (comma-separated). Examples: "
            "\"neuromorphometrics,suit\" or \"'neuromorphometrics','suit'\". "
            "Empty disables ROI export."
        ),
    )
    parser.add_argument(
        "--bin_dir",
        type=str,
        default="",
        help="Folder with CAT/AMAP binaries (required for --amap or --lesions)",
    )
    parser.add_argument(
        "--surf",
        action="store_true",
        help="Save partitioned segmentation maps for surface estimation.",
    )
    parser.add_argument(
        "--csf", action="store_true", help="Save also CSF segmentations."
    )
    parser.add_argument(
        "--mwp",
        action="store_true",
        help="Save modulated and warped segmentations.",
    )
    parser.add_argument(
        "--wp", action="store_true", help="Save warped segmentations."
    )
    parser.add_argument(
        "--p", action="store_true", help="Save native segmentations."
    )
    parser.add_argument(
        "--rp",
        action="store_true",
        help="Save affine registered segmentations.",
    )
    parser.add_argument(
        "--lesions", action="store_true", help="Save also WMH lesion maps (if available)."
    )
    parser.add_argument(
        "--bids", action="store_true", help="Use BIDS-like naming convention."
    )
    parser.add_argument(
        "--gz", action="store_true", help="Save compressed NIfTI outputs (.nii.gz)."
    )
    parser.add_argument(
        "--amap", action="store_true", help="Use AMAP segmentation (requires --bin_dir)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress output."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Do not delete temporary files."
    )
    parser.add_argument(
        "--vessel",
        type=float,
        default=0.4,
        help="Initial threshold to isolate WM for vessel removal",
    )

    skullstrip_group = parser.add_mutually_exclusive_group()
    skullstrip_group.add_argument(
        "--skullstrip-only",
        action="store_true",
        help=(
            "Only run skull stripping and save outputs to --mri_dir, then exit. "
            "Writes a skull-stripped volume and a brain mask."
        ),
    )
    skullstrip_group.add_argument(
        "--skip-skullstrip",
        action="store_true",
        help="Skip skull stripping (assume input is already skull-stripped).",
    )
    skullstrip_group.add_argument(
        "--no-skullstrip",
        action="store_true",
        dest="skip_skullstrip",
        help="Alias for --skip-skullstrip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generators",
    )
    return parser.parse_args()


def setup_device() -> tuple[torch.device, bool]:
    """Return the torch device and ``no_gpu`` flag."""

    if torch.cuda.is_available():
        return torch.device("cuda"), False
    if torch.backends.mps.is_available() and False:  # not yet fully supported
        return torch.device("mps"), False
    return torch.device("cpu"), True

class CustomPreprocess(Preprocess):
    """
    Custom class to override Preprocess
    Use linear interpolation for p0, which prevents negative values due to
    sinc-interpolation
    """

    def run_atlas_register(
        self, t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list, wj_affine
    ):
        voxel_vol = np.prod(p1_large.affine[np.diag_indices(3)])
        p1_large, p2_large, p3_large = [
            nifti_to_tensor(p).to(self.device) for p in [p1_large, p2_large, p3_large]
        ]
        inv_affine = torch.linalg.inv(
            torch.from_numpy(affine.values).float().to(self.device)
        )
        grid = F.affine_grid(
            inv_affine[None, :3],
            [1, 3, *t1.shape[:3]],
            align_corners=INTERP_KWARGS["align_corners"],
        )
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(self.device)
        atlases, warps = {}, {}

        atl_list = [
            "_".join(a.split("_")[:-1]) if a.endswith(("_affine", "_volumes")) else a
            for a in atlas_list
        ]
        for atl in atl_list:
            # Use absolute path for external atlas if exists
            if os.path.isabs(atl) and os.path.exists(atl):
                atlas_path = atl
                base_atl = os.path.splitext(os.path.basename(atl))[0]
            else:
                atlas_path = f"{TEMPLATE_PATH_T1PREP}/{atl}.nii.gz"
                base_atl = atl
            atlas = nib.as_closest_canonical(nib.load(atlas_path))
            header = atlas.header
            shape = tuple(shape_from_to(atlas, warp_yx))
            if shape not in warps:
                scaled_yx = F.interpolate(
                    yx.permute(0, 4, 1, 2, 3),
                    shape,
                    mode="trilinear",
                    align_corners=False,
                )
                warps[shape] = scaled_yx.permute(0, 2, 3, 4, 1)
            atlas = self.atlas_register(affine, warps[shape], atlas, t1.shape)
            if f"{atl}_affine" in atlas_list:
                atlases[f"{atl}_affine"] = atlas
            atlas_tensor = nifti_to_tensor(atlas).to(self.device)
            if f"{atl}_volumes" in atlas_list:
                # Check for csf file and create dummy ROIs if not found
                csv_path = (
                    f"{TEMPLATE_PATH_T1PREP}/{base_atl}.csv"
                    if not os.path.isabs(atl)
                    else os.path.splitext(atl)[0] + ".csv"
                )
                if os.path.exists(csv_path):
                    rois = pd.read_csv(csv_path, sep=";")[["ROIid", "ROIname"]]
                else:
                    # Fallback: Dummy-ROI-list with increasing IDs
                    labels = torch.unique(atlas_tensor).cpu().numpy()
                    labels = labels[labels > 0]
                    rois = pd.DataFrame(
                        {"ROIid": labels, "ROIname": [f"Region_{i}" for i in labels]}
                    )
                volumes = voxel_vol * get_volumes(
                    atlas_tensor, p1_large, p2_large, p3_large
                )
                volumes *= wj_affine[0] / 1000
                # Smart rounding
                volumes = np.vectorize(smart_round)(volumes)
                volumes = pd.DataFrame(
                    volumes, columns=["gmv_cm3", "wmv_cm3", "csfv_cm3", "region_cm3"]
                )
                atlases[f"{atl}_volumes"] = pd.concat([rois, volumes], axis=1)
            if atl in atlas_list:
                sample_kwargs = {
                    "mode": "nearest",
                    "align_corners": INTERP_KWARGS["align_corners"],
                }
                sampled_atlas = F.grid_sample(
                    atlas_tensor[None, None], grid, **sample_kwargs
                )[0, 0]
                atlases[atl] = reoriented_nifti(
                    sampled_atlas.cpu().numpy(), t1.affine, header
                )
        return atlases


def preprocess_input(t1: nib.Nifti1Image, no_gpu: bool, use_amap: bool):
    """Align the input volume and create the preprocessing object."""

    vol = t1.get_fdata().copy()
    vol = np.squeeze(vol)
    vol, affine_resamp, header_resamp, ras_affine = align_brain(
        vol, t1.affine, t1.header, np.eye(4), do_flip=0
    )
    t1 = nib.Nifti1Image(vol, affine_resamp, header_resamp)
    prep = CustomPreprocess(no_gpu)

    # This is a bit faster since for initial segmentation the sinc-interpolation
    # of the segmentations does not help and is slower
    # Furthermore, skip self.run_patch_models(x, p0) which takes a lot of time
    # and is not needed for Amap segmentation.
    if use_amap:
        prep.brain_segment = CustomBrainSegmentation(no_gpu=no_gpu)

    return t1, prep, ras_affine


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

def run_segment():
    """Run the full segmentation workflow."""

    args = parse_arguments()

    # Input/output parameters
    t1_name = args.input
    mri_dir = args.mri_dir
    atlas = args.atlas

    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    vessel = args.vessel
    verbose = args.verbose
    debug = args.debug

    # Check for GPU support
    device, no_gpu = setup_device()

    t1 = nib.load(t1_name)
    ext = "nii"
    count = 1
    end_count = 5
    # Preprocess volume and create preprocess object
    t1, prep, ras_affine = preprocess_input(t1, no_gpu, use_amap)

    brain, mask, count = skull_strip(prep, t1, verbose, count, end_count)
    # Step 3: Affine registration
    affine, brain_large, mask_large, affine_loss, count = affine_register(
        prep, brain, mask, verbose, 1, 2
    )
    print(affine)
    
    # Get atlas list (currently restricted to ROI estimation)
    atlas = tuple(x.strip(" '") for x in atlas.split(","))
    # Build atlas_list. Set atlas_list to None, if empty
    atlas_list = (
        tuple(f"{a}_volumes" for a in atlas) if any(atlas) and atlas != ("",) else None
    )

    # Prepare filenames and load input MRI data
    out_name = os.path.basename(os.path.basename(t1_name).replace(".nii", "")).replace(
        ".gz", ""
    )

    # Cleanup (e.g. remove vessels outside cerebellum, but are surrounded by CSF) 
    # to refine segmentation
    if vessel != 0:
        p1_large = nib.load(f"{mri_dir}/{out_name}_p1_large_pre_vessel_cleanup_tmp.{ext}")
        p1_large = nib.load(f"{mri_dir}/{out_name}_p1_large_pre_vessel_cleanup_tmp.{ext}")
        p2_large = nib.load(f"{mri_dir}/{out_name}_p2_large_pre_vessel_cleanup_tmp.{ext}")
        p3_large = nib.load(f"{mri_dir}/{out_name}_p3_large_pre_vessel_cleanup_tmp.{ext}")
        p0_value = 2*p1_large.get_fdata().copy() + 3*p2_large.get_fdata().copy() + p3_large.get_fdata().copy()
        p0_large = nib.Nifti1Image(p0_value, p1_large.affine, p1_large.header)
        
        atlas = get_atlas(
            t1,
            affine,
            p0_large.header,
            p0_large.affine,
            "cat_bloodvessels",
            None,
            device,
            is_label_atlas=False,
        )
        vessel_TPM = atlas.get_fdata().copy()

        atlas = get_atlas(
            t1,
            affine,
            p0_large.header,
            p0_large.affine,
            "csf_TPM",
            None,
            device,
            is_label_atlas=False,
        )
        csf_TPM = atlas.get_fdata().copy()

        # Only modify label image for surface processing
        if vessel > 0:
            p0_large, p1_cleanup, p2_cleanup, p3_cleanup = cleanup_vessels(
                p1_large, p2_large, p3_large, vessel, None, csf_TPM,
                vessel_TPM, brain_large,
                csf_shrink_iters=2, sulcal_min_depth_mm=0.6,
                sulcal_full_depth_mm=1.2, sulcal_gate=0.3,
                wm_close_iters=0, wm_min_island_voxels=10000
            )
            if debug:
                nib.save(p1_cleanup, f"{mri_dir}/{out_name}_p1_large_post_vessel_cleanup_tmp.{ext}")
                nib.save(p2_cleanup, f"{mri_dir}/{out_name}_p2_large_post_vessel_cleanup_tmp.{ext}")
                nib.save(p3_cleanup, f"{mri_dir}/{out_name}_p3_large_post_vessel_cleanup_tmp.{ext}")
        # This is a hidden feature for testing since vessel removal for VBM was 
        # not ending in better accuracy
        else:
            p0_large, p1_large, p2_large, p3_large = cleanup_vessels(
                p1_large, p2_large, p3_large, np.abs(vessel), None, None,
                vessel_TPM, brain_large
            )
            if debug:
                nib.save(p1_large, f"{mri_dir}/{out_name}_p1_large_post_vessel_cleanup_tmp.{ext}")
                nib.save(p2_large, f"{mri_dir}/{out_name}_p2_large_post_vessel_cleanup_tmp.{ext}")
                nib.save(p3_large, f"{mri_dir}/{out_name}_p3_large_post_vessel_cleanup_tmp.{ext}")
    else:
        gm = p1_large.get_fdata()
        wm = p2_large.get_fdata()
        csf = p3_large.get_fdata()
        gm, wm, csf = normalize_to_sum1(gm, wm, csf)
        tmp = csf + 2 * gm + 3 * wm
        p0_large = nib.Nifti1Image(tmp, p0_large.affine, p0_large.header)

    # We have to apply the initial mask again to the label
    p0_value = p0_large.get_fdata().copy()
    p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)
    nib.save(p0_large, f"{mri_dir}/{out_name}_p0_large_post_vessel_cleanup_tmp.{ext}")

if __name__ == "__main__":
    run_segment()
