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
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"        # CPU Fallback for MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # More GPU memory

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
import subprocess
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
from spline_resize import resize
from report import write_t1prep_report
from qa import estimate_qa
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    generate_binary_structure,
)
from utils import (
    smart_round,
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
    compute_euler_number,
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

def shell_progress(count, end_count, label, failed=0):
    script = ROOT_PATH / "scripts" / "progress_bar_multi.sh"
    subprocess.run(
        [str(script), "1", "", str(count), str(end_count), label, "40", str(failed)],
        check=False)
    return count + 1

class CustomBrainSegmentation(BrainSegmentation):
    """
    Custom class to override BrainSegmentation
    Furthermore use run_model function with linear interpolation for p0, which
    prevents negative values due to sinc-interpolation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # deepmriprep and deepbet can disagree on device selection on macOS,
        # which may leave scripted model tensors on CPU while inputs are on MPS.
        self.inference_device = (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else self.device
        )

    def __call__(self, x, mask):
        x = x.to(device=self.inference_device)
        x = x[:, :, 1:-2, 15:-12, :-3]
        x = scale_intensity(x)
        p0 = self.run_model(x)
        p0 = self.run_patch_models(x, p0)
        if self.fill_holes:
            mask_np = p0[0, 0].detach().cpu().numpy() > 0.9
            mask_filled = fill_voids.fill(mask_np)
            filled = (mask_np == 0) & (mask_filled == 1)
            if np.any(filled):
                filled_t = torch.from_numpy(filled).to(p0.device)
                p0[0, 0][filled_t] = 1.0
        # Keep return device aligned with caller tensors used in downstream indexing.
        return F.pad(p0, (0, 3, 15, 12, 1, 2)).to(mask.device)

    def run_patch_models(self, x, p0):
        x = x.to(device=self.inference_device)
        p0 = p0.to(device=self.inference_device)
        patch_p0 = torch.zeros(
            x.shape, device=self.inference_device
        )
        for i, (patch, weight) in enumerate(zip(self.patch_slices, self.patch_weights)):
            patch_inp = torch.cat([x[patch], p0[patch]], dim=1)
            patch_inp = patch_inp.flip(2) if i >= 18 else patch_inp
            with torch.no_grad():
                p0_patch = self.patch_models[i % 18](patch_inp)
            p0_patch = p0_patch.flip(2) if i >= 18 else p0_patch
            patch_p0[patch] += p0_patch * weight.to(self.inference_device)
        return patch_p0

    def run_model(self, x, scale_factor=1.5):
        x = x.to(device=self.inference_device)
        
        # Use linear interpolation, since sinc-interpolation caused issues with
        # negative values in label image and resulted in less accurate segmentations
        with torch.no_grad():
            p0 = self.model(
                F.interpolate(x, scale_factor=1 / scale_factor, **INTERP_KWARGS))
        return F.interpolate(p0, scale_factor=scale_factor, **INTERP_KWARGS)


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


def all_models_present():
    """Return ``True`` if all required model files are available."""

    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation pipeline."""

    parser = argparse.ArgumentParser(
        description="T1Prep volume pipeline: skull-stripping, segmentation and atlas ROI export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=str, help="Input NIfTI image")
    parser.add_argument("--mri_dir", required=True, type=str, help="Output folder for MRI volumes")
    parser.add_argument("--report_dir", required=True, type=str, help="Output folder for reports/logs")
    parser.add_argument("--label_dir", required=True, type=str, help="Output folder for labels/aux outputs")
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
        required=True,
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
        default=1.0,
        help="Use vessel removal",
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
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="End count for progress bar",
    )
    return parser.parse_args()


def setup_device() -> tuple[torch.device, bool]:
    """Return the torch device and ``no_gpu`` flag."""

    if torch.cuda.is_available():
        return torch.device("cuda"), False
    elif torch.backends.mps.is_available():
        return torch.device("mps"), False
    else:
        return torch.device("cpu"), True


def prepare_model_files() -> None:
    """Ensure required model files are present, downloading if needed."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not all_models_present():
        print("One or more model files are missing. Copying models...")
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
    prep = CustomPreprocess(no_gpu=no_gpu)

    # This is a bit faster since for initial segmentation the sinc-interpolation
    # of the segmentations does not help and is slower
    # Furthermore, CustomBrainSegmentation supports mps device
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
        count = shell_progress(count, end_count, 
            "Skull-stripping              ")
    output = prep.run_bet(t1)
    return output["brain"], output["mask"], count


def mask_from_skullstripped(brain: nib.Nifti1Image) -> nib.Nifti1Image:
    """Create a brain mask from a (already) skull-stripped volume."""

    data = np.asarray(brain.get_fdata(), dtype=np.float32)
    mask = np.isfinite(data) & (data > 0)
    if not bool(mask.any()):
        mask = np.isfinite(data) & (data != 0)
    mask = binary_closing(mask, generate_binary_structure(3, 3), 2)
    try:
        mask = fill_voids.fill(mask)
    except Exception:
        pass
    return nib.Nifti1Image(mask.astype(np.uint8), brain.affine, brain.header)


def save_skullstrip_only_outputs(
    brain: nib.Nifti1Image,
    use_bids: bool,
    mri_dir: str,
    out_name: str,
    ext: str,
) -> None:
    """Save skull-stripped image into the output folder."""

    code_vars = get_filenames(use_bids, out_name, "", "", "", ext)
    skullstripped_name = code_vars.get("skullstripped_volume", "")
    os.makedirs(mri_dir, exist_ok=True)
    brain_path = f"{mri_dir}/{skullstripped_name}"
    nib.save(brain, brain_path)


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
        count = shell_progress(count, end_count, 
            "Affine registration          ")
    output = prep.run_affine_register(brain, mask)
    return (
        output["affine"],
        output["brain_large"],
        output["mask_large"],
        output["affine_loss"],
        count,
    )


def run_amap_segmentation(
    bin_dir: str,
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
        os.path.join(bin_dir, "CAT_VolSanlm")
        + " "
        + f"{mri_dir}/{out_name}_brain_large_tmp.{ext}"
        + " "
        + f"{mri_dir}/{out_name}_brain_large.{ext}"
    )
    os.system(cmd)

    # Call AMAP and write tissue and label maps
    cmd = (
        os.path.join(bin_dir, "CAT_VolAmap")
        + f" -nowrite-corr -bias-fwhm 0 -cleanup 1 -mrf 0 "
        + "-h-ornlm 0.0 -multistep -write-seg 1 1 1 -sub 64 -label "
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

    if (use_amap or save_lesions) and not debug:
        remove_file(f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_seg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_seg_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")

    if not use_amap and not debug:
        remove_file(f"{mri_dir}/{out_name}_brain_large.{ext}")


def save_results(
    prep: CustomPreprocess,
    t1: nib.Nifti1Image,
    affine,
    p0_large: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wmh_large: nib.Nifti1Image,
    discrepancy_large: nib.Nifti1Image,
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
    if save_hemilabel or save_mwp or save_wp or save_rp or (atlas_list is not None):
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

    # Save lesion and discrepancy maps
    if save_lesions and wmh_large is not None:
        wmh_name = code_vars.get("WMH_volume", "")
        resample_and_save_nifti(
            wmh_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{wmh_name}",
        )
        discrepance_name = code_vars.get("Discrepance_volume", "")
        resample_and_save_nifti(
            discrepancy_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{discrepance_name}",
        )

    # Estimate raw volumes
    vol_gm = get_volume_native_space(p1_large, wj_affine[0])  # GM    (p1)
    vol_wm = get_volume_native_space(p2_large, wj_affine[0])  # WM    (p2)
    vol_csf = get_volume_native_space(p3_large, wj_affine[0])  # CSF   (p3)
    if save_lesions and wmh_large is not None:
        vol_wmh = get_volume_native_space(wmh_large, wj_affine[0])  # WMHs  (lesions)
    else:
        vol_wmh = 0

    # treat WMHs as part of WM
    vol_wm_incl = vol_wm + vol_wmh

    # Absolute volumes
    # Order: CSF-GM-WM(incl.WMH)-WMH
    vol_CGW = [vol_csf, vol_gm, vol_wm_incl]

    # TIV contains CSF + GM + WM (already incl. WMH!)
    vol_tiv = vol_csf + vol_gm + vol_wm_incl

    # Compute relative volumes as fractions
    # Fractions w. r. t. TIV
    vol_rel_CGW = [v / vol_tiv for v in vol_CGW]  # CSF, GM, WM+WMH

    # Lesion load: WMH fraction w. r. t. WM (+WMH)
    wmh_rel_to_wm = vol_wmh / vol_wm_incl

    # Mean intensities per tissue class
    mean_CGW = []
    for label in (1, 2, 3):  # p0_large: 1=CSF, 2=GM, 3=WM
        mask_label = np.round(p0_large.get_fdata()) == label
        mean_CGW.append(brain_large.get_fdata()[mask_label].mean())

    # Estimate image quality measures at native acquisition resolution.
    # CAT12 reads the ORIGINAL file (varargin{2}) and normalises to WM≈1 via
    # cat_vol_approx (lines 293-298 of cat_vol_qa201901x.m; the branch that
    # would use the bias-corrected image is permanently disabled with "if 0").
    # Resample p0 to native space; pass t1 (original) as the intensity image.
    vx_vol_orig = np.array(t1.header.get_zooms()[:3], dtype=np.float64)
    p0_native = F.grid_sample(
        nifti_to_tensor(p0_large)[None, None],
        grid_native,
        align_corners=INTERP_KWARGS["align_corners"],
    )[0, 0].numpy()
    # t1 has negative-diagonal affine after align_brain (do_flip=0); grid_native
    # produces p0_native in as_closest_canonical space (positive/RAS axes).
    # Must canonicalise t1 the same way so tissue masks align with intensities.
    t1_canonical = nib.as_closest_canonical(t1).get_fdata().astype(np.float32)
    qa_result = estimate_qa(
        p0_native,
        t1_canonical,
        vx_vol_orig,
        vx_vol_orig,
    )

    # Prepare dictionary
    summary = {
        "vol_CGW": {
            "value": [smart_round(x) for x in vol_CGW],
            "desc": "Tissue volumes in mL (CSF, GM, WM incl. WMH)",
        },
        "vol_rel_CGW": {
            "value": [smart_round(x) for x in vol_rel_CGW],
            "desc": "Relative tissue volumes ([CSF, GM, WM incl. WMH]/TIV)",
        },
        "mean_CGW": {
            "value": [smart_round(x) for x in mean_CGW],
            "desc": "Mean intensity per tissue (p0 labels 1-3)",
        },
        "vol_tiv": {
            "value": smart_round(vol_tiv),
            "desc": "Total intracranial volume in mL (CSF+GM+WM incl. WMH)",
        },
    }

    if save_lesions and wmh_large is not None:
        summary |= {
            "vol_WMH": {
                "value": vol_wmh,
                "desc": "WMH",
            },
            "WMH_rel_WM": {
                "value": smart_round(wmh_rel_to_wm),
                "desc": "WMH load relative to WM incl. WMH",
            },
        }

    # Write to JSON file
    summary |= qa_result
    report_name = code_vars.get("Report_file", "")
    with open(f"{report_dir}/{report_name}", "w") as f:
        json.dump(summary, f, indent=2)

    # Save non-linear registered data
    if save_hemilabel or save_mwp or save_wp or (atlas_list is not None):
        if verbose:
            count = shell_progress(count, end_count, 
                "Warping                      ")
        output_reg = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
        warp_yx = output_reg["warp_yx"]
        warp_xy = output_reg["warp_xy"]
        warp_mse = output_reg["warp_mse"]

        if atlas_list is not None:
            output_atlas = prep.run_atlas_register(
                t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list, wj_affine
            )

            # Convert each DataFrame to a list of dicts:
            atlas_json = {
                key.removesuffix("_volumes"): df.to_dict(orient="records")
                for key, df in output_atlas.items()
            }

            # Write to a single JSON file:
            atlas_name = code_vars.get("Atlas_ROI", "")
            with open(f"{label_dir}/{atlas_name}", "w") as f:
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
                # mwp3 = output_reg["mwp3"]
                # nib.save(mwp3, f"{mri_dir}/{csf_name}")

        if save_wp:
            gm_name = code_vars_warped.get("GM_volume", "")
            wm_name = code_vars_warped.get("WM_volume", "")
            wp1 = output_reg["wp1"]
            wp2 = output_reg["wp2"]
            nib.save(wp1, f"{mri_dir}/{gm_name}")
            nib.save(wp2, f"{mri_dir}/{wm_name}")
            if save_csf:
                csf_name = code_vars_warped.get("CSF_volume", "")
                # wp3 = output_reg["wp3"]
                # nib.save(wp3, f"{mri_dir}/{csf_name}")

        def_name = code_vars.get("Def_volume", "")
        nib.save(warp_xy, f"{mri_dir}/{def_name}")
        invdef_name = code_vars.get("invDef_volume", "")
        # nib.save(warp_yx, f"{mri_dir}/{invdef_name}")

        # Save hemispheric partition for surface estimation
        if save_hemilabel:
            if verbose:
                count = shell_progress(count, end_count, 
                    "Atlas creation               ")
            atlas = get_atlas(
                t1,
                affine,
                p0_large.header,
                p0_large.affine,
                "ibsr",
                warp_yx,
                device,
                is_label_atlas=True,
            )
            lh, rh = get_partition(p0_large, atlas)

            # Compute Euler numbers at GM/WM boundary for QA
            euler_lh = compute_euler_number(lh, threshold=2.5)
            euler_rh = compute_euler_number(rh, threshold=2.5)
            abs_euler = abs(euler_lh - 2) + abs(euler_rh - 2)

            # Append to existing JSON report
            report_name = code_vars.get("Report_file", "")
            report_path = f"{report_dir}/{report_name}"
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_data = json.load(f)
            else:
                report_data = {}
            qa = report_data.setdefault("qualitymeasures", {})
            qa["euler_lh"] = {
                "value": euler_lh,
                "desc": (
                    "Euler number of left hemisphere "
                    "(ideal = 2; values closer to 2 indicate less topological defects)"
                ),
            }
            qa["euler_rh"] = {
                "value": euler_rh,
                "desc": (
                    "Euler number of right hemisphere "
                    "(ideal = 2; values closer to 2 indicate less topological defects)"
                ),
            }
            qa["abs_euler"] = {
                "value": abs_euler,
                "desc": (
                    "Absolute Euler number for both hemispheres "
                    "(absolute difference to ideal Euler number of 2; ideal = 0;larger values indicate more topological defects)"
                ),
            }
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)

            if verbose:
                count = shell_progress(count, end_count, 
                    "Resampling                   ")

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    # Input/output parameters
    t1_name = args.input
    mri_dir = args.mri_dir
    report_dir = args.report_dir
    label_dir = args.label_dir
    bin_dir = args.bin_dir
    atlas = args.atlas

    # Processing options
    use_amap = args.amap
    use_bids = args.bids
    vessel = args.vessel
    verbose = args.verbose
    debug = args.debug
    skullstrip_only = args.skullstrip_only
    skip_skullstrip = args.skip_skullstrip

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
    end_count = args.count

    if save_gz:
        ext = "nii.gz"
    else:
        ext = "nii"

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

    # Track running time
    start = time.perf_counter()
    t1 = nib.load(t1_name)

    # Ensure required model files are available
    prepare_model_files()

    # Preprocess volume and create preprocess object
    t1, prep, ras_affine = preprocess_input(t1, no_gpu, use_amap)

    # Step 1: Skull-stripping (or skip)
    if skip_skullstrip:
        if verbose:
            count = shell_progress(count, end_count, 
                "Skull-stripping              ")
        brain = t1
        mask = mask_from_skullstripped(brain)
    else:
        brain, mask, count = skull_strip(prep, t1, verbose, count, end_count)

    if skullstrip_only:
        save_skullstrip_only_outputs(brain, use_bids, mri_dir, out_name, ext)
        return

    # Step 2: Initial bias-correction that is benefitial for strong signal
    # inhomogeneities (i.e. 7T data)
    brain = correct_bias_field(brain)

    # Step 3: Affine registration
    affine, brain_large, mask_large, affine_loss, count = affine_register(
        prep, brain, mask, verbose, count, end_count
    )

    inv_affine = torch.linalg.inv(torch.from_numpy(affine.values).float())

    # Ensure that minimum of brain is not negative (which can happen after sinc-interpolation)
    brain_value = brain_large.get_fdata().copy()
    mask_value = binary_closing(brain_value > 0.0, generate_binary_structure(3, 3), 7)
    min_brain = np.min(brain_value)
    if min_brain < 0:
        brain_value -= min_brain
    brain_value[~mask_value] = 0
    brain_large = nib.Nifti1Image(brain_value, brain_large.affine, brain_large.header)

    # Call SANLM filter for non-Amap approach and rename output to original name
    if not use_amap:
        nib.save(brain_large, f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
        cmd = (
            os.path.join(bin_dir, "CAT_VolSanlm")
            + " "
            + f"{mri_dir}/{out_name}_brain_large_tmp.{ext}"
            + " "
            + f"{mri_dir}/{out_name}_brain_large.{ext}"
        )
        os.system(cmd)
        brain_large = nib.load(f"{mri_dir}/{out_name}_brain_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")    

    # Step 4: Segmentation
    if verbose:
        count = shell_progress(
            count, end_count, "DeepMriPrep segmentation     ")
    output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    p0_large = output_seg["p0_large"]

    # Due to sinc-interpolation we have to change values below zero
    p0_value = p0_large.get_fdata()
    if np.min(p0_value) < 0:
        p0_value[p0_value < 0] = 0
        p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

    mask_large_value = p0_large.get_fdata() > 1E-3
        
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

    # Correct bias using label from deepmriprep
    brain_large = correct_bias_field(brain_large, p0_large)

    p0_large_orig = p0_large

    if use_amap:
        if verbose:
            count = shell_progress(count, end_count, 
                "Amap segmentation            ")
        brain_large, p0_large = run_amap_segmentation(
            bin_dir,
            p0_large,
            brain_large,
            mri_dir,
            out_name,
            ext,
            verbose,
            debug,
        )
    else:
        brain_large = apply_LAS(brain_large, p0_large)

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
            count = shell_progress(
                count, end_count, 
                    "Fine DeepMriPrep segmentation"
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
            discrepancy_large,
            wmh_value,
            ind_wmh,
        ) = handle_lesions(
            t1,
            affine,
            brain_large,
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
    else:
        discrepancy_large = None

    warp_template = nib.load(f"{DATA_PATH}/templates/Template_4_GS.nii.gz")
    wj_affine = (
        np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
    )

    wj_affine = pd.Series([wj_affine])

    # Cleanup (e.g. remove vessels outside cerebellum, but are surrounded by CSF) 
    # to refine segmentation
    if vessel > 0:
        atlas = get_atlas(
            t1,
            affine,
            p0_large.header,
            p0_large.affine,
            "neuromorphometrics",
            None,
            device,
            is_label_atlas=True,
        )
        
        # Exclude cerebellum + subcortical areas + hippocampus + amygdala
        excl_regions = get_regions_mask(atlas, "Neuromorphometrics",
            [
                "Left Cerebellum White Matter",
                "Right Cerebellum White Matter",
                "Left Cerebellum Exterior",
                "Right Cerebellum Exterior",
                "Cerebellar Vermal Lobules I-V",
                "Cerebellar Vermal Lobules VI-VII",
                "Cerebellar Vermal Lobules VIII-X",
                "4th Ventricle",
                "Left Amygdala",
                "Right Amygdala",
                "Left Caudate",
                "Right Caudate",
                "Left Hippocampus",
                "Right Hippocampus",
                "Left Pallidum",
                "Right Pallidum",
                "Left Putamen",
                "Right Putamen",
                "Left Thalamus Proper",
                "Right Thalamus Proper",
                "Left Ventral DC",
                "Right Ventral DC",
            ],
        )
        excl_regions = binary_dilation(excl_regions, generate_binary_structure(3, 3), 2)

        excl_ventricle = get_regions_mask(atlas, "Neuromorphometrics",
            [
                "Left Inf Lat Vent",
                "Right Inf Lat Vent",
                "Left Lateral Ventricle",
                "Right Lateral Ventricle",
                "Brain Stem",
            ],
        )
        excl_ventricle = binary_dilation(excl_ventricle, generate_binary_structure(3, 3), 5)
        excl_regions |= excl_ventricle

        p0_value_original = p0_large.get_fdata().copy()
        p0_large, p1_large, p2_large, p3_large = cleanup_vessels(
            p1_large, p2_large, p3_large, bin_dir, mri_dir, out_name, ext, 
            debug, excl_regions)
    else:
        gm = p1_large.get_fdata()
        wm = p2_large.get_fdata()
        csf = p3_large.get_fdata()
        gm, wm, csf = normalize_to_sum1(gm, wm, csf)
        tmp = csf + 2 * gm + 3 * wm
        p0_large = nib.Nifti1Image(tmp, p0_large.affine, p0_large.header)

    # We have to apply the initial mask again to the label
    p0_value = p0_large.get_fdata().copy()
    p0_value[mask_large_value == 0] = 0
    p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

    if debug and (vessel > 0):
        p0_value = p0_value_original - p0_value
        nib.save(nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header), 
            f"{mri_dir}/{out_name}_vessels_large.{ext}")
        
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
    else:
        wmh_large = None

    save_results(
        prep,
        t1,
        affine,
        p0_large,
        p1_large,
        p2_large,
        p3_large,
        wmh_large,
        discrepancy_large,
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

    # Write to log file
    end = time.perf_counter()
    text = f"Execution time of volume pipeline: {end - start:.1f}s.\n"
    code_vars = get_filenames(use_bids, out_name, "", "", "", ext)
    log_name = code_vars.get("Log_file", "")
    with open(f"{report_dir}/{log_name}", "a") as f:
        f.write(text)

    write_t1prep_report(report_dir, out_name, use_bids, t1_name)


if __name__ == "__main__":
    run_segment()
