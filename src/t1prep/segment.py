"""T1Prep volume pipeline (segmentation) CLI.

This module is primarily invoked by the bash pipeline in scripts/T1Prep.
It can also be run directly for debugging:

    python src/t1prep/segment.py --help

Typical usage (as done by scripts/T1Prep) provides output directories and
optionally atlas names:

    python src/t1prep/segment.py \
        --input sub-01_T1w.nii.gz --mri-dir out/mri --report-dir out/report --label-dir out/label \
        --atlas "'Neuromorphometrics','suit'" --bids
"""

import os
import sys
import warnings

# Suppress urllib3/LibreSSL warning before any transitive imports trigger it
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

if sys.platform == "darwin":
    # setdefault, not assignment: both are worth tuning per machine, and a 0.0
    # watermark (no ceiling at all) lets MPS allocate until the system swaps.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # CPU fallback for MPS
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # more GPU memory

import cat_surf
import torch
import argparse
import warnings
import shutil
import fill_voids
import json
import random
import time
import subprocess
import sentry_sdk
from dataclasses import dataclass
from typing import NamedTuple, Optional
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deep learning and image processing utilities
from deepbet.utils import reoriented_nifti
from deepmriprep.segment import BrainSegmentation
from deepmriprep.preprocess import Preprocess
from deepmriprep.utils import DATA_PATH, nifti_to_tensor, nifti_volume
from deepmriprep.atlas import get_volumes, shape_from_to
from torchreg.utils import INTERP_KWARGS
from pathlib import Path
from .report import write_t1prep_report
from .qa import estimate_qa
from scipy.ndimage import (
    binary_closing,
    generate_binary_structure,
)
from .transforms import (
    save_affine_itk_txt,
    save_deformation_h5,
    save_deformation_spm,
)
from .bids_derivatives import (
    write_dataset_description,
    write_sidecar,
)
from .utils import (
    smart_round,
    remove_file,
    resample_and_save_nifti,
    get_resampled_header,
    align_brain,
    get_filenames,
    get_volume_native_space,
    progress_bar,
    TEMPLATE_PATH_T1PREP,
)
from ._atlas import get_atlas
from ._intensity import (
    apply_LAS,
    correct_bias_field,
    correct_label_map,
    scale_intensity,
)
from ._lesions import handle_lesions
from ._partition import compute_euler_number, get_partition
from ._segment_utils import normalize_to_sum1
from .vessels import (
    apply_blood_vessel_correction,
    blood_vessel_prior,
    protected_regions,
)
from ._models import prepare_model_files
from ._conv_chunk import chunked_conv3d
from .nogm import run_segment_nogm_conventional
from .dura import remove_dura
from ._device import (
    mps_routing_requested,
    release_cache,
    resolve_device,
    route_deepmriprep,
    stage_device,
    stage_target_device,
)

ROOT_PATH = Path(__file__).resolve().parents[2]
TMP_PATH = ROOT_PATH / "tmp_models/"


def _progress_bar_script():
    """Locate ``progress_bar_multi.sh`` in both source-tree and installed layouts.

    Source-tree / editable installs keep it at ``<repo>/scripts/``; an installed
    venv ships it into ``<venv>/bin/`` via setuptools ``script-files`` (so
    ``parents[2]`` points into ``site-packages`` and the old ``ROOT_PATH/scripts``
    path does not exist).  Returns ``None`` when it cannot be found, so callers
    fall back to the pure-Python progress renderer rather than crashing.
    """
    candidates = (
        ROOT_PATH / "scripts" / "progress_bar_multi.sh",   # source tree / editable
        Path(sys.prefix) / "bin" / "progress_bar_multi.sh",  # installed venv bin
    )
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("progress_bar_multi.sh")
    return Path(found) if found else None


def shell_progress(count, end_count, label, failed=0):
    # When invoked from the pure-Python API (t1prep.run_t1prep), end_count is
    # 0 because the bash orchestrator's pre-scan that computes the step total
    # was bypassed.  Skip the shell renderer in that case to avoid a divide
    # by zero in progress_bar_multi.sh and stay free of bash dependencies.
    if end_count <= 0:
        return progress_bar(count, end_count, label, failed=bool(failed))
    script = _progress_bar_script()
    if script is None:
        # Shell renderer unavailable — fall back to the Python progress bar.
        return progress_bar(count, end_count, label, failed=bool(failed))
    subprocess.run(
        [str(script), "1", "", str(count), str(end_count), label, "40", str(failed)],
        check=False)
    return count + 1

class CustomBrainSegmentation(BrainSegmentation):
    """
    Custom class to override BrainSegmentation
    Furthermore use run_model function with linear interpolation for p0, which
    prevents negative values due to B-spline interpolation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Follow the device T1Prep selected, not the one deepmriprep pinned
        # itself to: unless `T1PREP_DEVICE=mps` routes everything, deepmriprep
        # stays on the CPU while this stage still belongs on the GPU.  Going
        # through `resolve_device` (rather than hard-coding MPS) is what keeps
        # `T1PREP_DEVICE=cpu` meaningful for a CPU reference run.
        requested, _ = resolve_device()
        self.inference_device = requested if requested.type != "cpu" else self.device

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
        
        # Use linear interpolation, since B-spline interpolation caused issues with
        # negative values in label image and resulted in less accurate segmentations
        with torch.no_grad():
            p0 = self.model(
                F.interpolate(x, scale_factor=1 / scale_factor, **INTERP_KWARGS))
        return F.interpolate(p0, scale_factor=scale_factor, **INTERP_KWARGS)


class CustomPreprocess(Preprocess):
    """
    Custom class to override Preprocess
    Use linear interpolation for p0, which prevents negative values due to
    B-spline interpolation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # deepbet picks its own device (it does support MPS) and moves inputs to
        # wherever its weights live, so the skull-strip needs no help from us.
        # This used to force it onto the CPU because the traced bbox model hit
        # `max_pool3d_with_indices`, which had no MPS kernel; that op is native
        # from torch 2.9 on, and on older builds the PYTORCH_ENABLE_MPS_FALLBACK
        # set at the top of this module covers it — verified on 2.8 and 2.13.
        # The branch was dead code anyway until `_device` started routing
        # deepmriprep to MPS, at which point it would have *demoted* the
        # skull-strip from MPS to CPU.

    def run_segment_brain(self, brain_large, mask, affine, mask_large):
        """Brain segmentation without deepmriprep's unused native-space ``p0``.

        Upstream additionally resamples ``p0_large`` back to native space with a
        quadratic B-spline, which costs ~4 s at 0.5 mm — over a tenth of the whole
        volume pipeline.  T1Prep never reads that output: the native label map
        is written later by ``resample_and_save_nifti`` with linear
        interpolation, which avoids the negative overshoot spline sampling
        produces on a label image.  ``p0_large`` is computed exactly as upstream
        does, so every downstream result is unchanged.
        """
        brain_large = nifti_to_tensor(brain_large)
        mask_large = nifti_to_tensor(mask_large)
        p0_large = self.brain_segment(
            brain_large[None, None].to(self.device),
            mask_large[None, None].to(self.device),
        )[0, 0]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        p0_large[mask_large == 0.0] = 0.0
        return {
            "p0_large": reoriented_nifti(
                p0_large.cpu().numpy(), **self.affine_template_metadata
            )
        }

    def run_warp_register(self, p0_large, p1_affine, p2_affine, wj_affine):
        """Warp registration, on the accelerator when this stage opted in.

        This is the one deepmriprep stage that can move to MPS as-is: it never
        calls ``spline_resize.grid_sample``, whose 27-tap gather over
        47-million-voxel volumes is what exhausts unified memory in the affine,
        brain and nogm stages.  Everything here happens at the 113x137x113 warp
        grid, so the stage gets both faster and smaller — measured 21.6 s /
        3.13 GB on CPU against 7.5 s / 2.28 GB on MPS.  Outputs are CPU-side
        NIfTIs either way; only floating-point rounding differs.
        """
        target = stage_target_device("warp", self.device)
        with stage_device(self, target):
            return super().run_warp_register(
                p0_large, p1_affine, p2_affine, wj_affine
            )

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
                # Case-insensitive fallback: scan the template directory when
                # the exact filename is not found (e.g. "neuromorphometrics"
                # vs the on-disk "Neuromorphometrics.nii.gz").
                if not os.path.exists(atlas_path):
                    atl_lower = atl.lower()
                    for fname in os.listdir(TEMPLATE_PATH_T1PREP):
                        if fname.lower() == f"{atl_lower}.nii.gz":
                            atlas_path = os.path.join(TEMPLATE_PATH_T1PREP, fname)
                            base_atl = os.path.splitext(os.path.splitext(fname)[0])[0]
                            break
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
            # AtlasRegistration pins tensors to its own (CPU) device; if
            # ``self.device`` is MPS/CUDA the warp would land on a different
            # device than the atlas tensor and grid_sample would error out.
            atlas = self.atlas_register(
                affine, warps[shape].to(self.atlas_register.device), atlas, t1.shape
            )
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation pipeline."""

    parser = argparse.ArgumentParser(
        description="T1Prep volume pipeline: skull-stripping, segmentation and atlas ROI export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=str, help="Input NIfTI image")
    parser.add_argument("--mri-dir", required=True, type=str, help="Output folder for MRI volumes")
    parser.add_argument("--report-dir", required=True, type=str, help="Output folder for reports/logs")
    parser.add_argument("--label-dir", required=True, type=str, help="Output folder for labels/aux outputs")
    parser.add_argument(
        "--atlas",
        type=str,
        default="",
        help=(
            "Atlases for ROI estimation (comma-separated). Examples: "
            "\"Neuromorphometrics,SUIT\" or \"'Neuromorphometrics','SUIT'\". "
            "Empty disables ROI export."
        ),
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
        "--save-h5",
        action="store_true",
        help=(
            "Save the T1w<->MNI deformations as ANTs/ITK composite HDF5 "
            "(.h5) files alongside the NIfTI y_ field, without switching the "
            "run to fMRIPrep output mode.  Requires the nitransforms package."
        ),
    )
    parser.add_argument(
        "--save-fmriprep",
        action="store_true",
        help=(
            "Save deformation fields as ANTs/ITK-compatible HDF5 (.h5) files "
            "in addition to the default NIfTI output. "
            "Requires the nitransforms package."
        ),
    )
    parser.add_argument(
        "--bids", action="store_true", help="Use BIDS-like naming convention."
    )
    parser.add_argument(
        "--gz", action="store_true", help="Save compressed NIfTI outputs (.nii.gz)."
    )
    parser.add_argument(
        "--amap", action="store_true", help="Use AMAP segmentation."
    )
    parser.add_argument(
        "--nogm-model",
        action="store_true",
        help=(
            "Remove non-cortical grey matter with the deepmriprep nogm model "
            "instead of the atlas-and-geometry rule in t1prep.nogm."
        ),
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
            "Only run skull stripping and save outputs to --mri-dir, then exit. "
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
    """Return the torch device and ``no_gpu`` flag.

    Also routes deepmriprep's stages to the same device.  Left alone,
    deepmriprep pins itself to ``'cuda' if cuda else 'cpu'`` at import, so on
    Apple Silicon its affine/nogm/warp/atlas stages would stay on the CPU while
    T1Prep believes it selected the GPU — see :mod:`t1prep._device`.  This runs
    before any model is constructed, which is when the device is read.
    """
    device, no_gpu = resolve_device()
    # MPS-everywhere is opt-in (T1PREP_DEVICE=mps): it is faster per stage but
    # currently exhausts unified memory on a full run — see `_device`.  Any
    # other resolved device (CPU, CUDA) is safe to route unconditionally.
    if device.type != "mps" or mps_routing_requested():
        route_deepmriprep(device)
    return device, no_gpu


def preprocess_input(t1: nib.Nifti1Image, no_gpu: bool, use_amap: bool):
    """Denoise and align the input volume and create the preprocessing object.

    Returns:
        Tuple ``(t1, t1_raw, prep, ras_affine)``.  ``t1`` is denoised and used
        for the whole pipeline; ``t1_raw`` shares its grid but keeps the
        original intensities and is only used for the image quality measures,
        which have to describe the acquisition rather than the denoised data
        (CAT12 reads the original file from disk for the same reason).
    """

    vol = t1.get_fdata().copy()
    vol = np.squeeze(vol)

    vol, affine_resamp, header_resamp, ras_affine = align_brain(
        vol, t1.affine, t1.header, np.eye(4), do_flip=0
    )
    t1 = nib.Nifti1Image(vol, affine_resamp, header_resamp)
    prep = CustomPreprocess(no_gpu=no_gpu)

    denoised = cat_surf.vol_sanlm(t1.get_fdata().astype(np.float32))
    # Keep the un-denoised intensities on the same grid (float32 to halve the
    # extra memory) — only the quality measures use them.
    t1_raw = nib.Nifti1Image(vol.astype(np.float32), t1.affine, t1.header)
    t1 = nib.Nifti1Image(denoised, t1.affine, t1.header)


    # This is a bit faster since for initial segmentation the B-spline interpolation
    # of the segmentations does not help and is slower.
    # Furthermore, CustomBrainSegmentation supports mps device
    prep.brain_segment = CustomBrainSegmentation(no_gpu=no_gpu)

    return t1, t1_raw, prep, ras_affine


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
    p0_large: nib.Nifti1Image,
    brain_large: nib.Nifti1Image,
    mri_dir: str,
    out_name: str,
    ext: str,
    verbose: bool,
    debug: bool,
    vessel: float = 0.0,
    protect: np.ndarray = None,
    bv_prior: np.ndarray = None,
    device: str = "cpu",
):
    """Execute the AMAP segmentation pipeline."""

    if verbose:
        print("Running AMAP segmentation")

    p0_large, brain_large = correct_label_map(brain_large, p0_large)
    brain_large = apply_LAS(brain_large, p0_large, verbose=bool(verbose and debug))

    # Blood vessel correction runs between LAS and AMAP, as in CAT12
    # (cat_main.m l. 399-444), so AMAP is fitted on an image where vessels no
    # longer sit at WM intensity.  Correcting the labels afterwards cannot
    # recover the tissue peaks AMAP has already been biased by.
    if vessel > 0:
        brain_large, p0_large = apply_blood_vessel_correction(
            brain_large,
            p0_large,
            strength=vessel,
            protect=protect,
            bv_prior=bv_prior,
            device=device,
            verbose=bool(verbose and debug),
            mri_dir=mri_dir,
            out_name=out_name,
            ext=ext,
        )

    nib.save(brain_large, f"{mri_dir}/{out_name}_brain_large.{ext}")
    nib.save(p0_large, f"{mri_dir}/{out_name}_seg_large.{ext}")

    vol = brain_large.get_fdata().astype(np.float32)
    lab = np.round(p0_large.get_fdata()).astype(np.uint8)
    vx = brain_large.header.get_zooms()[:3]
    _prob, lab_pve, mean = cat_surf.vol_amap(
        vol,
        lab,
        voxelsize=vx,
        weight_mrf=0.0,
        # Amap() normalises this itself (`sub = ROUND(sub / mean_voxelsize)`,
        # CAT_Amap.c), so `sub` is a distance in mm and must not be divided by
        # the voxel size here as well -- that made the sampling grid twice as
        # coarse as intended at 0.5 mm.
        sub=64,
        use_multistep=True,
        pve=True,
        verbose=bool(verbose and debug),
    )

    # vol_amap binds Amap() directly and therefore returns the raw 5-class PVE
    # result.  The 5 -> 3 conversion that CAT_VolAmap and CAT12's cat_amap both
    # apply afterwards is Pve5(), which is not part of Amap(), so it is done
    # here.  Class codes (CAT_Amap.h): 1=CSF, 2=CSF/GM, 3=GM, 4=GM/WM, 5=WM.
    # Pure classes go to their own map; a mixture voxel is split by its
    # intensity between the two tissues it lies between, using the pure-class
    # means Amap estimated.  Note this follows Pve5 in deciding on the hard
    # label rather than mixing the class posteriors.
    #
    # `mean` follows the same 5-class order, so the pure-class means are at
    # indices 0, 2 and 4 -- which is how Pve5 itself reads them
    # (mean[CSFLABEL - 1], mean[GMLABEL - 1], mean[WMLABEL - 1]).  Taking
    # 0, 1, 2 instead picks up the CSF/GM mixture mean as GM and the GM mean
    # as WM, which collapses the GM/WM band below to pure WM.
    m_csf, m_gm, m_wm = (float(mean[0]), float(mean[2]), float(mean[4]))
    p_csf = np.zeros(vol.shape, np.float32)
    p_gm = np.zeros(vol.shape, np.float32)
    p_wm = np.zeros(vol.shape, np.float32)
    p_csf[lab_pve == 1] = 1.0
    p_gm[lab_pve == 3] = 1.0
    p_wm[lab_pve == 5] = 1.0
    sel = lab_pve == 2
    w = np.clip((vol[sel] - m_csf) / max(m_gm - m_csf, 1e-6), 0.0, 1.0)
    p_csf[sel], p_gm[sel] = 1.0 - w, w
    sel = lab_pve == 4
    w = np.clip((vol[sel] - m_gm) / max(m_wm - m_gm, 1e-6), 0.0, 1.0)
    p_gm[sel], p_wm[sel] = 1.0 - w, w
    nib.save(
        nib.Nifti1Image(p_gm, brain_large.affine, brain_large.header),
        f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}",
    )
    nib.save(
        nib.Nifti1Image(p_wm, brain_large.affine, brain_large.header),
        f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}",
    )
    nib.save(
        nib.Nifti1Image(p_csf, brain_large.affine, brain_large.header),
        f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}",
    )
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
        remove_file(f"{mri_dir}/{out_name}_brain_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_seg_large.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        remove_file(f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")


@dataclass(frozen=True)
class OutputOptions:
    """Where the volume pipeline writes to, and which optional outputs it saves."""

    t1_name: str
    mri_dir: str
    label_dir: str
    report_dir: str
    out_name: str
    ext: str
    use_bids: bool
    save_p: bool
    save_rp: bool
    save_wp: bool
    save_mwp: bool
    save_hemilabel: bool
    save_lesions: bool
    save_csf: bool
    save_fmriprep: bool
    save_h5: bool
    atlas_list: Optional[tuple]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OutputOptions":
        """Collect the output settings from :func:`parse_arguments`."""
        # Get atlas list (currently restricted to ROI estimation)
        atlas = tuple(x.strip(" '") for x in args.atlas.split(","))
        # Build atlas_list. Set atlas_list to None, if empty
        atlas_list = (
            tuple(f"{a}_volumes" for a in atlas)
            if any(atlas) and atlas != ("",)
            else None
        )
        out_name = os.path.basename(
            os.path.basename(args.input).replace(".nii", "")
        ).replace(".gz", "")
        return cls(
            t1_name=args.input,
            mri_dir=args.mri_dir,
            label_dir=args.label_dir,
            report_dir=args.report_dir,
            out_name=out_name,
            ext="nii.gz" if args.gz else "nii",
            use_bids=args.bids,
            save_p=args.p,
            save_rp=args.rp,
            save_wp=args.wp,
            save_mwp=args.mwp,
            save_hemilabel=args.surf,
            save_lesions=args.lesions,
            save_csf=args.csf,
            save_fmriprep=args.save_fmriprep,
            save_h5=args.save_h5,
            atlas_list=atlas_list,
        )

    @property
    def needs_warp(self) -> bool:
        """Whether any requested output needs the non-linear registration."""
        return (
            self.save_hemilabel
            or self.save_mwp
            or self.save_wp
            or self.save_fmriprep
            or self.atlas_list is not None
        )


class _OutputNames(NamedTuple):
    """``get_filenames`` tables for each space and hemisphere."""

    native: dict
    affine: dict
    warped: dict
    warped_modulated: dict
    left: dict
    right: dict


def _output_names(opts: OutputOptions) -> _OutputNames:
    """Look up the output file names for every space the pipeline writes in."""
    use_bids, out_name, ext = opts.use_bids, opts.out_name, opts.ext
    native = get_filenames(use_bids, out_name, "", "", "", ext)
    space_affine = native.get("Affine_space", "")
    if use_bids:
        affine = get_filenames(use_bids, out_name, "", "", space_affine, ext)
    else:
        affine = get_filenames(use_bids, out_name, "", "_affine", space_affine, ext)
    return _OutputNames(
        native=native,
        affine=affine,
        warped=get_filenames(
            use_bids, out_name, "", "", native.get("Warp_space", ""), ext
        ),
        warped_modulated=get_filenames(
            use_bids, out_name, "", "", native.get("Warp_modulated_space", ""), ext
        ),
        left=get_filenames(use_bids, out_name, "left", "", "", ext),
        right=get_filenames(use_bids, out_name, "right", "", "", ext),
    )


def _to_template_space(img: nib.Nifti1Image, warp_template: nib.Nifti1Image):
    """Downsample a working-grid map by 3 onto the warp template grid."""
    data = F.interpolate(
        nifti_to_tensor(img)[None, None], scale_factor=1 / 3, **INTERP_KWARGS
    )[0, 0]
    return reoriented_nifti(data, warp_template.affine, warp_template.header)


def _save_native_outputs(
    opts: OutputOptions,
    names: _OutputNames,
    t1: nib.Nifti1Image,
    p0_large: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wmh_large,
    discrepancy_large,
    brain_large: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    grid_native,
) -> None:
    """Write the label, the bias-corrected T1w and the tissue maps in native space."""
    mri_dir = opts.mri_dir
    native = names.native

    def save(img, name, **kwargs):
        resample_and_save_nifti(
            img, grid_native, mask.affine, mask.header, f"{mri_dir}/{name}", **kwargs
        )

    label_name = native.get("Label_volume", "")
    mT1_name = native.get("mT1_volume", "")
    save(p0_large, label_name, clip=[0, 4])
    # ``desc-preproc_T1w`` keeps the skull under --fmriprep, matching what
    # fMRIPrep means by the name and what it coregisters BOLD against.  The
    # bias field is refitted on the whole head from the native label; the
    # spline is driven by the WM mask either way, so the tissue intensities
    # come out on the same scale as the skull-stripped image (WM at 1), and
    # scalp and skull are extrapolated rather than zeroed.
    full_head = None
    if opts.save_fmriprep:
        # Both images have to be in the same array order before the bias fit
        # can pair a voxel with its label, and the result has to go back into
        # the orientation the other native outputs are stored in -- which is
        # not RAS in general.  Writing a canonical array under ``mask.affine``
        # would flip it against its own affine.
        t1_canonical = nib.as_closest_canonical(t1)
        label_native = nib.as_closest_canonical(nib.load(f"{mri_dir}/{label_name}"))
        if np.allclose(t1_canonical.affine, label_native.affine, atol=1e-3) and (
            t1_canonical.shape[:3] == label_native.shape[:3]
        ):
            full_head = correct_bias_field(t1_canonical, label_native)
            # The native outputs share ``mask``'s uint8 header, which would
            # quantise a whole-head image whose scalp reaches ~5x the WM
            # intensity.  fMRIPrep stores this one as float32; so do we.
            header = mask.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(
                reoriented_nifti(
                    full_head.get_fdata().astype(np.float32), mask.affine, header
                ),
                f"{mri_dir}/{mT1_name}",
            )
        else:
            print(
                "Warning: the input does not share the native output grid; "
                "writing a skull-stripped desc-preproc_T1w instead."
            )
    if full_head is None:
        save(brain_large, mT1_name)
    if opts.save_fmriprep:
        # The skull-stripped, bias-corrected brain that used to be written as
        # desc-preproc_T1w is still useful, so keep it under its own name.
        save(brain_large, native.get("skullstripped_volume", ""))

    if opts.save_p:
        save(p1_large, native.get("GM_volume", ""), clip=[0, 1])
        save(p2_large, native.get("WM_volume", ""), clip=[0, 1])
        if opts.save_csf:
            save(p3_large, native.get("CSF_volume", ""), clip=[0, 1])

    if opts.save_lesions and wmh_large is not None:
        save(wmh_large, native.get("WMH_volume", ""))
        save(discrepancy_large, native.get("Discrepance_volume", ""))


def _write_volume_report(
    opts: OutputOptions,
    names: _OutputNames,
    t1: nib.Nifti1Image,
    t1_raw,
    p0_large: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wmh_large,
    brain_large: nib.Nifti1Image,
    grid_native,
    wj_affine: pd.Series,
) -> None:
    """Write tissue volumes, mean intensities and image quality to the report."""
    with_lesions = opts.save_lesions and wmh_large is not None

    # Estimate raw volumes
    vol_gm = get_volume_native_space(p1_large, wj_affine[0])  # GM    (p1)
    vol_wm = get_volume_native_space(p2_large, wj_affine[0])  # WM    (p2)
    vol_csf = get_volume_native_space(p3_large, wj_affine[0])  # CSF   (p3)
    if with_lesions:
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
    # CAT12 reads the ORIGINAL file (varargin{2}); both branches that would use
    # processed data are permanently disabled with "if 0", because the measures
    # have to describe the acquisition, not the pipeline output.  We therefore
    # pass ``t1_raw`` — same grid as ``t1`` but before SANLM denoising, which
    # otherwise removes exactly the high-frequency component NCR measures and
    # compresses the rating scale (a 9 % noise scan would rate like a 1 % one).
    # The bias field is *not* removed here either: estimate_qa approximates and
    # divides it out itself (cat_vol_approx, cat_vol_qa201901x.m:293-298) and
    # reports its strength as ICR, which a pre-corrected input would void.
    t1_qa = t1_raw if t1_raw is not None else t1
    vx_vol_orig = np.array(t1_qa.header.get_zooms()[:3], dtype=np.float64)
    p0_native = F.grid_sample(
        nifti_to_tensor(p0_large)[None, None],
        grid_native,
        align_corners=INTERP_KWARGS["align_corners"],
    )[0, 0].numpy()
    # t1 has negative-diagonal affine after align_brain (do_flip=0); grid_native
    # produces p0_native in as_closest_canonical space (positive/RAS axes).
    # Must canonicalise t1 the same way so tissue masks align with intensities.
    t1_canonical = nib.as_closest_canonical(t1_qa).get_fdata().astype(np.float32)
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

    if with_lesions:
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
    report_name = names.native.get("Report_file", "")
    with open(f"{opts.report_dir}/{report_name}", "w") as f:
        json.dump(summary, f, indent=2)


def _save_atlas_rois(
    prep: CustomPreprocess,
    opts: OutputOptions,
    names: _OutputNames,
    t1: nib.Nifti1Image,
    affine,
    warp_yx: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wj_affine: pd.Series,
) -> None:
    """Write the regional tissue volumes of all requested atlases to one JSON file."""
    output_atlas = prep.run_atlas_register(
        t1, affine, warp_yx, p1_large, p2_large, p3_large, opts.atlas_list, wj_affine
    )

    # Convert each DataFrame to a list of dicts:
    atlas_json = {
        key.removesuffix("_volumes"): df.to_dict(orient="records")
        for key, df in output_atlas.items()
    }

    # Write to a single JSON file:
    atlas_name = names.native.get("Atlas_ROI", "")
    with open(f"{opts.label_dir}/{atlas_name}", "w") as f:
        json.dump(atlas_json, f, indent=2)


def _save_warped_maps(opts: OutputOptions, names: _OutputNames, output_reg) -> None:
    """Write the warped (wp) and modulated warped (mwp) GM and WM maps.

    ``run_warp_register`` returns no warped CSF map, so ``--csf`` adds nothing
    here.
    """
    if opts.save_mwp:
        nib.save(
            output_reg["mwp1"],
            f"{opts.mri_dir}/{names.warped_modulated.get('GM_volume', '')}",
        )
        nib.save(
            output_reg["mwp2"],
            f"{opts.mri_dir}/{names.warped_modulated.get('WM_volume', '')}",
        )
    if opts.save_wp:
        nib.save(
            output_reg["wp1"], f"{opts.mri_dir}/{names.warped.get('GM_volume', '')}"
        )
        nib.save(
            output_reg["wp2"], f"{opts.mri_dir}/{names.warped.get('WM_volume', '')}"
        )


def _save_deformations(
    opts: OutputOptions,
    names: _OutputNames,
    warp_xy: nib.Nifti1Image,
    warp_yx: nib.Nifti1Image,
    affine,
    mask: nib.Nifti1Image,
) -> None:
    """Write the T1w<->MNI deformations: ITK composites and/or the SPM y_ field."""
    mri_dir = opts.mri_dir
    native = names.native

    if opts.save_fmriprep or opts.save_h5:
        # save deformation as fMRIPrep-compatible h5-file.  ``Def_h5_volume``
        # is the T1w-to-MNI direction (CAT12's ``y_``) and is built from
        # warp_xy, which maps template points back onto the subject;
        # ``invDef_h5_volume`` is the opposite direction, from warp_yx.
        def_h5_name = native.get("Def_h5_volume", "")
        save_deformation_h5(warp_xy, affine, mask, f"{mri_dir}/{def_h5_name}")
        invdef_h5_name = native.get("invDef_h5_volume", "")
        save_deformation_h5(
            warp_yx, affine, mask, f"{mri_dir}/{invdef_h5_name}", inverse=True
        )

        # The same pair retargeted at MNI152NLin6Asym, which fMRIPrep adds
        # to its normalisation targets for --cifti-output and would
        # otherwise spend a full ANTs registration computing.  The fixed
        # 2009cAsym<->6Asym warp is composed into the stored field, so
        # these stay two-element composites like the pair above.
        for code, disp_name, is_inverse in (
            ("Def6_h5_volume",
             "tpl-MNI152NLin6Asym_to-MNI152NLin2009cAsym_desc-disp_xfm.nii.gz", False),
            ("invDef6_h5_volume",
             "tpl-MNI152NLin2009cAsym_to-MNI152NLin6Asym_desc-disp_xfm.nii.gz", True),
        ):
            disp_path = os.path.join(TEMPLATE_PATH_T1PREP, disp_name)
            if not os.path.exists(disp_path):
                print(f"Warning: {disp_name} not found; skipping MNI152NLin6Asym transform.")
                continue
            save_deformation_h5(
                warp_yx if is_inverse else warp_xy,
                affine,
                mask,
                f"{mri_dir}/{native.get(code, '')}",
                inverse=is_inverse,
                template_displacement=nib.load(disp_path),
            )

    # save deformation as nifti-file
    if not opts.save_fmriprep:
        def_name = native.get("Def_volume", "")
        save_deformation_spm(warp_xy, affine, mask, f"{mri_dir}/{def_name}")


def _save_fmriprep_extras(
    opts: OutputOptions,
    names: _OutputNames,
    p0_large: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    grid_native,
) -> None:
    """Write the fMRIPrep-only outputs: fsnative transforms, dseg, mask, sidecars."""
    mri_dir = opts.mri_dir
    native = names.native

    # T1w <-> fsnative.  T1Prep reconstructs its surfaces directly on
    # the preprocessed T1w grid and has no separate FreeSurfer
    # conformed space, so both directions are the identity.  (This is
    # not the MNI affine: that one lives in the composites above, and
    # is a normalised grid transform rather than a millimetre one.)
    affine_txt_name = native.get("Affine_txt_volume", "")
    save_affine_itk_txt(np.eye(4), f"{mri_dir}/{affine_txt_name}")
    invaffine_txt_name = native.get("invAffine_txt_volume", "")
    save_affine_itk_txt(np.eye(4), f"{mri_dir}/{invaffine_txt_name}")

    # Save dseg in native space and reorder tissue class intensities
    dseg_value = np.round(p0_large.get_fdata().copy())
    ind_CSF = dseg_value == 1
    ind_GM = dseg_value == 2
    ind_WM = dseg_value == 3
    dseg_value[ind_CSF] = 3
    dseg_value[ind_GM] = 1
    dseg_value[ind_WM] = 2
    dseg_large = nib.Nifti1Image(dseg_value, p0_large.affine, p0_large.header)
    dseg_name = native.get("dseg_volume", "")
    resample_and_save_nifti(
        dseg_large,
        grid_native,
        mask.affine,
        mask.header,
        f"{mri_dir}/{dseg_name}",
        clip=[0, 4],
    )

    # simply use the clipped dseg image as mask image
    mask_name = native.get("mask_volume", "")
    resample_and_save_nifti(
        dseg_large,
        grid_native,
        mask.affine,
        mask.header,
        f"{mri_dir}/{mask_name}",
        clip=[0, 1],
    )

    # BIDS bookkeeping.  PyBIDS ignores a derivatives tree without a
    # dataset description, and the specification requires ``Type`` on
    # every mask, so without these fMRIPrep cannot see any of the
    # files above and recomputes them.
    raw_sources = [os.path.abspath(opts.t1_name)] if opts.t1_name else None
    write_dataset_description(mri_dir)
    write_sidecar(f"{mri_dir}/{mask_name}", Type="Brain", RawSources=raw_sources)
    write_sidecar(
        f"{mri_dir}/{native.get('mT1_volume', '')}",
        SkullStripped=False,
        RawSources=raw_sources,
    )
    write_sidecar(
        f"{mri_dir}/{native.get('skullstripped_volume', '')}",
        SkullStripped=True,
        RawSources=raw_sources,
    )


def _report_euler_numbers(report_path: str, lh: np.ndarray, rh: np.ndarray) -> None:
    """Add the hemispheres' Euler numbers at the GM/WM boundary to the report."""
    euler_lh = compute_euler_number(lh, threshold=2.5)
    euler_rh = compute_euler_number(rh, threshold=2.5)
    EC_abs = abs(euler_lh - 2) + abs(euler_rh - 2)

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
    qa["EC_abs"] = {
        "value": EC_abs,
        "desc": (
            "Absolute Euler number for both hemispheres "
            "(absolute difference to ideal Euler number of 2; ideal = 0; "
            "larger values indicate more topological defects)"
        ),
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)


def _save_hemispheres(
    opts: OutputOptions,
    names: _OutputNames,
    t1: nib.Nifti1Image,
    affine,
    p0_large: nib.Nifti1Image,
    warp_yx: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    grid_native,
    grid_target_res,
    affine_resamp,
    header_resamp,
    device,
    verbose: bool,
    count: int,
    end_count: int,
) -> None:
    """Write the hemisphere label maps that surface extraction starts from.

    Also writes the fMRIPrep ribbon mask and adds the hemispheres' Euler
    numbers to the report.
    """
    mri_dir = opts.mri_dir

    if verbose:
        count = shell_progress(count, end_count,
            "Atlas creation               ")
    atlas = get_atlas(
        t1,
        affine,
        p0_large.header,
        p0_large.affine,
        "IBSR",
        warp_yx,
        device,
        is_label_atlas=True,
    )
    # Locates the cortex the fills in ``get_partition`` must spare;
    # IBSR has no cortical parcellation to do that with.
    guard_atlas = get_atlas(
        t1,
        affine,
        p0_large.header,
        p0_large.affine,
        "Neuromorphometrics",
        warp_yx,
        device,
        is_label_atlas=True,
    )

    lh, rh = get_partition(p0_large, atlas, guard_atlas)
    del guard_atlas

    if opts.save_fmriprep:
        # Get the ribbon mask using lh and rh and masking GM
        ribbon_value = lh + rh
        ribbon_value = (ribbon_value > 2.5) & (ribbon_value < 3.5)
        ribbon_large = nib.Nifti1Image(ribbon_value, p0_large.affine, p0_large.header)
        ribbon_name = names.native.get("ribbon_volume", "")
        resample_and_save_nifti(
            ribbon_large,
            grid_native,
            mask.affine,
            mask.header,
            f"{mri_dir}/{ribbon_name}",
            round=True,
        )
        write_sidecar(f"{mri_dir}/{ribbon_name}", Type="ROI")

    report_name = names.native.get("Report_file", "")
    _report_euler_numbers(f"{opts.report_dir}/{report_name}", lh, rh)

    if verbose:
        count = shell_progress(count, end_count,
            "Resampling                   ")

    # The hemisphere labels go onto a 0.5 mm grid from a 0.5 mm source,
    # so this is a pure reslice at matched resolution -- the case where
    # the trilinear kernel blurs most.  Surface extraction thresholds
    # these maps at the GM/WM level, and trilinear displaces that
    # boundary by ~61 um (median, against a quintic reference) against
    # ~17 um for the B-spline.  ``clip_overshoot`` removes the ringing
    # the spline introduces at the sharp label edges, keeping the values
    # inside the [1, 3] range ``get_partition`` produces.
    for hemi, side_names in ((lh, names.left), (rh, names.right)):
        resample_and_save_nifti(
            nib.Nifti1Image(hemi, p0_large.affine, p0_large.header),
            grid_target_res,
            affine_resamp,
            header_resamp,
            f"{mri_dir}/{side_names.get('Hemi_volume', '')}",
            True,
            True,
            bspline=True,
            clip_overshoot=True,
        )


def save_results(
    prep: CustomPreprocess,
    opts: OutputOptions,
    *,
    t1: nib.Nifti1Image,
    t1_raw,
    affine,
    p0_large: nib.Nifti1Image,
    p1_large: nib.Nifti1Image,
    p2_large: nib.Nifti1Image,
    p3_large: nib.Nifti1Image,
    wmh_large,
    discrepancy_large,
    brain_large: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    grid_native,
    grid_target_res,
    affine_resamp,
    header_resamp,
    warp_template: nib.Nifti1Image,
    wj_affine: pd.Series,
    device,
    verbose: bool,
    count: int,
    end_count: int,
) -> None:
    """Save segmentation and atlas results to disk.

    Writes the affine-registered tissue maps, the native-space outputs and the
    volume/QA report.  If any requested output needs the non-linear
    registration (``opts.needs_warp``), it then runs the warp and writes the
    atlas ROI volumes, the warped maps, the deformation fields, the fMRIPrep
    extras and the hemisphere label maps.
    """
    names = _output_names(opts)

    # Get affine segmentations: saved as rp*, and the input of the warp
    p1_affine = p2_affine = p3_affine = None
    if opts.save_rp or opts.needs_warp:
        p1_affine = _to_template_space(p1_large, warp_template)
        p2_affine = _to_template_space(p2_large, warp_template)
        if opts.save_csf and opts.save_rp:
            p3_affine = _to_template_space(p3_large, warp_template)

    # Save affine registered data
    if opts.save_rp:
        nib.save(p1_affine, f"{opts.mri_dir}/{names.affine.get('GM_volume', '')}")
        nib.save(p2_affine, f"{opts.mri_dir}/{names.affine.get('WM_volume', '')}")
        if opts.save_csf:
            nib.save(
                p3_affine, f"{opts.mri_dir}/{names.affine.get('CSF_volume', '')}"
            )

    _save_native_outputs(
        opts, names, t1, p0_large, p1_large, p2_large, p3_large,
        wmh_large, discrepancy_large, brain_large, mask, grid_native,
    )
    _write_volume_report(
        opts, names, t1, t1_raw, p0_large, p1_large, p2_large, p3_large,
        wmh_large, brain_large, grid_native, wj_affine,
    )

    if not opts.needs_warp:
        return

    if verbose:
        count = shell_progress(count, end_count,
            "Warping                      ")
    # The warp model's 32-channel layer at the template grid would ask for
    # an 8 GB im2col buffer in one allocation on CPU, which is the peak of
    # the whole run.  Slab it: same result, roughly a quarter of the memory.
    with chunked_conv3d():
        output_reg = prep.run_warp_register(
            p0_large, p1_affine, p2_affine, wj_affine
        )
    release_cache(device)
    warp_yx = output_reg["warp_yx"]
    warp_xy = output_reg["warp_xy"]

    if opts.atlas_list is not None:
        _save_atlas_rois(
            prep, opts, names, t1, affine, warp_yx,
            p1_large, p2_large, p3_large, wj_affine,
        )
    _save_warped_maps(opts, names, output_reg)
    _save_deformations(opts, names, warp_xy, warp_yx, affine, mask)
    if opts.save_fmriprep:
        _save_fmriprep_extras(opts, names, p0_large, mask, grid_native)

    # Save hemispheric partition for surface estimation
    if opts.save_hemilabel or opts.save_fmriprep:
        _save_hemispheres(
            opts, names, t1, affine, p0_large, warp_yx, mask, grid_native,
            grid_target_res, affine_resamp, header_resamp, device,
            verbose, count, end_count,
        )


def run_segment():
    """Run the full segmentation workflow."""

    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    # Output locations and the optional outputs to write
    opts = OutputOptions.from_args(args)
    t1_name = opts.t1_name
    mri_dir, out_name, ext = opts.mri_dir, opts.out_name, opts.ext

    # Processing options
    use_amap = args.amap
    use_nogm_model = args.nogm_model
    vessel = args.vessel
    verbose = args.verbose
    debug = args.debug
    skullstrip_only = args.skullstrip_only
    skip_skullstrip = args.skip_skullstrip

    # Check for GPU support
    device, no_gpu = setup_device()

    sentry_sdk.init(
        dsn="https://ca6089ed5dc4326c6c69afc3684c5fc1@o4511309449068544.ingest.de.sentry.io/4511309454311504",
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=False,
    )
    
    # Set processing parameters
    target_res = np.array([0.5] * 3)  # Target resolution for resampling
    count = 1
    end_count = args.count

    # Track running time
    start = time.perf_counter()
    t1 = nib.load(t1_name)

    # Ensure required model files are available
    prepare_model_files()

    # Preprocess volume and create preprocess object
    t1, t1_raw, prep, ras_affine = preprocess_input(t1, no_gpu, use_amap)

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
        save_skullstrip_only_outputs(brain, opts.use_bids, mri_dir, out_name, ext)
        return

    # Step 2: Initial bias-correction that is benefitial for strong signal
    # inhomogeneities (i.e. 7T data)
    brain = correct_bias_field(brain)

    # Step 3: Affine registration
    affine, brain_large, mask_large, affine_loss, count = affine_register(
        prep, brain, mask, verbose, count, end_count
    )

    # Ensure that minimum of brain is not negative (which can happen after B-spline interpolation)
    brain_value = brain_large.get_fdata().copy()
    mask_value = binary_closing(brain_value > 0.0, generate_binary_structure(3, 3), 7)
    min_brain = np.min(brain_value)
    if min_brain < 0:
        brain_value -= min_brain
    brain_value[~mask_value] = 0
    brain_large = nib.Nifti1Image(brain_value, brain_large.affine, brain_large.header)

    # Step 4: Segmentation
    if verbose:
        count = shell_progress(
            count, end_count, "DeepMriPrep segmentation     ")
    # Unsplit, the widest layer of the segmentation model asks for a single
    # 20.7 GB im2col buffer at this grid — the largest allocation in the run.
    with chunked_conv3d():
        output_seg = prep.run_segment_brain(brain_large, mask, affine, mask_large)
    release_cache(device)
    p0_large = output_seg["p0_large"]

    # Due to B-spline interpolation we have to change values below zero
    p0_value = p0_large.get_fdata()
    if np.min(p0_value) < 0:
        p0_value[p0_value < 0] = 0
        p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

    mask_large_value = p0_large.get_fdata() > 1E-3
        
    # Prepare for resampling
    header_resamp, affine_resamp = get_resampled_header(
        brain.header, brain.affine, target_res, ras_affine
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

    # Dura the skull-strip left outside the CSF.  It is cleared from the image,
    # the label and the mask the final label is cut with, before LAS, AMAP, the
    # vessel correction and nogm see it.  See t1prep.dura.
    brain_large, p0_large, dura_large = remove_dura(
        brain_large, p0_large, verbose=bool(verbose and debug)
    )
    if dura_large is not None:
        mask_large_value &= ~dura_large
        if debug:
            nib.save(
                nib.Nifti1Image(dura_large.astype(np.uint8), p0_large.affine),
                f"{mri_dir}/{out_name}_dura_large.{ext}",
            )
        del dura_large

    p0_large_orig = p0_large

    # Regions the vessel correction must not touch, and the MRA-derived
    # spatial prior.  Both only depend on the grid, so they are built once
    # here and reused by the post-segmentation cleanup further down.
    if vessel > 0:
        protect = protected_regions(p0_large.affine, p0_large.shape, device)
        try:
            bv_prior = blood_vessel_prior(
                p0_large.affine, p0_large.shape, device=device
            )
        except (FileNotFoundError, OSError) as exc:
            if verbose:
                print(f"Blood vessel prior unavailable ({exc}); continuing without it")
            bv_prior = None
    else:
        protect, bv_prior = None, None

    if use_amap:
        if verbose:
            count = shell_progress(count, end_count,
                "Amap segmentation            ")
        brain_large, p0_large = run_amap_segmentation(
            p0_large,
            brain_large,
            mri_dir,
            out_name,
            ext,
            verbose,
            debug,
            vessel=vessel,
            protect=protect,
            bv_prior=bv_prior,
            device=device,
        )
    else:
        brain_large = apply_LAS(brain_large, p0_large, verbose=bool(verbose and debug))

        # The deepmriprep path refines p1/p2/p3 from p0_large in
        # run_segment_nogm below, so the correction has to land before that
        # call for the vessel removal to reach the tissue maps at all.
        if vessel > 0:
            brain_large, p0_large = apply_blood_vessel_correction(
                brain_large,
                p0_large,
                strength=vessel,
                protect=protect,
                bv_prior=bv_prior,
                device=device,
                verbose=bool(verbose and debug),
                mri_dir=mri_dir,
                out_name=out_name,
                ext=ext,
            )

    if debug:
        nib.save(brain_large, f"{mri_dir}/{out_name}_brain_large_tmp.{ext}")
        nib.save(p0_large, f"{mri_dir}/{out_name}_seg_large.{ext}")

    # Hoisted above the segmentation branch: the default nogm rule needs
    # it to express its volumes in native space, and nothing here depends on
    # the tissue maps it is computed alongside.
    warp_template = nib.load(f"{DATA_PATH}/templates/Template_4_GS.nii.gz")
    wj_affine = (
        np.linalg.det(affine.values) * nifti_volume(t1) / nifti_volume(warp_template)
    )

    if use_amap:
        # Load Amap label
        p1_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-GM_probseg.{ext}")
        p2_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-WM_probseg.{ext}")
        p3_large = nib.load(f"{mri_dir}/{out_name}_brain_large_label-CSF_probseg.{ext}")
    else:
        if use_nogm_model:
            # Call deepmriprep refinement of deepmriprep label
            if verbose:
                count = shell_progress(
                    count, end_count,
                        "Fine DeepMriPrep segmentation"
                )
            # Same story as the brain model: 15.2 GB in one block if left unsplit.
            with chunked_conv3d():
                output_nogm = prep.run_segment_nogm(p0_large, affine, t1)
            release_cache(device)
        else:
            # Atlas-and-geometry equivalent of the nogm model.  Measured on a
            # 0.5 mm subject: 5.5 s / 2.0 GB against 43.8 s / 5.6 GB, agreeing
            # with the model at Dice 0.74.  See t1prep.nogm.
            if verbose:
                count = shell_progress(
                    count, end_count,
                        "Remove non-cortical GM"
                )
            # No device argument: the rule is numpy/scipy throughout and its
            # one torch op is a nearest-neighbour atlas sample, which an
            # accelerator would only round-trip.  MPS has no
            # ``grid_sampler_3d`` kernel, so routing it there additionally
            # depends on PYTORCH_ENABLE_MPS_FALLBACK to land back on the CPU.
            output_nogm = run_segment_nogm_conventional(
                p0_large,
                wj_affine=wj_affine,
                verbose=bool(verbose and debug),
            )

        # Load probability maps for GM, WM, CSF
        p1_large = output_nogm["p1_large"]
        p2_large = output_nogm["p2_large"]
        p3_large = output_nogm["p3_large"]

    if use_amap or opts.save_lesions:
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
            use_amap,
            device,
        )
    else:
        discrepancy_large = None

    wj_affine = pd.Series([wj_affine])

    # Rebuild the label from the (possibly lesion-corrected) tissue maps.
    gm, wm, csf = normalize_to_sum1(
        p1_large.get_fdata(), p2_large.get_fdata(), p3_large.get_fdata()
    )
    p0_large = nib.Nifti1Image(
        csf + 2 * gm + 3 * wm, p0_large.affine, p0_large.header
    )

    # We have to apply the initial mask again to the label
    p0_value = p0_large.get_fdata().copy()
    p0_value[mask_large_value == 0] = 0
    p0_large = nib.Nifti1Image(p0_value, p0_large.affine, p0_large.header)

    if use_amap or opts.save_lesions:
        p0_value = p0_large.get_fdata().copy()
        np.clip(p0_value, 0, 3, out=p0_value)
        p0_value[ind_wmh] += wmh_value[ind_wmh]
        np.clip(p0_value, 0, 4, out=p0_value)
        # These maps live on the working ("large") grid, so they keep that
        # grid's affine.  Stamping the native image resampled to ``target_res``
        # on them instead put ``p0_large`` ~180 mm off in world space, and
        # ``get_atlas`` places the atlas from exactly that affine -- the
        # hemisphere atlas then landed outside the volume, so ``get_partition``
        # returned empty hemispheres and the crop below had nothing to crop to.
        p0_affine, p0_header = p0_large.affine, p0_large.header
        p0_large = nib.Nifti1Image(p0_value, p0_affine, p0_header)
        wmh_large = nib.Nifti1Image(wmh_value, p0_affine, p0_header)
    else:
        wmh_large = None

    save_results(
        prep,
        opts,
        t1=t1,
        t1_raw=t1_raw,
        affine=affine,
        p0_large=p0_large,
        p1_large=p1_large,
        p2_large=p2_large,
        p3_large=p3_large,
        wmh_large=wmh_large,
        discrepancy_large=discrepancy_large,
        brain_large=brain_large,
        mask=mask,
        grid_native=grid_native,
        grid_target_res=grid_target_res,
        affine_resamp=affine_resamp,
        header_resamp=header_resamp,
        warp_template=warp_template,
        wj_affine=wj_affine,
        device=device,
        verbose=verbose,
        count=count,
        end_count=end_count,
    )

    final_cleanup(mri_dir, out_name, ext, use_amap, opts.save_lesions, debug)

    # Write to log file
    end = time.perf_counter()
    text = f"Execution time of volume pipeline: {end - start:.1f}s.\n"
    code_vars = get_filenames(opts.use_bids, out_name, "", "", "", ext)
    log_name = code_vars.get("Log_file", "")
    with open(f"{opts.report_dir}/{log_name}", "a") as f:
        f.write(text)

    write_t1prep_report(opts.report_dir, out_name, opts.use_bids, t1_name)


if __name__ == "__main__":
    run_segment()
