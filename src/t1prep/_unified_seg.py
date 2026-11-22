"""Unified segmentation in place of deepbet, the affine fit and deepmriprep's networks.

``--unified-seg`` hands the first stages of the volume pipeline to
``cat_surf.vol_unified_seg`` (cat-surf >= 1.1.9): SPM's unified segmentation
-- a mixture of Gaussians per class, a smooth bias field and tissue priors --
whose priors, a series of shooting templates with six classes (GM, WM, CSF,
bone, soft tissue, air), are deformed onto the image by geodesic shooting.
The model covers the whole head, so it replaces three stages:

- the affine registration to MNI152NLin2009cAsym (deepmriprep's torchreg fit);
- the label map with partial volumes and the bias-corrected image
  (deepmriprep's segmentation networks and T1Prep's first bias correction);
- the brain mask, from the posteriors of GM, WM and CSF (deepbet's networks).

The segmentation is computed on the 0.5 mm working grid, where deepmriprep's
networks segment, not at the resolution of the scan.  That takes two passes:

1. The affine registration, on the unfiltered scan as it is.
   ``vol_unified_seg`` registers first and from the unfiltered image on
   sampling lattices of its own, so the settings of :data:`AFFINE_PASS`
   change nothing of the affine (it came out bit-identical); they only make
   the rest of this pass, which is discarded, as short as the library allows
   (10.5 s instead of 39.5 s on IXI199, the affine itself 2.7 s).
2. The segmentation, on the scan denoised by T1Prep's initial SANLM filter
   and resampled onto the working grid through that affine (quadratic
   B-spline, as the affine registration resamples the brain), with the
   affine fixed: the label map, the corrected image and the posteriors come
   out on the working grid and are not resampled again.

The second pass runs with :data:`UNIFIED_SETTINGS`; everything else keeps
cat-surf's defaults (``cat_surf.vol_unified_seg_defaults()``).

The SANLM filter runs on the scan at its own resolution, before the
resampling, and the library's own filter is off.  On the resampled scan the
noise is no longer white: the library estimated it at 0.19% of the range on
IXI199 instead of 2.06%, its filter hardly filtered, and the GM/WM boundary
of the label map, which follows the intensity of every voxel through the
partial volume model, came out with two to three times the topology
defects (Euler numbers of the hemispheres 508/602 against 290/404 on
IXI199, 1644/1492 against 560/512 on HR075).  The parameters are then
estimated from the filtered image, which CAT-Surface found to narrow the
Gaussian of white matter; adding the scan's own noise back for the
estimation (``noise``) did not help: a better label RMSE on two of three
images, but more defects on two, and worse in every measure on the phantom
below.

A segmentation at the resolution of the scan, its label map resampled
onto the working grid by B-spline, had fewer defects still (136/216 on
IXI199, 46/28 against 180/106 on ADNI_005), but on the simulated HR075 with
its known ground truth (``evaluation/data/phantom``, SNR 25, bias 45) GM and
WM were less accurate than at 0.5 mm: soft Dice 0.906/0.943 against
0.916/0.949 (the networks: 0.886/0.930); see ``docs/usage.md``.
"""

from __future__ import annotations

import cat_surf
import fill_voids
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from ._atlas import resolve_template_file
from .utils import bspline_grid_sample, reoriented_nifti

#: The six-class shooting templates the priors come from, coarse to fine.
#: They share the grid of CAT12's ``Template_*_GS`` (1.5 mm,
#: MNI152NLin2009cAsym) and hold GM, WM, CSF, bone, soft tissue and air.
UNIFIED_TEMPLATES = tuple(f"Template_{k}_GS_6class" for k in range(5))

#: T1Prep's setting of the segmentation: estimate the parameters on voxels
#: 2 mm apart (cat-surf: 3 mm) and allow a bias field as fine as 30 mm FWHM
#: (cat-surf: 60 mm), which 7 T images need.  The image is denoised by
#: T1Prep's initial SANLM filter before, at the resolution of the scan, so
#: the library's filter stays off.
UNIFIED_SETTINGS = {"sanlm": False, "samp": 2.0, "biasfwhm": 30.0}

#: The first pass, of which only the affine is kept (see the module notes).
#: ``nits`` below 5 is refused by the library.
AFFINE_PASS = {"sanlm": False, "samp": 8.0, "lowres": 6.0, "biasfwhm": 0.0,
               "tol": 1e-2, "nits": 10}

#: The first cat-surf release with ``vol_unified_seg``.
MIN_CAT_SURF = "1.1.9"

#: Posterior of the brain (GM + WM + CSF) at which a voxel joins the mask.
BRAIN_THRESHOLD = 0.5

#: 26-connectivity, as deepbet keeps its largest component.
_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def check_available() -> None:
    """Raise if the unified segmentation cannot run: cat-surf too old, templates missing."""
    if not hasattr(cat_surf, "vol_unified_seg"):
        raise RuntimeError(
            f"--unified-seg needs cat-surf >= {MIN_CAT_SURF}, but "
            f"{getattr(cat_surf, '__version__', 'an older version')} is installed: "
            "pip install --upgrade cat-surf"
        )
    for name in UNIFIED_TEMPLATES:
        try:
            resolve_template_file(name, ".nii.gz")
        except FileNotFoundError:
            raise RuntimeError(
                f"--unified-seg needs the six-class shooting templates, but {name}.nii.gz "
                "is not in T1Prep's template folder.  They are not part of the "
                "PyPI package yet; run T1Prep from a source checkout."
            ) from None


def unified_templates():
    """Load the six-class shooting templates.

    Returns:
        The five templates as float32 arrays (X, Y, Z, 6) in their stored
        orientation, coarse to fine, and their voxel-to-world matrix.  The
        classes of every voxel are renormalised to sum to one, which the
        rounding of the stored values may have upset by a few 1e-5.
    """
    images = [nib.load(resolve_template_file(name, ".nii.gz")) for name in UNIFIED_TEMPLATES]
    templates = []
    for img in images:
        tpl = img.get_fdata(dtype=np.float32)
        tpl /= tpl.sum(axis=-1, keepdims=True)
        templates.append(tpl)
    return templates, images[0].affine


def _voxels_from_grid(shape) -> np.ndarray:
    """4x4 matrix from ``grid_sample`` coordinates to voxel indices of a RAS array.

    ``grid_sample`` reads a (D, H, W) volume with the channels of its grid in
    reversed order -- the first one runs along W, the last axis -- and, with
    ``align_corners``, puts -1 and +1 on the centres of the corner voxels.
    """
    half = (np.asarray(shape[:3], dtype=np.float64) - 1.0) / 2.0
    matrix = np.eye(4)
    matrix[:3, :3] = 0.0
    for axis in range(3):
        matrix[axis, 2 - axis] = half[axis]
        matrix[axis, 3] = half[axis]
    return matrix


def grid_affine(world, image_mat, image_shape, grid_mat, grid_shape) -> np.ndarray:
    """The affine of deepmriprep's registration from a world-space mapping.

    Args:
        world: 4x4 mapping from image to template millimetres (SPM's
            ``Affine``, as ``vol_unified_seg`` returns it).
        image_mat, image_shape: Voxel-to-world matrix and shape of the image
            in RAS order.
        grid_mat, grid_shape: The same for the working grid, in template
            millimetres.

    Returns:
        The 4x4 matrix that maps the ``grid_sample`` coordinates of the
        working grid to those of the image (``align_corners=True``), which
        is what ``Preprocess.run_affine_register`` returns as ``affine``.
    """
    return (
        np.linalg.inv(_voxels_from_grid(image_shape))
        @ np.linalg.inv(np.asarray(image_mat, dtype=np.float64))
        @ np.linalg.inv(np.asarray(world, dtype=np.float64))
        @ np.asarray(grid_mat, dtype=np.float64)
        @ _voxels_from_grid(grid_shape)
    )


def _sampling_grid(affine: np.ndarray, shape) -> torch.Tensor:
    """``grid_sample`` positions of the voxels of a grid of *shape* under *affine*."""
    return F.affine_grid(
        torch.from_numpy(np.asarray(affine, dtype=np.float64)[None, :3]).float(),
        [1, 3, *shape[:3]],
        align_corners=True,
    )


def affine_registration(data, mat, templates, template_mat, verbose=False):
    """The first pass: the affine registration of the scan to the templates.

    Args:
        data: The unfiltered scan in RAS order, float32.
        mat: Its voxel-to-world matrix.
        templates, template_mat: :func:`unified_templates`.
        verbose: Print what the library does.

    Returns:
        ``(world, params)``: the 4x4 mapping from scan to template
        millimetres and the library's summary of the pass.
    """
    result = cat_surf.vol_unified_seg(
        data, templates, mat, template_mat, prob=False, label=False,
        corrected=False, verbose=int(verbose), **AFFINE_PASS,
    )
    return np.asarray(result["params"]["affine"], dtype=np.float64), result["params"]


def segment(t1: nib.Nifti1Image, t1_raw: nib.Nifti1Image, working_grid: nib.Nifti1Image,
            verbose: bool = False, on_step=None) -> dict:
    """Unified segmentation of the input on the 0.5 mm working grid.

    The scan goes in in RAS storage order: the lattices the affine and the
    parameters are estimated on follow the stored voxels, and a copy of the
    same scan in another storage order gave other results.

    Args:
        t1: The input denoised by T1Prep's SANLM filter: what is segmented.
        t1_raw: The input as it came, on the grid of *t1*: what the affine
            is estimated from.
        working_grid: The 0.5 mm working grid (``Preprocess.affine_template``).
        verbose: Print the stages and parameters of the fit.
        on_step: Called with ``"segmentation"`` between the two passes
            (progress reporting).

    Returns:
        A dict: ``affine`` (4x4, :func:`grid_affine`), ``prob`` (posteriors of
        the six classes times 255), ``label`` (0 background, 1 CSF, 2 GM,
        3 WM, with partial volumes) and ``corrected`` (the denoised,
        bias-corrected scan) on the RAS-ordered working grid, ``grid``
        (that grid as an image), ``world`` (scan to template millimetres),
        ``native_mat`` and ``native_shape`` (the RAS-ordered grid of the
        scan), ``params`` and ``affine_params`` (cat-surf's summaries of the
        two passes).
    """
    check_available()
    image = nib.as_closest_canonical(t1_raw)
    data = np.nan_to_num(image.get_fdata(dtype=np.float32))
    denoised = nib.as_closest_canonical(t1)
    if denoised.shape[:3] != image.shape[:3] or not np.allclose(denoised.affine, image.affine):
        raise ValueError("the denoised and the unfiltered input must share their grid")
    grid_img = nib.as_closest_canonical(working_grid)
    shape = grid_img.shape[:3]
    templates, template_mat = unified_templates()

    world, affine_params = affine_registration(
        data, image.affine, templates, template_mat, verbose=verbose
    )
    affine = grid_affine(world, image.affine, data.shape, grid_img.affine, shape)
    if on_step is not None:
        on_step("segmentation")

    # The denoised scan on the working grid.  Background that is exactly 0
    # (defaced, stripped or zero-filled) stays 0, which the library reads as
    # no data.
    del data
    data = np.nan_to_num(denoised.get_fdata(dtype=np.float32))
    on_grid = bspline_grid_sample(
        torch.from_numpy(data)[None, None],
        _sampling_grid(affine, shape),
        align_corners=True,
        prefilter=True,
        mask_value=0,
    )[0, 0].numpy()
    del data
    result = cat_surf.vol_unified_seg(
        on_grid,
        templates,
        grid_img.affine,
        template_mat,
        affine=np.eye(4),
        affine_mode="fixed",
        prob=True,
        label=True,
        corrected=True,
        verbose=int(verbose),
        **UNIFIED_SETTINGS,
    )
    del templates, on_grid
    return {
        "affine": affine,
        "prob": result["prob"],
        "label": result["label"],
        "corrected": result["corrected"],
        "grid": grid_img,
        "world": world,
        "native_mat": np.asarray(image.affine, dtype=np.float64),
        "native_shape": tuple(image.shape[:3]),
        "params": result["params"],
        "affine_params": affine_params,
    }


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    """The largest 26-connected component of *mask*, its cavities filled."""
    labels, n = ndimage.label(mask, structure=_CONNECTIVITY)
    if n == 0:
        raise RuntimeError("the unified segmentation found no brain")
    if n > 1:
        mask = labels == np.bincount(labels[labels > 0]).argmax()
    return fill_voids.fill(mask)


def brain_mask(prob: np.ndarray, threshold: float = BRAIN_THRESHOLD) -> np.ndarray:
    """The brain: where GM, WM and CSF together reach *threshold*.

    Kept as deepbet keeps its mask: the largest 26-connected component, with
    the cavities inside it filled.

    Args:
        prob: (X, Y, Z, 6) posteriors times 255, GM, WM and CSF first.
        threshold: Posterior of the brain at which a voxel belongs to it.

    Returns:
        A boolean array on the grid of *prob*.
    """
    brain = prob[..., :3].sum(axis=-1, dtype=np.uint16)
    return _largest_filled(brain >= np.uint16(np.ceil(threshold * 255)))


def to_native(mask_large: np.ndarray, seg: dict) -> np.ndarray:
    """The brain mask of the working grid on the grid of the scan (RAS order)."""
    grid = _sampling_grid(np.linalg.inv(seg["affine"]), seg["native_shape"])
    sampled = F.grid_sample(
        torch.from_numpy(mask_large.astype(np.float32))[None, None],
        grid, mode="bilinear", align_corners=True,
    )[0, 0].numpy()
    return _largest_filled(sampled >= 0.5)


def to_working_grid(mask: np.ndarray, seg: dict) -> np.ndarray:
    """A mask on the grid of the scan (RAS order), on the working grid."""
    grid = _sampling_grid(seg["affine"], seg["grid"].shape)
    return F.grid_sample(
        torch.from_numpy(mask.astype(np.float32))[None, None],
        grid, mode="bilinear", align_corners=True,
    )[0, 0].numpy() >= 0.5


def native_brain(t1: nib.Nifti1Image, mask: np.ndarray):
    """The skull-stripped image and its mask, as the skull-stripping stage returns them.

    Args:
        t1: The denoised input.
        mask: Its brain mask in RAS order.

    Returns:
        ``(brain, mask)`` in the storage order, affine and header of *t1*;
        the mask as uint8.
    """
    data = nib.as_closest_canonical(t1).get_fdata(dtype=np.float32)
    data[~mask] = 0.0
    brain = reoriented_nifti(data, t1.affine, t1.header)
    mask_img = reoriented_nifti(mask.astype(np.uint8), t1.affine, t1.header)
    mask_img.header.set_data_dtype(np.uint8)
    return brain, mask_img


def working_grid_maps(seg: dict, mask_large: np.ndarray):
    """Hand the segmentation on in the form of deepmriprep's stages.

    Args:
        seg: Result of :func:`segment`.
        mask_large: The brain mask on the working grid.

    Returns:
        ``(affine, brain_large, p0_large)``: the affine as a 4x4 DataFrame and
        the corrected brain and the label map on the working grid, both cut
        to the mask.
    """
    grid_img = seg["grid"]
    brain = np.where(mask_large, np.nan_to_num(seg["corrected"]), 0.0).astype(np.float32)
    label = np.where(mask_large, seg["label"], 0.0).astype(np.float32)
    header_fp32 = grid_img.header.copy()
    header_fp32.set_data_dtype(np.float32)
    return (
        pd.DataFrame(seg["affine"]),
        reoriented_nifti(brain, grid_img.affine, header_fp32),
        reoriented_nifti(label, grid_img.affine, grid_img.header),
    )


def report_entry(params: dict, affine_params: dict = None, world=None) -> dict:
    """What the run report records of the unified segmentation.

    Args:
        params: cat-surf's summary of the segmentation (second pass).
        affine_params: Its summary of the affine registration (first pass).
        world: The affine from scan to template millimetres.  The volumes of
            the classes are cat-surf's, measured on the working grid in
            template space; they are scaled to the scan with it.  Bone, soft
            tissue and air are counted inside the box of the working grid
            only.
    """
    shooting = params.get("shooting", {})
    affine_params = affine_params or {}
    scale = 1.0
    if world is not None:
        scale = 1.0 / abs(float(np.linalg.det(np.asarray(world, dtype=np.float64)[:3, :3])))
    volumes = [round(v * scale, 2) for v in params.get("volume_ml", [])]
    return {
        "method": "unified",
        "templates": UNIFIED_TEMPLATES[0].replace("_0_", "_<k>_"),
        "resolution_mm": 0.5,
        "denoising": "SANLM at the resolution of the scan, before resampling",
        "settings": dict(UNIFIED_SETTINGS),
        "affine_mutual_information": affine_params.get("affine_mutual_information"),
        "log_likelihood": params.get("ll"),
        "noise_estimated": affine_params.get("noise_estimated"),
        "volume_native_ml": dict(zip(params.get("classes", []), volumes)),
        "shooting_iterations": shooting.get("iterations"),
        "time_s": (params.get("time_s", {}).get("total") or 0.0)
        + (affine_params.get("time_s", {}).get("total") or 0.0),
    }
