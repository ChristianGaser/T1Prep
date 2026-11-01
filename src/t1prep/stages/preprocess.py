"""T1Prep's ``Preprocess``: deepmriprep's stages as one object.

=========================  ===================================================
``run_bet``                deepbet's networks (:mod:`t1prep.stages.skullstrip`)
``run_affine_register``    deepmriprep's affine registration (torchreg), with
                           the B-spline resampling in libCAT
``run_segment_brain``      deepmriprep's networks (:mod:`t1prep.stages.segmentation`)
``run_warp_register``      geodesic shooting (:mod:`t1prep._shoot`)
``run_atlas_register``     atlas labels in native space and ROI tissue volumes
=========================  ===================================================

The networks run on the device T1Prep selected, the registrations on
:func:`t1prep._device.stage_device`.  ``segment.py`` drives the stages and
everything T1Prep adds between them.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchreg.affine import AffineRegistration
from torchreg.metrics import dice_score
from torchreg.utils import INTERP_KWARGS

from .._atlas import (
    resolve_atlas,
    resolve_template_file,
    roi_table,
    roi_tissue_volumes,
    shape_from_to,
    warp_atlas,
)
from .._device import interpolate, release_cache, stage_device
from .._shoot import shoot_register
from ..utils import bspline_grid_sample, nifti_to_tensor, reoriented_nifti, smart_round
from .segmentation import BrainSegmentation
from .skullstrip import BrainExtraction

__all__ = ["Preprocess"]

#: deepmriprep's affine registration (its ``AFFINE_KWARGS``): Adam over
#: translation, rotation, zoom and shear at two scales of the working grid.
AFFINE_SETTINGS = {
    "scales": (24, 12),
    "iterations": (500, 100),
    "with_shear": True,
    "verbose": False,
    "learning_rate": 1e-3,
    "padding_mode": "zeros",
}


class MSEAndDice:
    """deepmriprep's affine cost on two-channel images.

    ``(1 - alpha)`` times the mean squared difference of the second channels
    (the masks) minus ``alpha`` times the Dice score of the first (the
    images), exactly as deepmriprep weighs them.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(self, x, y):
        mse = ((x[:, 1:] - y[:, 1:]) ** 2).mean()
        return (1 - self.alpha) * mse - self.alpha * dice_score(x[:, :1], y[:, :1])


def _working_grid_template(affine_template, warp_template):
    """deepmriprep's 0.5 mm affine template, with the geometry of its content.

    ``Template_05mm_bet`` is the 1.5 mm warp template upsampled three times
    with the corner voxel centres aligned, so its 339x411x339 voxels span the
    warp template's 168x204x168 mm between those centres -- not the
    169x205x169 mm deepmriprep's 0.5 mm header claims.  Registration, warp and
    ``_to_template_space`` work in normalised coordinates and follow the
    content, but millimetres read from that header were 0.6% too large:
    atlases and priors placed through the affine landed up to 1 mm off, and
    every voxel size handed to a filter or a size threshold was overstated.
    Volumes were right either way, because ``wj_affine`` divides by the same
    box.

    The returned image shares the file's data; only its affine and header
    change (voxel size 168/338, 204/410, 168/338 mm).
    """
    template = nib.as_closest_canonical(affine_template)
    warp = nib.as_closest_canonical(warp_template)
    first = warp.affine[:3, 3]
    last = warp.affine[:3, :3] @ (np.asarray(warp.shape[:3]) - 1.0) + first
    mat = np.eye(4)
    mat[:3, :3] = np.diag((last - first) / (np.asarray(template.shape[:3]) - 1.0))
    mat[:3, 3] = first
    return nib.Nifti1Image(template.dataobj, mat, template.header)


class Preprocess:
    """deepmriprep's stages as T1Prep runs them.

    Args:
        no_gpu: Keep every stage on the CPU.
        affine_template: The 0.5 mm working grid; default: deepmriprep's
            affine template with the geometry of its content.
        warp_template: The 1.5 mm template grid of the deformations
            (CAT12's ``Template_4_GS``).
    """

    def __init__(self, no_gpu: bool = False, affine_template=None, warp_template=None):
        if warp_template is None:
            warp_template = nib.load(resolve_template_file("Template_4_GS", ".nii.gz"))
        if affine_template is None:
            affine_template = _working_grid_template(
                nib.load(resolve_template_file("Template_05mm_bet", ".nii.gz")),
                warp_template,
            )
        self.affine_template = affine_template
        self.warp_template = warp_template
        self.device = torch.device("cpu") if no_gpu else stage_device()
        self.brain_extract = BrainExtraction(no_gpu)
        self.brain_segment = BrainSegmentation(no_gpu)

    def run_bet(self, t1):
        """Skull-strip *t1*: ``{"brain", "mask"}`` on its grid."""
        brain, mask = self.brain_extract(t1)
        self.brain_extract.release()
        return {"brain": brain, "mask": mask}

    def run_affine_register(self, brain_nifti, mask_nifti):
        """Affine registration of the brain onto the working grid.

        As deepmriprep's, except that the brain is resampled with T1Prep's
        :func:`~t1prep.utils.bspline_grid_sample`: libCAT on the CPU, the same
        samples as ``spline_resize``'s 27-tap torch gather in ~0.4 s instead
        of ~6 s, and more accurate; on an accelerator still ``spline_resize``.

        Returns:
            ``{"affine", "brain_large", "mask_large", "affine_loss"}``: the
            4x4 matrix mapping the working grid's normalised coordinates
            into the input's, the brain and its mask on the working grid, and
            the final cost.
        """
        align = INTERP_KWARGS["align_corners"]
        brain = nifti_to_tensor(brain_nifti)[None, None]
        mask = nifti_to_tensor(mask_nifti)[None, None]
        brain = torch.nan_to_num(brain, nan=0)
        max_value = np.quantile(brain[mask.bool()], 0.95)
        x = torch.cat([(brain / max_value).clip(max=1.0), mask], dim=1)
        template = nifti_to_tensor(self.affine_template)[None, None]
        # nifti_to_tensor() left a float32 copy of the template (180 MB) in
        # its cache; it is never read again
        self.affine_template.uncache()
        y = torch.cat([template, (template > 0.0).float()], dim=1)
        registration = AffineRegistration(
            **AFFINE_SETTINGS, dissimilarity_function=MSEAndDice(0.5)
        )
        registration(x.to(self.device), y.to(self.device), return_moved=False)
        affine = registration.get_affine().detach()
        grid = F.affine_grid(affine, [1, 3, *template.shape[-3:]], align_corners=align)
        mask_large = F.grid_sample(
            mask.to(self.device), grid, align_corners=align, mode="bilinear"
        )[0, 0]
        brain_large = bspline_grid_sample(
            brain.to(self.device),
            grid,
            align_corners=align,
            prefilter=True,
            mask_value=0,
        )[0, 0]
        release_cache(self.device)
        template_affine, header = (
            self.affine_template.affine,
            self.affine_template.header,
        )
        header_fp32 = header.copy()
        header_fp32.set_data_dtype(np.float32)
        return {
            "affine": pd.DataFrame(
                np.vstack((affine[0].cpu().numpy(), [[0, 0, 0, 1]]))
            ),
            "brain_large": reoriented_nifti(
                brain_large.cpu().numpy(), template_affine, header_fp32
            ),
            "mask_large": reoriented_nifti(
                mask_large.cpu().numpy(), template_affine, header
            ),
            "affine_loss": pd.Series([registration._loss]),
        }

    def run_segment_brain(self, brain_large, mask_large):
        """The label map ``p0`` on the working grid: ``{"p0_large"}``.

        deepmriprep also resampled ``p0_large`` back to native space by
        quadratic B-spline (~4 s at 0.5 mm); T1Prep writes the native label
        map later by linear interpolation, which avoids the negative overshoot
        spline sampling produces on a label image.
        """
        brain = nifti_to_tensor(brain_large)
        mask = nifti_to_tensor(mask_large)
        p0_large = self.brain_segment(brain[None, None].to(self.device))[0, 0]
        self.brain_segment.release()
        p0_large[mask == 0.0] = 0.0
        return {
            "p0_large": reoriented_nifti(
                p0_large.cpu().numpy(),
                self.affine_template.affine,
                self.affine_template.header,
            )
        }

    def run_warp_register(self, p1_affine, p2_affine, wj_affine, **options):
        """Non-linear registration by geodesic shooting, not deepmriprep's network.

        Returns the keys deepmriprep's warp returned and T1Prep reads, in the
        same convention (see :func:`t1prep._shoot.shoot_register`); *options*
        replace the shooting's settings.
        """
        return shoot_register(p1_affine, p2_affine, wj_affine, **options)

    def run_atlas_register(
        self, t1, affine, warp_yx, p1_large, p2_large, p3_large, atlas_list, wj_affine
    ):
        """Atlases and their ROI tissue volumes, for each name in *atlas_list*.

        ``<atlas>`` gives the atlas in native space, ``<atlas>_affine`` on the
        working grid, ``<atlas>_volumes`` a table of GM, WM, CSF and region
        volumes (cm^3, in native space through *wj_affine*).  An atlas is a
        name in T1Prep's template folder or the absolute path of a NIfTI with
        its region table beside it.
        """
        align = INTERP_KWARGS["align_corners"]
        voxel_vol = np.prod(p1_large.affine[np.diag_indices(3)])
        p_large = [
            nifti_to_tensor(p).to(self.device) for p in (p1_large, p2_large, p3_large)
        ]
        inv_affine = torch.linalg.inv(
            torch.from_numpy(affine.values).float().to(self.device)
        )
        native_grid = F.affine_grid(
            inv_affine[None, :3], [1, 3, *t1.shape[:3]], align_corners=align
        )
        warp_yx = nib.as_closest_canonical(warp_yx)
        yx = nifti_to_tensor(warp_yx)[None].to(self.device)
        # one deformation per atlas resolution, shared by the atlases at it
        warps, out = {}, {}
        names = [
            a.rsplit("_", 1)[0] if a.endswith(("_affine", "_volumes")) else a
            for a in atlas_list
        ]
        for name in dict.fromkeys(names):
            atlas_path, csv_path = resolve_atlas(name)
            atlas = nib.as_closest_canonical(nib.load(atlas_path))
            template_ids = np.unique(np.asarray(atlas.dataobj)).astype(np.int64)
            header = atlas.header
            shape = tuple(shape_from_to(atlas, warp_yx))
            if shape not in warps:
                scaled = interpolate(
                    yx.permute(0, 4, 1, 2, 3),
                    shape,
                    mode="trilinear",
                    align_corners=False,
                )
                warps[shape] = scaled.permute(0, 2, 3, 4, 1)
            atlas = warp_atlas(atlas, warps[shape], self.device)
            if f"{name}_affine" in atlas_list:
                out[f"{name}_affine"] = atlas
            atlas_tensor = nifti_to_tensor(atlas).to(self.device)
            if f"{name}_volumes" in atlas_list:
                # one row per label of the atlas, named from its table
                rois = roi_table(csv_path, template_ids)
                volumes = voxel_vol * roi_tissue_volumes(
                    atlas_tensor, *p_large, rois["ROIid"]
                )
                volumes *= wj_affine[0] / 1000
                volumes = np.vectorize(smart_round)(volumes)
                volumes = pd.DataFrame(
                    volumes, columns=["gmv_cm3", "wmv_cm3", "csfv_cm3", "region_cm3"]
                )
                out[f"{name}_volumes"] = pd.concat([rois, volumes], axis=1)
            if name in atlas_list:
                native = F.grid_sample(
                    atlas_tensor[None, None],
                    native_grid,
                    mode="nearest",
                    align_corners=align,
                )
                out[name] = reoriented_nifti(
                    native[0, 0].cpu().numpy(), t1.affine, header
                )
        return out
