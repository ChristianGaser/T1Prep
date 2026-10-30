"""Non-linear registration to MNI152NLin2009cAsym by geodesic shooting.

T1Prep's affine-registered GM and WM maps on the 1.5 mm template grid are
registered onto CAT12's shooting templates (``Template_0..4_GS``) by
``cat_surf.vol_shoot``, CAT-Surface's C port of CAT26's optimised geodesic
shooting (``run_Shooting`` in ``cat_main_registration.m``, on SPM's shooting
primitives).  It replaces deepmriprep's warp network, which imitated CAT12's
deformations and fell short of them: on Mindboggle-101 the shooting carries
manual cortical labels between subjects with a Dice of 0.742 (leave-one-out)
with the default setting and 0.770 with ``--fmriprep``'s, against the
network's 0.729 and CAT12's own pipeline's 0.741; see
``evaluation/results/README.md``.

The deformations are handed on in the form the rest of T1Prep consumes, which
is deepmriprep's: ``grid_sample`` grids on the template grid, in normalised
coordinates (``align_corners=True``, -1 and +1 at the corner voxel centres)
with the channels in reversed axis order, stored through ``reoriented_nifti``.
"""

from __future__ import annotations

import cat_surf
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from deepbet.utils import reoriented_nifti
from scipy.ndimage import distance_transform_edt

from ._atlas import resolve_template_file

#: CAT12's shooting templates for MNI152NLin2009cAsym, coarse to fine.
SHOOT_TEMPLATES = tuple(f"Template_{k}_GS" for k in range(5))

#: T1Prep's default, CAT12's own setting: ``regstr`` 0.5 (on to the next
#: resolution after about three Gauss-Newton iterations) with CAT12's
#: regularisation, the setting for VBM.  Spelled out rather than left to
#: cat-surf's defaults, so that the registration stays the evaluated one.
SHOOT_SETTINGS = {"regstr": 0.5, "reg_scale": 1.0}

#: The setting ``--fmriprep`` runs: every scheduled iteration with half of
#: CAT12's regularisation, the best label accuracy in T1Prep's evaluation
#: (+0.028 Dice over the default on Mindboggle-101).  Not for VBM: its
#: Jacobians vary more between subjects, which the modulated maps inherit
#: (in 60 children and 60 adults the mean T for age fell by 12 % and 9 %).
SHOOT_SETTINGS_FMRIPREP = {"regstr": 1.0, "reg_scale": 0.5}


def shooting_templates():
    """Load CAT12's shooting templates.

    Returns:
        The five templates as float32 arrays (X, Y, Z, K) in their stored
        orientation, coarse to fine, and their voxel-to-world matrix.
    """
    images = [nib.load(resolve_template_file(name, ".nii.gz")) for name in SHOOT_TEMPLATES]
    return [img.get_fdata(dtype=np.float32) for img in images], images[0].affine


def _canonical(array, affine):
    """Reorder the first three axes of *array* from *affine*'s storage to RAS."""
    return nib.apply_orientation(array, nib.io_orientation(affine))


def _normalised(mm, affine, shape):
    """World millimetres -> ``grid_sample`` coordinates of the grid (*affine*, *shape*).

    -1 and +1 are the centres of the grid's corner voxels (``align_corners``)
    in RAS order, and the channels come out reversed (z, y, x), as
    ``grid_sample`` indexes a (D, H, W) volume.
    """
    ornt = nib.io_orientation(affine)
    ras = np.asarray(affine, dtype=np.float64) @ nib.orientations.inv_ornt_aff(ornt, shape)
    first = ras[:3, 3]
    last = ras[:3, :3] @ (np.asarray(shape[:3], dtype=np.float64) - 1.0) + first
    norm = 2.0 * (mm - first) / (last - first) - 1.0
    return np.ascontiguousarray(norm[..., ::-1], dtype=np.float32)


def _fill_undefined(grid):
    """Give each undefined position the displacement of the nearest defined one.

    SPM's inverse (``invdef``) leaves the positions no template voxel maps to
    undefined (NaN): ~3% of the grid, nearly all within two voxels of its
    border.  The atlas warping and the ITK inverse transform need a finite
    field, as deepmriprep's always was.
    """
    bad = ~np.isfinite(grid).all(axis=-1)
    if not bad.any():
        return grid
    shape = grid.shape[:3]
    axes = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape]
    identity = np.stack(np.meshgrid(*axes, indexing="ij")[::-1], axis=-1)
    nearest = tuple(i[bad] for i in distance_transform_edt(
        bad, return_distances=False, return_indices=True))
    filled = grid.copy()
    filled[bad] = identity[bad] + (grid - identity)[nearest]
    return filled


def shoot_register(p1_affine, p2_affine, wj_affine, **options):
    """Register the affine GM and WM maps onto the templates by geodesic shooting.

    Unless *options* say otherwise, the shooting runs with
    :data:`SHOOT_SETTINGS`.  The warped maps are pulled through the
    deformation as deepmriprep pulled them (trilinear, reflected at the
    border), and the modulated ones multiply them by the Jacobian determinant
    of the deformation and the affine volume factor.

    Args:
        p1_affine: GM probability on the 1.5 mm template grid
            (``_to_template_space``).
        p2_affine: WM probability on the same grid.
        wj_affine: Native volume per template volume of the affine
            registration (a scalar or a one-element series).
        **options: Keywords of ``cat_surf.vol_shoot`` that replace or add to
            :data:`SHOOT_SETTINGS` (``regstr``, ``reg_scale``,
            ``n_threads``, ...).

    Returns:
        A dict of NIfTIs: ``warp_xy`` (for every template voxel the position
        in the affine maps it pulls from) and ``warp_yx`` (for every voxel of
        the affine maps its position in template space) as ``grid_sample``
        grids, the warped maps ``wp1``/``wp2``, the modulated ones
        ``mwp1``/``mwp2`` and ``jac``, the Jacobian determinant of the
        deformation (affine-space mm per template mm) on the template grid;
        plus ``info``, the shooting's own
        summary (iterations per level, Jacobian range).
    """
    gm = np.asarray(p1_affine.dataobj, dtype=np.float32)
    wm = np.asarray(p2_affine.dataobj, dtype=np.float32)
    mat = np.asarray(p1_affine.affine, dtype=np.float64)
    templates, tmat = shooting_templates()
    tshape = templates[0].shape[:3]
    res = cat_surf.vol_shoot(
        gm, wm, templates, mat, tmat, inverse=True, **{**SHOOT_SETTINGS, **options}
    )
    del templates

    # y: for every template voxel the input voxel it pulls from; iy: for every
    # input voxel the template voxel it maps to (both 0-based, stored order)
    y_mm = res["y"] @ mat[:3, :3].T + mat[:3, 3]
    warp_xy = _normalised(_canonical(y_mm, tmat), mat, gm.shape)
    iy_mm = res["iy"] @ tmat[:3, :3].T + tmat[:3, 3]
    warp_yx = _fill_undefined(_normalised(_canonical(iy_mm, mat), tmat, tshape))
    # The shooting's Jacobian is in voxel units; in millimetres it is free of
    # how the two grids store their axes (maps stored RAS against CAT's LAS
    # templates would flip its sign) and of their voxel sizes
    jac = _canonical(res["jac"], tmat) * (np.linalg.det(mat[:3, :3]) / np.linalg.det(tmat[:3, :3]))
    del y_mm, iy_mm

    maps = torch.from_numpy(
        np.stack([_canonical(gm, mat), _canonical(wm, mat)]).astype(np.float32)
    )[None]
    wp = F.grid_sample(
        maps, torch.from_numpy(warp_xy)[None], align_corners=True, padding_mode="reflection"
    )[0].numpy()
    mwp = wp * jac[None] * float(np.asarray(wj_affine, dtype=np.float64).ravel()[0])

    header_fp32 = p1_affine.header.copy()
    header_fp32.set_data_dtype(np.float32)
    header_uint8 = p1_affine.header.copy()
    header_uint8.set_data_dtype(np.uint8)
    return {
        "warp_xy": reoriented_nifti(warp_xy, tmat, header_fp32),
        "warp_yx": reoriented_nifti(warp_yx, mat, header_fp32),
        "wp1": reoriented_nifti(wp[0], tmat, header_uint8),
        "wp2": reoriented_nifti(wp[1], tmat, header_uint8),
        "mwp1": reoriented_nifti(mwp[0], tmat, header_fp32),
        "mwp2": reoriented_nifti(mwp[1], tmat, header_fp32),
        "jac": reoriented_nifti(jac, tmat, header_fp32),
        "info": res["info"],
    }
