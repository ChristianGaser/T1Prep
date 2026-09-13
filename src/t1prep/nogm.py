"""Conventional removal of non-cortical grey matter (the ``nogm`` step).

``deepmriprep`` refines its label map with a learned "no grey matter" model
(``segmentation_nogm_model.pt``): a UNet is run over two overlapping
128x288x256 patches of ``p0_large`` and, wherever it fires, the grey matter of
that voxel is deleted and split half into white matter and half into CSF
(``deepmriprep/segment.py``, ``NoGMSegmentation.apply_nogm``).  This module
reproduces that decision with explicit anatomy instead, so the two patch
forwards -- 89% of the stage's wall clock and effectively all of its 5.6 GB
peak -- can be skipped.  It is T1Prep's default; ``--nogm-model`` runs the
UNet instead.

What the model actually does
----------------------------
Measured against the model's own output on a 0.5 mm subject (see
:data:`SANDWICH_MIN` for the numbers), the mask is not a diffuse correction.
It removes grey-matter-valued voxels from a short list of places:

    region                     GM voxels removed
    4th ventricle                     85.5 %
    brain stem                        61.4 %
    lateral ventricles             38 - 57 %
    ventral diencephalon           21 - 24 %
    3rd ventricle                      8.1 %
    cerebral white matter          1.2 - 1.6 %
    every cortical region             < 0.3 %

That is the classic partial-volume failure: a voxel that is half white matter
and half CSF has grey-matter intensity, so an intensity-driven segmentation
labels the ventricle rims, the brain stem and the periventricular white matter
as grey matter.  There is no cortex there to protect, which is why CAT12
handles the same problem geometrically.

The CAT12 strategy this follows
-------------------------------
``cat_vol_partvol.m`` builds an individual label map ``Ya1`` from an atlas plus
the tissue classes, detects the ventricles by Laplace region growing from the
atlas ``LAB.VT`` seed against a non-ventricle counter-seed (l. 451-514), and
hands the result on as the filling mask ``YMF`` (l. 921).  ``cat_main`` then
fills that region (``cat_surf_createCS_fun.m``, ``fillVentricle``) so the
surface never follows periventricular grey matter, and ``cat_main1639.m``
l. 1417-1431 shows the cheap version of the same construction: atlas
ventricle + basal ganglia, Laplace-relaxed against white matter, closed and
smoothed.

Two parts of that transfer directly and are used here: the **atlas admission
region**, which is what keeps the correction away from cortex, and the
**tissue-composition test** inside it.  What is not ported is CAT12's full
``Ya1`` partitioning, which T1Prep does not have; as in :mod:`t1prep.vessels`,
the equivalent constraint comes from the Neuromorphometrics atlas resampled
through its real affine.

The test itself
---------------
A grey-matter-valued voxel is a white-matter/CSF partial volume when its
neighbourhood contains **both** white matter and CSF.  Measured medians inside
the regions the model targets, at a 2 mm box:

    group                    p0    f_WM   f_CSF
    brain stem, removed    1.98    0.36    0.36
    brain stem, kept       2.00    0.15    0.06
    ventricles, removed    1.96    0.37    0.31
    ventricles, kept       1.95    0.03    0.24
    cerebral WM, removed   2.06    0.42    0.25
    cerebral WM, kept      2.01    0.24    0.00

``p0`` is identical in both columns, so this is geometry and not intensity:
``min(f_WM, f_CSF)`` separates the two cleanly while the label value does not.

The confound that costs the most is that **the cortical ribbon is also a
white-matter/CSF sandwich** -- grey matter with white matter beneath and CSF
above is the definition of cortex.  What separates them is how much grey
matter is around: a cortical voxel sits inside a 2.5 mm band of its own kind,
a periventricular partial volume sits in a one-voxel film.  Hence the
:data:`GM_FRACTION_MAX` ceiling, which is worth 0.03-0.09 Dice on its own and
removes the pre-/postcentral gyrus false positives that dominate without it.

What the correction cannot affect
---------------------------------
The reassignment is *exactly* neutral on the label map, which bounds what
replacing it can break.  ``segment.py`` rebuilds ``p0_large`` from the tissue
maps as ``csf + 2 * gm + 3 * wm``, and sending ``gm -> 0``, ``wm -> wm + gm/2``,
``csf -> csf + gm/2`` changes it by

    (gm/2) * 1  -  gm * 2  +  (gm/2) * 3  =  0

for any ``gm``, with no thresholds involved.  Two full pipeline runs on the
same subject, one per method, confirmed it: the native ``p0`` maps came out
bit-identical.  So neither the surfaces nor cortical thickness can move,
whichever way this step is computed; only ``p1``/``p2``/``p3`` change, and with
them the modulated warped maps and the reported tissue volumes.  On that
subject the reported volumes moved by 0.05% (GM 734.76 -> 735.13 mL) and TIV
not at all.

Accuracy
--------
Against the model's own mask on one 0.5 mm subject the agreement is Dice
0.74 (sensitivity 0.71, precision 0.78, 12.3 vs 13.7 cm^3 corrected).  The
model is the reference only in the sense of being the thing replaced --
neither map is ground truth.  The constants were chosen on a plateau rather
than a peak (every setting in the sweep scored 0.72-0.74), so they are not
knife-edge.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    label as _connected_components,
    uniform_filter,
)

from ._segment_utils import (
    _resample_to,
    _resolve_template_file,
    get_regions_mask,
)

__all__ = [
    "noncortical_gm_regions",
    "noncortical_gm_mask",
    "apply_noncortical_gm",
    "run_segment_nogm_conventional",
]


# ---------------------------------------------------------------------------
# Tissue thresholds on the p0 label scale (background 0, CSF 1, GM 2, WM 3).
#
# These are the detector's own notion of "is white matter here", deliberately
# separate from the tent functions ``one_hot`` uses to build p1/p2/p3: the
# neighbourhood test needs a crisp yes/no per voxel, while the tissue maps
# need the partial volume preserved.  The midpoints between tissue classes are
# used, so a voxel counts as white matter only once it is past the GM/WM
# boundary.
# ---------------------------------------------------------------------------
CSF_RANGE = (0.05, 1.25)
GM_RANGE = (1.25, 2.75)
WM_MIN = 2.75

#: Radius of the box the tissue composition is measured over, in mm.
#: 2.0 mm was measured against 2.5 and 3.0: the larger boxes blur the thin
#: periventricular film into its surroundings and cost 0.05-0.13 Dice at every
#: threshold.  Cortical grey matter is ~2.5 mm thick, so this stays inside one
#: ribbon and the :data:`GM_FRACTION_MAX` ceiling can still tell them apart.
SANDWICH_RADIUS_MM = 2.0

#: Minimum of the white-matter and CSF neighbourhood fractions required to
#: call a voxel a partial volume rather than tissue.  Using the minimum rather
#: than the product or the sum is what makes it a *sandwich* test: both
#: neighbours have to be present, and neither can substitute for the other.
#: Measured 0.36/0.36 (brain stem), 0.37/0.31 (ventricles) and 0.42/0.25
#: (cerebral WM) for voxels the model removes, against 0.15/0.06, 0.03/0.24
#: and 0.24/0.00 for those it keeps.
SANDWICH_MIN = 0.12

#: Ceiling on the grey-matter fraction of the same neighbourhood.  This is the
#: cortex guard: without it the pre- and postcentral gyri, where the ribbon is
#: thinnest, contribute the largest block of false positives.
GM_FRACTION_MAX = 0.55

#: Margin grown around the atlas admission region, in mm.  The atlas is warped
#: only affinely here, so some slack is needed; 1.5 mm measured best against
#: 0, 3 and 5 mm.  99.6% of the model's mask falls inside the grown region.
ADMISSION_MARGIN_MM = 1.5

#: Connected components smaller than this are dropped, in mm^3.  CAT12 does the
#: same with ``cat_vol_morph(..., 'l', [10 0.1])`` after its region growing.
#: The effect is small (+0.002 Dice) but it removes speckle that would
#: otherwise punch isolated holes into the tissue maps.
MIN_COMPONENT_MM3 = 20.0

#: Regions where grey matter is a partial-volume artefact rather than tissue.
#: This is the admission region -- nothing outside it is ever corrected, which
#: is what makes the correction safe for the cortex.  It mirrors the CAT12
#: labels that feed ``YMF``: ``LAB.VT`` (ventricles), ``LAB.BS`` (brain stem)
#: and the diencephalon, plus the cerebral white matter for the
#: periventricular rim.
NONCORTICAL_REGIONS = (
    "Brain Stem",
    "3rd Ventricle",
    "4th Ventricle",
    "Left Lateral Ventricle",
    "Right Lateral Ventricle",
    "Left Inf Lat Vent",
    "Right Inf Lat Vent",
    "Left Ventral DC",
    "Right Ventral DC",
    "Left Cerebral White Matter",
    "Right Cerebral White Matter",
)

# The cerebellum is deliberately *not* in that list, for the same reason
# :mod:`t1prep.vessels` gives it the widest protection margin: its folia are
# thin laminae separated by CSF, so every cerebellar voxel looks like a
# white-matter/CSF sandwich and the test cannot be trusted there.  The model
# removes only 6.5% of cerebellar white-matter grey matter, and admitting the
# region cost 0.013 Dice while contributing 18% of all false positives.


def _tissue_fractions(p0, vx):
    """Neighbourhood tissue composition, as fractions in ``[0, 1]``.

    Args:
        p0: Label map on the 0..3 scale.
        vx: Voxel size in mm (isotropic, or the smallest edge).

    Returns:
        ``(f_wm, f_csf, f_gm)``, each the fraction of the surrounding box
        occupied by that tissue.
    """
    radius = max(1, int(round(SANDWICH_RADIUS_MM / float(vx))))
    size = 2 * radius + 1
    wm = (p0 > WM_MIN).astype(np.float32)
    csf = ((p0 > CSF_RANGE[0]) & (p0 < CSF_RANGE[1])).astype(np.float32)
    gm = ((p0 > GM_RANGE[0]) & (p0 < GM_RANGE[1])).astype(np.float32)
    return (
        uniform_filter(wm, size),
        uniform_filter(csf, size),
        uniform_filter(gm, size),
    )


def noncortical_gm_regions(target_affine, target_shape, device="cpu"):
    """Atlas region where grey matter is treated as a partial-volume artefact.

    Resamples Neuromorphometrics through its real affine -- as
    :func:`t1prep.vessels.protected_regions` does, and for the same reason: a
    plain resize assumes a shared field of view, which this atlas does not
    share with the working grid, and a few millimetres of offset is the
    difference between guarding the cortex and guarding most of it.

    Args:
        target_affine: Affine of the grid to build the mask on.
        target_shape: Shape of that grid.
        device: Device used for the resampling.  This is the only torch op in
            the module and there is nothing to gain by moving it: it is a
            single nearest-neighbour sample of a label volume, and everything
            downstream is numpy.  MPS has no ``grid_sampler_3d`` kernel, so
            passing it there works only with ``PYTORCH_ENABLE_MPS_FALLBACK``
            set -- which lands the work back on the CPU regardless.

    Returns:
        Boolean array, ``True`` where correction is permitted.
    """
    target_shape = tuple(int(v) for v in np.asarray(target_shape)[:3])
    target_affine = np.asarray(target_affine, dtype=float)
    labels = _resample_to(
        nib.load(_resolve_template_file("Neuromorphometrics", ".nii.gz")),
        target_affine,
        target_shape,
        device=device,
        nearest=True,
    )
    atlas = nib.Nifti1Image(np.round(labels).astype(np.int16), target_affine)
    mask = get_regions_mask(atlas, "Neuromorphometrics", list(NONCORTICAL_REGIONS))

    vx = np.sqrt((target_affine[:3, :3] ** 2).sum(axis=0))
    iters = max(1, int(round(ADMISSION_MARGIN_MM / float(min(vx)))))
    return binary_dilation(mask, generate_binary_structure(3, 3), iters)


def noncortical_gm_mask(p0_large, device="cpu", admission=None, verbose=False):
    """Grey matter to be reassigned, the conventional equivalent of ``nogm``.

    A voxel is admitted only when all four conditions hold: it is
    grey-matter-valued, it lies inside the atlas admission region, its
    neighbourhood holds both white matter and CSF, and that neighbourhood is
    not itself mostly grey matter.  The last two are the partial-volume test
    and the cortex guard respectively; see the module docstring for the
    measurements behind them.

    Args:
        p0_large: Label map on the working grid, as a NIfTI image.
        device: Device used to resample the atlas.
        admission: Precomputed admission region, to avoid resampling the atlas
            twice when the caller already has one.  ``None`` builds it.
        verbose: Print the corrected volume.

    Returns:
        Boolean array with the shape of *p0_large*.
    """
    p0 = np.asanyarray(p0_large.dataobj, dtype=np.float32)
    vx = np.sqrt((np.asarray(p0_large.affine)[:3, :3] ** 2).sum(axis=0))

    if admission is None:
        try:
            admission = noncortical_gm_regions(
                p0_large.affine, p0.shape, device=device
            )
        except (FileNotFoundError, OSError) as exc:
            # Same posture as the blood-vessel prior in segment.py: degrade
            # rather than abort.  Without the atlas the geometric test still
            # works, it just loses the cortex guard the region provides.
            if verbose:
                print(f"Atlas unavailable ({exc}); continuing without it")
            admission = np.ones(p0.shape, dtype=bool)

    f_wm, f_csf, f_gm = _tissue_fractions(p0, min(vx))
    mask = (
        (p0 > GM_RANGE[0])
        & (p0 < GM_RANGE[1])
        & admission
        & (np.minimum(f_wm, f_csf) > SANDWICH_MIN)
        & (f_gm < GM_FRACTION_MAX)
    )

    voxel_vol = float(np.prod(vx))
    if MIN_COMPONENT_MM3 > 0 and mask.any():
        labelled, n = _connected_components(mask, generate_binary_structure(3, 3))
        if n:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            keep = np.flatnonzero(sizes * voxel_vol >= MIN_COMPONENT_MM3)
            mask = np.isin(labelled, keep)

    if verbose:
        print(f"Non-cortical GM removed: {mask.sum() * voxel_vol / 1000:.1f} cm^3")
    return mask


def _one_hot_gm_wm_csf(p0):
    """Tissue maps from a label map, matching ``deepmriprep.segment.one_hot``.

    Upstream builds four tent functions over the classes 0..3 and then keeps
    channels 2, 3 and 1.  Only those three are ever used, so they are built
    directly here -- the discarded background channel is 188 MB at 0.5 mm.

    Args:
        p0: Label map on the 0..3 scale.

    Returns:
        ``(gm, wm, csf)`` float32 arrays in ``[0, 1]``.
    """
    p0c = np.clip(p0, None, 3.0)
    out = []
    for centre in (2, 3, 1):  # GM, WM, CSF
        chan = np.zeros_like(p0c)
        sel = (p0c > centre - 1) & (p0c <= centre + 1)
        chan[sel] = 1.0 - np.abs(p0c[sel] - centre)
        out.append(chan)
    return tuple(out)


def apply_noncortical_gm(gm, wm, csf, mask):
    """Reassign the flagged grey matter, matching ``apply_nogm``.

    Upstream zeroes the grey matter and adds half of it to each of the other
    two classes, which keeps the three maps summing to what they summed to
    before.  Done in place; the arrays are hundreds of megabytes each.

    Args:
        gm: Grey-matter map, modified in place.
        wm: White-matter map, modified in place.
        csf: CSF map, modified in place.
        mask: Voxels to reassign.

    Returns:
        ``(gm, wm, csf)``, the same arrays.
    """
    share = gm[mask] / 2.0
    wm[mask] += share
    csf[mask] += share
    gm[mask] = 0.0
    return gm, wm, csf


def run_segment_nogm_conventional(
    p0_large, wj_affine=None, device="cpu", admission=None, verbose=False
):
    """Drop-in replacement for ``Preprocess.run_segment_nogm``.

    Returns the keys T1Prep reads from the deepmriprep version -- the three
    tissue maps on the working grid plus the grey-matter and intracranial
    volumes.  The native-resolution maps upstream also returns are not built:
    T1Prep resamples its own, and the ``spline_resize.grid_sample`` that
    produced them is 9% of the stage's wall clock and the source of its
    resident peak.

    Args:
        p0_large: Label map on the working grid, as a NIfTI image.
        wj_affine: Jacobian of the affine registration, used to express the
            volumes in native space.  ``None`` reports them on the working
            grid instead.
        device: Device used to resample the atlas.
        admission: Precomputed admission region, see
            :func:`noncortical_gm_mask`.
        verbose: Print the corrected volume.

    Returns:
        Dict with ``p1_large``, ``p2_large``, ``p3_large``, ``gmv`` and
        ``tiv``.
    """
    mask = noncortical_gm_mask(
        p0_large, device=device, admission=admission, verbose=verbose
    )

    p0 = np.asanyarray(p0_large.dataobj, dtype=np.float32)
    gm, wm, csf = _one_hot_gm_wm_csf(p0)
    gm, wm, csf = apply_noncortical_gm(gm, wm, csf, mask)

    affine, header = p0_large.affine, p0_large.header
    voxel_vol = float(np.abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    scale = 1e-3 * voxel_vol * (1.0 if wj_affine is None else float(wj_affine))
    gmv = float(gm.sum()) * scale
    tiv = gmv + (float(wm.sum()) + float(csf.sum())) * scale

    return {
        "p1_large": nib.Nifti1Image(gm, affine, header),
        "p2_large": nib.Nifti1Image(wm, affine, header),
        "p3_large": nib.Nifti1Image(csf, affine, header),
        "gmv": gmv,
        "tiv": tiv,
    }
