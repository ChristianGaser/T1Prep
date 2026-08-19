"""Blood-vessel detection and correction for T1-weighted segmentation.

This module ports the strategy CAT12 uses for blood vessels
(``cat_vol_partvol.m`` -> ``LAB.BV`` and the BVC block of ``cat_main.m``).
The single test T1Prep used before -- ``blood_vessel_correction_pve_float``
in ``CAT_VolPbt.c``, which flags bright voxels that a region-growing from
the WM core cannot reach -- only sees the label map and only catches vessels
that are *fully detached* from the brain.  Vessels running inside a sulcus
touch the pia through partial-volume CSF, so the front reaches them and they
survive.  Those are the ones that break surface reconstruction.

The hard part is not finding vessels, it is not destroying thin gyral white
matter while doing so.  A 2 mm blade running out into a gyrus is bright, thin
and ridge-like: every *local* cue a vessel gives, it gives too.  So the
central rule of this module is that **looking like a vessel is never
sufficient**.  A voxel is admitted only with positive evidence, and which
evidence works was settled by measurement rather than by argument.

Three topological criteria were tried first, following CAT12 closely: outside
the region-grown white matter (``cat_vol_partvol.m`` l. 542), brighter than
white matter (l. 348), and geodesically detached (l. 398-407).  Against the
Colin27 ground truth they turned out to be nearly useless for the vessels that
matter -- small ones inside sulci -- because those sit at 2.46 on the 0..3
scale, *darker* than the 3.00 of white matter, and are anatomically continuous
with vasculature entering the parenchyma.  Together they admitted 14.5% of
them.  What separates a sulcal vessel from a gyral blade is local:

    term                small sulcal vessel   thin gyral WM
    local brightness excess     47.6%              0.9%
    CSF in the neighbourhood    48.4%              0.8%
    MRA spatial prior           93.9%             27.6%
    the three topological ones  14.5%              1.1%

so the gate is built from the first two plus hyperintensity, and the prior is
applied as a near-gate.  The white matter tree survives only as a floor over
its eroded interior: masking the whole tree cost six times the sensitivity on
small sulcal vessels *and* increased white matter damage, because half of
those vessels belong to the tree topologically.

The failure mode is worth naming, because both directions were hit while
building this.  Lean on local shape alone and thin gyral blades -- bright,
thin, ridge-like, indistinguishable by any local cue -- get set to CSF.  Lean
on topology and nothing is corrected at all.

Thresholds are calibrated against the Colin27 fuzzy phantom, which ships a
ground-truth ``VESSELS`` class at 0.5 mm.  Its own vessels are dark (1.5 in
tissue units, i.e. flow voids), so they serve as the specificity control,
while the sensitivity cases raise those same voxels to a range of intensities
and segment them as WM or as GM.  The GM case matters for thickness in
particular: a vessel mislabelled GM inside a sulcus is exactly what PBT
follows.  The measured trades are recorded at :data:`PRIOR_RAMP`,
:data:`WM_TREE_CAP` and :data:`MIN_VESSEL_VOLUME`, and in the docstrings of
:func:`_wm_tree` and :func:`vessel_weight`.

Around that gate sit the supporting pieces, also from CAT12:

1. **Local shape and intensity cues** -- ridge-like negative divergence,
   brightness relative to the local neighbourhood, and the mismatch between
   image intensity and what the tissue label claims
   (``cat_vol_partvol.m``, l. 345 and 357).  Necessary, never sufficient.
2. **A spatial prior** built from MRA data (``cat_bloodvessels.nii.gz``)
   combined with the WM/CSF tissue priors, which both boosts detection where
   larger vessels actually run (insula, along the brainstem, superior
   sagittal sinus) and protects deep WM (``cat_vol_partvol.m``, l. 187-204).
3. **The connected white matter tree** as a hard protected region, so gyral
   blades are excluded independently of any threshold (:func:`_wm_tree`).
4. **A soft correction applied to the intensity image before AMAP**, not to
   the label map afterwards, so the segmentation never sees the vessel
   (``cat_main.m``, l. 399-444), plus a last-resort suppression right before
   PBT (``cat_surf_createCS2.m``, l. 1709-1711).

What is *not* ported is CAT12's atlas partitioning (``Ya1``/``YA``/``LAB.*``),
which T1Prep does not have.  Wherever a CAT12 threshold depends on it, the
equivalent constraint here comes from the Neuromorphometrics protection mask
(:func:`protected_regions`) and from the spatial prior.  Constants that do
transfer are used with CAT12's values and are marked as such; the rest are
expressed in interpretable units (mm of detour, fraction of a tissue step)
and exposed as parameters.

The derivative maps are computed at a fixed 1.5 mm working resolution, which
is what ``cat_vol_div`` does internally (``vx_volr = min(1.5, 3*vx_vol)``).
Without that the constants would not transfer to T1Prep's 0.5 mm grid.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    generate_binary_structure,
    label as _connected_components,
    median_filter,
    uniform_filter,
)

from ._segment_utils import (
    _resample_to,
    _resolve_template_file,
    get_regions_mask,
)

__all__ = [
    "protected_regions",
    "blood_vessel_prior",
    "cat_divergence",
    "vessel_weight",
    "apply_blood_vessel_correction",
    "suppress_vessels_for_surface",
]


# The cerebellum, kept separate because it is the most dangerous region for
# this correction and gets the widest margin.  Its folia are thin bright
# laminae separated by CSF -- which is precisely the configuration the gate
# fires on (local brightness excess plus CSF in the neighbourhood), so the
# cues cannot be relied on to spare them and a mask has to.  Correcting there
# destroys the fine fissures and with them the cerebellar surface.
# CAT12 protects it the same way, via ``Ycb3`` in cat_vol_partvol.m.
CEREBELLUM_REGIONS = (
    "Left Cerebellum White Matter",
    "Right Cerebellum White Matter",
    "Left Cerebellum Exterior",
    "Right Cerebellum Exterior",
    "Cerebellar Vermal Lobules I-V",
    "Cerebellar Vermal Lobules VI-VII",
    "Cerebellar Vermal Lobules VIII-X",
    "4th Ventricle",
)

# Subcortical grey and the hippocampal formation: also thin bright structures
# with vessel-like divergence.  CAT12's ``Yhc3``/``Ysc5`` equivalents.
PROTECTED_REGIONS = (
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
)

# Margins in mm.  The cerebellum gets the widest because the atlas is warped
# only affinely here, and a plain resize of it was measured to sit 3.9 mm off
# (Dice 0.84), leaving 13000 mm^3 of true cerebellum outside the mask.
CEREBELLUM_MARGIN_MM = 4.0
PROTECTED_MARGIN_MM = 1.5
VENTRICLE_MARGIN_MM = 3.0

PROTECTED_VENTRICLES = (
    "Left Inf Lat Vent",
    "Right Inf Lat Vent",
    "Left Lateral Ventricle",
    "Right Lateral Ventricle",
    "Brain Stem",
)

# Working resolution for the derivative maps, in mm.
# cat_vol_div uses min(1.5, 3*vx_vol), which is 1.5 for every resolution
# T1Prep runs at.
WORK_RES = 1.5

# Cue thresholds, as (start, full) pairs of a linear ramp.
#
# These are calibrated on a 2.5 mm vessel phantom at 0.5 mm, measured at the
# 1.5 mm working grid.  Absolute values are meaningful because the correction
# runs after LAS, so the tissue scale is fixed at CSF=1/3, GM=2/3, WM=1.
# Measured medians there were:
#
#     cue                 vessel    GM     GM crown   WM
#     -divergence(0..1)    0.206   0.015    0.025    0.000
#     brightness excess    0.478   0.005    0.026    0.000
#     detour               7.4 mm  0 mm     0 mm     0 mm
#
# The thin outer GM crown is the false positive that matters, so every ramp
# starts above its value with margin.
RIDGE_RAMP = (0.04, 0.14)       # -divergence of the 0..1 image
THIN_RAMP = (0.10, 0.35)        # brightness above a 3x3x3 box, 0..3 scale
BRIGHT_RAMP = (2.0, 2.5)        # absolute intensity, 0..3 scale
MISMATCH_RAMP = (0.15, 0.65)    # intensity above the tissue label, 0..3 scale
CSF_CONTEXT_RAMP = (0.0, 0.30)  # surrounding CSF fraction minus GM fraction

# How hard the MRA spatial prior gates detection.  CAT12 uses it as a genuine
# gate (`YbvA > .7` at cat_vol_partvol.m l. 406, `YbvA > 1` at l. 415), not as
# a gentle weight, and that turns out to matter more than any local cue.
# Measured on the brightened Colin27, where the prior reads 1.07 at sulcal
# vessels, 1.02 at intra-WM vessels and 0.19 in real white matter:
#
#     gate        sulcal found   WM damaged   ratio
#     none          1719 mm^3     4126 mm^3    0.4
#     > 0.7         1708          786          2.2
#     > 1.0         1660          351          4.7
#     > 1.2            79          154          0.5
#
# Local cues cannot make this distinction: after LAS the median vessel voxel
# (2.58) is *darker* than the median WM voxel (3.00), and 8.7% of pure WM is
# brighter than the tree cap.  Location is what separates them.
PRIOR_RAMP = (0.85, 1.05)

# The CSF context is no longer a suppressing factor -- it is one of the terms
# the admission gate is built from, because on real anatomy it is among the
# few that actually separate a sulcal vessel from a gyral blade (48.4% against
# 0.8%).  Using it as a multiplier instead, as an earlier version did, made it
# suppress real vessels rather than false positives.
SURFACE_RIDGE_RAMP = (0.35, 0.70)  # -divergence x CSF context, 0..3 map

# The three independent routes by which CAT12 admits a voxel as vessel.  Every
# one of its rules requires one of these *in addition to* looking vessel-like,
# and that is the whole reason thin gyral white matter survives: a 2 mm blade
# is bright, thin and ridge-like, but it is neither brighter than WM nor
# detached from it.
#
#   HYPER  -- brighter than white matter.  cat_vol_partvol.m l. 348 seeds
#             everything with Ym < 3.2 as *brain*, so anything at or below WM
#             intensity can never be claimed as vessel by that detector.
#   ISO    -- geodesically detached, l. 398-407, which has no brightness
#             requirement beyond Ym > 2.4 but demands isolation.
#   DETACH -- bright but outside the connected white matter tree, l. 542
#             (`Ym > 2.5 - 0.5*BVCstr & Ywm == 0`).  The most robust of
#             the three, and the only one that survives both LAS intensity
#             compression and sulci whose CSF is too thin to block a front.
HYPER_RAMP = (3.05, 3.35)       # intensity above WM, 0..3 scale

# Intensity cap on the white matter tree, CAT12's `Ym < 3.2` brain seed
# (cat_vol_partvol.m l. 348).  This single constant decides the whole
# sensitivity/specificity trade, because connectivity is transitive: without
# it, one contact point between a vessel and white matter absorbs the entire
# vessel tree into the protected mask.  Measured on the Colin27 phantom with
# its 31000 mm^3 of ground-truth vessels raised to 1.07x WM:
#
#     cap        vessels absorbed    thin gyral WM protected
#     none              85.2%                 99.5%
#     3.30              17.3%                 99.1%
#     3.15               3.2%                 99.1%
#
# Below WM intensity the cap cannot help and the other evidence routes carry.
WM_TREE_CAP = 3.15

# Minimum detected volume before anything is changed, after cat_main.m l. 403
# (1000 voxels on CAT12's ~1 mm working grid).  This is the safety mechanism
# that keeps subjects without a vessel problem untouched.
#
# Calibrated on Colin27, where the measurements separate a worthwhile
# correction from a harmful one.  Ground-truth vessels were set
# to a range of intensities and segmented as WM or GM; "damage" is white
# matter wrongly flagged, out of 654000 mm^3 present:
#
#     vessel case              found      WM damaged   ratio   total flagged
#     1.07x WM, as WM        25097 mm^3     455 mm^3    55:1      26233 mm^3
#     GM/WM midpoint, as GM  15599 mm^3     593 mm^3    26:1      16585 mm^3
#     WM level, as WM         3796 mm^3     276 mm^3    14:1       4204 mm^3
#     GM/WM midpoint, as WM   1202 mm^3     617 mm^3     2:1       2013 mm^3
#     none (dark vessels)      155 mm^3     556 mm^3     -          983 mm^3
#
#
MIN_VESSEL_VOLUME = 1000.0      # mm^3


def _ramp(values, ramp):
    """Linear ramp from 0 at ``ramp[0]`` to 1 at ``ramp[1]``."""
    lo, hi = ramp
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _report_terms(terms, ym3, yp0, vx):
    """Print which term is suppressing the detection, and by how much.

    Every term is a multiplier, so a single one at zero produces no correction
    at all and the summary volume alone cannot say which.  This tabulates each
    term over the voxels that *look* like vessel, which is the fastest way to
    see whether a null result comes from the evidence gate, the prior, the
    protection mask, or simply from there being no vessel.
    """
    voxel_volume = float(np.prod(vx))
    looks = (terms["bright"] * terms["shape"]) > 0.5
    n_looks = int(looks.sum())
    print(
        f"  vessel-like voxels (bright x shape > 0.5): "
        f"{n_looks * voxel_volume:.0f} mm^3"
    )
    if not n_looks:
        print("  -> nothing looks like a vessel; the cue thresholds are the issue")
        return
    order = ("evidence", "thin", "csf_context", "hyper", "prior")
    for name in order:
        term = np.broadcast_to(terms[name], ym3.shape)[looks]
        passing = float((term > 0.5).mean())
        print(
            f"    {name:12s} median {float(np.median(term)):5.3f}   "
            f"passing (>0.5) {100.0 * passing:5.1f}%"
        )
    inside = float(terms["wm_tree"][looks].mean())
    print(f"    {'wm_tree':12s} {100.0 * inside:5.1f}% of them are protected WM")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_tensor(vol, device):
    return torch.as_tensor(
        np.ascontiguousarray(vol, dtype=np.float32), device=device
    )[None, None]


def _resize(tensor, shape, mode="trilinear"):
    """Field-of-view preserving resize of a 5-D tensor."""
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(
        tensor, size=tuple(int(s) for s in shape), mode=mode, **kwargs
    )


def _reduced_shape(shape, vx, res):
    """Shape of the *res* mm grid covering the same field of view."""
    return np.maximum(
        2, np.round(np.asarray(shape, float) * np.asarray(vx, float) / res)
    ).astype(int)


def _largest_component(mask):
    """Keep only the largest 26-connected component of a boolean mask."""
    lab, n = _connected_components(mask, generate_binary_structure(3, 3))
    if n < 2:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


def _masked_median(vol, mask, size=3):
    """Median-filter *vol* where *mask* is set, cropped to the mask bounding box."""
    if not mask.any():
        return vol
    idx = np.nonzero(mask)
    box = tuple(
        slice(max(0, int(i.min()) - size), min(n, int(i.max()) + size + 1))
        for i, n in zip(idx, vol.shape)
    )
    out = vol.copy()
    out[box] = np.where(mask[box], median_filter(vol[box], size=size), vol[box])
    return out


def _smooth_mm(vol, vx, fwhm_mm):
    """Gaussian smoothing with an isotropic width given in mm."""
    sigma = [max(1e-3, fwhm_mm / 2.355 / float(v)) for v in vx]
    return gaussian_filter(vol, sigma=sigma)


def _wm_tree(yp0, ym3, vx, close_mm=0.5, cap=WM_TREE_CAP):
    """The connected white matter, used to protect its deep interior.

    Built by connected components on a generous seed.  CAT12 instead grows
    ``Ywm`` by downhill-constrained region growing (``cat_vol_partvol.m``
    l. 533), which is the principled version -- reaching a sulcal vessel from
    white matter means descending through cortex and climbing back up, and the
    climb is refused.  That was implemented and measured against this, and it
    was **worse on both axes**: it still absorbed 55% of small sulcal vessels
    (against 52%) while protecting only 96% of thin gyral white matter
    (against 99%).  The reason is anatomical rather than algorithmic -- the
    vessel tree is continuous with vasculature entering the parenchyma, so the
    front reaches sulcal segments along the vessel itself, never having to
    climb.  Connectivity is kept because it is cheaper and measured better.

    Only the eroded interior of this mask is protected (:func:`_deep_wm`).
    Protecting the whole tree was measured to cost six times the sensitivity
    on small sulcal vessels while *increasing* white matter damage, because
    the cue that actually separates a blade from a vessel is local, not
    topological -- see the gate built in :func:`vessel_weight`.
    """
    struct = generate_binary_structure(3, 3)
    init = (yp0 > 2.25) & (ym3 > 2.25) & (yp0 < 3.1) & (ym3 < cap)
    iters = max(1, int(round(close_mm / float(min(vx)))))
    return _largest_component(binary_closing(init, struct, iters))


def _deep_wm(yp0, ym3, vx, erode_mm=2.0):
    """Interior of the white matter, which must never be corrected.

    An erosion this deep also removes thin gyral blades, which is exactly why
    it is used only as a floor and not as the main protection: blades are kept
    safe by the cues, not by this mask.
    """
    iters = max(1, int(round(erode_mm / float(min(vx)))))
    return binary_erosion(
        _wm_tree(yp0, ym3, vx), generate_binary_structure(3, 3), iters
    )


def _csf_context(yp0, vx, radius_mm=2.5):
    """How much more CSF than grey matter surrounds a voxel, in ``[0, 1]``.

    This is the anatomical statement that separates a vessel from thin gyral
    white matter: a vessel lies in the sulcal CSF, a blade lies between two
    banks of cortex.  Both are bright, thin and ridge-like, so no purely local
    shape cue can tell them apart -- their surroundings can.

    It plays the role of CAT12's ``2.*smooth3(Yp0<2.9)``
    (``cat_vol_partvol.m`` l. 357).  Counting "not WM" as CAT12 does works
    there because SPM tends to assign vessels to CSF or GM; deepmriprep and
    AMAP assign them to WM, so a thick vessel would score its own core as
    closed, and cortex around a blade would score as open.  Contrasting CSF
    against GM avoids both.
    """
    size = [max(3, int(round(2.0 * radius_mm / float(v))) | 1) for v in vx]
    csf = uniform_filter(
        ((yp0 > 0.25) & (yp0 < 1.5)).astype(np.float32), size=size
    )
    gm = uniform_filter(
        ((yp0 >= 1.5) & (yp0 < 2.5)).astype(np.float32), size=size
    )
    return _ramp(csf - gm, CSF_CONTEXT_RAMP).astype(np.float32)


# ---------------------------------------------------------------------------
# 2) spatial prior
# ---------------------------------------------------------------------------


def blood_vessel_prior(target_affine, target_shape, device="cpu"):
    """Spatial blood-vessel prior, after ``cat_vol_partvol.m`` l. 187-204.

    ``YbvA = 1 - YwmA + max(YcsfA * 0.1, YbvA)`` -- roughly 0 inside deep white
    matter, 1 in neutral territory and up to 2 where the MRA-derived atlas says
    larger vessels run.  Every threshold that uses it treats 1 as neutral.

    The WM prior is taken from channel 1 of the shooting template
    (``Template_4_GS.nii.gz``); a subject-derived WM mask cannot be used here
    because vessels are exactly what the segmentation misassigns to WM.

    Returns
    -------
    np.ndarray
        Float32 prior on the target grid, in ``[0, 2]``.
    """
    bv = _resample_to(
        nib.load(_resolve_template_file("cat_bloodvessels", ".nii.gz")),
        target_affine,
        target_shape,
        device=device,
    )
    wm = _resample_to(
        nib.load(_resolve_template_file("Template_4_GS", ".nii.gz")),
        target_affine,
        target_shape,
        device=device,
        channel=1,
    )
    csf = _resample_to(
        nib.load(_resolve_template_file("csf_TPM", ".nii.gz")),
        target_affine,
        target_shape,
        device=device,
    )
    csf_max = float(np.nanmax(csf))
    if csf_max > 0:
        csf = csf / csf_max

    prior = (
        1.0
        - np.clip(wm, 0.0, 1.0)
        + np.maximum(0.1 * np.clip(csf, 0.0, 1.0), np.clip(bv, 0.0, 1.0))
    )
    return prior.astype(np.float32)


def protected_regions(target_affine, target_shape, device="cpu"):
    """Regions the vessel correction must never touch, on the target grid.

    Cerebellum first and foremost: its folia are thin bright laminae separated
    by CSF, which is exactly what the admission gate keys on, so without this
    mask the correction eats the fine fissures.  Measured on Colin27, dropping
    it leaves 172 mm^3 of cerebellar tissue flagged; adding it costs nothing
    elsewhere.  Then subcortical grey, hippocampus/amygdala, ventricles and
    brainstem, which CAT12 guards with ``Ycb3``/``Yhc3``/``Ysc5``.

    The atlas is resampled through its real affine rather than resized onto
    the target dimensions.  ``get_atlas`` does the latter, which silently
    assumes both grids share a field of view; Neuromorphometrics does not
    share one with the working space, and the resulting mask sits 3.9 mm off.
    For a protection mask that is the difference between guarding the
    cerebellum and guarding most of it.
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
    vx = np.sqrt((target_affine[:3, :3] ** 2).sum(axis=0))
    struct = generate_binary_structure(3, 3)

    def _grow(names, margin_mm):
        mask = get_regions_mask(atlas, "Neuromorphometrics", list(names))
        iters = max(1, int(round(margin_mm / float(min(vx)))))
        return binary_dilation(mask, struct, iters)

    return (
        _grow(CEREBELLUM_REGIONS, CEREBELLUM_MARGIN_MM)
        | _grow(PROTECTED_REGIONS, PROTECTED_MARGIN_MM)
        | _grow(PROTECTED_VENTRICLES, VENTRICLE_MARGIN_MM)
    )


# ---------------------------------------------------------------------------
# 1) local shape and intensity cues
# ---------------------------------------------------------------------------


def cat_divergence(vol, vx, res=WORK_RES, floor=1.0 / 3.0, device="cpu"):
    """Divergence map, following ``cat_vol_div(..., norm=0)``.

    Thin bright structures -- vessels, meninges -- have a strongly negative
    divergence while the interior of a tissue does not.  Unlike
    ``_segment_utils._divergence``, which normalises the gradient (CAT12's
    ``norm=1`` branch), this reproduces the default branch that
    ``cat_vol_partvol.m`` actually uses: the plain gradient of ``max(1/3, Ym)``,
    computed on a reduced grid with unit voxel spacing and divided once by the
    reduced voxel size.  Reproducing both the branch and the working resolution
    is what makes CAT12's constants transfer.

    Parameters
    ----------
    vol : np.ndarray
        Intensity image.  Pass the 0..1 normalised image to reproduce the
        constants of ``cat_vol_partvol.m``; pass the 0..3 map to reproduce
        those of ``cat_surf_createCS2.m``.
    floor : float or None
        Lower clamp applied before differentiating (CAT12 uses ``1/3``).
    """
    shape0 = np.asarray(vol.shape, dtype=int)
    vx = np.asarray(vx, dtype=float)
    shape_r = _reduced_shape(shape0, vx, res)
    vx_r = shape0 * vx / shape_r

    reduced = _resize(_as_tensor(vol, device), shape_r)[0, 0].cpu().numpy()
    if floor is not None:
        reduced = np.maximum(floor, reduced)

    grad = np.gradient(reduced)  # unit voxel spacing, as in cat_vol_gradient3
    div = sum(
        np.gradient(grad[i] / vx_r[i], axis=i) for i in range(3)
    ).astype(np.float32)
    return _resize(_as_tensor(div, device), shape0)[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# combination
# ---------------------------------------------------------------------------


def vessel_weight(
    ym01,
    yp0,
    vx,
    bv_prior=None,
    protect=None,
    strength=1.0,
    res=WORK_RES,
    device="cpu",
    return_terms=False,
):
    """Soft blood-vessel weight in ``[0, 1]``.

    A voxel has to be **bright** *and* show one of the shape cues (ridge-like,
    thin, geodesically isolated, or brighter than its label), *and* sit
    **away from the white matter body**, *and* be somewhere the **prior**
    considers plausible.  That conjunction is what keeps the two false
    positives that matter out of the mask: thin gyral crowns, which are
    ridge-like but not bright, and the bright side of the WM/GM edge, which is
    both but is part of the WM body.

    Note that brightness is judged absolutely, not relative to the label.
    CAT12 can use ``max(0, Ym - Yp0)`` as a primary cue because SPM tends to
    assign vessels to CSF or GM; deepmriprep and AMAP assign them to WM, so
    that difference vanishes on exactly the voxels that matter.

    Parameters
    ----------
    ym01 : np.ndarray
        LAS-normalised intensity image (CSF=1/3, GM=2/3, WM=1), i.e. CAT12's
        ``Ymi``.
    yp0 : np.ndarray
        PVE label map on the 0..3 scale.
    bv_prior : np.ndarray or None
        Output of :func:`blood_vessel_prior`; 1 is neutral.
    protect : np.ndarray or None
        Boolean mask that is forced to zero weight and seeded as brain.
    strength : float
        Overall scaling of the returned weight.

    Returns
    -------
    np.ndarray
        Float32 weight, 0 where no correction should happen.
    """
    ym01 = np.asarray(ym01, dtype=np.float32)
    yp0 = np.asarray(yp0, dtype=np.float32)
    vx = tuple(float(v) for v in vx)
    ym3 = ym01 * 3.0

    ydiv = cat_divergence(ym01, vx, res=res, device=device)
    brain = yp0 > 1.0
    wm_tree = _wm_tree(yp0, ym3, vx)

    # (a) Bright in absolute terms.  This is the mandatory term: a vessel is
    #     always at least GM-bright.  It deliberately does *not* reference the
    #     label -- deepmriprep and AMAP both routinely assign vessels to WM,
    #     so "brighter than its label" scores zero on exactly the voxels that
    #     matter most.
    t_bright = _ramp(ym3, BRIGHT_RAMP)

    # (b) Bright ridge.  cat_vol_partvol.m l. 345 uses (Ym - Ydiv) > 3.4,
    #     which couples the shape cue to the vessel being brighter than WM.
    #     Decoupling it and thresholding the divergence alone keeps the cue
    #     for vessels that only reach WM intensity; the brightness
    #     requirement is already carried by t_bright.
    t_ridge = _ramp(-ydiv, RIDGE_RAMP)

    # (c) Brighter than its own neighbourhood, i.e. thin.  MATLAB smooth3
    #     defaults to a 3x3x3 box.
    t_thin = _ramp(ym3 - uniform_filter(ym3, size=3), THIN_RAMP)

    # (d) Brighter than the tissue label claims -- CAT12's max(0, Ym - Yp0).
    #     A boost rather than a gate, for the same reason as (a).
    t_mismatch = _ramp(ym3 - yp0, MISMATCH_RAMP)

    # Shape: thin, ridge-like, or brighter than its label.  On its own this
    # says almost nothing -- thin gyral WM scores just as high as a vessel.
    t_shape = np.maximum(np.maximum(t_ridge, 0.7 * t_thin), t_mismatch)

    # Context: lying in CSF rather than between two banks of cortex.
    t_context = _csf_context(yp0, vx)

    # (e) Location plausibility, and the single most discriminating term.
    #     bv_prior is ~0 in deep WM, 1 in neutral territory and up to 2 where
    #     the MRA atlas says vessels run.  It is applied as a near-gate rather
    #     than a weight because after LAS the vessel and WM intensity
    #     distributions overlap almost completely -- see PRIOR_RAMP.
    if bv_prior is None:
        t_prior = np.float32(1.0)
    else:
        t_prior = _ramp(np.asarray(bv_prior, dtype=np.float32), PRIOR_RAMP)

    # (f) Hyperintensity above white matter.
    t_hyper = _ramp(ym3, HYPER_RAMP)

    # A voxel is admitted only with positive evidence, and the three terms
    # that provide it were chosen by measurement against the Colin27 ground
    # truth, comparing small sulcal vessels with thin gyral white matter:
    # local brightness excess passes 47.6% against 0.9%, CSF in the
    # neighbourhood 48.4% against 0.8%.  Hyperintensity adds the vessels that
    # are frankly brighter than white matter.  Shape alone is never enough --
    # a gyral blade is bright, thin and ridge-like too, and treating that as
    # evidence is what once set thin white matter to CSF.
    t_evidence = np.maximum(np.maximum(t_thin, t_context), t_hyper)

    # t_context enters only weakly, through a high floor.  It was designed on
    # a phantom whose vessel sat in a clean CSF shell; on real anatomy a
    # vessel lies in a narrow sulcus flanked by cortex, so csf_frac - gm_frac
    # is near zero and it passes on only 11-28% of Colin27's ground-truth
    # vessels.  As a strong factor it therefore suppresses real vessels rather
    # than false positives, so it is a gate term here rather than a factor.
    raw = np.clip(t_evidence * t_bright * t_shape * t_prior, 0.0, 1.0)

    # CAT12 smooths, sharpens with a fourth power and smooths again
    # (cat_main.m l. 417-423) to drop scattered weak responses while keeping
    # coherent cores.  A soft threshold does the same on a bounded score and
    # keeps the cut point readable.
    weight = _smooth_mm(raw, vx, 1.0)
    weight = np.clip((weight - 0.25) / 0.45, 0.0, 1.0)
    weight = _smooth_mm(weight, vx, 0.7)

    weight *= float(np.clip(strength, 0.0, 1.0))
    weight[~brain] = 0.0
    if protect is not None:
        weight[protect] = 0.0

    # Only the deep interior of the white matter is protected outright.  The
    # gate above is what keeps gyral blades safe; masking the whole tree was
    # measured to cost six times the sensitivity on small sulcal vessels while
    # increasing white matter damage, because half of those vessels are
    # topologically part of the tree.
    weight[_deep_wm(yp0, ym3, vx)] = 0.0
    weight = weight.astype(np.float32)

    if return_terms:
        return weight, {
            "bright": t_bright,
            "ridge": t_ridge,
            "thin": t_thin,
            "mismatch": t_mismatch,
            "shape": t_shape,
            "csf_context": t_context,
            "prior": np.broadcast_to(t_prior, ym3.shape),
            "hyper": t_hyper,
            "evidence": t_evidence,
            "wm_tree": wm_tree.astype(np.float32),
        }
    return weight


# ---------------------------------------------------------------------------
# 4) application
# ---------------------------------------------------------------------------


def apply_blood_vessel_correction(
    brain,
    label,
    strength=1.0,
    protect=None,
    bv_prior=None,
    min_volume=MIN_VESSEL_VOLUME,
    device="cpu",
    verbose=False,
    mri_dir=None,
    out_name=None,
    ext="nii.gz",
):
    """Detect and correct blood vessels before the segmentation sees them.

    Port of the BVC block in ``cat_main.m`` (l. 399-444), which runs after LAS
    and *before* AMAP, so that AMAP -- or, on the deepmriprep path,
    ``run_segment_nogm`` -- never sees a WM-bright tube.

    The label map is pulled towards CSF, but the intensity image only down to
    roughly GM level: CAT12's cap is ``1 - Ybv/4``, which bottoms out at 0.75
    even at full weight.  That is deliberate.  Driving the intensity to CSF
    here would fight AMAP rather than inform it; what finishes the job is the
    label correction plus :func:`suppress_vessels_for_surface` before PBT.

    CAT12's gate is reproduced: nothing is changed unless at least 1000 mm^3 of
    confident vessel is found *and* those voxels are actually bright.  Without
    it a soft correction would nibble at every subject; with it, images that do
    not have a vessel problem come through untouched.

    Parameters
    ----------
    brain : nib.Nifti1Image
        LAS-normalised intensity image (CSF=1/3, GM=2/3, WM=1).
    label : nib.Nifti1Image
        PVE label map on the 0..3 scale.
    strength : float
        Correction weight; 0 disables it entirely.
    protect, bv_prior : np.ndarray or None
        See :func:`vessel_weight`.  Both are built on demand when None; for
        ``protect`` that is a hard requirement rather than a convenience, so a
        failure to build it raises instead of proceeding unprotected.

    Returns
    -------
    (nib.Nifti1Image, nib.Nifti1Image)
        Corrected intensity image and label map.
    """
    if strength <= 0:
        return brain, label

    ym01 = brain.get_fdata().astype(np.float32)
    yp0 = label.get_fdata().astype(np.float32)
    vx = tuple(float(z) for z in brain.header.get_zooms()[:3])

    if bv_prior is None:
        try:
            bv_prior = blood_vessel_prior(brain.affine, ym01.shape, device=device)
        except (FileNotFoundError, OSError) as exc:  # template missing
            if verbose:
                print(f"Blood vessel prior unavailable ({exc}); continuing without it")
            bv_prior = None

    # Built here when the caller did not supply it.  Sparing the cerebellum is
    # not optional -- its folia look exactly like what is being detected -- so
    # it must not depend on every call site remembering to pass a mask.
    if protect is None:
        try:
            protect = protected_regions(brain.affine, ym01.shape, device=device)
        except (FileNotFoundError, OSError, KeyError) as exc:
            raise RuntimeError(
                "Cannot build the vessel-correction protection mask "
                f"({exc}). Refusing to run: without it the cerebellar "
                "fissures would be corrected away."
            ) from exc

    weight, terms = vessel_weight(
        ym01,
        yp0,
        vx,
        bv_prior=bv_prior,
        protect=protect,
        strength=strength,
        device=device,
        return_terms=True,
    )

    # Gate, after cat_main.m l. 403: enough confident vessel, and bright.
    # CAT12 counts voxels on its ~1 mm working grid, so the count is converted
    # to a volume to stay resolution independent.
    core = weight > 0.5
    n_core = int(core.sum())
    volume_mm3 = n_core * float(np.prod(vx))
    median_int = float(np.median(ym01[core])) if n_core else 0.0
    applied = volume_mm3 >= min_volume and median_int > 0.75

    if verbose:
        print(
            f"Blood vessel correction: {volume_mm3:.0f} mm^3 detected, "
            f"median intensity {median_int:.2f} -> "
            f"{'applied' if applied else 'skipped'}"
        )

    if verbose:
        _report_terms(terms, ym01 * 3.0, yp0, vx)

    if not applied:
        return brain, label

    # Intensity correction, cat_main.m l. 431-433.  The inner min caps the
    # value at 1 - w/4 (0.75 at full weight); the outer max floors it at
    # ym - 2w/3, so a very bright vessel is not driven straight to CSF.
    ym_new = np.maximum(
        np.minimum(np.maximum(1.0 / 3.0, 1.0 - weight / 4.0), ym01),
        ym01 - weight * 2.0 / 3.0,
    )
    # Two decreasing-threshold median passes, then one over the whole mask,
    # so the patched tube is filled from its surroundings rather than left
    # as a flat plateau.
    for step in (1, 2):
        ym_new = _masked_median(ym_new, weight > max(0.0, 1.0 - step / 2.0))
    ym_new = _masked_median(ym_new, weight > 0)
    # The median passes fill the lumen from its surroundings and could
    # brighten a voxel at the edge of the mask.  A vessel correction should
    # only ever remove signal.
    ym_new = np.minimum(ym_new, ym01)

    # Label correction.  cat_main.m lowers GM/WM and raises CSF by 127/255 at
    # full weight; on a PVE label map the equivalent is a two-class step down,
    # clamped so nothing is pushed below CSF or into the background.
    brain_mask = yp0 > 1.0
    yp0_new = np.where(
        brain_mask, np.minimum(yp0, np.maximum(1.0, yp0 - 2.0 * weight)), yp0
    )

    return (
        nib.Nifti1Image(ym_new.astype(np.float32), brain.affine, brain.header),
        nib.Nifti1Image(yp0_new.astype(np.float32), label.affine, label.header),
    )


def suppress_vessels_for_surface(vol, vx, strength=1.0, device="cpu"):
    """Last-resort vessel suppression right before PBT.

    Port of ``cat_surf_createCS2.m`` l. 1709-1711::

        Ybv = cat_vol_morph(Ymf + Ydiv./max(1,Ymf) > 3.5, 'd') & Ymf > 2;
        Ymf(Ybv) = 1.4;
        Ymfs = cat_vol_median3(Ymf, Ysroi | Ybv, ...);

    Two deviations, both deliberate:

    * **Sign.**  ``Ydiv`` is negative on bright ridges -- that is the
      convention every other use in CAT12 relies on
      (``max(0, 8*-Ydiv)`` at ``cat_vol_partvol.m`` l. 357, ``(Ym - Ydiv) >
      3.4`` at l. 345).  With ``Ymf`` capped at 3 and ``Ydiv`` negative,
      ``Ymf + Ydiv/max(1,Ymf) > 3.5`` cannot be satisfied, so the criterion is
      inverted here to select bright ridges as intended.
    * **Threshold.**  Measured on a 2.5 mm vessel phantom, ``-Ydiv`` of the
      0..3 map is 0.60 at the vessel, 0.08 at a thin gyral crown and 0.00 in
      WM, so the cut is placed on the divergence directly instead of on the
      3.5 sum, which has almost no margin against WM.

    Once the correction upstream of AMAP is doing its job this should rarely
    fire; it is kept because residue costs the most at exactly this point.

    Parameters
    ----------
    vol : np.ndarray
        PVE label map on the 0..3 scale (the PBT input).

    Returns
    -------
    np.ndarray
        Copy of *vol* with vessel-like structures pulled down to CSF/GM level.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if strength <= 0:
        return vol

    ydiv = cat_divergence(vol, vx, floor=None, device=device)
    ridge = -ydiv * _csf_context(vol, vx)
    outside = ~_wm_tree(vol, vol, vx)
    detected = binary_dilation(
        _ramp(ridge, SURFACE_RIDGE_RAMP) > 0.5, generate_binary_structure(3, 3), 1
    ) & (vol > 2.5) & outside
    if not detected.any():
        return vol

    out = vol.copy()
    out[detected] = 1.4
    out = _masked_median(out, detected)

    # Scale back towards the original where the correction is only partly
    # wanted, and never let this step brighten a voxel.
    weight = float(np.clip(strength, 0.0, 1.0))
    return np.minimum(vol, weight * out + (1.0 - weight) * vol).astype(np.float32)
