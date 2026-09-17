"""Hemisphere partition of the label map for surface extraction.

Splits the PVE label into left and right hemispheres, fills the deep grey
nuclei and the ventricles with white matter, and measures the topology of
the result.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    generate_binary_structure,
)

from ._atlas import resolve_template_file
from .utils import find_largest_cluster


def _octagon_dilation(mask, iterations, mask_region=None):
    """Dilate with an alternating 6/26 neighbourhood.

    Repeatedly dilating with ``generate_binary_structure(3, 3)`` grows a *cube*
    (Chebyshev ball): a 10-iteration dilation reaches 10 voxels along the axes
    but 17 along the body diagonal, so the resulting boundary is faceted.
    Alternating the 26- and 6-connected structuring elements grows an octagon
    instead, which is within ~8% of a sphere, at the same cost per iteration.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array to dilate.
    iterations : int
        Number of dilation steps.
    mask_region : np.ndarray, optional
        If given, the front may only expand into voxels that are ``True`` here,
        which turns the dilation into a geodesic propagation.

    Returns
    -------
    np.ndarray
        Boolean array holding the dilated mask.
    """
    struct26 = generate_binary_structure(3, 3)
    struct6 = generate_binary_structure(3, 1)
    out = mask
    for i in range(iterations):
        prev = out
        out = binary_dilation(
            out, struct26 if i % 2 else struct6, 1, mask=mask_region
        )
        if mask_region is not None and out.sum() == prev.sum():
            break  # front has converged, further steps cannot change it
    return out


#: Neuromorphometrics structures whose grey matter no fill may enter, however
#: thin it is.  The hippocampal tail and the medial hippocampal head sit
#: directly under the IBSR thalamus and ventral-DC labels, which are fill
#: seeds, so without this the seeds alone turned ~1% of the hippocampus into
#: white matter.
GUARD_MTL_REGIONS = (
    "Left Hippocampus",
    "Right Hippocampus",
    "Left Amygdala",
    "Right Amygdala",
)


#: Neuromorphometrics numbers its cortical parcels from 100 upwards; every
#: label below that is subcortical, ventricular, white matter or cerebellar.
NMM_FIRST_CORTICAL_ID = 100


def cortex_guard(p0_data, guard_atlas):
    """Grey matter that the white-matter fills of ``get_partition`` must spare.

    The fills are driven by the warped IBSR atlas, whose only notion of cortex
    is a label that misregistration moves around.  Dilating that label enough
    to absorb the error protects periventricular tissue as well: at 0.5 mm the
    partial-volume band along the ventricle wall is several voxels thick, reads
    as grey matter, and lies within 2 mm of the atlas cortex.  Guarding it left
    ~2 cm^3 of that band unfilled, ~0.4 cm^3 of it inside the ventricles, and
    raised the handle count of the white-matter volume by 70-100%.

    So the guard is built from the subject's tissue instead and only *located*
    by the atlas, a second one drawn on cortical grey matter
    (Neuromorphometrics):

    - **cortex**: grey matter that survives a one-voxel opening, inside a
      cortical parcel.  The opening keeps the ribbon (>= ~1.5 mm thick) and
      drops partial-volume films, which is what lets the ventricle fill still
      absorb the wall and the choroid plexus.  No margin is added around the
      parcels: every margin tried (1 mm, and 2 or 4 mm along thick grey
      matter) left 10-30 times more tissue unfilled outside the parcels than
      it protected inside them, some of it inside the ventricles.
    - **medial temporal lobe**: all grey matter in hippocampus and amygdala.
      No thickness test here -- the tail of the hippocampus is thin, and it is
      exactly where the atlas seeds overlap it.

    Against the unguarded fills, the cortical grey matter turned into white
    matter drops from 541 to 1 mm^3 (HR075) and from 694 to 1 mm^3 (IXI199),
    and hippocampus and amygdala are no longer touched at all.  On atrophic
    brains (613 -> 397 and 791 -> 201 mm^3) what is left is mostly the
    enlarged occipital and temporal horns, filled correctly under parcels the
    warp has shifted onto them.

    Parameters
    ----------
    p0_data : np.ndarray
        PVE label map (0 = background, 1 = CSF, 2 = GM, 3 = WM).
    guard_atlas : nib.Nifti1Image
        Neuromorphometrics label volume resampled onto the same grid.

    Returns
    -------
    np.ndarray
        Boolean array, ``True`` on grey matter that must not be filled.
    """
    # Matched on the stored integer labels: ``get_regions_mask`` would go
    # through ``get_fdata`` and a ~380 MB float64 copy on the working grid.
    labels = np.asanyarray(guard_atlas.dataobj)
    rois = pd.read_csv(resolve_template_file("Neuromorphometrics", ".csv"), sep=";")
    mtl_ids = rois.ROIid[rois.ROIname.isin(GUARD_MTL_REGIONS)].tolist()

    gm = (p0_data >= 1.5) & (p0_data < 2.5)
    thick_gm = binary_opening(gm, generate_binary_structure(3, 1), 1)
    cortex = thick_gm & (labels >= NMM_FIRST_CORTICAL_ID)
    mtl = gm & np.isin(labels, mtl_ids)
    return cortex | mtl


def ventricle_fill(
    p0_data, atlas_data, regions, vx=0.5, reach_mm=5.0, guard=None
):
    """Grow the atlas ventricle labels through the subject's own ventricle.

    The fill that makes the hemisphere maps usable for surface extraction has
    to cover the whole ventricular system, and dilating the warped atlas label
    by a fixed margin does not: on a brain with enlarged ventricles the label
    sits inside the real cavity and the margin runs out before the roof, which
    leaves a CSF band under the cingulate that PBT then tracks as a sulcus.
    Worse, the same misregistration puts the atlas *cortical* label over that
    band, so the ``gm_mask`` veto blocks it a second time.

    Growing the label geodesically instead -- through non-WM voxels only --
    follows the cavity rather than a distance: the front flows along the
    ventricle to its true extent whatever the size, and stops at the wall
    because white matter blocks it.  ``reach_mm`` is therefore not the size of
    the ventricle but the margin of error allowed on the atlas: it bounds how
    far the fill can stray if the registration is off, and 5 mm is far too
    short to cross the corpus callosum into the interhemispheric fissure.

    The hippocampus and amygdala are held out of the propagation region: the
    temporal horn abuts the hippocampus without an intervening wall of white
    matter, so they are the one place the front has to be stopped by name
    rather than by tissue class.

    Parameters
    ----------
    p0_data : np.ndarray
        PVE label map (0 = background, 1 = CSF, 2 = GM, 3 = WM).
    atlas_data : np.ndarray
        IBSR label volume resampled onto the same grid.
    regions : dict
        Mapping from IBSR ``ROIabbr`` to ``ROIid``.
    vx : float, optional
        Voxel size in mm, used to convert ``reach_mm`` into steps.
    reach_mm : float, optional
        Geodesic budget for the growth.  Raising it past ~7 mm lets the front
        reach the callosal sulcus and the medial cortex, so it is deliberately
        tight.
    guard : np.ndarray, optional
        Grey matter the front may neither start in nor cross, from
        :func:`cortex_guard`.

    Returns
    -------
    np.ndarray
        Boolean array, ``True`` on the ventricular system as segmented.
    """
    ventricles = ["lLatVen", "rLatVen", "lInfLatVen", "rInfLatVen"]
    seed = np.isin(atlas_data, [regions[r] for r in ventricles])
    # Grow through everything that is not white matter.  The ventricle is
    # CSF, but on brains like this the segmentation calls parts of its
    # interior GM, so keying on CSF alone would stop at the first mislabelled
    # voxel.
    region = p0_data < 2.5
    # The archicortex is the exception, and it has to be named explicitly.
    # "Not white matter" is a wall everywhere around the ventricular system
    # except at the temporal horn, which touches the hippocampus directly
    # with no white matter in between: a GM-permissive front seeded there
    # does not stop at the ventricle wall but flows along the hippocampus and
    # fills roughly a third of it with WM, on every subject, however good the
    # registration.  The surface then follows that fill and PBT reports the
    # medial temporal lobe as locally thin.  Barring only hippocampus and
    # amygdala leaves the roof fix above untouched -- both are inferior
    # structures, far from the roof of the ventricular body, so neither can
    # block the front where it needs to reach.
    keep_out = ["lHip", "rHip", "lAmy", "rAmy"]
    region = region & ~binary_dilation(
        np.isin(atlas_data, [regions[r] for r in keep_out]),
        generate_binary_structure(3, 3),
        2,
    )
    # "Not white matter" is no wall at all where the white matter between the
    # ventricle and a sulcus is thinner than the partial-volume blur: behind
    # the atrium (calcar avis, collateral trigone) and under the frontal horns
    # the front used to cross it and fill the lingual gyrus, precuneus,
    # fusiform and subcallosal cortex along a straight 5 mm edge, 0.5-0.8 cm^3
    # per subject.  The guard stops it at the ribbon while still letting it
    # through the thin partial-volume films the fill exists to absorb.
    if guard is not None:
        region &= ~guard
    # Seeds outside the region would be kept by the dilation regardless, and
    # the atlas ventricle is wide enough to lie on cortex and hippocampus.
    seed &= region
    return _octagon_dilation(seed, int(round(reach_mm / vx)), region)


def get_partition(p0_large, atlas, guard_atlas=None):
    """Partition a segmentation into left and right hemispheres.

    The deep grey nuclei and the ventricles are filled with white matter so
    that the white surface passes over them, and everything outside the
    hemisphere (cerebellum, brainstem, the other side) is set to CSF.

    Parameters
    ----------
    p0_large : nib.Nifti1Image
        PVE label map on the working grid.
    atlas : nib.Nifti1Image
        IBSR label volume resampled onto the same grid; drives the partition
        and seeds the fills.
    guard_atlas : nib.Nifti1Image, optional
        Neuromorphometrics label volume on the same grid.  When given, no fill
        may write white matter into cortical or medial temporal grey matter
        (see :func:`cortex_guard`).  Without it the fills are bounded by the
        IBSR cortex label alone, which misregistration defeats.

    Returns
    -------
    tuple of np.ndarray
        ``(lh, rh)`` label maps in ``[1, 3]``.
    """
    rois = pd.read_csv(resolve_template_file("IBSR", ".csv"), sep=";")[
        ["ROIid", "ROIabbr"]
    ]
    regions = dict(zip(rois.ROIabbr, rois.ROIid))

    bin_struct3 = generate_binary_structure(3, 3)
    atlas_data = atlas.get_fdata().copy()
    atlas_mask = atlas_data > 0
    atlas_mask = binary_dilation(atlas_mask, bin_struct3, 3)

    p0_data = p0_large.get_fdata().copy()
    # Cortical ribbon plus the two archicortical structures that border it.
    # Every fill below is vetoed by this mask, so its margin has to be
    # commensurate with how far those fills reach.  At 2 voxels (1 mm) it was
    # not: the temporal horn seed touches the hippocampus, so on any brain
    # whose warp is off by more than a millimetre there -- which is most of
    # them, the medial temporal lobe being where nonlinear registration is
    # worst -- the fill was free to grow straight through the structure.
    gm_regions = ["lCbrGM", "rCbrGM", "lAmy", "lHip", "rAmy", "rHip"]
    gm_mask = np.isin(atlas_data, [regions[r] for r in gm_regions])
    gm_mask = binary_dilation(gm_mask, bin_struct3, 4)

    left_regions = [
        "lCbrWM",
        "lCbrGM",
        "lLatVen",
        "lInfLatVen",
        "lThaPro",
        "lCau",
        "lPut",
        "lPal",
        "lHip",
        "lAmy",
        "lAcc",
        "lVenDC",
    ]
    right_regions = [r.replace("l", "r", 1) for r in left_regions]

    left = np.isin(atlas_data, [regions[r] for r in left_regions])
    right = np.isin(atlas_data, [regions[r] for r in right_regions])

    bin_struct3 = generate_binary_structure(3, 3)
    left = binary_opening(left, bin_struct3, 3)
    left = binary_closing(left, bin_struct3, 3)

    lh = binary_dilation(left, bin_struct3, 5) & ~right
    rh = binary_dilation(right, bin_struct3, 5) & ~left

    left = binary_closing(lh, bin_struct3, 2) & ~rh
    right = binary_closing(rh, bin_struct3, 2) & ~left

    excl_regions = ["lCbeWM", "lCbeGM", "rCbeWM", "rCbeGM", "b3thVen", "b4thVen"]
    exclude = np.isin(atlas_data, [regions[r] for r in excl_regions])
    exclude = binary_dilation(exclude, bin_struct3, 1)
    # The brainstem has to be cut away or the surface runs down into it, but
    # the cut must not take the surrounding cortex along.  A blind 5-step
    # 26-connected dilation did exactly that: the cube reaches 4.3 mm along
    # its diagonals and the midbrain is wrapped by parahippocampal cortex at
    # that distance, so ~1.4 cm^3 of ribbon and ~0.1 cm^3 of hippocampus were
    # forced to CSF on every subject, registration error or not.  Because
    # ``exclude`` is applied last it beat both the fill and the subject's own
    # labels, and PBT then measured the truncated ribbon as locally thin.
    # Growing through non-cortical tissue covers the same brainstem -- the
    # seed is kept, only the front is barred from entering ``gm_mask``.
    exclude = exclude | _octagon_dilation(
        np.isin(atlas_data, regions["bBst"]), 5, ~gm_mask
    )
    exclude = exclude | ~atlas_mask

    wm_regions = [
        "lThaPro",
        "lCau",
        "lPut",
        "lPal",
        "lAcc",
        "lLatVen",
        "lInfLatVen",
        "rThaPro",
        "rCau",
        "rPut",
        "rPal",
        "rAcc",
        "rLatVen",
        "rInfLatVen",
        # Subcortical, directly against the hippocampus, and the only region
        # in ``left_regions``/``right_regions`` that reached neither this list
        # nor ``excl_regions``.  What covered it was incidental -- the
        # thalamus fill from one side, the over-dilated brainstem from the
        # other -- and between them they left ~39% of it at GM level, a
        # 5 cm^3 slab of unfilled tissue against the medial temporal lobe for
        # the white surface to wander into.
        "lVenDC",
        "rVenDC",
    ]

    # Geodesic growth of the atlas structures through non-cortical tissue,
    # which is what covers the deep grey nuclei.  Subtracting ``gm_mask``
    # after a blind dilation got this wrong twice over: the cube first reached
    # 8.7 mm along its diagonals, far enough for the temporal horn seed to
    # cross the entire hippocampus, and the subtraction then clipped the seeds
    # themselves wherever the atlas cortex label overlapped them, leaving the
    # lateral putamen unfilled.  Passing the veto as the propagation region
    # fixes both: the seeds survive intact, and the front stops at the ribbon
    # rather than being carved back out of it afterwards.
    wm_fill = np.isin(atlas_data, [regions[r] for r in wm_regions])
    fill_region = ~gm_mask

    # The subject-level guard is the one veto the seeds do not survive.  It
    # can afford to: it holds only thick cortical grey matter and the
    # archicortex, never a deep nucleus, so it does not reopen the putamen
    # problem above -- but the IBSR thalamus and ventral DC do extend into
    # the hippocampal tail and head, and keeping those seeds intact is what
    # filled them.
    guard = None
    if guard_atlas is not None:
        guard = cortex_guard(p0_data, guard_atlas)
        wm_fill &= ~guard
        fill_region &= ~guard
    wm_fill = _octagon_dilation(wm_fill, 10, fill_region)

    # The ventricles get a subject-driven fill on top.  The veto above is
    # exactly what used to block their roof: on a brain with enlarged
    # ventricles the warp puts the atlas cortical label over the roof, so the
    # one place the blind dilation still had to reach was the one place it was
    # forbidden from.  The guard does not reintroduce that block: the roof
    # band is CSF and partial volume, neither of which it contains.
    vx = float(np.mean(p0_large.header.get_zooms()[:3])) or 0.5
    wm_fill = wm_fill | ventricle_fill(
        p0_data, atlas_data, regions, vx=vx, guard=guard
    )

    lh = np.copy(p0_data)
    lh[lh < 1] = 1
    lh[wm_fill] = 3
    lh[exclude | right] = 1

    rh = np.copy(p0_data)
    rh[rh < 1] = 1
    rh[wm_fill] = 3
    rh[exclude | left] = 1

    mask = (lh > 1) | (rh > 1)
    mask = binary_closing(mask, bin_struct3, 1)
    mask = binary_opening(mask, bin_struct3, 3)
    mask = find_largest_cluster(mask)
    mask = binary_dilation(mask, bin_struct3, 1)
    lh[~mask] = 1
    rh[~mask] = 1

    return lh, rh


def compute_euler_number(vol, threshold=2.5):
    """Compute the Euler number of a 3D volume at a given threshold.

    Thresholds the volume at the given level (default 2.5, the GM/WM
    boundary) and computes the Euler characteristic of the resulting
    binary cubical cell complex using the formula:

        chi = V - E + F - C

    where V = foreground voxels, E = foreground edges (6-connected
    adjacent pairs), F = foreground faces (2x2 blocks), and C =
    foreground cubes (2x2x2 blocks).

    For a topologically perfect hemisphere (single connected component,
    no handles), chi = 1.  Each topological defect (handle/tunnel)
    decreases chi by 1.

    Parameters
    ----------
    vol : np.ndarray
        3-D label or probability array (e.g. from ``get_partition``).
    threshold : float, optional
        Iso-level that separates foreground from background (default 2.5).

    Returns
    -------
    int
        Euler characteristic of the binary volume.
    """
    b = vol >= threshold

    # Vertices (foreground voxels)
    V = int(np.count_nonzero(b))

    # Edges along each axis
    ex = int(np.count_nonzero(b[:-1] & b[1:]))
    ey = int(np.count_nonzero(b[:, :-1] & b[:, 1:]))
    ez = int(np.count_nonzero(b[:, :, :-1] & b[:, :, 1:]))
    E = ex + ey + ez

    # Faces (2x2 blocks in each plane)
    fxy = int(np.count_nonzero(
        b[:-1, :-1] & b[1:, :-1] & b[:-1, 1:] & b[1:, 1:]
    ))
    fxz = int(np.count_nonzero(
        b[:-1, :, :-1] & b[1:, :, :-1] & b[:-1, :, 1:] & b[1:, :, 1:]
    ))
    fyz = int(np.count_nonzero(
        b[:, :-1, :-1] & b[:, 1:, :-1] & b[:, :-1, 1:] & b[:, 1:, 1:]
    ))
    F = fxy + fxz + fyz

    # Cubes (2x2x2 blocks)
    C = int(np.count_nonzero(
        b[:-1, :-1, :-1] & b[1:, :-1, :-1]
        & b[:-1, 1:, :-1] & b[1:, 1:, :-1]
        & b[:-1, :-1, 1:] & b[1:, :-1, 1:]
        & b[:-1, 1:, 1:] & b[1:, 1:, 1:]
    ))

    return V - E + F - C
