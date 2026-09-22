"""Remove dura mater that the skull-strip leaves outside the CSF.

deepbet keeps a thin rim of dura in many scans, heaviest over the vertex and
along the falx.  Walking outwards from the cortex it reads GM -> dark CSF ->
bright -> background: one or two voxels of GM-bright tissue lying on the
*outside* of the subarachnoid CSF.  The deepmriprep label calls almost all of
it CSF, so grey matter and the surfaces are not affected, but it stays in the
brain mask, in the bias-corrected image and in the CSF volume, and so in the
TIV.

The rule is layer order, read off two distance maps.  A voxel is dura when it
is brighter than the CSF/GM midpoint, lies more than :data:`GAP_MM` from the
brain tissue, and is closer to the edge of the mask than to that tissue: there
has to be CSF between it and the cortex, and it has to be on the far side of
that CSF.  What lies outside a dura voxel goes with it (:data:`PEEL_MM`).  The
subarachnoid CSF underneath is kept.

Measured on the working grid of 18 test subjects (among them a 7T scan, an
infant and three AD cases), the step removes 8-142 ml from the mask, of which
2-40 ml is CSF by label and at most 0.21 ml is labelled GM or WM, in about
3 s.  On IXI199 the reported TIV drops by 1.0% while GM changes by 0.12 ml.
The sinus at the top of the falx is thicker than :data:`BAND_MM` and only
its upper part goes; a second pass would take the rest, but would also
follow the falx down the interhemispheric fissure.

A connectivity rule was tried first and rejected: grayscale reconstruction
from white-matter seeds (an h-dome) found 0.7 ml where layer order found
10.3 ml on the same subject.  Wherever the dura touches a gyral crown the
partial volume forms a bright bridge and the reconstruction leaks through it,
which any connectedness criterion will do.

The rule trusts the CSF/GM contrast its threshold is taken from.  When that
contrast is weak it would flag ordinary CSF, so the step is skipped below
:data:`MIN_CONTRAST`.
"""

from __future__ import annotations

import fill_voids
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    distance_transform_edt,
    generate_binary_structure,
    label as _connected_components,
)

__all__ = ["dura_mask", "remove_dura"]


#: Resolution the distance maps are computed at, in mm.  The thresholds below
#: were set on ~1 mm data, and on the 0.5 mm working grid this is also 8x less
#: memory for the distance transforms.
DETECT_RES = 1.0

#: CSF gap required between a dura voxel and the brain tissue, in mm.
GAP_MM = 1.0

#: Only voxels this close to the edge of the mask can be dura, in mm.
BAND_MM = 4.0

#: Voxels no deeper than their nearest dura voxel and within this distance of
#: it are removed with it, in mm.  This takes the partial volume between the
#: dura and the background.  A fixed-size window (the deepest dura voxel in a
#: 5x5x5 box) was tried first; it removed the CSF underneath as well.
PEEL_MM = 2.0

#: Lowest relative contrast (GM - CSF) / (WM - CSF) the rule runs at.  The 18
#: test subjects measured 0.48-0.54 on the image this step sees (bias
#: corrected, before LAS), so this is a margin for inputs whose tissue classes
#: are too close to set the threshold from; none of the test data comes near
#: it.  Measure on that image: LAS can change the CSF level a lot -- one scan
#: reads CSF 0.34 here but 0.79 in its final bias-corrected output.
MIN_CONTRAST = 0.35

#: Fewest voxels a tissue class needs for its intensity level to be used.
MIN_LEVEL_VOXELS = 1000

_6 = generate_binary_structure(3, 1)
_26 = generate_binary_structure(3, 3)


def _voxel_size(img):
    """Voxel edge lengths of a NIfTI image, in mm."""
    return np.sqrt((np.asarray(img.affine, float)[:3, :3] ** 2).sum(axis=0))


def _bounding_box(mask, margin):
    """Slices of the bounding box of *mask*, grown by *margin* voxels."""
    box = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        lo = max(0, int(hits[0]) - margin)
        hi = min(mask.shape[axis], int(hits[-1]) + margin + 1)
        box.append(slice(lo, hi))
    return tuple(box)


def _block_mean(vol, factors):
    """Mean over non-overlapping blocks, zero-padded to whole blocks."""
    vol = np.pad(vol, [(0, (-n) % f) for n, f in zip(vol.shape, factors)])
    blocks = []
    for n, f in zip(vol.shape, factors):
        blocks += [n // f, f]
    return vol.reshape(blocks).mean(axis=(1, 3, 5), dtype=np.float32)


def _upsample(vol, factors, shape):
    """Trilinear inverse of :func:`_block_mean`, cropped to *shape*."""
    size = [n * f for n, f in zip(vol.shape, factors)]
    tensor = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    up = F.interpolate(
        tensor[None, None], size=size, mode="trilinear", align_corners=False
    )[0, 0].numpy()
    return up[: shape[0], : shape[1], : shape[2]]


def _largest_component(mask):
    """Keep only the largest 26-connected component of a boolean mask."""
    lab, n = _connected_components(mask, _26)
    if n < 2:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


def _tissue_levels(brain, p0, inside):
    """Median intensity of CSF, GM and WM, or ``None`` if one is too sparse."""
    levels = []
    for lo, hi in ((0.9, 1.1), (1.9, 2.1), (2.9, np.inf)):
        values = brain[inside & (p0 > lo) & (p0 < hi)]
        if values.size < MIN_LEVEL_VOXELS:
            return None
        levels.append(float(np.median(values)))
    return tuple(levels)


def _detect(brain, p0, inside, vx, verbose=False):
    """Dura rule on a ~1 mm grid; see the module docstring.

    Returns:
        ``(removed, levels)`` -- boolean mask of the voxels to remove and the
        CSF, GM and WM intensity levels, or ``None`` if there is nothing to
        remove or the contrast guard stops the rule.
    """
    levels = _tissue_levels(brain, p0, inside)
    if levels is None:
        if verbose:
            print("Dura removal skipped: too few CSF, GM or WM voxels")
        return None
    csf, gm, wm = levels
    contrast = (gm - csf) / (wm - csf) if wm > csf else 0.0
    if contrast < MIN_CONTRAST:
        if verbose:
            print(
                f"Dura removal skipped: CSF/GM contrast {contrast:.2f} "
                f"is below {MIN_CONTRAST:.2f}"
            )
        return None

    # Opened, so that a one-voxel sheet of dura the label calls grey matter
    # does not count as brain tissue.
    tissue = _largest_component(binary_opening(p0 >= 1.5, _6))
    if not tissue.any():
        return None

    threshold = 0.5 * (csf + gm)
    d_tissue = distance_transform_edt(~tissue, sampling=vx)
    d_edge = distance_transform_edt(inside, sampling=vx)
    dura = (
        inside
        & (brain > threshold)
        & (d_tissue > GAP_MM)
        & (d_edge < d_tissue)
        & (d_edge < BAND_MM)
    )
    del d_tissue
    if not dura.any():
        return None

    # Peel what lies outside each dura voxel: anything close to it that is no
    # deeper below the edge than the nearest dura voxel itself.
    d_dura, nearest = distance_transform_edt(
        ~dura, sampling=vx, return_indices=True
    )
    depth = d_edge[tuple(nearest)]
    del nearest
    removed = (dura | (inside & (d_dura <= PEEL_MM) & (d_edge <= depth))) & ~tissue

    # Tidy the new edge: drop the slivers the peel leaves standing, and any
    # piece it cut off.  ``tissue`` is itself an opened set, so the opening
    # cannot eat into it.  Confined to near the dura, so nothing changes
    # where there was no dura to remove.
    kept = binary_opening(inside & ~removed, _6)
    kept = fill_voids.fill(_largest_component(kept))
    removed = inside & ~kept & (d_dura <= 2 * PEEL_MM) & ~tissue
    return removed, levels


def dura_mask(brain, p0, vx, verbose=False):
    """Find dura left outside the CSF by the skull-strip.

    Args:
        brain: Skull-stripped, bias-corrected T1w intensities; zero outside
            the brain mask.
        p0: Label map on the same grid (CSF 1, GM 2, WM 3).
        vx: Voxel size in mm.
        verbose: Print the removed volume, or why the rule did not run.

    Returns:
        Boolean mask on the grid of *brain* of the voxels to remove from the
        brain mask, or ``None`` if there is nothing to remove.
    """
    brain = np.asarray(brain, dtype=np.float32)
    p0 = np.asarray(p0, dtype=np.float32)
    vx = np.asarray(vx, dtype=float)
    inside = brain > 0
    if not inside.any():
        return None

    # Detect at ~DETECT_RES on the bounding box of the brain.
    factors = [max(1, int(round(DETECT_RES / v))) for v in vx]
    box = _bounding_box(inside, 2 * max(factors))
    full_shape = inside.shape
    brain, p0, inside = brain[box], p0[box], inside[box]
    coarse_inside = _block_mean(inside, factors) > 0.5
    found = _detect(
        _block_mean(brain, factors),
        _block_mean(p0, factors),
        coarse_inside,
        vx * factors,
        verbose=verbose,
    )
    if found is None:
        return None
    coarse_removed, (csf, gm, wm) = found
    threshold = 0.5 * (csf + gm)

    # Place the edge on the full grid, within one coarse voxel of the dura.
    # Taking the complement of what the coarse grid keeps, rather than what
    # it removes, also takes the fringe outside the coarse mask; without it
    # a thin shell of dura stays behind at the edge.
    shape = brain.shape
    zone = _upsample(binary_dilation(coarse_removed, _26), factors, shape) > 0.5
    kept = _upsample(coarse_inside & ~coarse_removed, factors, shape) > 0.5
    removed = zone & inside & ~kept
    del kept
    if max(factors) > 1:
        # Block averaging mixes the inner fringe of the dura with the CSF
        # under it and pulls it below threshold; grow back into it.
        bright = zone & inside & (brain > threshold) & (p0 < 1.5)
        removed = binary_dilation(
            removed, _6, iterations=max(factors), mask=bright
        ) | removed

    if verbose:
        ml = removed.sum() * np.prod(vx) / 1000.0
        print(
            f"Dura removal: {ml:.1f} ml removed on the working grid "
            f"(CSF {csf:.2f}, GM {gm:.2f}, WM {wm:.2f})"
        )
    if not removed.any():
        return None
    out = np.zeros(full_shape, dtype=bool)
    out[box] = removed
    return out


def remove_dura(brain_large, p0_large, verbose=False):
    """Clear dura left outside the CSF from the brain image and the label.

    Args:
        brain_large: Bias-corrected brain on the working grid.
        p0_large: Label map on the same grid.
        verbose: Print the removed volume, or why nothing was removed.

    Returns:
        ``(brain_large, p0_large, removed)`` -- the two images with the dura
        set to zero, and the boolean mask of what was removed (``None`` and
        the inputs unchanged if nothing was).
    """
    removed = dura_mask(
        np.asanyarray(brain_large.dataobj),
        np.asanyarray(p0_large.dataobj),
        _voxel_size(p0_large),
        verbose=verbose,
    )
    if removed is None:
        return brain_large, p0_large, None

    # Copies: the caller may still hold the input images.
    brain = np.array(brain_large.dataobj)
    p0 = np.array(p0_large.dataobj)
    brain[removed] = 0
    p0[removed] = 0
    return (
        nib.Nifti1Image(brain, brain_large.affine, brain_large.header),
        nib.Nifti1Image(p0, p0_large.affine, p0_large.header),
        removed,
    )
