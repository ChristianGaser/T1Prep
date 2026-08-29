"""Tests for the myelination correction hook in :mod:`t1prep.segment`."""

import nibabel as nib
import numpy as np

from t1prep.segment import apply_myelin_correction

# The settings that actually move labels.  The library defaults
# (n_median_filter=1, max_correction=0.5) produce zero isovalue crossings
# here: one median pass erases a sparse correction field, and a cap of 0.5
# lands a voxel at PVE 3.0 exactly on the WM isovalue rather than across it.
# See test_library_defaults_do_not_reach_the_isovalue.
CORR = dict(erosion_mm=6.0, n_median_filter=0, max_correction=1.0)


def _phantom(myelinated=True, wmh=False, noise=2.0, seed=0):
    """A layered slab, optionally with a myelinated deep-cortical band.

    Labels follow the PVE convention (1=CSF, 2=GM, 3=WM).  The WM slab is
    deliberately much thicker than the myelinated band, so the eroded
    deep-WM core the correction calibrates on is genuine white matter --
    with a thin slab the core lands inside the band and the reference is
    contaminated by the very tissue being looked for.

    ``noise`` matters: the gradient criterion reads a percentile of the
    boundary-band gradient, and on a perfectly flat phantom that percentile
    is 0, so nothing is ever flagged.
    """
    rng = np.random.default_rng(seed)
    n = 64
    pve = np.zeros((n, n, n), np.float32)
    t1 = np.zeros((n, n, n), np.float32)

    x = np.arange(n)[:, None, None]
    blk = np.zeros((n, n, n), bool)
    blk[:, 8:56, 8:56] = True
    for lo, hi, lab, val in ((6, 40, 3.0, 110.0), (40, 46, 2.0, 60.0),
                             (46, 50, 1.0, 20.0)):
        sel = np.broadcast_to((x >= lo) & (x < hi), (n, n, n)) & blk
        pve[sel] = lab
        t1[sel] = val

    if myelinated:
        yy, zz = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        disc = ((yy - 32) ** 2 + (zz - 32) ** 2) <= 100
        band = np.broadcast_to((x >= 36) & (x < 40), (n, n, n)) & disc[None] & blk
        t1[band] = 85.0

    if wmh:
        # A hypointense white-matter lesion, coded above WM as the pipeline
        # does, deep inside the WM slab.
        pve[20:24, 20:24, 20:24] = 3.6
        t1[20:24, 20:24, 20:24] = 88.0

    if noise:
        t1 = t1 + rng.normal(0.0, noise, t1.shape).astype(np.float32) * (pve > 0)

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return nib.Nifti1Image(pve, affine), nib.Nifti1Image(t1, affine)


def test_noise_free_healthy_cortex_is_untouched():
    """Where the intensities agree exactly with the labels, nothing moves."""
    p0, brain = _phantom(myelinated=False, noise=0.0)
    before = p0.get_fdata().astype(np.float32)
    out = apply_myelin_correction(p0, brain, **CORR)
    assert np.array_equal(out.get_fdata().astype(np.float32), before)


def test_myelinated_band_is_moved_toward_gm():
    """A band of WM-labelled but GM-dark cortex is pushed back to GM."""
    p0, brain = _phantom()
    before = p0.get_fdata().astype(np.float32)
    after = apply_myelin_correction(p0, brain, **CORR).get_fdata().astype(np.float32)

    delta = after - before
    assert delta.min() < 0.0, "nothing was corrected"
    assert delta.max() <= 0.0, "labels moved away from GM"
    # A correction only reaches the surface if it crosses the WM isovalue.
    assert ((before > 2.5) & (after <= 2.5)).sum() > 0


def test_library_defaults_do_not_reach_the_isovalue():
    """The stock defaults are inert here, and that is worth pinning.

    The correction is visible downstream only where it takes voxels across
    the WM isovalue at 2.5.  With ``n_median_filter=1`` and
    ``max_correction=0.5`` nothing crosses, so a run at the defaults leaves
    every surface and every thickness exactly as it found them.
    """
    p0, brain = _phantom()
    before = p0.get_fdata().astype(np.float32)
    after = apply_myelin_correction(
        p0, brain, erosion_mm=6.0).get_fdata().astype(np.float32)
    assert ((before > 2.5) & (after <= 2.5)).sum() == 0


def test_geometry_is_preserved():
    """The result keeps the input's grid, shape and affine."""
    p0, brain = _phantom()
    out = apply_myelin_correction(p0, brain, **CORR)
    assert out.shape == p0.shape
    assert np.array_equal(out.affine, p0.affine)


def test_wm_hyperintensities_are_left_alone():
    """WMH are T1-dark and coded above WM, so they must be excluded.

    Without the guard they land in the boundary band, look exactly like
    myelinated cortex, and get relabelled as grey matter.
    """
    p0, brain = _phantom(wmh=True)
    before = p0.get_fdata().astype(np.float32)
    after = apply_myelin_correction(p0, brain, **CORR).get_fdata().astype(np.float32)

    lesion = before > 3.0
    assert lesion.any(), "the phantom has no lesion to test"
    assert np.array_equal(after[lesion], before[lesion])
