"""Tests for the myelination thickness correction wired into T1Prep."""

import os

import nibabel as nib
import numpy as np

from t1prep.segment import estimate_myelin_thickness_offset
from t1prep.surface_estimation import _load_thickness_offset


def _phantom(myelinated=True, wmh=False, noise=1.0, seed=0):
    """A layered slab whose GM/WM transition is widened inside a disc.

    The label follows the intensity, as a classifier would derive it, so the
    label boundary and the intensity boundary coincide everywhere and only
    the *width* of the transition differs -- which is the observable.
    """
    rng = np.random.default_rng(seed)
    n = 64
    x = np.broadcast_to(np.arange(n)[:, None, None], (n, n, n))
    yy, zz = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    disc = ((yy - 32) ** 2 + (zz - 32) ** 2) <= 100

    half = np.broadcast_to(
        np.where(disc[None, :, :], 1.5 if myelinated else 0.5, 0.5), (n, n, n))
    d = np.clip((39.5 - x) / half, -1.0, 1.0)
    t1v = np.where(x < 46, 85.0 + 25.0 * d, np.where(x < 50, 20.0, 0.0))
    labv = np.where(x < 46, np.clip(2.0 + (t1v - 60.0) / 50.0, 2.0, 3.0),
                    np.where(x < 50, 1.0, 0.0))

    blk = np.zeros((n, n, n), bool)
    blk[:, 8:56, 8:56] = True
    lab = np.zeros((n, n, n), np.float32)
    t1 = np.zeros((n, n, n), np.float32)
    lab[blk] = labv[blk]
    t1[blk] = t1v[blk]

    if wmh:
        # A hypointense lesion, coded above WM as the pipeline does.
        lab[20:24, 20:24, 20:24] = 3.6
        t1[20:24, 20:24, 20:24] = 88.0

    if noise:
        t1 = t1 + rng.normal(0.0, noise, t1.shape).astype(np.float32) * (lab > 0)

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return (nib.Nifti1Image(lab, affine),
            nib.Nifti1Image(t1.astype(np.float32), affine),
            np.broadcast_to(disc[None, :, :], (n, n, n)))


def test_the_widened_patch_is_found():
    """The correction lands where the transition is widened."""
    p0, brain, disc = _phantom()
    off = estimate_myelin_thickness_offset(
        p0, brain, width_pct=80.0).get_fdata().astype(np.float32)

    assert off.max() > 0.1, "nothing was corrected"
    # The disc is ~14% of this phantom's boundary, so where the correction
    # mass lands says more than a ratio of means.
    assert off[disc].sum() > 0.6 * off.sum()


def test_a_uniform_boundary_is_not_corrected():
    """Where every transition is equally sharp there is nothing to correct."""
    p0, brain, _ = _phantom(myelinated=False, noise=0.0)
    off = estimate_myelin_thickness_offset(p0, brain).get_fdata()
    assert not off.any()


def test_the_correction_is_one_sided():
    """Myelination can only make the ribbon look thinner, never thicker."""
    p0, brain, _ = _phantom()
    off = estimate_myelin_thickness_offset(p0, brain, width_pct=80.0)
    assert off.get_fdata().min() >= 0.0


def test_geometry_is_preserved():
    p0, brain, _ = _phantom()
    off = estimate_myelin_thickness_offset(p0, brain, width_pct=80.0)
    assert off.shape == p0.shape
    assert np.array_equal(off.affine, p0.affine)


def test_wm_hyperintensities_do_not_drive_it():
    """WMH are coded above WM and are T1-dark; they must not read as a boundary.

    Without the clip they would look exactly like the myelinated cortex the
    measurement is hunting for.  With it, adding a lesion leaves the whole
    correction field unchanged.
    """
    clean, brain_c, _ = _phantom()
    lesion, brain_l, _ = _phantom(wmh=True)

    a = estimate_myelin_thickness_offset(
        clean, brain_c, width_pct=80.0).get_fdata()
    b = estimate_myelin_thickness_offset(
        lesion, brain_l, width_pct=80.0).get_fdata()

    assert np.abs(b - a).max() < 1e-3, "the lesion perturbed the correction"


class _Log:
    def info(self, *a, **k):
        pass


def test_offset_is_resampled_onto_the_target_grid(tmp_path):
    """The consumer resamples the whole-brain map onto the hemisphere grid.

    The two grids differ -- the hemisphere maps are resliced to the target
    resolution and then cropped -- so this is where they are reconciled.
    """
    src = np.zeros((32, 32, 32), np.float32)
    src[8:24, 8:24, 8:24] = 0.5
    nib.save(nib.Nifti1Image(src, np.diag([1.0, 1.0, 1.0, 1.0])),
             os.path.join(tmp_path, "sub_thickness_offset.nii"))

    # Half the voxel size, shifted origin: a different grid in both senses.
    target_aff = np.diag([0.5, 0.5, 0.5, 1.0])
    target_aff[:3, 3] = [4.0, 4.0, 4.0]
    target = nib.Nifti1Image(np.zeros((40, 40, 40), np.float32), target_aff)

    out = _load_thickness_offset(str(tmp_path), "sub", "nii", target, _Log())

    assert out is not None
    assert out.shape == target.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert abs(float(out.max()) - 0.5) < 1e-5
    # The block spans src voxels 8..24, i.e. 8..24 mm; in the target that is
    # (8 - 4)/0.5 = 8 to (24 - 4)/0.5 = 40, so it fills from index 8 on.
    assert out[20, 20, 20] > 0.49
    assert out[2, 2, 2] == 0.0


def test_no_correction_file_means_no_correction(tmp_path):
    """Not asking for --myelin must leave the thickness estimation alone."""
    target = nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4))
    assert _load_thickness_offset(str(tmp_path), "sub", "nii", target,
                                  _Log()) is None
