"""Tests for the conventional non-cortical grey matter rule in :mod:`t1prep.nogm`.

The phantom is a concentric CSF/GM/WM brain with a ventricle carved out of the
white-matter core, and a one-voxel film of grey-matter-valued label between the
ventricle CSF and the surrounding white matter -- the partial-volume artefact
the ``nogm`` step exists to remove.

As in :mod:`tests.test_vessels`, the negative control carries more weight than
the detection.  The cortical ribbon is also grey matter between white matter
and CSF, so a rule built only on the sandwich test deletes the cortex; the
guard against that is the grey-matter fraction ceiling, and
``test_cortical_ribbon_is_never_touched`` is the regression test for it.

The atlas admission region is bypassed throughout (``admission`` is passed
explicitly) because the phantom is not in MNI space, so Neuromorphometrics
would place its regions arbitrarily on it.  What is under test here is the
geometric rule; the admission region is a separate, purely spatial mask.
"""

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import nogm  # noqa: E402

VX = 0.5
N = 128
C = N // 2


def _phantom():
    """Concentric brain with a ventricle and a periventricular GM film.

    Returns:
        ``(p0_img, film, ribbon)`` -- the label map, the mask of the artificial
        white-matter/CSF partial volume, and the mask of the cortical ribbon
        that must survive.
    """
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
    r = np.sqrt((zz - C) ** 2 + (yy - C) ** 2 + (xx - C) ** 2) * VX

    p0 = np.zeros((N, N, N), np.float32)
    p0[r < 30.0] = 3.0                     # white matter core
    ribbon = (r >= 30.0) & (r < 32.5)      # 2.5 mm cortical ribbon
    p0[ribbon] = 2.0
    p0[(r >= 32.5) & (r < 35.0)] = 1.0     # surrounding CSF

    # Ventricle carved out of the white matter, with a one-voxel film of
    # GM-valued label at its border: half white matter, half CSF by volume,
    # so an intensity-driven segmentation calls it grey matter.
    rv = np.sqrt((zz - C) ** 2 + (yy - C) ** 2 + ((xx - C) * 2.0) ** 2) * VX
    p0[rv < 10.0] = 1.0
    film = (rv >= 10.0) & (rv < 10.0 + VX)
    p0[film] = 2.0

    affine = np.diag([VX, VX, VX, 1.0]).astype(np.float64)
    affine[:3, 3] = -C * VX
    return nib.Nifti1Image(p0, affine), film, ribbon


class NoGMMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p0, cls.film, cls.ribbon = _phantom()
        cls.admission = np.ones(cls.p0.shape, dtype=bool)
        cls.mask = nogm.noncortical_gm_mask(
            cls.p0, admission=cls.admission
        )

    def test_periventricular_film_is_detected(self):
        """The WM/CSF partial volume is what the rule is for."""
        hit = (self.mask & self.film).sum() / self.film.sum()
        self.assertGreater(hit, 0.5, f"only {100 * hit:.1f}% of the film found")

    def test_cortical_ribbon_is_never_touched(self):
        """The ribbon is also a WM/CSF sandwich; deleting it is the failure mode.

        Without the grey-matter fraction ceiling this is what the sandwich test
        does to the cortex, so the bound is deliberately tight.
        """
        hit = (self.mask & self.ribbon).sum() / self.ribbon.sum()
        self.assertLess(hit, 0.02, f"{100 * hit:.1f}% of the ribbon removed")

    def test_white_matter_and_csf_are_not_flagged(self):
        """Only grey-matter-valued voxels are ever reassigned."""
        p0 = np.asanyarray(self.p0.dataobj)
        self.assertFalse(self.mask[p0 > nogm.GM_RANGE[1]].any())
        self.assertFalse(self.mask[p0 < nogm.GM_RANGE[0]].any())

    def test_admission_region_is_respected(self):
        """Nothing outside the admission region is corrected, whatever the cues."""
        half = np.zeros(self.p0.shape, dtype=bool)
        half[: N // 2] = True
        masked = nogm.noncortical_gm_mask(self.p0, admission=half)
        self.assertFalse(masked[N // 2:].any())
        self.assertTrue(masked[: N // 2].any())


class ApplyTests(unittest.TestCase):
    """The reassignment must match ``deepmriprep``'s ``apply_nogm`` exactly."""

    def test_tissue_sum_is_preserved(self):
        p0, _, _ = _phantom()
        arr = np.asanyarray(p0.dataobj, dtype=np.float32)
        gm, wm, csf = nogm._one_hot_gm_wm_csf(arr)
        before = gm + wm + csf
        mask = nogm.noncortical_gm_mask(
            p0, admission=np.ones(arr.shape, dtype=bool)
        )
        gm, wm, csf = nogm.apply_noncortical_gm(gm, wm, csf, mask)
        np.testing.assert_allclose(gm + wm + csf, before, atol=0, rtol=0)

    def test_label_map_is_unchanged(self):
        """The reassignment is exactly neutral on ``p0``.

        ``segment.py`` rebuilds the label map as ``csf + 2*gm + 3*wm``, and the
        even split changes that by ``gm/2 - 2*gm + 3*gm/2 == 0``.  This is what
        makes the step unable to move the surfaces, so it is worth pinning: a
        future change to the split ratio would silently start doing so.
        """
        rng = np.random.default_rng(0)
        gm, wm, csf = (rng.random(5000).astype(np.float32) for _ in range(3))
        total = gm + wm + csf
        gm, wm, csf = gm / total, wm / total, csf / total
        before = csf + 2 * gm + 3 * wm
        mask = np.ones(gm.shape, dtype=bool)
        gm, wm, csf = nogm.apply_noncortical_gm(gm, wm, csf, mask)
        np.testing.assert_allclose(csf + 2 * gm + 3 * wm, before, atol=1e-6)

    def test_grey_matter_is_split_evenly(self):
        gm = np.array([[[0.8]]], np.float32)
        wm = np.array([[[0.1]]], np.float32)
        csf = np.array([[[0.1]]], np.float32)
        mask = np.array([[[True]]])
        gm, wm, csf = nogm.apply_noncortical_gm(gm, wm, csf, mask)
        self.assertEqual(gm[0, 0, 0], 0.0)
        self.assertAlmostEqual(float(wm[0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(csf[0, 0, 0]), 0.5, places=6)

    def test_one_hot_matches_deepmriprep(self):
        """Guards the hand-rolled three-channel build against upstream."""
        try:
            import torch
            from deepmriprep.segment import one_hot
        except ImportError:  # pragma: no cover - deepmriprep is optional here
            self.skipTest("deepmriprep not available")
        rng = np.random.default_rng(0)
        p0 = rng.uniform(0.0, 3.0, size=(6, 7, 8)).astype(np.float32)
        gm, wm, csf = nogm._one_hot_gm_wm_csf(p0)
        ref = one_hot(torch.from_numpy(p0)[None, None])
        np.testing.assert_allclose(gm, ref[0, 2].numpy(), atol=1e-6)
        np.testing.assert_allclose(wm, ref[0, 3].numpy(), atol=1e-6)
        np.testing.assert_allclose(csf, ref[0, 1].numpy(), atol=1e-6)


class RunTests(unittest.TestCase):
    def test_returns_the_keys_segment_reads(self):
        p0, _, _ = _phantom()
        out = nogm.run_segment_nogm_conventional(
            p0, wj_affine=1.0, admission=np.ones(p0.shape, dtype=bool)
        )
        for key in ("p1_large", "p2_large", "p3_large", "gmv", "tiv"):
            self.assertIn(key, out)
        for key in ("p1_large", "p2_large", "p3_large"):
            self.assertEqual(out[key].shape, p0.shape)
            np.testing.assert_allclose(out[key].affine, p0.affine)
        self.assertGreater(out["tiv"], out["gmv"])

    def test_wj_affine_scales_the_volumes(self):
        p0, _, _ = _phantom()
        kw = dict(admission=np.ones(p0.shape, dtype=bool))
        one = nogm.run_segment_nogm_conventional(p0, wj_affine=1.0, **kw)
        two = nogm.run_segment_nogm_conventional(p0, wj_affine=2.0, **kw)
        self.assertAlmostEqual(two["gmv"], 2 * one["gmv"], places=4)
        self.assertAlmostEqual(two["tiv"], 2 * one["tiv"], places=4)


if __name__ == "__main__":
    unittest.main()
