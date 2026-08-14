"""Tests for the blood-vessel correction in :mod:`t1prep.vessels`.

The phantom is concentric CSF/GM/WM spheres carrying thin radial white matter
blades, plus a vessel tree in the CSF shell.

The negative controls carry more weight here than the detection does.  Thin
gyral white matter is bright, thin and ridge-like -- indistinguishable from a
vessel by any local cue -- and an earlier version of this correction set it to
CSF across the brain.  ``test_thin_gyral_white_matter_is_never_touched`` is
the regression guard for that, and it is why the phantom has blades at all: a
solid WM ball cannot express the failure.
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

from t1prep import vessels  # noqa: E402

VX = (0.5, 0.5, 0.5)
N = 160
C = N // 2


def _phantom(with_vessel=True, seed=0):
    """Concentric brain with thin gyral WM blades and a sulcal vessel tree.

    The blades matter as much as the vessel.  A 2 mm blade running out into a
    gyrus is bright, thin and ridge-like -- every local cue a vessel gives --
    and an earlier version of this correction set exactly those to CSF.  A
    phantom with a solid WM ball cannot catch that regression.

    The vessel tubes run parallel to z at a fixed distance from the axis, so
    they stay inside the CSF shell along their whole length.  A tube whose
    radius drifts would touch white matter somewhere and then be protected as
    part of the WM tree, which is correct behaviour but tests nothing.
    """
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    radius = np.sqrt((zz - C) ** 2 + (yy - C) ** 2 + (xx - C) ** 2) * VX[0]
    angle = np.arctan2(yy - C, xx - C)

    label = np.zeros((N, N, N), np.float32)
    label[radius < 34] = 1.0  # CSF
    label[radius < 30] = 2.0  # GM ribbon
    label[radius < 20] = 3.0  # WM body

    # thin radial WM blades (~2 mm) reaching out through the GM ribbon
    blade = (
        (np.abs(np.sin(angle * 6)) < 0.055)
        & (radius >= 20)
        & (radius < 28)
        & (np.abs((zz - C) * VX[2]) < 30)
    )
    label[blade] = 3.0

    image = np.zeros_like(label)
    image[label > 0] = label[label > 0] / 3.0

    # vessel tubes in the CSF shell, clear of both GM and WM
    vessel = np.zeros((N, N, N), bool)
    for dx, dy, rad in ((0, 62, 1.75), (22, 58, 1.75), (-22, 58, 1.75)):
        tube = np.sqrt(
            ((xx - (C + dx)) * VX[0]) ** 2 + ((yy - (C + dy)) * VX[1]) ** 2
        ) < rad
        vessel |= tube & (np.abs((zz - C) * VX[2]) < 14)
    if with_vessel:
        image[vessel] = 1.02  # about WM intensity
        label[vessel] = 3.0  # the segmentation calls it WM -- the real case

    rng = np.random.default_rng(seed)
    image = np.clip(image + rng.normal(0, 0.01, image.shape), 0, None)
    return image.astype(np.float32), label, vessel, blade, radius


class BloodVesselCorrectionTests(unittest.TestCase):
    """Detection, the safety gate, and the correction itself."""

    @classmethod
    def setUpClass(cls):
        cls.affine = np.diag(list(VX) + [1.0])
        cls.image, cls.label, cls.vessel, cls.blade, cls.radius = _phantom(True)
        # The phantom is not in MNI space, so the shipped spatial prior would
        # be meaningless here; a neutral prior isolates the local cues.
        cls.neutral = np.ones_like(cls.image)
        cls.weight = vessels.vessel_weight(
            cls.image, cls.label, VX, bv_prior=cls.neutral
        )
        cls.gm = (cls.label > 1.5) & (cls.label < 2.5) & ~cls.vessel
        cls.crown = cls.gm & (cls.radius > 27)
        cls.wm_core = cls.radius < 18

    def _brain(self):
        return nib.Nifti1Image(self.image, self.affine)

    def _label(self):
        return nib.Nifti1Image(self.label, self.affine)

    # -- detection ---------------------------------------------------------

    def test_vessel_is_detected(self):
        self.assertGreater(self.weight[self.vessel].mean(), 0.5)

    def test_detected_volume_matches_the_vessel(self):
        """The mask must not sprawl beyond the structure it is correcting.

        The lower bound is deliberately loose.  This phantom's vessel sits in
        a clean CSF shell, which is the easy case; calibration is driven by
        the Colin27 measurements, where the target is a 1-2 mm vessel inside a
        sulcus and recall is around 20%.  Tightening this bound would mean
        tuning against the phantom instead of the anatomy.
        """
        found = float((self.weight > 0.5).sum()) * float(np.prod(VX))
        truth = float(self.vessel.sum()) * float(np.prod(VX))
        self.assertGreater(found, 0.35 * truth)
        self.assertLess(found, 1.5 * truth)

    # -- negative controls -------------------------------------------------

    def test_grey_matter_is_spared(self):
        self.assertLess(self.weight[self.gm].mean(), 0.02)

    def test_thin_gyral_crown_is_spared(self):
        self.assertLess(self.weight[self.crown].mean(), 0.02)

    def test_white_matter_body_is_never_touched(self):
        self.assertEqual(self.weight[self.wm_core].max(), 0.0)

    def test_thin_gyral_white_matter_is_never_corrected(self):
        """The regression this design exists to prevent.

        Thin gyral WM is bright, thin and ridge-like, so shape cues alone rate
        it exactly like a vessel.  It survives because it is neither brighter
        than WM nor detached from the white matter tree.

        The assertion is that no blade voxel reaches the correction threshold,
        not that every one scores exactly zero: the phantom blades are 1 mm
        wide, thinner than real gyral WM, and a small sub-threshold response
        there is expected.  On Colin27, where the anatomy is real, this
        corresponds to 0.7% of the 41000 mm^3 of thin gyral WM present picking
        up any weight at all.
        """
        self.assertLess(self.weight[self.blade].max(), 0.5)
        self.assertLess(self.weight[self.blade].mean(), 0.01)

    def test_vessel_free_image_passes_through_unchanged(self):
        image, label, _, _, _ = _phantom(with_vessel=False, seed=1)
        brain_out, label_out = vessels.apply_blood_vessel_correction(
            nib.Nifti1Image(image, self.affine),
            nib.Nifti1Image(label, self.affine),
            strength=1.0,
            bv_prior=self.neutral,
        )
        np.testing.assert_array_equal(label_out.get_fdata(), label)
        np.testing.assert_array_equal(brain_out.get_fdata(), image)

    def test_gate_blocks_a_sub_threshold_detection(self):
        """The phantom tree is 763 mm^3, below CAT12's 1000 mm^3 gate."""
        brain_out, _ = vessels.apply_blood_vessel_correction(
            self._brain(), self._label(), strength=1.0, bv_prior=self.neutral
        )
        np.testing.assert_array_equal(brain_out.get_fdata(), self.image)

    def test_strength_zero_is_a_no_op(self):
        brain_in, label_in = self._brain(), self._label()
        brain_out, label_out = vessels.apply_blood_vessel_correction(
            brain_in, label_in, strength=0.0
        )
        self.assertIs(brain_out, brain_in)
        self.assertIs(label_out, label_in)

    # -- correction --------------------------------------------------------

    def _corrected(self):
        return vessels.apply_blood_vessel_correction(
            self._brain(),
            self._label(),
            strength=1.0,
            bv_prior=self.neutral,
            min_volume=300.0,
        )

    def test_vessel_intensity_is_reduced(self):
        """CAT12 caps at 1 - Ybv/4, i.e. just above GM, not at CSF.

        The rest of the job is done by the label correction below and by the
        surface-time net; pushing the intensity to CSF here would fight AMAP.
        """
        brain_out, _ = self._corrected()
        after = brain_out.get_fdata()[self.vessel].mean()
        self.assertLess(after, 0.833)  # below the GM/WM midpoint
        self.assertGreater(after, 1.0 / 3.0)

    def test_vessel_label_is_pulled_towards_csf(self):
        """The vessel must end up below GM, not merely reduced."""
        _, label_out = self._corrected()
        before = self.label[self.vessel].mean()
        after = label_out.get_fdata()[self.vessel].mean()
        self.assertAlmostEqual(before, 3.0, places=2)
        self.assertLess(after, 2.0)

    def test_correction_leaves_grey_matter_alone(self):
        _, label_out = self._corrected()
        after = label_out.get_fdata()
        self.assertLess(abs(after[self.gm].mean() - self.label[self.gm].mean()), 0.01)
        np.testing.assert_allclose(after[self.wm_core], self.label[self.wm_core])

    def test_correction_only_ever_removes_signal(self):
        brain_out, _ = self._corrected()
        self.assertTrue((brain_out.get_fdata() <= self.image + 1e-5).all())


class GeodesicIsolationTests(unittest.TestCase):
    """The cue that separates an attached vessel from ordinary tissue."""

    @classmethod
    def setUpClass(cls):
        cls.image, cls.label, cls.vessel, cls.blade, cls.radius = _phantom(True)

    def test_vessel_is_more_isolated_than_grey_matter(self):
        ym3 = self.image * 3
        seed = (ym3 >= 1.7) & (self.label > 1.0) & ~self.vessel
        detour = vessels.geodesic_isolation(
            ym3, seed, ym3 < 1.7, VX, limit_strict=0.03, limit_loose=0.5
        )
        gm = (self.label > 1.5) & (self.label < 2.5) & ~self.vessel
        self.assertGreater(detour[self.vessel].mean(), 4.0)
        self.assertLess(detour[gm].mean(), 0.5)


class SurfaceNetTests(unittest.TestCase):
    """The last-resort suppression applied right before PBT."""

    @classmethod
    def setUpClass(cls):
        _, cls.label, cls.vessel, cls.blade, cls.radius = _phantom(True)
        cls.out = vessels.suppress_vessels_for_surface(cls.label, VX)
        cls.wm_core = cls.radius < 18
        cls.crown = (
            (cls.label > 1.5) & (cls.label < 2.5) & ~cls.vessel & (cls.radius > 27)
        )

    def test_vessel_is_lowered(self):
        self.assertLess(self.out[self.vessel].mean(), 2.0)

    def test_gyral_crown_is_spared(self):
        self.assertLess(abs(self.out[self.crown].mean() - 2.0), 0.02)

    def test_white_matter_body_is_spared(self):
        np.testing.assert_allclose(self.out[self.wm_core], 3.0)

    def test_never_brightens(self):
        self.assertTrue((self.out <= self.label + 1e-5).all())

    def test_strength_zero_is_a_no_op(self):
        out = vessels.suppress_vessels_for_surface(self.label, VX, strength=0.0)
        np.testing.assert_array_equal(out, self.label)


if __name__ == "__main__":
    unittest.main()
