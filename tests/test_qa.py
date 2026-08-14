"""Tests for the image quality measures in :mod:`t1prep.qa`.

The tests use a small synthetic phantom (concentric CSF/GM/WM spheres) so
that the ground truth is known: adding a smooth multiplicative field must
not change the noise rating, and adding noise must not change the
inhomogeneity rating.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import qa  # noqa: E402


def _phantom(shape=(96, 96, 96)):
    """Concentric CSF/GM/WM spheres with CAT12-like tissue intensities."""
    centre = np.array(shape) / 2.0 - 0.5
    grid = np.ogrid[: shape[0], : shape[1], : shape[2]]
    radius = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, centre)))

    p0 = np.zeros(shape, dtype=np.float32)
    p0[radius < 30] = 1.0
    p0[radius < 25] = 2.0
    p0[radius < 17] = 3.0
    # Soften the boundaries so that partial volume effects are present.
    p0 = gaussian_filter(p0, 0.7)
    p0[radius >= 31] = 0.0

    intensity = np.zeros(shape, dtype=np.float32)
    for level, value in ((1.0, 300.0), (2.0, 700.0), (3.0, 1000.0)):
        intensity += np.clip(1.0 - np.abs(p0 - level), 0.0, 1.0) * value
    return p0, intensity.astype(np.float32)


def _smooth_field(shape, amplitude, seed=0):
    """Smooth multiplicative field with mean one and the given amplitude."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(shape).astype(np.float32), 12.0)
    field /= np.abs(field).max()
    return 1.0 + amplitude * field


def _measure(p0, intensity, vx=1.0):
    vx_vol = np.full(3, float(vx))
    result = qa.estimate_qa(p0, intensity, vx_vol, vx_vol)["qualitymeasures"]
    return {k: v["value"] for k, v in result.items()}


class TestLowLevelHelpers(unittest.TestCase):
    """The CAT12 building blocks must match their reference definitions."""

    def test_localstat_sd_uses_the_six_neighbourhood(self):
        rng = np.random.default_rng(1)
        vol = rng.standard_normal((9, 8, 7))
        mask = rng.random((9, 8, 7)) > 0.3

        got = qa._localstat_sd(vol, mask)

        # Brute-force reference: sample SD over the voxels with Euclidean
        # distance <= 1 that belong to the mask (cat_vol_localstat, F_STD).
        offsets = [(0, 0, 0), (-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        expected = np.zeros_like(got)
        for x in range(vol.shape[0]):
            for y in range(vol.shape[1]):
                for z in range(vol.shape[2]):
                    if not mask[x, y, z]:
                        continue
                    values = []
                    for dx, dy, dz in offsets:
                        i, j, k = x + dx, y + dy, z + dz
                        if not (0 <= i < vol.shape[0] and 0 <= j < vol.shape[1]
                                and 0 <= k < vol.shape[2]):
                            continue
                        if mask[i, j, k]:
                            values.append(vol[i, j, k])
                    if len(values) > 1:
                        expected[x, y, z] = np.std(values, ddof=1)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_localstat_sd_ignores_the_diagonal_neighbours(self):
        # A single bright diagonal neighbour must not contribute.
        vol = np.zeros((5, 5, 5))
        vol[3, 3, 2] = 100.0
        mask = np.ones((5, 5, 5), dtype=bool)
        self.assertAlmostEqual(qa._localstat_sd(vol, mask)[2, 2, 2], 0.0)

    def test_reduce_meanm_averages_defined_voxels_only(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        vol[0, 0, 0] = 2.0
        vol[0, 0, 1] = 4.0
        vol[1, 1, 1] = 6.0
        reduced = qa._reduce_meanm(vol, np.array([2, 2, 2]))
        self.assertEqual(reduced.shape, (2, 2, 2))
        # Three defined voxels in the first block -> their mean.
        self.assertAlmostEqual(float(reduced[0, 0, 0]), 4.0)
        # Blocks with fewer than two defined voxels are dropped.
        self.assertEqual(float(reduced[1, 1, 1]), 0.0)

    def test_disterode_removes_a_shell_of_the_given_width(self):
        mask = np.zeros((21, 21, 21), dtype=bool)
        mask[5:16, 5:16, 5:16] = True
        eroded = qa._disterode(mask, 1.0, np.ones(3))
        self.assertTrue(eroded[10, 10, 10])
        self.assertFalse(eroded[5, 10, 10])
        self.assertTrue(eroded[7, 10, 10])

    def test_smooth_fwhm_interprets_its_argument_as_fwhm(self):
        # A delta smoothed with FWHM f must have half its peak at f/2 voxels.
        vol = np.zeros((41, 41, 41), dtype=np.float32)
        vol[20, 20, 20] = 1.0
        smoothed = qa._smooth_fwhm(vol, 8.0)
        peak = smoothed[20, 20, 20]
        self.assertAlmostEqual(smoothed[24, 20, 20] / peak, 0.5, places=2)


class TestQualityMeasures(unittest.TestCase):
    """End-to-end behaviour of :func:`t1prep.qa.estimate_qa`."""

    @classmethod
    def setUpClass(cls):
        cls.p0, cls.intensity = _phantom()

    def test_measures_are_invariant_to_global_scaling(self):
        base = _measure(self.p0, self.intensity)
        scaled = _measure(self.p0, self.intensity * 137.0)
        for key in ("NCR", "ICR", "contrastr", "res_ECR"):
            self.assertAlmostEqual(base[key], scaled[key], places=3,
                                   msg=f"{key} is not scale invariant")

    def test_noise_rating_is_invariant_to_a_bias_field(self):
        rng = np.random.default_rng(7)
        noisy = self.intensity + rng.standard_normal(
            self.intensity.shape).astype(np.float32) * 20.0
        plain = _measure(self.p0, noisy)
        biased = _measure(
            self.p0, noisy * _smooth_field(noisy.shape, 0.35))
        # A 35 % inhomogeneity must not move NCR by more than a few percent.
        self.assertLess(abs(biased["NCR"] - plain["NCR"]) / plain["NCR"], 0.10)
        # ... but it must be visible in ICR.
        self.assertGreater(biased["ICR"], plain["ICR"] * 1.3)

    def test_ncr_increases_with_noise(self):
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(self.intensity.shape).astype(np.float32)
        values = [_measure(self.p0, self.intensity + noise * level)["NCR"]
                  for level in (5.0, 20.0, 40.0)]
        self.assertTrue(all(np.diff(values) > 0), f"NCR not monotone: {values}")

    def test_res_ecr_increases_when_the_image_is_blurred(self):
        rng = np.random.default_rng(5)
        noisy = self.intensity + rng.standard_normal(
            self.intensity.shape).astype(np.float32) * 10.0
        sharp = _measure(self.p0, noisy)["res_ECR"]
        blurred = _measure(self.p0, gaussian_filter(noisy, 1.5))["res_ECR"]
        self.assertGreater(blurred, sharp)

    def test_result_structure_and_grades(self):
        result = qa.estimate_qa(self.p0, self.intensity,
                                np.ones(3), np.ones(3))
        measures = result["qualitymeasures"]
        for key in ("NCR", "CNR", "ICR", "contrastr", "res_RMS", "res_ECR"):
            self.assertIn("value", measures[key])
            self.assertIn("mark", measures[key])
            self.assertIn("desc", measures[key])
        for key in ("IQR", "SIQR"):
            self.assertIn("value", measures[key])
            self.assertIn("grade", measures[key])
        self.assertEqual(measures["res_RMS"]["value"], 1.0)
        # CNR is the reciprocal of NCR (both are rounded) and shares its mark.
        self.assertAlmostEqual(
            measures["CNR"]["value"] / (1.0 / measures["NCR"]["value"]),
            1.0, places=1)
        self.assertEqual(measures["CNR"]["mark"], measures["NCR"]["mark"])

    def test_degenerate_input_returns_undefined_measures(self):
        empty = np.zeros((40, 40, 40), dtype=np.float32)
        measures = qa.estimate_qa(
            empty, empty, np.ones(3), np.ones(3))["qualitymeasures"]
        self.assertIsNone(measures["NCR"]["value"])
        self.assertEqual(measures["IQR"]["grade"], "NA")


class TestRatingScale(unittest.TestCase):
    """The mark/grade helpers must stay consistent with the bounds."""

    def test_mark_maps_the_bounds_to_one_and_six(self):
        for name, (best, worst) in qa._RATING_BOUNDS.items():
            self.assertAlmostEqual(qa._mark(best, best, worst), 1.0,
                                   msg=f"{name} best bound")
            self.assertAlmostEqual(qa._mark(worst, best, worst), 6.0,
                                   msg=f"{name} worst bound")

    def test_mark_is_clamped(self):
        self.assertEqual(qa._mark(-1e6, 0.0, 1.0), 0.5)
        self.assertEqual(qa._mark(1e6, 0.0, 1.0), 10.5)
        self.assertTrue(np.isnan(qa._mark(float("nan"), 0.0, 1.0)))

    def test_iqr_is_dominated_by_the_worst_mark(self):
        self.assertGreater(qa._iqr([1.0, 6.0], power=8), 5.0)
        self.assertAlmostEqual(qa._iqr([2.0, 2.0], power=8), 2.0)

    def test_grades_and_scores(self):
        self.assertEqual(qa.mark_to_grade(1.0), "A+")
        self.assertEqual(qa.mark_to_grade(5.9), "D")
        self.assertEqual(qa.mark_to_rps(1.0), 95.0)
        self.assertEqual(qa.mark_to_rps(10.5), 0.0)


if __name__ == "__main__":
    unittest.main()
