"""LAS with tissue peaks that are given instead of estimated (``apply_LAS(peaks=...)``).

A caller that knows the intensity of a pure tissue -- the mean of the mixture
of Gaussians a unified segmentation estimated -- hands it over, and that is
where the tissue ends up: the global mapping runs through it and the local
field of the tissue is held at that level.  Without ``peaks`` nothing changes.
"""

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import _intensity, apply_LAS  # noqa: E402

CSF, GM, WM = 100.0, 250.0, 400.0


def _shells(n=64):
    """A WM ball in a GM shell in a CSF shell, under a bias field of +-15% along x."""
    axis = np.arange(n, dtype=np.float32) - (n - 1) / 2
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    label = np.zeros((n, n, n), np.float32)
    label[r < 28] = 1
    label[r < 22] = 2
    label[r < 14] = 3
    pure = np.array([0, CSF, GM, WM], np.float32)[label.astype(int)]
    noise = np.random.default_rng(0).normal(0, 2, label.shape)
    image = ((pure * (1 + 0.15 * x / (n / 2)) + noise) * (label > 0)).astype(np.float32)
    inside = {"CSF": (r > 23.5) & (r < 26.5), "GM": (r > 15.5) & (r < 20.5), "WM": r < 12}
    return nib.Nifti1Image(image, np.eye(4)), nib.Nifti1Image(label, np.eye(4)), inside


class TestGivenPeaks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t1, cls.label, cls.inside = _shells()

    def las(self, peaks=None):
        out = 3 * np.asarray(apply_LAS(self.t1, self.label, peaks=peaks).dataobj)
        return {name: out[mask] for name, mask in self.inside.items()}

    def test_without_peaks_every_tissue_sits_at_its_label(self):
        out = self.las()
        for value, name in enumerate(("CSF", "GM", "WM"), start=1):
            self.assertAlmostEqual(float(np.median(out[name])), value, delta=0.01)

    def test_a_given_gm_peak_is_where_pure_gm_ends_up(self):
        # GM of 250 between CSF at 100 and a GM peak of 262.5 is 92% GM, and
        # with a peak of 237.5 it is 8% of the way from GM to WM at 400
        for peak, expected in ((262.5, 1 + 150 / 162.5), (237.5, 2 + 12.5 / 162.5)):
            with self.subTest(peak=peak):
                out = self.las((None, peak, None))
                self.assertAlmostEqual(float(np.median(out["GM"])), expected, delta=0.01)
                self.assertAlmostEqual(float(np.median(out["CSF"])), 1.0, delta=0.01)
                self.assertAlmostEqual(float(np.median(out["WM"])), 3.0, delta=0.01)

    def test_the_local_field_still_follows_the_bias(self):
        # left in, the bias gives GM an SD of 0.083 on this scale (the global
        # mapping alone)
        free = self.las()["GM"].std()
        held = self.las((None, 262.5, None))["GM"].std()
        self.assertLess(free, 0.03)
        self.assertLess(held, 0.03)

    def test_the_true_peaks_change_nothing(self):
        np.testing.assert_allclose(self.las((CSF, GM, WM))["GM"], self.las()["GM"], atol=0.01)

    def test_the_global_mapping_runs_through_the_given_peak(self):
        _, estimated = _intensity.global_intensity_norm(self.t1, self.label)
        image, stats = _intensity.global_intensity_norm(self.t1, self.label,
                                                        peaks=(None, 262.5, None))
        self.assertEqual(stats["gm"], 262.5)
        self.assertEqual(stats["csf"], estimated["csf"])
        self.assertEqual(stats["wm"], estimated["wm"])
        src = np.asarray(self.t1.dataobj)
        between = (src > stats["csf"]) & (src < 262.5)
        expected = (1 + (src[between] - stats["csf"]) / (262.5 - stats["csf"])) / 3
        np.testing.assert_allclose(np.asarray(image.dataobj)[between], expected, rtol=1e-5)

    def test_peaks_out_of_order_are_refused(self):
        for peaks in ((None, 1e6, None), (300.0, None, None), (None, None, 200.0)):
            with self.subTest(peaks=peaks):
                with self.assertRaisesRegex(ValueError, "increase from CSF to GM to WM"):
                    apply_LAS(self.t1, self.label, peaks=peaks)

    def test_three_values_are_needed(self):
        with self.assertRaisesRegex(ValueError, "three values"):
            apply_LAS(self.t1, self.label, peaks=(250.0,))


if __name__ == "__main__":
    unittest.main()
