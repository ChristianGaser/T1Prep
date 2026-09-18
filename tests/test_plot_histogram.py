"""The parts of ``t1prep.plot_histogram`` that carry numbers.

The drawing itself is left to matplotlib; what is checked here is what the
plot is made of -- the MATLAB binning rule, the palettes taken from
``cat_io_colormaps``, the shortened legend labels and the statistics of the
printed table.
"""

import re
import sys
import unittest
from pathlib import Path

import numpy as np

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from t1prep import plot_histogram as cph
except Exception as exc:  # pragma: no cover - depends on scipy/matplotlib
    raise unittest.SkipTest(f"plot_histogram unavailable: {exc}")


class TestTheBinningFollowsMatlab(unittest.TestCase):
    """``hist_centers`` reproduces MATLAB's ``hist(y, centers)``."""

    def setUp(self):
        self.centers = np.array([0.0, 1.0, 2.0, 3.0])

    def test_borders_are_the_midpoints_between_centers(self):
        counts = cph.hist_centers(np.array([0.4, 0.6, 1.4, 1.6]), self.centers)
        np.testing.assert_array_equal(counts, [1, 2, 1, 0])

    def test_a_value_on_a_border_goes_to_the_upper_bin(self):
        counts = cph.hist_centers(np.array([0.5, 1.5, 2.5]), self.centers)
        np.testing.assert_array_equal(counts, [0, 1, 1, 1])

    def test_the_outer_bins_are_open(self):
        # Everything below the first and above the last center is counted,
        # which is why both are dropped again before plotting
        counts = cph.hist_centers(np.array([-1e6, 1e6]), self.centers)
        np.testing.assert_array_equal(counts, [1, 0, 0, 1])


class TestThePalettes(unittest.TestCase):
    """``colormap`` resizes as ``cat_io_colormaps`` does."""

    def test_a_categorical_palette_is_truncated(self):
        three = cph.colormap("nejm", 3)
        self.assertEqual(three.shape, (3, 3))
        np.testing.assert_allclose(three[0], [0xBC / 255, 0x3C / 255, 0x29 / 255])

    def test_more_colours_than_the_palette_holds_are_interpolated(self):
        many = cph.colormap("nejm", 40)
        self.assertEqual(many.shape, (40, 3))
        # The ends of the palette are kept, only the middle is filled in
        np.testing.assert_allclose(many[0], cph.colormap("nejm")[0])
        np.testing.assert_allclose(many[-1], cph.colormap("nejm")[-1])

    def test_an_unknown_name_is_refused(self):
        with self.assertRaises(ValueError):
            cph.colormap("no-such-palette", 3)


class TestTheLegendLabels(unittest.TestCase):
    """Only the part that tells the inputs apart ends up in the legend."""

    def test_the_common_prefix_and_suffix_are_dropped(self):
        labels = cph.shorten_names(["/d/sub-01_T1w.nii", "/d/sub-02_T1w.nii"])
        self.assertEqual(labels, ["1", "2"])

    def test_paths_differing_only_in_their_folder_keep_the_folder(self):
        self.assertEqual(cph.shorten_names(["/a/x/p1.nii", "/a/y/p1.nii"]),
                         ["x", "y"])

    def test_a_single_input_keeps_its_file_name(self):
        self.assertEqual(cph.shorten_names(["/a/b/p1.nii"]), ["p1.nii"])

    def test_identical_names_fall_back_to_the_file_name(self):
        # Nothing distinguishing is left, so the bare name is more useful
        self.assertEqual(cph.shorten_names(["/a/p1.nii", "/a/p1.nii"]),
                         ["p1.nii", "p1.nii"])

    def test_statistic_maps_are_recognised_by_their_name(self):
        self.assertTrue(cph._is_statistic_map("/d/spmT_0001.nii"))
        self.assertTrue(cph._is_statistic_map("/d/D_0001.nii.gz"))
        self.assertFalse(cph._is_statistic_map("/d/mwp1sub-01.nii"))


class TestTheFittedCurves(unittest.TestCase):
    """``fit_density`` returns a density, whatever family is asked for."""

    def setUp(self):
        self.sample = np.random.default_rng(0).normal(2.0, 1.0, 20000)
        self.x = np.linspace(-4.0, 8.0, 500)

    def _area(self, dist, sample=None, x=None):
        x = self.x if x is None else x
        density = cph.fit_density(self.sample if sample is None else sample,
                                  dist, x)
        return float(np.trapezoid(density, x)) if hasattr(np, "trapezoid") \
            else float(np.trapz(density, x))

    def test_the_kernel_estimate_integrates_to_one(self):
        self.assertAlmostEqual(self._area("kernel"), 1.0, places=2)

    def test_a_parametric_fit_integrates_to_one(self):
        self.assertAlmostEqual(self._area("normal"), 1.0, places=2)

    def test_a_positive_family_is_refused_for_negative_data(self):
        with self.assertRaises(ValueError):
            cph.fit_density(self.sample, "gamma", self.x)

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            cph.fit_density(self.sample, "no-such-distribution", self.x)


class TestTheZeroBackground(unittest.TestCase):
    """A file's zero background is dropped, a handful of zeros is kept."""

    def test_a_large_background_becomes_nan(self):
        data = np.zeros(1000, dtype=np.float32)
        data[:100] = 1.0
        self.assertEqual(np.isnan(cph._drop_zero_background(data)).sum(), 900)

    def test_a_few_zeros_are_data(self):
        data = np.ones(1000, dtype=np.float32)
        data[:5] = 0.0
        self.assertFalse(np.isnan(cph._drop_zero_background(data)).any())


class TestTheResult(unittest.TestCase):
    """What ``plot_histogram`` reports about the data it drew."""

    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise unittest.SkipTest(f"matplotlib unavailable: {exc}")
        matplotlib.use("Agg")

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_a_matrix_is_split_along_its_smaller_dimension(self):
        data = np.random.default_rng(1).normal(size=(4, 5000))
        self.assertEqual(cph.plot_histogram(data, verbose=False).hist.shape[0], 4)
        self.assertEqual(
            cph.plot_histogram(data.T, verbose=False).hist.shape[0], 4)

    def test_the_statistics_describe_the_sample(self):
        sample = np.random.default_rng(2).normal(3.0, 2.0, 50000)
        result = cph.plot_histogram(sample, names=["one"], verbose=False)
        stats = result.stats[0]
        self.assertEqual(stats.name, "one")
        self.assertAlmostEqual(stats.mean, 3.0, places=1)
        self.assertAlmostEqual(stats.std, 2.0, places=1)
        self.assertAlmostEqual(stats.effect_size, 1.5, places=1)
        self.assertIsNone(stats.th5)

    def test_without_a_fit_the_curve_is_the_histogram(self):
        data = [np.random.default_rng(3).normal(size=5000)]
        result = cph.plot_histogram(data, dist="none", verbose=False)
        np.testing.assert_allclose(result.fit, result.hist)
        # ... and all inputs are pooled into a second figure
        self.assertIn("pooled", result.figures)

    def test_normalized_histograms_sum_to_one(self):
        data = [np.random.default_rng(4).normal(size=n) for n in (2000, 9000)]
        result = cph.plot_histogram(data, dist="none", xrange=(-6, 6),
                                    bins=100, verbose=False)
        # The plotted range excludes the open outer bins, so a little of the
        # mass sits outside it
        for row in result.hist:
            self.assertAlmostEqual(row.sum(), 1.0, places=3)

    def test_the_average_gets_its_own_figure(self):
        data = [np.random.default_rng(5).normal(size=4000) for _ in range(3)]
        result = cph.plot_histogram(data, mean=True, verbose=False)
        self.assertIn("mean", result.figures)
        self.assertEqual(result.figures["mean"].number, 11)

    def test_mixing_file_names_and_arrays_is_refused(self):
        with self.assertRaises(ValueError):
            cph.plot_histogram(["file.nii", np.zeros(10)])


class TestTheCommandLine(unittest.TestCase):
    """The tool follows the shared T1Prep command-line conventions.

    A bare call has to print the synopsis; that is checked for every tool at
    once in ``test_cli_help.TestEveryToolFollowsIt``, which this one is
    registered with.
    """

    def test_only_the_double_dash_spelling_is_advertised(self):
        # The legacy '-v'/'-h' spellings still parse, but the help names
        # only the '--' form -- the same rule the other tools follow
        text = cph.build_parser().format_help()
        single_dash = re.findall(r"(?<![\w-])-(?!-)[A-Za-z][\w-]*", text)
        self.assertEqual(single_dash, [],
                         f"single-dash spellings advertised: {single_dash}")

    def test_it_answers_version_with_the_release(self):
        with self.assertRaises(SystemExit) as caught:
            cph.main(["--version"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
