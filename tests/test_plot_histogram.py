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
        self.assertAlmostEqual(self._area("gaussian_kde"), 1.0, places=2)

    def test_gaussian_kde_is_scipys_estimate(self):
        from scipy import stats

        expected = stats.gaussian_kde(self.sample)(self.x)
        density = cph.fit_density(self.sample, "gaussian_kde", self.x)
        np.testing.assert_allclose(density, expected, atol=1e-3 * expected.max())

    def test_gaussian_kde_survives_a_constant_sample(self):
        # scipy itself refuses it: the covariance cannot be inverted
        density = cph.fit_density(np.full(100, 2.0), "gaussian_kde", self.x)
        self.assertTrue(np.isfinite(density).all())

    def test_a_parametric_fit_integrates_to_one(self):
        self.assertAlmostEqual(self._area("normal"), 1.0, places=2)

    def test_a_positive_family_is_refused_for_negative_data(self):
        with self.assertRaises(ValueError):
            cph.fit_density(self.sample, "gamma", self.x)

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            cph.fit_density(self.sample, "no-such-distribution", self.x)


class TestTheKernelDensity(unittest.TestCase):
    """``kernel_density`` with the Improved Sheather-Jones bandwidth."""

    @staticmethod
    def _error(sample, x, truth, bandwidth):
        """Integrated squared error relative to that of the truth itself."""
        density = cph.kernel_density(sample, x, bandwidth)
        return np.sum((density - truth) ** 2) / np.sum(truth ** 2)

    def test_a_normal_sample_gets_the_optimal_bandwidth(self):
        sample = np.random.default_rng(6).normal(size=100000)
        # The bandwidth that minimises the asymptotic MISE of a normal density
        optimal = (4.0 / (3.0 * sample.size)) ** 0.2
        self.assertAlmostEqual(cph.kernel_bandwidth(sample) / optimal, 1.0,
                               delta=0.05)

    def test_separated_modes_are_resolved(self):
        rng = np.random.default_rng(7)
        sample = np.concatenate([rng.normal(-2, 0.5, 5000),
                                 rng.normal(2, 0.5, 5000)])
        x = np.linspace(-4, 4, 400)
        truth = 0.5 * (np.exp(-0.5 * ((x + 2) / 0.5) ** 2)
                       + np.exp(-0.5 * ((x - 2) / 0.5) ** 2)) / (0.5 * np.sqrt(2 * np.pi))
        chosen = self._error(sample, x, truth, cph.kernel_bandwidth(sample))
        rule = self._error(sample, x, truth, cph._rule_of_thumb(sample))
        # Silverman's rule oversmooths the two modes; ISJ does not
        self.assertLess(chosen, rule / 5)

    def test_a_lattice_does_not_comb_the_curve(self):
        from scipy.special import ndtri

        # A uint8 p1 map: values on multiples of 1/255, most of them at 1.
        # Normal quantiles instead of random draws leave no sampling noise
        # that could pass for a comb
        count = 400000
        smooth = np.clip(0.55 + 0.15 * ndtri((np.arange(count) + 0.5) / count), 0, 1)
        sample = np.concatenate([np.round(smooth * 255) / 255, np.ones(600000)])
        on_lattice = np.arange(50, 200) / 255.0
        between = (np.arange(49, 200) + 0.5) / 255.0
        level = cph.kernel_density(sample, on_lattice)
        gaps = cph.kernel_density(sample, between)
        # A comb puts a spike on every lattice value; the mean of the two
        # neighbouring gaps cancels the slope of the density itself.  Plain
        # ISJ collapses here, to a bandwidth of a few millionths, and even
        # half a lattice step leaves a 3% comb
        ratio = level / (0.5 * (gaps[:-1] + gaps[1:]))
        self.assertLess(np.abs(ratio - 1.0).max(), 0.002)

    def test_binning_matches_the_exact_sum(self):
        rng = np.random.default_rng(9)
        sample = np.concatenate([rng.normal(0, 1, 30000), rng.normal(4, 0.3, 10000)])
        x = np.linspace(-4, 6, 300)
        width = cph.kernel_bandwidth(sample)
        exact = np.exp(-0.5 * ((x[:, None] - sample[None, :]) / width) ** 2).sum(1)
        exact /= sample.size * width * np.sqrt(2 * np.pi)
        np.testing.assert_allclose(cph.kernel_density(sample, x), exact,
                                   atol=1e-3 * exact.max())

    def test_degenerate_samples_still_get_a_bandwidth(self):
        for sample in (np.full(50, 3.0), np.array([1.0, 2.0]),
                       np.array([0.0, 1.0, 1.0, 1.0])):
            self.assertGreater(cph.kernel_bandwidth(sample), 0.0)


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

    def _ridges(self, **options):
        rng = np.random.default_rng(8)
        data = [rng.normal(loc, scale, 6000)
                for loc, scale in ((0, 1), (2, 0.5), (-1, 2))]
        result = cph.plot_histogram(data, names=["a", "b", "c"], ridgeline=True,
                                    verbose=False, **options)
        return result, result.figures["histogram"].axes[0]

    def test_a_ridgeline_puts_the_first_input_at_the_top(self):
        _, axes = self._ridges()
        ticks = dict(zip([t.get_text() for t in axes.get_yticklabels()],
                         axes.get_yticks()))
        self.assertEqual(ticks, {"a": 2.0, "b": 1.0, "c": 0.0})
        self.assertIsNone(axes.get_legend())

    def test_the_tallest_ridge_reaches_the_requested_height(self):
        result, axes = self._ridges(ridge_height=1.2, rawline=0)
        # Ridge outlines are the solid lines; b is the narrowest, hence tallest
        outlines = [line for line in axes.lines if line.get_linestyle() == "-"]
        self.assertEqual(len(outlines), 3)
        heights = [line.get_ydata().max() - base
                   for line, base in zip(outlines, (2.0, 1.0, 0.0))]
        self.assertAlmostEqual(max(heights), 1.2)
        self.assertEqual(int(np.argmax(heights)), 1)
        # The curves themselves are what the overlay would have drawn
        self.assertEqual(result.fit.shape, (3, len(result.x)))

    def test_a_ridgeline_scales_to_the_drawn_range(self):
        # Cut off the peak of b: the tallest visible ridge then sets the scale
        _, axes = self._ridges(ridge_height=1.0, rawline=0, xlim=(-6, 1))
        outlines = [line for line in axes.lines if line.get_linestyle() == "-"]
        visible = []
        for line, base in zip(outlines, (2.0, 1.0, 0.0)):
            x, y = line.get_xdata(), line.get_ydata()
            visible.append((y[(x >= -6) & (x <= 1)] - base).max())
        self.assertAlmostEqual(max(visible), 1.0)

    def test_long_ridge_labels_are_shortened_in_the_middle(self):
        label = "TH5=3.1416 " + "x" * 60 + "_subject.nii"
        short = cph._ellipsize(label)
        self.assertEqual(len(short), cph._RIDGE_LABEL_LENGTH)
        self.assertTrue(short.startswith("TH5=3.1416"))
        self.assertTrue(short.endswith("_subject.nii"))
        self.assertEqual(cph._ellipsize("short.nii"), "short.nii")

    def test_each_ridge_can_get_its_own_scale(self):
        _, axes = self._ridges(ridge_height=0.8, rawline=0, ridge_scale="own")
        outlines = [line for line in axes.lines if line.get_linestyle() == "-"]
        heights = [line.get_ydata().max() - base
                   for line, base in zip(outlines, (2.0, 1.0, 0.0))]
        np.testing.assert_allclose(heights, 0.8)

    def test_an_unknown_ridge_scale_is_refused(self):
        with self.assertRaises(ValueError):
            self._ridges(ridge_scale="tallest")

    def test_a_ridgeline_needs_a_positive_height(self):
        with self.assertRaises(ValueError):
            self._ridges(ridge_height=0)


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

    def test_the_ridgeline_options_reach_the_plot(self):
        args = cph.build_parser().parse_args(
            ["--ridgeline", "--ridge-height", "2", "--ridge-scale", "own", "a.nii"])
        self.assertTrue(args.ridgeline)
        self.assertEqual(args.ridge_height, 2.0)
        self.assertEqual(args.ridge_scale, "own")
        self.assertEqual(cph.build_parser().parse_args(["a.nii"]).ridge_scale,
                         "shared")
        self.assertEqual(args.files, ["a.nii"])

    def test_it_answers_version_with_the_release(self):
        with self.assertRaises(SystemExit) as caught:
            cph.main(["--version"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
