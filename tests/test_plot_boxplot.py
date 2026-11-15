"""The parts of ``t1prep.plot_boxplot`` that carry numbers.

The drawing is matplotlib's own; what is checked here is what it is given --
the statistics of ``cat_plot_boxplot`` (quartiles, whiskers, the two kinds of
outliers, the notch), the text reader with its columns, the order and the
place of the boxes -- and that the calls into matplotlib that differ between
its versions draw the same thing.
"""

import contextlib
import io
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover - a hard dependency
    raise unittest.SkipTest(f"matplotlib unavailable: {exc}")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Outside the guard: a broken module has to fail, not skip
from t1prep import plot_boxplot as cbp  # noqa: E402
from t1prep import plot_histogram as cph  # noqa: E402


def cat_stat_prctile(x, p):
    """A line-by-line port of CAT12's ``cat_stat_prctile.m``."""
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(x)])
    n = x.size
    r = np.clip(np.asarray(p, dtype=float) / 100 * n + 0.5, 1, n)
    lo = np.floor(r).astype(int)
    w = r - lo
    y = x[lo - 1].copy()
    inner = w > 0
    y[inner] = y[inner] + w[inner] * (x[lo[inner]] - y[inner])
    return y


def write(folder, name, text):
    """Write *text* to *folder*/*name* and return the path."""
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestThePercentilesFollowCat12(unittest.TestCase):
    """``prctile`` is ``cat_stat_prctile``, not numpy's default."""

    def test_they_match_cat_stat_prctile(self):
        rng = np.random.default_rng(0)
        percentiles = [0, 1, 25, 50, 75, 99, 100]
        for n in (1, 2, 3, 4, 5, 7, 10, 19, 100, 1001):
            with self.subTest(n=n):
                x = rng.normal(size=n)
                np.testing.assert_allclose(cbp.prctile(x, percentiles),
                                           cat_stat_prctile(x, percentiles),
                                           rtol=0, atol=1e-12)

    def test_small_samples_differ_from_numpys_default(self):
        # The reason not to take matplotlib's statistics: another box
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(cbp.prctile(x, 25), 1.5)
        self.assertEqual(np.percentile(x, 25), 1.75)


class TestTheBoxStatistics(unittest.TestCase):
    """Whiskers, outliers and notch as ``cat_plot_boxplot`` computes them."""

    # Sorted: -20 -10 1 2 3 4 5 6 7 8 20 30, so Q1 = 1.5, median = 4.5,
    # Q3 = 7.5, IQR = 6 and the whiskers reach 9: limits at -7.5 and 16.5,
    # the extreme outliers beyond -16.5 and 25.5.  The NaN shifts the
    # positions behind it, which the outlier indices have to follow.
    SAMPLE = (5, np.nan, -20, 1, 30, 2, 3, -10, 4, 6, 20, 7, 8)

    def test_the_quartiles_and_the_whiskers(self):
        stats = cbp.box_stats(self.SAMPLE, name="x")
        self.assertEqual(stats.n, 12)
        self.assertEqual((stats.q1, stats.median, stats.q3), (1.5, 4.5, 7.5))
        self.assertEqual((stats.min, stats.max), (-20.0, 30.0))
        # The last values within reach, not the fences themselves
        self.assertEqual((stats.whisker_low, stats.whisker_high), (1.0, 8.0))

    def test_the_two_kinds_of_outliers(self):
        stats = cbp.box_stats(self.SAMPLE)
        np.testing.assert_array_equal(stats.extreme_low, [2])   # -20
        np.testing.assert_array_equal(stats.mild_low, [7])      # -10
        np.testing.assert_array_equal(stats.mild_high, [10])    # 20
        np.testing.assert_array_equal(stats.extreme_high, [4])  # 30
        np.testing.assert_array_equal(stats.outliers, [2, 4, 7, 10])

    def test_the_notch_is_the_confidence_interval_of_the_median(self):
        stats = cbp.box_stats(self.SAMPLE)
        reach = 1.57 * 6 / np.sqrt(12)
        self.assertAlmostEqual(stats.ci_low, 4.5 - reach)
        self.assertAlmostEqual(stats.ci_high, 4.5 + reach)

    def test_the_notch_stays_inside_the_box(self):
        stats = cbp.box_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((stats.ci_low, stats.ci_high), (stats.q1, stats.q3))

    def test_no_whiskers_make_every_value_outside_the_box_extreme(self):
        stats = cbp.box_stats(np.arange(1.0, 9.0), whisker=0)
        np.testing.assert_array_equal(stats.extreme_low, [0, 1])
        np.testing.assert_array_equal(stats.extreme_high, [6, 7])
        self.assertEqual(stats.mild_low.size + stats.mild_high.size, 0)
        self.assertEqual((stats.whisker_low, stats.whisker_high),
                         (stats.q1, stats.q3))

    def test_every_value_is_inside_or_an_outlier(self):
        sample = np.random.default_rng(1).standard_t(2, 2000)
        stats = cbp.box_stats(sample)
        inside = np.ones(sample.size, dtype=bool)
        inside[stats.outliers] = False
        reach = 1.5 * (stats.q3 - stats.q1)
        self.assertTrue(np.all(sample[inside] >= stats.q1 - reach))
        self.assertTrue(np.all(sample[inside] <= stats.q3 + reach))
        self.assertEqual(stats.whisker_low, sample[inside].min())
        self.assertEqual(stats.whisker_high, sample[inside].max())

    def test_an_empty_or_a_single_value_set(self):
        empty = cbp.box_stats([np.nan, np.inf])
        self.assertEqual(empty.n, 0)
        self.assertTrue(np.isnan(empty.median))
        single = cbp.box_stats([3.0])
        self.assertEqual((single.q1, single.median, single.q3), (3.0, 3.0, 3.0))
        self.assertEqual(single.outliers.size, 0)

    def test_negative_whiskers_are_refused(self):
        with self.assertRaises(ValueError):
            cbp.box_stats([1.0, 2.0], whisker=-1)


class TestTheTextReader(unittest.TestCase):
    """Text files, with the matrix input of the MATLAB original."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_a_missing_value_keeps_its_place(self):
        # What CAT_GrepJson writes for a report without the field
        path = write(self.folder.name, "tiv.txt", "1500\nN/A\n1600\n")
        (values,), names = cbp.read_text(path)
        np.testing.assert_array_equal(values, [1500, np.nan, 1600])
        self.assertIsNone(names)

    def test_zeros_are_values_not_background(self):
        path = write(self.folder.name, "wmh.txt", "0\n0\n0\n2.5\n")
        (values,), _ = cbp.read_text(path)
        np.testing.assert_array_equal(values, [0, 0, 0, 2.5])
        # ... unlike in CAT_PlotHistogram, which drops them
        self.assertEqual(int(np.isnan(cph.load_data(path)[0]).sum()), 3)

    def test_comments_and_words_are_skipped(self):
        text = "# 12 subjects\n% from CAT_GrepJson\nsub-01 1.5\nsub-02 2.5\n"
        (values,), _ = cbp.read_text(write(self.folder.name, "a.txt", text))
        np.testing.assert_array_equal(values, [1.5, 2.5])

    def test_columns_are_named_by_the_header(self):
        text = "subject TIV GM\nsub-01 1500 650\nsub-02 1400 600\n"
        sets, names = cbp.read_text(write(self.folder.name, "a.txt", text),
                                    columns=True)
        self.assertEqual(names, ["TIV", "GM"])
        np.testing.assert_array_equal(sets[0], [1500, 1400])
        np.testing.assert_array_equal(sets[1], [650, 600])

    def test_columns_without_a_header_are_numbered(self):
        text = "316.99 664.79 554.82\n265.53 572.68 522.35\n"
        sets, names = cbp.read_text(write(self.folder.name, "cgw.txt", text),
                                    columns=True)
        self.assertEqual(names, ["1", "2", "3"])
        self.assertEqual(len(sets), 3)

    def test_ragged_rows_cannot_be_columns(self):
        path = write(self.folder.name, "a.txt", "1 2\n3 4\n\n5\n")
        with self.assertRaisesRegex(ValueError, "line 4"):
            cbp.read_text(path, columns=True)
        # ... but are fine as one data set
        np.testing.assert_array_equal(cbp.read_text(path)[0][0], [1, 2, 3, 4, 5])

    def test_a_file_without_numbers_is_refused(self):
        with self.assertRaises(ValueError):
            cbp.read_text(write(self.folder.name, "a.txt", "no numbers here\n"))

    def test_the_labels_of_columns(self):
        one = write(self.folder.name, "g1.txt", "CSF GM\n1 2\n3 4\n")
        two = write(self.folder.name, "g2.txt", "CSF GM\n5 6\n7 8\n")
        self.assertEqual(cbp._read_files([one], columns=True)[1], ["CSF", "GM"])
        self.assertEqual(cbp._read_files([one, two], columns=True)[1],
                         ["1:CSF", "1:GM", "2:CSF", "2:GM"])
        self.assertEqual(cbp._read_files([one, two], columns=False)[1],
                         ["1", "2"])

    def test_a_lone_missing_value_stands_for_a_whole_row(self):
        # CAT_GrepJson writes one N/A for a report without vol_abs_CGW
        for text in ("1 2 3\nN/A\n4 5 6\n", "N/A\n1 2 3\n4 5 6\n"):
            with self.subTest(text=text):
                sets, _ = cbp.read_text(write(self.folder.name, "cgw.txt", text),
                                        columns=True)
                self.assertEqual(len(sets), 3)
                self.assertEqual([int(np.isnan(s).sum()) for s in sets], [1] * 3)

    def test_a_header_names_the_columns_past_a_missing_row(self):
        text = "CSF GM WM\nN/A\n1 2 3\n"
        _, names = cbp.read_text(write(self.folder.name, "cgw.txt", text),
                                 columns=True)
        self.assertEqual(names, ["CSF", "GM", "WM"])

    def test_null_is_missing_too(self):
        # jq's answer for {"value": null}
        path = write(self.folder.name, "tiv.txt", "1500\nnull\n1600\n")
        np.testing.assert_array_equal(cbp.read_text(path)[0][0],
                                      [1500, np.nan, 1600])

    def test_a_byte_order_mark_is_ignored(self):
        path = Path(self.folder.name) / "bom.txt"
        path.write_bytes(b"\xef\xbb\xbf1500\n1600\n")
        np.testing.assert_array_equal(cbp.read_text(str(path))[0][0],
                                      [1500, 1600])
        path.write_bytes(b"\xef\xbb\xbf# 12 subjects\n1\n")
        np.testing.assert_array_equal(cbp.read_text(str(path))[0][0], [1])

    def test_a_missing_file_is_named(self):
        with self.assertRaisesRegex(FileNotFoundError, "no such file"):
            cbp._read_files([str(Path(self.folder.name) / "lh.thickness.x")],
                            columns=False)

    def test_text_files_need_their_extension(self):
        path = write(self.folder.name, "tiv.csv", "1500\n1600\n")
        with self.assertRaisesRegex(ValueError, r"extension \.txt"):
            cbp._read_files([path], columns=False)


class TestTheInputs(unittest.TestCase):
    """The forms ``plot_boxplot`` takes from Python."""

    def tearDown(self):
        plt.close("all")

    def test_a_list_of_numbers_is_one_data_set(self):
        result = cbp.plot_boxplot([1500, 1600, 1550], verbose=False)
        self.assertEqual([s.n for s in result.stats], [3])

    def test_a_list_of_arrays_is_one_data_set_each(self):
        result = cbp.plot_boxplot([[1.0, 2.0], np.arange(3.0)], verbose=False)
        self.assertEqual([s.n for s in result.stats], [2, 3])

    def test_a_mapping_brings_its_names(self):
        result = cbp.plot_boxplot({"CSF": [1, 2, 3], "GM": [4, 5, 6]},
                                  verbose=False)
        self.assertEqual(result.labels, ["CSF", "GM"])

    def test_pandas_columns_are_data_sets(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - a hard dependency
            self.skipTest("pandas unavailable")
        frame = pd.DataFrame({"TIV": [1500.0, 1600, 1550],
                              "GM": [650.0, 700, 600]})
        result = cbp.plot_boxplot(frame, verbose=False)
        self.assertEqual(result.labels, ["TIV", "GM"])
        self.assertEqual([s.n for s in result.stats], [3, 3])
        result = cbp.plot_boxplot(frame["TIV"], verbose=False)
        self.assertEqual(result.labels, ["TIV"])
        self.assertEqual([s.n for s in result.stats], [3])

    def test_options_have_to_be_named(self):
        with self.assertRaises(TypeError):
            cbp.plot_boxplot([1.0, 2.0], None, "violin")

    def test_a_single_name_is_not_split_into_letters(self):
        result = cbp.plot_boxplot([1.0, 2.0, 3.0], names="TIV", verbose=False)
        self.assertEqual(result.labels, ["TIV"])


class TestThePlot(unittest.TestCase):
    """What ``plot_boxplot`` draws, and where."""

    def tearDown(self):
        plt.close("all")

    def data(self):
        rng = np.random.default_rng(2)
        return [rng.normal(loc, 1.0, 400) for loc in (1.0, 3.0, 2.0)]

    def test_a_matrix_has_one_data_set_per_column(self):
        result = cbp.plot_boxplot(np.zeros((2, 5)) + np.arange(5), verbose=False)
        self.assertEqual(len(result.stats), 5)
        self.assertEqual([s.n for s in result.stats], [2] * 5)

    def test_the_boxes_are_sorted_by_their_median(self):
        result = cbp.plot_boxplot(self.data(), sort="ascending", verbose=False)
        self.assertEqual(result.order, [0, 2, 1])
        np.testing.assert_array_equal(result.positions, [1, 3, 2])
        result = cbp.plot_boxplot(self.data(), sort="descending", verbose=False)
        self.assertEqual(result.order, [1, 2, 0])

    def test_the_first_box_goes_on_top_when_horizontal(self):
        result = cbp.plot_boxplot(self.data(), horizontal=True, verbose=False,
                                  names=["a", "b", "c"])
        np.testing.assert_array_equal(result.positions, [3, 2, 1])
        axes = result.figure.axes[0]
        ticks = dict(zip([t.get_text() for t in axes.get_yticklabels()],
                         axes.get_yticks()))
        self.assertEqual(ticks, {"a": 3.0, "b": 2.0, "c": 1.0})

    def test_colours_stay_with_their_data_when_sorted(self):
        result = cbp.plot_boxplot(self.data(), sort="descending", verbose=False)
        palette = cbp.colormap(cbp.DEFAULT_PALETTE, 3)
        boxes = result.figure.axes[0].patches
        self.assertEqual(len(boxes), 3)
        for k, box in enumerate(boxes):
            x = box.get_path().vertices[:, 0]
            self.assertAlmostEqual((x.min() + x.max()) / 2, result.positions[k])
            np.testing.assert_allclose(box.get_facecolor()[:3], palette[k])

    def test_the_matrix_is_the_s_of_cat_plot_boxplot(self):
        result = cbp.plot_boxplot(self.data(), verbose=False)
        s = result.matrix()
        self.assertEqual(s.shape, (7, 3))
        first = result.stats[0]
        np.testing.assert_array_equal(
            s[:, 0], [first.min, first.q1, first.median, first.q3, first.max,
                      first.ci_low, first.ci_high])

    def test_an_empty_data_set_is_left_blank(self):
        result = cbp.plot_boxplot([np.arange(10.0), np.full(5, np.nan)],
                                  verbose=False)
        self.assertEqual([s.n for s in result.stats], [10, 0])
        self.assertEqual(len(result.figure.axes[0].patches), 1)
        with self.assertRaises(ValueError):
            cbp.plot_boxplot([np.full(5, np.nan)], verbose=False)

    def test_wrong_option_values_are_refused(self):
        data = self.data()
        for options in ({"style": "pie"}, {"show_data": "all"}, {"sort": "up"},
                        {"width": 0}, {"alpha": 2}, {"ylim": [1]},
                        {"names": ["one"]}):
            with self.subTest(**{k: str(v) for k, v in options.items()}), \
                    self.assertRaises(ValueError):
                cbp.plot_boxplot(data, verbose=False, **options)

    def test_connected_data_sets_need_the_same_size(self):
        with self.assertRaises(ValueError):
            cbp.plot_boxplot([np.arange(5.0), np.arange(6.0)], connect=True,
                             verbose=False)
        result = cbp.plot_boxplot([np.arange(5.0), np.arange(5.0) + 1],
                                  connect=True, verbose=False)
        from matplotlib.collections import LineCollection

        lines = [c for c in result.figure.axes[0].collections
                 if isinstance(c, LineCollection)]
        self.assertEqual(len(lines[-1].get_segments()), 5)

    def test_a_violin_is_scaled_within_the_value_range_drawn(self):
        values = np.r_[np.zeros(5000), np.random.default_rng(3).normal(5, 1, 500)]
        coords, density = cbp._density(values, (2.0, 8.0))
        self.assertGreaterEqual(coords.min(), 2.0)
        self.assertLessEqual(coords.max(), 8.0)
        # The spike at zero flattens the whole violin to a sliver at 5 ...
        whole_coords, whole = cbp._density(values)
        peak = whole[np.argmin(np.abs(whole_coords - 5.0))]
        self.assertLess(peak / whole.max(), 0.01)
        # ... but out of view it no longer sets the width
        self.assertLess(density.max(), whole.max() / 100)
        self.assertGreater(density[np.argmin(np.abs(coords - 5.0))]
                           / density.max(), 0.5)

    def test_every_style_draws_without_a_warning(self):
        data = self.data()
        for style in cbp.STYLES:
            for horizontal in (False, True):
                with self.subTest(style=style, horizontal=horizontal):
                    with warnings.catch_warnings():
                        # Covers the deprecated vert= of matplotlib >= 3.10
                        warnings.simplefilter("error")
                        cbp.plot_boxplot(data, style=style, notch=True,
                                         horizontal=horizontal, verbose=False)
                    plt.close("all")

    def test_long_labels_are_cut_but_kept_apart(self):
        names = [f"sub-{i:02d}_ses-1_acq-mprage_run-1_desc-preproc_T1w"
                 for i in range(20)]
        data = [np.arange(5.0) + i for i in range(20)]
        result = cbp.plot_boxplot(data, names=names, verbose=False)
        shown = [t.get_text() for t in result.figure.axes[0].get_xticklabels()]
        self.assertLess(max(len(t) for t in shown), len(names[0]))
        self.assertEqual(len(set(shown)), 20)
        # The result keeps the full names
        self.assertEqual(result.labels, names)

    def test_outliers_are_listed_by_their_position(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cbp.plot_boxplot([TestTheBoxStatistics.SAMPLE], names=["x"],
                             verbose=False, list_outliers=True)
        rows = [line.split("\t")[1].split() for line in
                out.getvalue().splitlines() if line.strip().startswith("x\t")]
        # 1-based positions, ordered, with their values and kinds
        self.assertEqual([row[0] for row in rows], ["3", "5", "8", "11"])
        self.assertEqual([row[1] for row in rows], ["-20", "30", "-10", "20"])
        self.assertEqual(" ".join(rows[0][2:]), "extreme, low")

    def test_the_outliers_of_a_volume_are_listed_by_voxel(self):
        volume = np.random.default_rng(5).normal(0, 1, (3, 4, 5))
        volume[1, 2, 3] = 100.0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = cbp.plot_boxplot([volume], names=["v"], verbose=False,
                                      list_outliers=True)
        self.assertIn(int(np.ravel_multi_index((1, 2, 3), volume.shape)),
                      result.stats[0].extreme_high)
        rows = [line.split("\t")[1].split() for line in
                out.getvalue().splitlines() if line.strip().startswith("v\t")]
        # Counted from 1, as SPM shows voxels
        self.assertIn(["2,3,4", "100", "extreme,", "high"], rows)

    def test_a_set_too_uniform_for_a_violin_stays_visible(self):
        # One violin, and a constant set whose flat box the white median of
        # the thin box would hide
        result = cbp.plot_boxplot([np.r_[1.0, 2, 3, 4], np.full(5, 2.0)],
                                  style="violinbox", verbose=False)
        marks = [line for line in result.figure.axes[0].lines
                 if np.allclose(line.get_ydata(), 2.0)
                 and np.all(np.abs(np.asarray(line.get_xdata()) - 2.0) <= 0.5)
                 and matplotlib.colors.to_rgb(line.get_color()) != (1, 1, 1)]
        self.assertTrue(marks)

    def test_connected_values_are_picked_where_there_are_values(self):
        # A volume is mostly NaN background; the lines go where the brain is
        first = np.full(20000, np.nan)
        first[::10] = 1.0
        result = cbp.plot_boxplot([first, first + 1.0], connect=True,
                                  verbose=False)
        from matplotlib.collections import LineCollection

        lines = [c for c in result.figure.axes[0].collections
                 if isinstance(c, LineCollection)][-1]
        drawn = [s for s in lines.get_segments() if np.isfinite(s).all()]
        self.assertEqual(len(drawn), 2000)

    def test_an_open_end_of_the_value_range(self):
        result = cbp.plot_boxplot(self.data(), ylim=(np.nan, 2.5), verbose=False)
        low, high = result.figure.axes[0].get_ylim()
        self.assertEqual(high, 2.5)
        self.assertLess(low, min(s.min for s in result.stats))


@unittest.skipUnless(cbp._accepts(matplotlib.axes.Axes.violin, "side"),
                     "matplotlib < 3.9 has no half violins to compare with")
class TestTheHalfViolinFallback(unittest.TestCase):
    """matplotlib < 3.9 gets the half violins drawn by hand, the same way."""

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def bodies(result):
        from matplotlib.collections import PolyCollection

        return [c.get_paths()[0].vertices for c in result.figure.axes[0].collections
                if isinstance(c, PolyCollection)]

    def test_it_draws_what_side_high_draws(self):
        rng = np.random.default_rng(4)
        data = [rng.normal(0, 1, 500), rng.gamma(2, 1, 500)]
        real = cbp._accepts
        for horizontal in (True, False):
            with self.subTest(horizontal=horizontal):
                native = self.bodies(cbp.plot_boxplot(
                    data, style="density", horizontal=horizontal, verbose=False))
                # As matplotlib < 3.9: no side keyword, and Axes.violin must
                # not be reached, or the halves were not drawn by hand
                with mock.patch.object(
                        cbp, "_accepts",
                        lambda method, name: name != "side" and real(method, name)), \
                        mock.patch.object(matplotlib.axes.Axes, "violin",
                                          side_effect=AssertionError("called")):
                    by_hand = self.bodies(cbp.plot_boxplot(
                        data, style="density", horizontal=horizontal,
                        verbose=False))
                self.assertEqual(len(native), 2)
                self.assertEqual(len(by_hand), len(native))
                for a, b in zip(native, by_hand):
                    np.testing.assert_allclose(a, b)


class TestTheCommandLine(unittest.TestCase):
    """``CAT_PlotBoxplot`` end to end, writing the figure to a file."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(plt.close, "all")

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = cbp.main(list(argv))
        return status, out.getvalue() + err.getvalue()

    def test_it_saves_the_figure(self):
        data = write(self.folder.name, "cgw.txt", "1 5 9\n2 6 8\n3 7 7\n4 8 6\n")
        target = str(Path(self.folder.name) / "box")
        status, text = self.run_main("--columns", "--names", "CSF, GM, WM",
                                     "--save", target, data)
        self.assertEqual(status, 0)
        self.assertTrue(Path(target + ".png").is_file())
        self.assertIn("CSF", text)
        self.assertIn("WM", text)

    def test_a_wrong_number_of_names_is_an_error(self):
        data = write(self.folder.name, "a.txt", "1\n2\n3\n")
        status, text = self.run_main("--names", "a,b", "--save",
                                     str(Path(self.folder.name) / "x.png"), data)
        self.assertEqual(status, 1)
        self.assertIn("names has 2 entries for 1 boxes", text)

    def test_space_separated_names_get_a_hint(self):
        data = write(self.folder.name, "cgw.txt", "1 2 3\n4 5 6\n")
        status, text = self.run_main("--names", "CSF", "GM", "WM", data)
        self.assertEqual(status, 1)
        self.assertIn("no such file: GM", text)
        self.assertIn("comma-separated", text)

    def test_options_are_not_abbreviated(self):
        # CAT_GrepJson's --out would otherwise pass for --outliers
        data = write(self.folder.name, "a.txt", "1\n2\n3\n")
        with self.assertRaises(SystemExit) as stop, \
                contextlib.redirect_stderr(io.StringIO()):
            cbp.main(["--out", "x.png", data])
        self.assertEqual(stop.exception.code, 2)

    def test_a_dot_in_the_name_is_not_an_extension(self):
        data = write(self.folder.name, "a.txt", "1\n2\n3\n")
        target = Path(self.folder.name) / "qc_v1.2"
        status, _ = self.run_main("--save", str(target), data)
        self.assertEqual(status, 0)
        self.assertTrue(Path(str(target) + ".png").is_file())
        status, _ = self.run_main("--save", str(target) + ".pdf", data)
        self.assertTrue(Path(str(target) + ".pdf").is_file())

    def test_a_figure_that_cannot_be_saved_is_an_error(self):
        data = write(self.folder.name, "a.txt", "1\n2\n3\n")
        status, text = self.run_main(
            "--save", str(Path(self.folder.name) / "missing" / "box.png"), data)
        self.assertEqual(status, 1)
        self.assertIn("cannot save", text)


if __name__ == "__main__":
    unittest.main()
