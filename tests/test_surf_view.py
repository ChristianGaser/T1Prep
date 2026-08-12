import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Importing the viewer pulls in Qt; keep it headless for CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from t1prep.gui.cat_surf_view import (
        _is_scalars_only_gifti,
        convert_filename_to_mesh,
        format_p_value_label,
        is_logp_overlay,
        is_overlay_file,
        logp_colorbar_ticks,
        parse_args,
    )
except Exception as exc:  # pragma: no cover - depends on optional GUI deps
    raise unittest.SkipTest(f"cat_surf_view unavailable: {exc}")


class TestLogPDetection(unittest.TestCase):

    def test_logp_filenames(self):
        """Files with 'log' in their name are treated as -log10(p) maps."""
        for name in (
            "logP_age_(polynomial_of_degree_3)_pFWE0.1_k0.gii",
            "/tmp/stat/logP_0001.gii",
            "LogP_0001.gii",
        ):
            self.assertTrue(is_logp_overlay(name), name)

    def test_non_logp_filenames(self):
        for name in ("lh.thickness.subj", "spmT_0001.gii", "", None):
            self.assertFalse(is_logp_overlay(name), name)


class TestPValueLabels(unittest.TestCase):

    def test_thresholds(self):
        """The tick labels match the p-values shown by cat_surf_results."""
        self.assertEqual(format_p_value_label(-math.log10(0.05)), "0.05")
        self.assertEqual(format_p_value_label(2), "0.01")
        self.assertEqual(format_p_value_label(3), "0.001")
        self.assertEqual(format_p_value_label(7), "0.0000001")

    def test_negative_tail_and_exponential(self):
        self.assertEqual(format_p_value_label(-2), "-0.01")
        self.assertEqual(format_p_value_label(8), "1e-08")
        self.assertEqual(format_p_value_label(-8), "-1e-08")

    def test_compact_form_keeps_the_named_thresholds(self):
        """Crowded bars shorten the labels but keep 0.05/0.01/0.001 intact."""
        self.assertEqual(format_p_value_label(-math.log10(0.05), True), "0.05")
        self.assertEqual(format_p_value_label(2, True), "0.01")
        self.assertEqual(format_p_value_label(3, True), "0.001")
        self.assertEqual(format_p_value_label(4, True), "1e-04")
        self.assertEqual(format_p_value_label(-7, True), "-1e-07")

    def test_zero_has_no_p_value(self):
        self.assertEqual(format_p_value_label(0), "")


class TestLogPTicks(unittest.TestCase):

    def test_one_sided_threshold(self):
        """Thresholded at p<0.05 the first tick sits at -log10(0.05)."""
        ticks = logp_colorbar_ticks(0.0, 8.0, (-1.3, 1.3))
        labels = [format_p_value_label(t) for t in ticks]
        self.assertAlmostEqual(ticks[0], 1.30103, places=4)
        self.assertEqual(labels[:3], ["0.05", "0.01", "0.001"])

    def test_two_sided_threshold(self):
        ticks = logp_colorbar_ticks(-8.0, 8.0, (-1.3, 1.3))
        labels = [format_p_value_label(t) for t in ticks]
        self.assertIn("-0.05", labels)
        self.assertIn("0.05", labels)
        self.assertNotIn(0.0, ticks)

    def test_unthresholded_uses_integer_steps(self):
        ticks = logp_colorbar_ticks(0.0, 4.0, (0.0, -1.0))
        self.assertEqual(ticks, [1.0, 2.0, 3.0, 4.0])

    def test_ticks_stay_inside_range(self):
        ticks = logp_colorbar_ticks(2.0, 8.0, (-100.0, 6.0))
        self.assertTrue(all(2.0 <= t <= 8.0 for t in ticks))

    def test_empty_range(self):
        self.assertEqual(logp_colorbar_ticks(0.0, 0.0, None), [])
        self.assertEqual(logp_colorbar_ticks(0.0, -1.0, (-1.3, 1.3)), [])


class TestScalarsOnlyGifti(unittest.TestCase):
    """A .gii holding values but no surface must be treated as an overlay.

    CAT12/TFCE statistic results (TFCE_log_pFWE_0001.gii) carry a free-form
    name and are not always accompanied by the SPM.mat that used to be the
    only marker, so they were mistaken for meshes and failed to load.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import nibabel as nib
        except Exception as exc:  # pragma: no cover - optional dependency
            raise unittest.SkipTest(f"nibabel unavailable: {exc}")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.overlay = Path(cls._tmp.name) / "TFCE_log_pFWE_0001.gii"
        # 2 x 32492 values: a 32k template overlay covering both hemispheres
        values = np.zeros(2 * 32492, dtype=np.float32)
        img = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(values, intent="NIFTI_INTENT_SHAPE")]
        )
        nib.save(img, str(cls.overlay))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_detected_as_overlay(self):
        self.assertTrue(_is_scalars_only_gifti(str(self.overlay)))
        self.assertTrue(is_overlay_file(str(self.overlay)))

    def test_mesh_falls_back_to_matching_template(self):
        """No surface in the folder: the value count picks the template."""
        mesh = Path(convert_filename_to_mesh(str(self.overlay)))
        self.assertNotEqual(mesh, self.overlay)
        self.assertTrue(mesh.exists(), mesh)
        self.assertIn("templates_surfaces_32k", mesh.parts)

    def test_parse_args_keeps_it_as_overlay(self):
        opts = parse_args([str(self.overlay)])
        self.assertEqual(opts.overlay, str(self.overlay))
        self.assertTrue(Path(opts.mesh_left).exists())
        self.assertNotEqual(Path(opts.mesh_left), self.overlay)

    def test_real_mesh_is_not_an_overlay(self):
        template = (
            _SRC / "t1prep" / "data" / "templates_surfaces_32k" / "lh.central.freesurfer.gii"
        )
        if not template.exists():  # pragma: no cover - depends on packaged data
            self.skipTest("template surfaces not installed")
        self.assertFalse(_is_scalars_only_gifti(str(template)))
        self.assertFalse(is_overlay_file(str(template)))


if __name__ == "__main__":
    unittest.main()
