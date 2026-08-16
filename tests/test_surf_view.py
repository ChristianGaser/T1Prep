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
        UNDERLAY_GREYS,
        VIEWER_KEYS,
        default_overlay_range,
        ClusterTableDialog,
        Viewer,
        _is_scalars_only_gifti,
        atlas_border_lines,
        available_surface_atlases,
        convert_filename_to_mesh,
        find_surface_clusters,
        read_annotation,
        vertex_areas,
        format_p_value_label,
        is_combined_hemisphere_mesh,
        is_logp_overlay,
        is_overlay_file,
        logp_colorbar_ticks,
        parse_args,
        read_mesh_pair,
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


class TestLogPThresholds(unittest.TestCase):
    """The p-value thresholds offered for -log10(p) overlays."""

    def test_table_matches_cat_surf_results(self):
        from t1prep.gui.cat_surf_view import LOGP_THRESHOLDS
        labels = [label for label, _ in LOGP_THRESHOLDS]
        self.assertEqual(labels, ["none", "p<0.05", "p<0.01", "p<0.001"])
        values = [value for _, value in LOGP_THRESHOLDS]
        self.assertEqual(values[0], 0.0)
        for value, p in zip(values[1:], (0.05, 0.01, 0.001)):
            self.assertAlmostEqual(value, -math.log10(p), places=6)

    def test_labels_match_the_colorbar(self):
        """The entry and the tick it produces must say the same p-value."""
        from t1prep.gui.cat_surf_view import LOGP_THRESHOLDS
        for label, value in LOGP_THRESHOLDS[1:]:
            self.assertEqual(f"p<{format_p_value_label(value)}", label)


class TestThresholdWidget(unittest.TestCase):
    """The control-panel entry follows, and drives, the clip window."""

    @classmethod
    def setUpClass(cls):
        try:
            from PySide6 import QtWidgets
            cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            from t1prep.gui.cat_surf_view import ControlPanel
            cls.panel = ControlPanel()
        except Exception as exc:  # pragma: no cover - needs a Qt platform plugin
            raise unittest.SkipTest(f"Qt unavailable: {exc}")

    def test_hidden_until_asked_for(self):
        self.panel.set_threshold_visible(False)
        self.assertFalse(self.panel.threshold_row.isVisibleTo(self.panel))
        self.panel.set_threshold_visible(True)
        self.assertTrue(self.panel.threshold_row.isVisibleTo(self.panel))

    def test_selection_follows_the_clip_window(self):
        self.panel.set_threshold_from_clip((-2.0, 2.0))
        self.assertEqual(self.panel.threshold.currentText(), "p<0.01")
        self.panel.set_threshold_from_clip((-1.3, 1.3))
        self.assertEqual(self.panel.threshold.currentText(), "p<0.05")
        self.panel.set_threshold_from_clip((-3.0, 3.0))
        self.assertEqual(self.panel.threshold.currentText(), "p<0.001")

    def test_other_clips_show_as_none(self):
        for clip in ((0.0, 0.0), (-4.0, 4.0), (-100.0, 6.0), (0.0, -1.0)):
            self.panel.set_threshold_from_clip(clip)
            self.assertEqual(self.panel.threshold.currentText(), "none", clip)


class TestPresets(unittest.TestCase):
    """-preset applies several options at once."""

    MESH = str(_SRC / "t1prep" / "data" / "templates_surfaces_32k"
               / "lh.central.freesurfer.gii")

    def _opts(self, *args):
        return parse_args([self.MESH, *args])

    def test_preset_one(self):
        from t1prep.gui.cat_surf_view import C3
        opts = self._opts("-preset", "1")
        self.assertEqual(opts.colormap, C3)
        self.assertEqual(opts.discrete, 16)

    def test_defaults_are_untouched_without_a_preset(self):
        from t1prep.gui.cat_surf_view import JET
        opts = self._opts()
        self.assertEqual(opts.colormap, JET)
        self.assertEqual(opts.discrete, 0)

    def test_explicit_options_win(self):
        """A preset must not overrule what the user typed, in either order."""
        from t1prep.gui.cat_surf_view import FIRE
        self.assertEqual(self._opts("-preset", "1", "-dsc", "8").discrete, 8)
        self.assertEqual(self._opts("-preset", "1", "-fire").colormap, FIRE)
        self.assertEqual(self._opts("-fire", "-preset", "1").colormap, FIRE)
        # …and the rest of the preset still applies
        self.assertEqual(self._opts("-preset", "1", "-fire").discrete, 16)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._opts("-preset", "99")

    def test_every_preset_is_documented(self):
        from t1prep.gui.cat_surf_view import PRESETS, PRESET_HELP
        self.assertEqual(sorted(PRESETS), sorted(PRESET_HELP))
        for number in PRESETS:
            self.assertTrue(PRESET_HELP[number].strip(), number)


class TestHemisphereLookup(unittest.TestCase):
    """The partner surface is found from either side."""

    def test_hemisphere_is_recognised(self):
        from t1prep.gui.cat_surf_view import hemisphere_of
        for name, side in (("lh.central.subj.gii", "lh"),
                           ("rh.central.subj.gii", "rh"),
                           ("sub-01_hemi-L_midthickness.surf.gii", "lh"),
                           ("sub-01_hemi-R_midthickness.surf.gii", "rh"),
                           ("left_hemisphere.gii", "lh"),
                           ("right_hemisphere.gii", "rh"),
                           ("mesh.central.gii", None),
                           ("TFCE_log_pFWE_0001.gii", None)):
            self.assertEqual(hemisphere_of(name), side, name)

    def test_counterpart_works_in_both_directions(self):
        from t1prep.gui.cat_surf_view import _hemi_counterpart
        pairs = (("lh.central.subj.gii", "rh.central.subj.gii"),
                 ("sub-01_hemi-L_desc-thickness.gii", "sub-01_hemi-R_desc-thickness.gii"),
                 ("left_hemisphere.gii", "right_hemisphere.gii"))
        for left, right in pairs:
            self.assertEqual(_hemi_counterpart(Path(left)).name, right)
            self.assertEqual(_hemi_counterpart(Path(right)).name, left)

    def test_no_hemisphere_has_no_counterpart(self):
        from t1prep.gui.cat_surf_view import _hemi_counterpart
        self.assertIsNone(_hemi_counterpart(Path("mesh.central.Template_T1.gii")))

    def test_pair_is_ordered_left_then_right(self):
        """What was selected decides which of the two is the left one."""
        from t1prep.gui.cat_surf_view import order_by_hemisphere
        self.assertEqual(order_by_hemisphere("lh.central.gii", "own", "other"),
                         ("own", "other"))
        self.assertEqual(order_by_hemisphere("rh.central.gii", "own", "other"),
                         ("other", "own"))
        # nothing to order without a partner
        self.assertEqual(order_by_hemisphere("rh.central.gii", "own", None),
                         ("own", None))

    def test_template_follows_the_hemisphere(self):
        """A right-hemisphere overlay falls back to the right template."""
        from t1prep.gui.cat_surf_view import _template_mesh_for_points
        left = _template_mesh_for_points(32492, "lh")
        right = _template_mesh_for_points(32492, "rh")
        if left is None or right is None:  # pragma: no cover - packaged data
            self.skipTest("template surfaces not installed")
        self.assertTrue(left.name.startswith("lh."))
        self.assertTrue(right.name.startswith("rh."))


class TestMeshTitle(unittest.TestCase):
    """Several surfaces are numbered in the title, like several overlays."""

    def _title(self, path, meshes, index=0, custom=None):
        import types
        from t1prep.gui.cat_surf_view import Viewer
        stub = types.SimpleNamespace(
            opts=types.SimpleNamespace(title=custom),
            mesh_list=list(meshes),
            current_mesh_index=index,
        )
        return Viewer._mesh_title(stub, path)

    def test_single_surface_has_no_counter(self):
        self.assertNotIn("(", self._title("/data/lh.central.gii", ["/data/lh.central.gii"]))

    def test_several_surfaces_are_counted(self):
        meshes = [f"/data/sub-{i}/lh.central.gii" for i in range(4)]
        self.assertTrue(self._title(meshes[0], meshes).endswith("(1/4)"))
        self.assertTrue(self._title(meshes[3], meshes, index=3).endswith("(4/4)"))

    def test_explicit_title_wins(self):
        meshes = ["/data/a.gii", "/data/b.gii"]
        self.assertEqual(self._title(meshes[0], meshes, custom="my title"), "my title")


class TestCombinedHemisphereMesh(unittest.TestCase):
    """CAT12 stores both hemispheres in one file (mesh.central.*).

    Without splitting them, the viewer treated the pair as a single left
    hemisphere and the six-view montage showed only part of the surface.
    """

    TEMPLATES = _SRC / "t1prep" / "data" / "templates_surfaces_32k"

    def setUp(self):
        if not self.TEMPLATES.exists():  # pragma: no cover - depends on packaged data
            self.skipTest("template surfaces not installed")

    def test_name_detection(self):
        self.assertTrue(is_combined_hemisphere_mesh("mesh.central.Template_T1.gii"))
        self.assertTrue(is_combined_hemisphere_mesh("mesh.inflated.freesurfer.gii"))
        self.assertFalse(is_combined_hemisphere_mesh("lh.central.freesurfer.gii"))
        self.assertFalse(is_combined_hemisphere_mesh("s12.mesh.thickness.subj.gii"))

    def test_combined_mesh_is_split(self):
        poly_l, poly_r = read_mesh_pair(str(self.TEMPLATES / "mesh.central.Template_T1.gii"))
        self.assertIsNotNone(poly_r)
        self.assertEqual(poly_l.GetNumberOfPoints(), 32492)
        self.assertEqual(poly_r.GetNumberOfPoints(), 32492)
        # No triangle may be lost when the halves are separated
        self.assertEqual(poly_l.GetNumberOfPolys() + poly_r.GetNumberOfPolys(), 129960)

    def test_single_hemisphere_keeps_its_sibling(self):
        poly_l, poly_r = read_mesh_pair(str(self.TEMPLATES / "lh.central.freesurfer.gii"))
        self.assertIsNotNone(poly_r)
        self.assertEqual(poly_l.GetNumberOfPoints(), 32492)
        self.assertEqual(poly_r.GetNumberOfPoints(), 32492)


if __name__ == "__main__":
    unittest.main()


def _grid_mesh(rows=12, columns=12, spacing=1.0):
    """A flat triangulated sheet, so areas and clusters have known answers."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    points = vtkPoints()
    for j in range(rows):
        for i in range(columns):
            points.InsertNextPoint(i * spacing, j * spacing, 0.0)
    triangles = vtkCellArray()
    for j in range(rows - 1):
        for i in range(columns - 1):
            a = j * columns + i
            b, c, d = a + 1, a + columns, a + columns + 1
            for triangle in ((a, b, c), (b, d, c)):
                triangles.InsertNextCell(3)
                for vertex in triangle:
                    triangles.InsertCellPoint(vertex)
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(triangles)
    return poly


class TestVertexAreas(unittest.TestCase):
    """Cluster sizes are reported in mm², which needs a per-vertex area."""

    def test_the_areas_add_up_to_the_surface(self):
        poly = _grid_mesh(rows=10, columns=10, spacing=2.0)
        areas = vertex_areas(poly)
        self.assertEqual(len(areas), 100)
        # 9 x 9 squares of 2 x 2 mm
        self.assertAlmostEqual(areas.sum(), 9 * 9 * 4.0, places=5)

    def test_a_mesh_without_triangles_has_no_area(self):
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkPolyData
        poly = vtkPolyData()
        points = vtkPoints()
        points.InsertNextPoint(0.0, 0.0, 0.0)
        poly.SetPoints(points)
        self.assertEqual(list(vertex_areas(poly)), [0.0])


class TestClusterFinding(unittest.TestCase):
    """Suprathreshold clusters, as a statistical map is finally reported."""

    def setUp(self):
        self.poly = _grid_mesh(rows=12, columns=12, spacing=1.0)
        self.values = np.zeros(144)
        # two blobs of opposite sign, well apart on the sheet
        for index in (13, 14, 25, 26):
            self.values[index] = 5.0
        for index in (117, 118, 129, 130):
            self.values[index] = -4.0

    def test_both_tails_are_found(self):
        clusters = find_surface_clusters(self.poly, self.values, threshold=1.0)
        self.assertEqual(len(clusters), 2)
        self.assertEqual([round(c['peak_value'], 1) for c in clusters], [5.0, -4.0])
        self.assertEqual([c['vertices'] for c in clusters], [4, 4])

    def test_the_peak_is_the_strongest_vertex(self):
        self.values[26] = 9.0
        clusters = find_surface_clusters(self.poly, self.values, threshold=1.0)
        self.assertEqual(clusters[0]['peak_vertex'], 26)
        self.assertAlmostEqual(clusters[0]['peak_value'], 9.0)

    def test_touching_blobs_of_opposite_sign_stay_apart(self):
        """A positive and a negative finding are two findings, not one."""
        values = np.zeros(144)
        values[13] = values[14] = 3.0
        values[25] = values[26] = -3.0      # neighbours of the two above
        clusters = find_surface_clusters(self.poly, values, threshold=1.0)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(round(c['peak_value']) for c in clusters), [-3, 3])

    def test_the_area_is_in_square_millimetres(self):
        clusters = find_surface_clusters(self.poly, self.values, threshold=1.0)
        areas = vertex_areas(self.poly)
        self.assertAlmostEqual(clusters[0]['area'],
                               float(areas[[13, 14, 25, 26]].sum()), places=6)

    def test_small_clusters_can_be_dropped(self):
        self.values[70] = 5.0            # a single-vertex speck
        self.assertEqual(len(find_surface_clusters(self.poly, self.values, 1.0)), 3)
        big = find_surface_clusters(self.poly, self.values, 1.0, min_area=1.0)
        self.assertEqual(len(big), 2)

    def test_nothing_above_the_threshold(self):
        self.assertEqual(find_surface_clusters(self.poly, self.values, 99.0), [])

    def test_they_come_ordered_by_peak(self):
        self.values[70] = 7.0
        peaks = [abs(c['peak_value'])
                 for c in find_surface_clusters(self.poly, self.values, 1.0)]
        self.assertEqual(peaks, sorted(peaks, reverse=True))


class TestSurfaceAtlases(unittest.TestCase):
    """Region names come from the .annot files shipped with T1Prep."""

    def test_the_shipped_atlases_are_listed(self):
        names = [name for name, _ in available_surface_atlases()]
        self.assertTrue(any("aparc_DK40" in name for name in names), names)
        for _name, path in available_surface_atlases():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.basename(path).startswith("lh."))

    def test_an_annotation_reads_as_labels_and_names(self):
        atlases = dict((n, p) for n, p in available_surface_atlases())
        path = next((p for n, p in atlases.items() if "aparc_DK40" in n), None)
        if path is None:
            self.skipTest("DK40 atlas not installed")
        labels, names = read_annotation(path)
        self.assertEqual(len(labels), 32492)          # the 32k template
        self.assertIn("insula", names)
        self.assertGreaterEqual(labels.min(), -1)     # -1 = unlabelled
        self.assertLess(labels.max(), len(names))

    def test_an_unreadable_file_says_so(self):
        with tempfile.NamedTemporaryFile(suffix=".annot") as handle:
            handle.write(b"not an annotation")
            handle.flush()
            with self.assertRaises(RuntimeError):
                read_annotation(handle.name)


class TestPickReadout(unittest.TestCase):
    """What the status bar says about the marked vertex."""

    class _Stub:
        """A viewer, reduced to what the readout touches."""
        _pick_text = Viewer._pick_text
        _pick_value = Viewer._pick_value
        _atlas_region = Viewer._atlas_region

        def __init__(self, poly, values=None, atlas=None, logp=False):
            from vtkmodules.util.numpy_support import numpy_to_vtk
            self.poly_l, self.poly_r = poly, None
            self._y_shift_l = self._y_shift_r = 0.0
            self.scal_l = (numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
                           if values is not None else None)
            self.scal_r = None
            self._atlas = atlas
            self._cursor_vertex = None
            self._logp = logp

        def _uses_logp_scale(self):
            return self._logp

    def setUp(self):
        self.poly = _grid_mesh(rows=4, columns=4)

    def test_nothing_marked_says_nothing(self):
        stub = self._Stub(self.poly)
        self.assertEqual(stub._pick_text(), "")

    def test_hemisphere_vertex_and_position(self):
        stub = self._Stub(self.poly)
        stub._cursor_vertex = (0, 5)
        text = stub._pick_text()
        self.assertIn("lh vertex 5", text)
        self.assertIn("(1.0, 1.0, 0.0) mm", text)

    def test_the_overlay_value_is_reported(self):
        stub = self._Stub(self.poly, values=np.arange(16) * 0.5)
        stub._cursor_vertex = (0, 6)
        self.assertIn("value 3", stub._pick_text())

    def test_a_logp_value_also_as_a_p_value(self):
        stub = self._Stub(self.poly, values=np.full(16, 2.0), logp=True)
        stub._cursor_vertex = (0, 6)
        self.assertIn("p 0.01", stub._pick_text())

    def test_the_atlas_region_is_named(self):
        atlas = {'name': 'DK40', 'names': ['unknown', 'insula'],
                 'labels': {0: np.array([1, 0] * 8)}}
        stub = self._Stub(self.poly, atlas=atlas)
        stub._cursor_vertex = (0, 0)
        self.assertIn("DK40: insula", stub._pick_text())
        stub._cursor_vertex = (0, 1)       # 'unknown' is not a region
        self.assertIn("DK40: -", stub._pick_text())


class TestClusterTableColumns(unittest.TestCase):
    """The table shows the p-value and the atlas only when they exist."""

    class _Viewer:
        def __init__(self, logp=False, atlas=None):
            self._logp, self._atlas = logp, atlas

        def _uses_logp_scale(self):
            return self._logp

    def _dialog(self, viewer):
        dialog = ClusterTableDialog.__new__(ClusterTableDialog)
        dialog.viewer = viewer
        return dialog

    def test_a_plain_overlay(self):
        columns = self._dialog(self._Viewer()).columns()
        self.assertEqual(columns, ["hemi", "peak", "vertex", "x", "y", "z",
                                   "vertices", "area (mm²)"])

    def test_a_logp_overlay_gets_a_p_column(self):
        self.assertIn("p", self._dialog(self._Viewer(logp=True)).columns())

    def test_an_atlas_adds_its_own_column(self):
        dialog = self._dialog(self._Viewer(atlas={'name': 'DK40'}))
        self.assertEqual(dialog.columns()[-1], "DK40")

    def test_the_cells_line_up_with_the_columns(self):
        dialog = self._dialog(self._Viewer(logp=True, atlas={'name': 'DK40'}))
        cluster = {'hemi': 'lh', 'peak_value': 3.0, 'peak_vertex': 7,
                   'mm': (1.0, 2.0, 3.0), 'vertices': 12, 'area': 34.5,
                   'region': 'insula'}
        self.assertEqual(len(dialog._cells(cluster)), len(dialog.columns()))
        self.assertEqual(dialog._cells(cluster)[2], "0.001")   # p for 3.0


class TestViewerKeys(unittest.TestCase):
    """Keys the viewer handles must be kept from VTK's own bindings."""

    def test_the_keys_the_viewer_acts_on_are_claimed(self):
        for key in ("r", "R", "u", "d", "l", "o", "b", "g", "h", "m",
                    "Left", "Right"):
            self.assertIn(key, VIEWER_KEYS)

    def test_vtk_keys_the_viewer_does_not_use_are_left_alone(self):
        for key in ("w", "s", "3", "f", "p"):
            self.assertNotIn(key, VIEWER_KEYS)

    def test_every_documented_shortcut_says_what_it_does(self):
        for keys, what in Viewer.SHORTCUTS:
            self.assertTrue(keys and what)


class TestSurfaceTypes(unittest.TestCase):
    """Switching between central, inflated and the other siblings."""

    class _Stub:
        available_surface_types = Viewer.available_surface_types
        current_surface_type = Viewer.current_surface_type

        class _Opts:
            mesh_left = None

        def __init__(self, path):
            self.opts = self._Opts()
            self.opts.mesh_left = path

    def setUp(self):
        import nibabel as nib
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")])
        for name in ("lh.central.sub-01.gii", "lh.inflated.sub-01.gii",
                     "lh.patch.sub-01.gii", "lh.sphere.sub-01.gii"):
            nib.save(mesh, str(self.tmp / name))
        # Named like a surface, but holding scalars — this is what used to be
        # offered and then failed to load
        scalars = nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(np.zeros(3, dtype=np.float32),
                                     intent="NIFTI_INTENT_SHAPE")])
        for name in ("lh.mc.sub-01.gii", "lh.sqrtsulc.sub-01.gii"):
            nib.save(scalars, str(self.tmp / name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_the_three_useful_surfaces_are_offered(self):
        stub = self._Stub(str(self.tmp / "lh.central.sub-01.gii"))
        found = dict(stub.available_surface_types())
        self.assertEqual(sorted(found), ["central", "inflated", "patch"])
        self.assertTrue(found["inflated"].endswith("lh.inflated.sub-01.gii"))

    def test_scalar_files_named_like_surfaces_are_not_offered(self):
        """lh.mc.* and lh.sqrtsulc.* hold values, not geometry."""
        stub = self._Stub(str(self.tmp / "lh.central.sub-01.gii"))
        offered = dict(stub.available_surface_types())
        self.assertNotIn("mc", offered)
        self.assertNotIn("sqrtsulc", offered)

    def test_the_current_one_is_known(self):
        stub = self._Stub(str(self.tmp / "lh.inflated.sub-01.gii"))
        self.assertEqual(stub.current_surface_type(), "inflated")
        self.assertIsNone(
            self._Stub(str(self.tmp / "lh.sphere.sub-01.gii")).current_surface_type())

    def test_a_name_without_a_type_token(self):
        stub = self._Stub(str(self.tmp / "surface.gii"))
        self.assertEqual(stub.available_surface_types(), [])
        self.assertIsNone(stub.current_surface_type())


class TestFlatSurfaces(unittest.TestCase):
    """A flat map is shown once per hemisphere, not rotated into six views."""

    def test_a_flat_patch_is_recognised(self):
        poly = _grid_mesh(rows=6, columns=6)      # lies in z = 0
        self.assertTrue(Viewer.is_flat_surface(poly))

    def test_a_folded_surface_is_not(self):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        source = vtkSphereSource()
        source.SetRadius(50.0)
        source.Update()
        self.assertFalse(Viewer.is_flat_surface(source.GetOutput()))

    def test_nothing_loaded(self):
        self.assertFalse(Viewer.is_flat_surface(None))

    def test_the_layout_has_one_view_per_hemisphere(self):
        stub = type("_Stub", (), {
            '_montage_layout': Viewer._montage_layout,
            'is_flat_surface': staticmethod(Viewer.is_flat_surface),
            '_MONTAGE_ORDER': Viewer._MONTAGE_ORDER,
            'poly_l': _grid_mesh(rows=6, columns=6),
        })()
        order, posx, posy, rotx, rotz = stub._montage_layout()
        self.assertEqual(order, (0, 1))
        self.assertEqual(len(posx), 2)
        # Both start at the origin; _separate_flat_views moves the right one
        # once its rotated bounds are known (see TestMontageHemispheres)
        self.assertEqual(posx, [0.0, 0.0])
        self.assertEqual(rotx, [270, 270])
        # turned opposite ways, so the second map mirrors the first
        self.assertEqual(rotz, [90, -90])

    def test_a_folded_surface_keeps_the_six_views(self):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        source = vtkSphereSource(); source.SetRadius(50.0); source.Update()
        stub = type("_Stub", (), {
            '_montage_layout': Viewer._montage_layout,
            'is_flat_surface': staticmethod(Viewer.is_flat_surface),
            '_MONTAGE_ORDER': Viewer._MONTAGE_ORDER,
            'poly_l': source.GetOutput(),
        })()
        order, posx, _posy, _rotx, _rotz = stub._montage_layout()
        self.assertEqual(order, Viewer._MONTAGE_ORDER)
        self.assertEqual(len(posx), 6)


class TestZoomLock(unittest.TestCase):
    """The mouse must not change the zoom: a right-click leaves the style
    zooming on every later mouse move, which makes clicking a vertex
    impossible."""

    def test_locked_unless_asked_otherwise(self):
        self.assertFalse(parse_args(["lh.central.gii"]).free_zoom)
        self.assertTrue(parse_args(["lh.central.gii", "-free-zoom"]).free_zoom)

    def test_the_zoom_keys_are_claimed_from_vtk(self):
        for key in ("plus", "equal", "minus"):
            self.assertIn(key, VIEWER_KEYS)

    def test_zooming_is_documented(self):
        self.assertTrue(any("zoom" in what.lower() for _keys, what in Viewer.SHORTCUTS))


class TestUnderlay(unittest.TestCase):
    """What the surface is shaded with, separate from which surface it is."""

    class _Stub:
        _underlay_file = Viewer._underlay_file
        available_underlays = Viewer.available_underlays
        UNDERLAYS = Viewer.UNDERLAYS

        class _Opts:
            mesh_left = None

        def __init__(self, path):
            self.opts = self._Opts()
            self.opts.mesh_left = path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        for name in ("lh.central.sub-01.gii", "lh.mc.sub-01.gii",
                     "lh.sqrtsulc.sub-01.gii"):
            (self.tmp / name).write_text("")
        self.mesh = str(self.tmp / "lh.central.sub-01.gii")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_shading_files_are_found_next_to_the_surface(self):
        stub = self._Stub(self.mesh)
        self.assertTrue(str(stub._underlay_file("mc")).endswith("lh.mc.sub-01.gii"))
        self.assertTrue(
            str(stub._underlay_file("sqrtsulc")).endswith("lh.sqrtsulc.sub-01.gii"))

    def test_all_three_are_offered_when_the_files_are_there(self):
        offered = self._Stub(self.mesh).available_underlays()
        self.assertEqual(offered, [("Mean curvature", "mc"),
                                   ("Sulcal depth", "sqrtsulc"),
                                   ("None", None)])

    def test_sulcal_depth_needs_its_file(self):
        (self.tmp / "lh.sqrtsulc.sub-01.gii").unlink()
        offered = dict((label, token)
                       for label, token in self._Stub(self.mesh).available_underlays())
        self.assertNotIn("Sulcal depth", offered)
        # curvature can be computed from the mesh, so it is always offered
        self.assertIn("Mean curvature", offered)
        self.assertIn("None", offered)

    def test_a_surface_without_siblings(self):
        stub = self._Stub(str(self.tmp / "surface.gii"))
        self.assertIsNone(stub._underlay_file("mc"))
        self.assertEqual([token for _l, token in stub.available_underlays()],
                         ["mc", None])


class TestMontageHemispheres(unittest.TestCase):
    """Each view shows its own hemisphere, and flat maps are kept apart."""

    class _Stub:
        _separate_flat_views = Viewer._separate_flat_views

        def __init__(self, left, right):
            self._montage_bkg = [left, right]
            self._montage_ov = [None, None]

    @staticmethod
    def _actor(x_offset):
        from vtkmodules.vtkFiltersSources import vtkCubeSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        source = vtkCubeSource()
        source.SetXLength(100.0); source.SetYLength(60.0); source.SetZLength(1.0)
        source.Update()
        mapper = vtkPolyDataMapper(); mapper.SetInputData(source.GetOutput())
        actor = vtkActor(); actor.SetMapper(mapper)
        actor.AddPosition(x_offset, 0.0, 0.0)
        return actor

    def test_overlapping_maps_are_moved_apart(self):
        left, right = self._actor(0.0), self._actor(0.0)   # exactly on top
        stub = self._Stub(left, right)
        stub._separate_flat_views()
        self.assertGreaterEqual(right.GetBounds()[0], left.GetBounds()[1])

    def test_they_end_up_level_with_each_other(self):
        left, right = self._actor(0.0), self._actor(0.0)
        right.AddPosition(0.0, 40.0, 0.0)
        stub = self._Stub(left, right)
        stub._separate_flat_views()
        left_centre = 0.5 * (left.GetBounds()[2] + left.GetBounds()[3])
        right_centre = 0.5 * (right.GetBounds()[2] + right.GetBounds()[3])
        self.assertAlmostEqual(left_centre, right_centre, places=5)

    def test_the_overlay_clone_follows_its_surface(self):
        left, right = self._actor(0.0), self._actor(0.0)
        overlay = self._actor(0.0)
        stub = self._Stub(left, right)
        stub._montage_ov = [None, overlay]
        stub._separate_flat_views()
        self.assertAlmostEqual(overlay.GetBounds()[0], right.GetBounds()[0], places=5)

    def test_a_missing_hemisphere_is_survived(self):
        stub = self._Stub(self._actor(0.0), None)
        stub._separate_flat_views()      # must not raise


class TestOverlayRange(unittest.TestCase):
    """The default colour range, as cat_surf_results.m picks it.

    Its three rules: symmetric scaling for two-sided data, whole numbers for
    -log10(p) maps, and a range that starts at the threshold when nothing
    lies below the negative one — otherwise part of the colormap is spent on
    values that are hidden anyway.
    """

    def test_a_thresholded_one_sided_map_starts_at_the_threshold(self):
        values = np.array([0.0, 2.0, 5.7])
        self.assertEqual(default_overlay_range(values, threshold=1.3, logp=True),
                         (1.3, 6.0))

    def test_without_a_threshold_it_starts_at_the_data(self):
        values = np.array([0.0, 2.0, 5.7])
        self.assertEqual(default_overlay_range(values, threshold=0.0, logp=True),
                         (0.0, 6.0))

    def test_two_sided_data_is_scaled_symmetrically(self):
        values = np.array([-4.2, 0.0, 5.7])
        self.assertEqual(default_overlay_range(values, threshold=1.3, logp=True),
                         (-6.0, 6.0))
        # the same without the rounding of a p-map
        low, high = default_overlay_range(values, threshold=0.0, logp=False)
        self.assertAlmostEqual(low, -5.7)
        self.assertAlmostEqual(high, 5.7)

    def test_a_p_map_gets_whole_numbers(self):
        low, high = default_overlay_range(np.array([0.3, 4.2]), 0.0, logp=True)
        self.assertEqual((low, high), (0.0, 5.0))

    def test_ordinary_data_is_left_as_it_is(self):
        low, high = default_overlay_range(np.array([0.6, 2.5, 4.1]), 0.0, False)
        self.assertAlmostEqual(low, 0.6)
        self.assertAlmostEqual(high, 4.1)

    def test_nothing_to_scale(self):
        self.assertIsNone(default_overlay_range(np.array([])))
        self.assertIsNone(default_overlay_range(np.array([2.0, 2.0])))
        self.assertIsNone(default_overlay_range(np.array([np.nan, np.inf])))


class TestCurvatureShading(unittest.TestCase):
    """The underlay greys, following cat_surf_results.m."""

    def test_the_relief_keeps_its_order(self):
        values = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        shaded = Viewer.shade_from_curvature(values)
        self.assertTrue(np.all(np.diff(shaded) > 0))

    def test_the_greys_stay_inside_the_band(self):
        rng = np.random.default_rng(0)
        shaded = Viewer.shade_from_curvature(rng.normal(0, 20, 5000))
        self.assertGreaterEqual(shaded.min(), UNDERLAY_GREYS[0] - 1e-9)
        self.assertLessEqual(shaded.max(), UNDERLAY_GREYS[1] + 1e-9)

    def test_the_square_root_pulls_in_the_tails(self):
        """A few extreme vertices must not flatten everything else."""
        values = np.concatenate([np.linspace(-1, 1, 99), [500.0]])
        shaded = Viewer.shade_from_curvature(values)
        with_root = float(shaded[:99].max() - shaded[:99].min())
        # what a plain linear mapping of the same data would leave for the bulk
        band = UNDERLAY_GREYS[1] - UNDERLAY_GREYS[0]
        linear = band * (values[:99].max() - values[:99].min()) / (500.0 + 1.0)
        self.assertGreater(with_root, 5 * linear)

    def test_sulcal_depth_is_inverted(self):
        values = np.array([0.0, 1.0, 4.0, 9.0])
        plain = Viewer.shade_from_curvature(values)
        inverted = Viewer.shade_from_curvature(values, invert=True)
        self.assertTrue(np.all(np.diff(inverted) < 0))     # deep = dark
        self.assertAlmostEqual(float(plain[0] + inverted[0]),
                               float(UNDERLAY_GREYS[0] + UNDERLAY_GREYS[1]), places=6)

    def test_a_flat_surface_does_not_divide_by_zero(self):
        shaded = Viewer.shade_from_curvature(np.zeros(10))
        self.assertTrue(np.all(np.isfinite(shaded)))


class TestRangeFollowsThreshold(unittest.TestCase):
    """Changing the threshold moves the lower end of the range with it.

    cat_surf_results sets clim right after clip, so the colours always span
    the values that are actually shown — unless the range was given by hand.
    """

    class _Stub:
        _rescale_overlay_to_clip = Viewer._rescale_overlay_to_clip
        actor_ov_l = actor_ov_r = None

        class _Opts:
            fix_scaling = False

        def __init__(self, auto, user_set=False, fix_scaling=False):
            self.opts = self._Opts()
            self.opts.fix_scaling = fix_scaling
            self._user_set_range = user_set
            self._auto = auto
            self.overlay_range = [0.0, 6.0]
            self.applied = 0

        def _auto_overlay_range(self):
            return self._auto

        def _apply_clip_to_overlay_luts(self):
            self.applied += 1

    def test_the_range_follows(self):
        stub = self._Stub((3.0, 6.0))
        stub._rescale_overlay_to_clip()
        self.assertEqual(stub.overlay_range, [3.0, 6.0])
        self.assertEqual(stub.applied, 1)      # the clipped colours are redone

    def test_a_range_given_by_hand_is_kept(self):
        stub = self._Stub((3.0, 6.0), user_set=True)
        stub._rescale_overlay_to_clip()
        self.assertEqual(stub.overlay_range, [0.0, 6.0])

    def test_a_range_fixed_across_overlays_is_kept(self):
        stub = self._Stub((3.0, 6.0), fix_scaling=True)
        stub._rescale_overlay_to_clip()
        self.assertEqual(stub.overlay_range, [0.0, 6.0])

    def test_nothing_to_scale_changes_nothing(self):
        for auto in (None, (5.0, 5.0), (6.0, 3.0)):
            stub = self._Stub(auto)
            stub._rescale_overlay_to_clip()
            self.assertEqual(stub.overlay_range, [0.0, 6.0])


class TestFoldedCurvature(unittest.TestCase):
    """Shading comes from the folded surface, whatever surface is shown.

    An inflated or flattened surface is smooth by construction, so its own
    curvature carries no relief — cat_surf_results shades every surface with
    the curvature of the folded one.
    """

    TEMPLATES = (Path(__file__).resolve().parents[1] / "src" / "t1prep" / "data"
                 / "templates_surfaces_32k")

    class _Stub:
        _folded_curvature = Viewer._folded_curvature
        available_surface_types = Viewer.available_surface_types
        current_surface_type = Viewer.current_surface_type

        class _Opts:
            mesh_left = None

        def __init__(self, path, poly_l, poly_r=None):
            self.opts = self._Opts()
            self.opts.mesh_left = path
            self.poly_l, self.poly_r = poly_l, poly_r

    def setUp(self):
        if not (self.TEMPLATES / "lh.inflated.freesurfer.gii").exists():
            self.skipTest("surface templates not installed")

    @staticmethod
    def _own_curvature(poly):
        from vtkmodules.vtkFiltersGeneral import vtkCurvatures
        from vtkmodules.util.numpy_support import vtk_to_numpy
        curvature = vtkCurvatures()
        curvature.SetInputData(poly)
        curvature.SetCurvatureTypeToMean()
        curvature.Update()
        return vtk_to_numpy(curvature.GetOutput().GetPointData().GetScalars())

    def test_an_inflated_surface_is_shaded_with_the_folded_one(self):
        path = str(self.TEMPLATES / "lh.inflated.freesurfer.gii")
        inflated = read_mesh_pair(path)[0]
        stub = self._Stub(path, inflated)
        shading = stub._folded_curvature()[0]
        own = self._own_curvature(inflated)
        self.assertEqual(len(shading), inflated.GetNumberOfPoints())
        # the inflated surface has almost no relief of its own
        self.assertGreater(shading.std(), 3 * own.std())

    def test_the_folded_surface_uses_its_own(self):
        path = str(self.TEMPLATES / "lh.central.freesurfer.gii")
        central = read_mesh_pair(path)[0]
        stub = self._Stub(path, central)
        shading = stub._folded_curvature()[0]
        np.testing.assert_allclose(shading, self._own_curvature(central), atol=1e-9)

    def test_a_surface_without_a_central_sibling(self):
        """Nothing to borrow from: the displayed surface has to do."""
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            path = str(Path(tmp) / "lh.inflated.sub-01.gii")
            shutil.copy(self.TEMPLATES / "lh.inflated.freesurfer.gii", path)
            poly = read_mesh_pair(path)[0]
            stub = self._Stub(path, poly)
            shading = stub._folded_curvature()[0]
            np.testing.assert_allclose(shading, self._own_curvature(poly), atol=1e-9)


class TestAtlasBorders(unittest.TestCase):
    """The boundaries between atlas regions, drawn on the surface.

    cat_surf_results draws the 0.5-isocontour of each region; the same lines
    come out of one pass over the triangles.
    """

    def setUp(self):
        # a 10 x 10 sheet of unit squares, split into a left and a right region
        self.poly = _grid_mesh(rows=10, columns=10, spacing=1.0)
        self.labels = np.zeros(100, dtype=int)
        for vertex in range(100):
            if vertex % 10 >= 5:            # x >= 5 is the second region
                self.labels[vertex] = 1

    @staticmethod
    def _points(border):
        from vtkmodules.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(border.GetPoints().GetData())

    def test_a_boundary_becomes_line_segments(self):
        border = atlas_border_lines(self.poly, self.labels)
        self.assertGreater(border.GetNumberOfLines(), 0)
        self.assertEqual(border.GetNumberOfPoints(), 2 * border.GetNumberOfLines())

    def test_the_lines_lie_on_the_boundary(self):
        border = atlas_border_lines(self.poly, self.labels)
        x = self._points(border)[:, 0]
        # the two regions meet between x = 4 and x = 5
        self.assertGreaterEqual(x.min(), 4.0 - 1e-9)
        self.assertLessEqual(x.max(), 5.0 + 1e-9)

    def test_one_region_has_no_boundary(self):
        border = atlas_border_lines(self.poly, np.zeros(100, dtype=int))
        self.assertEqual(border.GetNumberOfLines(), 0)

    def test_three_regions_meeting_are_closed_through_the_centre(self):
        labels = self.labels.copy()
        labels[55] = 2                      # a third region inside the second
        border = atlas_border_lines(self.poly, labels)
        self.assertGreater(border.GetNumberOfLines(),
                           atlas_border_lines(self.poly, self.labels).GetNumberOfLines())

    def test_labels_that_do_not_fit_the_surface(self):
        self.assertEqual(
            atlas_border_lines(self.poly, np.zeros(7, dtype=int)).GetNumberOfLines(), 0)
        self.assertEqual(atlas_border_lines(self.poly, None).GetNumberOfLines(), 0)
        self.assertEqual(atlas_border_lines(None, self.labels).GetNumberOfLines(), 0)

    def test_a_real_atlas_gives_borders_for_every_region(self):
        atlases = dict(available_surface_atlases())
        path = next((p for n, p in atlases.items() if "aparc_DK40" in n), None)
        if path is None:
            self.skipTest("DK40 atlas not installed")
        templates = (Path(__file__).resolve().parents[1] / "src" / "t1prep" / "data"
                     / "templates_surfaces_32k" / "lh.central.freesurfer.gii")
        if not templates.exists():
            self.skipTest("surface templates not installed")
        poly = read_mesh_pair(str(templates))[0]
        labels, _names = read_annotation(path)
        border = atlas_border_lines(poly, labels)
        self.assertGreater(border.GetNumberOfLines(), 1000)


class TestAtlasHemisphereOrder(unittest.TestCase):
    """An rh.* annotation names the right hemisphere, whichever was chosen.

    Picking the rh file (through "Other…" or by dropping it on the window)
    used to put its labels on the left surface, which draws every region — and
    every border — in the wrong place.
    """

    ATLASES = (Path(__file__).resolve().parents[1] / "src" / "t1prep" / "data"
               / "atlases_surfaces_32k")

    class _Stub:
        set_atlas = Viewer.set_atlas
        _update_pick_label = lambda self: None
        _build_border_actors = lambda self: None
        _build_montage = lambda self: None
        show_borders = False
        poly_l = poly_r = None

        def __init__(self):
            self._atlas = None
            self.atlas_path = None

    def setUp(self):
        self.left = self.ATLASES / "lh.aparc_DK40.freesurfer.annot"
        self.right = self.ATLASES / "rh.aparc_DK40.freesurfer.annot"
        if not (self.left.exists() and self.right.exists()):
            self.skipTest("surface atlases not installed")

    def _labels(self, chosen):
        stub = self._Stub()
        stub.set_atlas(str(chosen))
        return stub._atlas['labels']

    def test_choosing_either_file_fills_the_slots_the_same_way(self):
        from_left = self._labels(self.left)
        from_right = self._labels(self.right)
        np.testing.assert_array_equal(from_left[0], from_right[0])
        np.testing.assert_array_equal(from_left[1], from_right[1])

    def test_the_left_slot_holds_the_left_hemisphere(self):
        expected, _names = read_annotation(str(self.left))
        for chosen in (self.left, self.right):
            np.testing.assert_array_equal(self._labels(chosen)[0], expected)
