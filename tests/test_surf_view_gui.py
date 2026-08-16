"""Integration tests for CAT_SurfView, driving a real viewer window.

The unit tests in test_surf_view.py cover the pure functions; these cover the
wiring that broke in practice — the shading following a surface switch, the
overlay surviving it, the range following the threshold, the montage layout
and the interaction guards.

A viewer needs a window system: QVTKRenderWindowInteractor uses a
QOpenGLWidget, which the "offscreen" Qt platform cannot host (it aborts rather
than raising).  The module therefore skips itself unless a real platform is
available, which is also what keeps it out of the way on a headless CI box.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DATA = _SRC / "t1prep" / "data"
_SURFACES = _DATA / "templates_surfaces_32k"
_ATLASES = _DATA / "atlases_surfaces_32k"

if os.environ.get("QT_QPA_PLATFORM", "") in ("offscreen", "minimal"):
    raise unittest.SkipTest("no window system for an OpenGL viewer")
if not (_SURFACES / "lh.central.freesurfer.gii").exists():
    raise unittest.SkipTest("surface templates not installed")

try:
    from PySide6 import QtWidgets
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from t1prep.gui import cat_surf_view as sv
except Exception as exc:  # pragma: no cover - depends on optional GUI deps
    raise unittest.SkipTest(f"cat_surf_view unavailable: {exc}")


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])


def _blob_overlay(path: Path, vertices, centres_and_peaks):
    """Write a -log10(p)-like overlay with Gaussian blobs at given vertices."""
    import nibabel as nib
    values = np.zeros(len(vertices), dtype=np.float32)
    for centre, peak in centres_and_peaks:
        distance = ((vertices - vertices[centre]) ** 2).sum(axis=1)
        values += (peak * np.exp(-distance / 120.0)).astype(np.float32)
    nib.save(nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(values)]), str(path))
    return path


class _ViewerTest(unittest.TestCase):
    """Shared setup: one temporary directory and overlays for the whole class."""

    MESH = str(_SURFACES / "lh.central.freesurfer.gii")
    ATLAS = str(_ATLASES / "lh.aparc_DK40.freesurfer.annot")

    @classmethod
    def setUpClass(cls):
        import tempfile
        import nibabel as nib
        cls.app = _application()
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        vertices = nib.load(cls.MESH).darrays[0].data
        # one-sided (positive only) and two-sided maps, both named as -log10(p)
        cls.one_sided = str(_blob_overlay(tmp / "lh.logP_one.gii", vertices,
                                          [(1000, 6.0), (9000, 2.4)]))
        cls.two_sided = str(_blob_overlay(tmp / "lh.logP_two.gii", vertices,
                                          [(1000, 6.0), (20000, -4.0)]))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def viewer(self, *argv):
        """A shown, rendered viewer; closed again when the test ends."""
        view = sv.Viewer(sv.parse_args(list(argv)))
        view.show()
        self.app.processEvents()
        view._post_init_render()
        for _ in range(2):
            self.app.processEvents()
        self.addCleanup(view.close)
        return view

    def settle(self):
        for _ in range(2):
            self.app.processEvents()

    @staticmethod
    def shading(view):
        scalars = view.curv_l_out.GetPointData().GetScalars()
        return vtk_to_numpy(scalars) if scalars is not None else None


class TestUnderlayFollowsTheSurface(_ViewerTest):
    """The shading is applied from the start and survives a surface switch."""

    def test_the_surface_is_shaded_at_startup(self):
        view = self.viewer(self.MESH)
        greys = self.shading(view)
        low, high = sv.UNDERLAY_GREYS
        self.assertGreaterEqual(greys.min(), low - 1e-6)
        self.assertLessEqual(greys.max(), high + 1e-6)

    def test_switching_the_surface_keeps_the_shading(self):
        view = self.viewer(self.MESH)
        before = self.shading(view).copy()
        surfaces = dict(view.available_surface_types())
        if "inflated" not in surfaces:
            self.skipTest("no inflated template")
        view.switch_surface_type(surfaces["inflated"])
        self.settle()
        np.testing.assert_allclose(self.shading(view), before, atol=1e-6)

    def test_an_inflated_surface_keeps_the_relief_of_the_folded_one(self):
        view = self.viewer(self.MESH)
        surfaces = dict(view.available_surface_types())
        if "inflated" not in surfaces:
            self.skipTest("no inflated template")
        view.switch_surface_type(surfaces["inflated"])
        self.settle()
        # its own curvature would be nearly flat; the folded one is not
        self.assertGreater(self.shading(view).std(), 0.02)

    def test_the_underlays_can_be_switched(self):
        view = self.viewer(self.MESH)
        for _label, token in view.available_underlays():
            view.set_underlay(token)
            self.settle()
            self.assertEqual(view.underlay, token)
            if token is None:
                self.assertFalse(view.actor_bkg_l.GetMapper().GetScalarVisibility())
                for channel, expected in zip(view.actor_bkg_l.GetProperty().GetColor(),
                                             sv.UNDERLAY_PLAIN_GREY):
                    self.assertAlmostEqual(channel, expected, places=5)
            else:
                self.assertTrue(view.actor_bkg_l.GetMapper().GetScalarVisibility())


class TestOverlaySurvivesSurfaceSwitch(_ViewerTest):
    """Switching the surface used to leave the overlay behind."""

    def test_the_values_stay_on_the_new_surface(self):
        view = self.viewer(self.MESH, "-overlay", self.one_sided)
        surfaces = dict(view.available_surface_types())
        for kind in ("inflated", "central"):
            if kind not in surfaces:
                continue
            view.switch_surface_type(surfaces[kind])
            self.settle()
            attached = view.poly_l.GetPointData().GetScalars()
            self.assertIsNotNone(attached, f"overlay lost on {kind}")
            self.assertEqual(attached.GetNumberOfTuples(),
                             view.poly_l.GetNumberOfPoints())
            self.assertAlmostEqual(float(np.abs(vtk_to_numpy(attached)).max()),
                                   6.0, places=3)


class TestRangeFollowsTheThreshold(_ViewerTest):
    """cat_surf_results sets the colour limits right after the clip window."""

    def test_a_one_sided_map_starts_at_the_threshold(self):
        view = self.viewer(self.MESH, "-overlay", self.one_sided,
                           "-clip", "-1.3", "1.3")
        self.assertAlmostEqual(view.overlay_range[0], 1.3, places=3)
        for index, expected in ((2, 2.0), (3, 3.0), (0, 0.0)):
            view.ctrl.threshold.setCurrentIndex(index)
            self.settle()
            self.assertAlmostEqual(view.overlay_range[0], expected, places=3)
            self.assertAlmostEqual(view.ctrl.range_min.value(), expected, places=3)

    def test_a_two_sided_map_stays_symmetric(self):
        view = self.viewer(self.MESH, "-overlay", self.two_sided,
                           "-clip", "-1.3", "1.3")
        for index in (0, 1, 2, 3):
            view.ctrl.threshold.setCurrentIndex(index)
            self.settle()
            low, high = view.overlay_range
            self.assertAlmostEqual(low, -high, places=6)

    def test_a_range_given_on_the_command_line_is_kept(self):
        view = self.viewer(self.MESH, "-overlay", self.one_sided,
                           "-range", "0", "8")
        view.ctrl.threshold.setCurrentIndex(3)
        self.settle()
        self.assertEqual([round(v, 3) for v in view.overlay_range], [0.0, 8.0])


class TestMontageWiring(_ViewerTest):
    """Six views for a folded surface, two mirrored maps for a flat one."""

    def test_a_folded_surface_gets_six_views(self):
        view = self.viewer(self.MESH)
        self.assertEqual(len(view._montage_order), 6)
        self.assertEqual(sum(a is not None for a in view._montage_bkg), 6)

    def test_a_one_hemisphere_overlay_is_not_shown_on_the_other(self):
        view = self.viewer(self.MESH, "-overlay", self.one_sided)
        shown = sum(a is not None for a in view._montage_ov)
        self.assertEqual(shown, 3, "the overlay belongs to one hemisphere only")

    def test_a_flat_map_is_shown_once_per_hemisphere_and_mirrored(self):
        patch = _SURFACES / "lh.patch.freesurfer.gii"
        if not patch.exists():
            self.skipTest("no patch template")
        view = self.viewer(str(patch))
        self.assertEqual(len(view._montage_order), 2)
        left, right = view._montage_bkg[0], view._montage_bkg[1]
        self.assertGreaterEqual(right.GetBounds()[0], left.GetBounds()[1],
                                "the two flat maps overlap")
        # turned opposite ways: one is the mirror image of the other
        self.assertAlmostEqual(left.GetOrientation()[2], 90.0, places=3)
        self.assertAlmostEqual(right.GetOrientation()[2], -90.0, places=3)


class TestAtlasAndBorders(_ViewerTest):
    """Region names at the cursor, and the borders drawn on the surface."""

    def test_the_region_under_the_cursor_is_named(self):
        view = self.viewer(self.MESH, "-overlay", self.one_sided)
        view.set_atlas(self.ATLAS)
        view.go_to_peak()
        self.settle()
        text = view._pick_text()
        self.assertIn("lh vertex", text)
        self.assertIn("aparc_DK40", text)

    def test_borders_appear_and_follow_the_atlas(self):
        view = self.viewer(self.MESH)
        view.set_atlas(self.ATLAS)
        view.set_atlas_borders(True)
        self.settle()
        self.assertIsNotNone(view.actor_border_l)
        first = view.actor_border_l.GetMapper().GetInput().GetNumberOfLines()
        self.assertGreater(first, 1000)
        other = _ATLASES / "lh.aparc_a2009s.freesurfer.annot"
        if other.exists():
            view.set_atlas(str(other))
            self.settle()
            self.assertNotEqual(
                view.actor_border_l.GetMapper().GetInput().GetNumberOfLines(), first)

    def test_borders_follow_a_surface_switch(self):
        view = self.viewer(self.MESH)
        view.set_atlas(self.ATLAS)
        view.set_atlas_borders(True)
        surfaces = dict(view.available_surface_types())
        if "inflated" not in surfaces:
            self.skipTest("no inflated template")
        view.switch_surface_type(surfaces["inflated"])
        self.settle()
        drawn = view.actor_border_l.GetMapper().GetInput()
        bounds = drawn.GetBounds()
        surface = view.poly_l.GetBounds()
        for axis in range(3):
            self.assertGreaterEqual(bounds[2 * axis], surface[2 * axis] - 1.0)
            self.assertLessEqual(bounds[2 * axis + 1], surface[2 * axis + 1] + 1.0)

    def test_switching_the_atlas_off_removes_them(self):
        view = self.viewer(self.MESH)
        view.set_atlas(self.ATLAS)
        view.set_atlas_borders(True)
        view.set_atlas(None)
        self.settle()
        self.assertIsNone(view.actor_border_l)


class TestClusterTable(_ViewerTest):
    """The suprathreshold clusters, as the table shows them."""

    def test_the_peaks_are_found_and_named(self):
        view = self.viewer(self.MESH, "-overlay", self.two_sided,
                           "-clip", "-1.3", "1.3")
        view.set_atlas(self.ATLAS)
        clusters = view.collect_clusters(view._default_cluster_threshold(), 10.0)
        self.assertGreaterEqual(len(clusters), 2)
        peaks = sorted(round(c["peak_value"]) for c in clusters)
        self.assertEqual(peaks[0], -4)
        self.assertEqual(peaks[-1], 6)
        for cluster in clusters:
            self.assertEqual(cluster["hemi"], "lh")
            self.assertGreater(cluster["area"], 0.0)

    def test_a_higher_threshold_finds_fewer(self):
        view = self.viewer(self.MESH, "-overlay", self.two_sided)
        many = view.collect_clusters(1.3, 0.0)
        few = view.collect_clusters(5.0, 0.0)
        self.assertGreater(len(many), len(few))


class TestInteractionGuards(_ViewerTest):
    """The mouse must not zoom, and the viewer's keys are its own."""

    @staticmethod
    def _press(view, key):
        view.iren.SetKeySym(key)
        view.iren.SetKeyCode(key if len(key) == 1 else "\0")
        view.iren.KeyPressEvent()
        view.iren.CharEvent()

    def test_the_wheel_and_a_right_drag_leave_the_zoom_alone(self):
        view = self.viewer(self.MESH)
        camera = view.ren.GetActiveCamera()
        before = camera.GetViewAngle(), tuple(camera.GetPosition())
        view.iren.SetEventPosition(300, 300)
        view.iren.MouseWheelForwardEvent()
        view.iren.RightButtonPressEvent()
        view.iren.SetEventPosition(300, 400)
        view.iren.MouseMoveEvent()
        self.settle()
        self.assertEqual((camera.GetViewAngle(), tuple(camera.GetPosition())), before)

    def test_the_keys_zoom_instead(self):
        view = self.viewer(self.MESH)
        camera = view.ren.GetActiveCamera()
        before = camera.GetViewAngle()
        self._press(view, "plus")
        self.settle()
        self.assertLess(camera.GetViewAngle(), before)

    def test_r_rotates_without_vtk_resetting_the_view(self):
        view = self.viewer(self.MESH)
        camera = view.ren.GetActiveCamera()
        camera.Dolly(0.25)
        view.ren.ResetCameraClippingRange()
        distance = camera.GetDistance()
        self._press(view, "r")
        self.settle()
        self.assertAlmostEqual(camera.GetDistance(), distance, places=3)

    def test_unlocking_gives_the_mouse_its_zoom_back(self):
        view = self.viewer(self.MESH, "-free-zoom")
        camera = view.ren.GetActiveCamera()
        # the trackball style dollies the camera; the keys change the view angle
        before = camera.GetDistance()
        view.iren.SetEventPosition(300, 300)
        view.iren.MouseWheelForwardEvent()
        self.settle()
        self.assertNotAlmostEqual(camera.GetDistance(), before, places=3)


class TestScreenshots(_ViewerTest):
    """The rendered image is what everything above is for."""

    def test_a_png_is_written(self):
        import tempfile
        view = self.viewer(self.MESH, "-overlay", self.one_sided, "-colorbar")
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "shot.png")
            view.save_png(target)
            self.assertTrue(os.path.exists(target))
            with open(target, "rb") as handle:
                self.assertEqual(handle.read(4), b"\x89PNG")


if __name__ == "__main__":
    unittest.main()
