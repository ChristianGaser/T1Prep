"""The viewers as picture makers: nothing on the screen, the right picture.

``CAT_VolView --headless --screenshot`` and ``CAT_SurfView --output`` write a
PNG and end.  A script runs them over a study, the launcher for every picture
of its results panel -- so what one call does to the desktop is multiplied.

* The volume viewer went off screen only for its last render, after it had
  initialised its window: the window was on the screen for a moment.
* The surface viewer showed its window, took the keyboard focus and sat in
  the Dock until it was done.
* Both were applications as far as macOS is concerned, with Python's icon
  appearing in the Dock for each picture.

Each case runs in a process of its own: staying in the background changes the
process for good, which the other tests of a session must not inherit.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TEMPLATE = _SRC / "t1prep" / "data" / "templates_surfaces_32k" / "lh.central.freesurfer.gii"

try:
    import nibabel as nib
    import vtkmodules  # noqa: F401
    import PySide6  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on optional GUI deps
    raise unittest.SkipTest(f"the viewers are unavailable: {exc}")


def run_python(program, *arguments, platform=None, timeout=300):
    """Run *program* in a fresh interpreter; return (process, the JSON it printed)."""
    environment = dict(os.environ, T1PREP_NO_APPS="1", PYTHONPATH=os.pathsep.join(
        [str(_SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    environment.pop("QT_QPA_PLATFORM", None)
    if platform:
        environment["QT_QPA_PLATFORM"] = platform
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(program), *map(str, arguments)],
                          capture_output=True, text=True, timeout=timeout, env=environment)
    line = next((text for text in done.stdout.splitlines() if text.startswith("RESULT ")), None)
    return done, (json.loads(line[7:]) if line else None)


class _WithATemporaryFolder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)


class TestTheVolumeViewerWritesItsPictureOffScreen(_WithATemporaryFolder):
    def setUp(self):
        super().setUp()
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = -16.0
        i, j, k = np.mgrid[0:32, 0:32, 0:32]
        radius = np.sqrt((i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2)
        nib.save(nib.Nifti1Image(np.clip(100 - 3 * radius, 0, None).astype(np.float32),
                                 affine), str(self.tmp / "image.nii.gz"))
        # A label map: 3 inside 6 voxels, 2 up to 10, 1 up to 13, like p0
        label = (radius < 13).astype(np.float32) + (radius < 10) + (radius < 6)
        nib.save(nib.Nifti1Image(label.astype(np.float32), affine),
                 str(self.tmp / "label.nii.gz"))

    PROGRAM = '''
        import json, os, sys
        from t1prep.gui import cat_vol_view as viewer

        seen = {}
        setup = viewer.CatImageViewer.setup
        render = viewer.CatImageViewer.render

        def watched_setup(self, *args, **kwargs):
            # before the interactor is initialised and anything is drawn
            seen["off_screen_at_setup"] = bool(self.render_window.GetOffScreenRendering())
            return setup(self, *args, **kwargs)

        def watched_render(self, *args, **kwargs):
            seen["levels"] = [entry["level"] for entry in self.contours]
            seen["colours"] = len({entry["color"] for entry in self.contours})
            seen["crosshair"] = bool(self.show_crosshair)
            return render(self, *args, **kwargs)

        viewer.CatImageViewer.setup = watched_setup
        viewer.CatImageViewer.render = watched_render
        seen["status"] = viewer.main(sys.argv[1:])
        seen["written"] = os.path.isfile(sys.argv[sys.argv.index("--screenshot") + 1])
        print("RESULT " + json.dumps(seen))
    '''

    def render(self, *options):
        done, seen = run_python(
            self.PROGRAM, self.tmp / "image.nii.gz", *options, "--headless",
            "--no-info", "--size", "120", "--screenshot", self.tmp / "picture.png",
            platform="offscreen")
        if seen is None:  # pragma: no cover - no OpenGL in this environment
            self.skipTest(f"rendering unavailable: {done.stderr[-400:]}")
        self.assertEqual(seen["status"], 0, done.stderr)
        self.assertTrue(seen["written"])
        return seen

    def test_it_is_off_screen_before_anything_is_drawn(self):
        seen = self.render("--contour", self.tmp / "label.nii.gz")
        self.assertTrue(seen["off_screen_at_setup"])

    def test_both_borders_of_the_grey_matter(self):
        # --contour-level outlines a volume at each level it names
        seen = self.render("--contour", self.tmp / "label.nii.gz",
                           "--contour-level", "1.5", "2.5")
        self.assertEqual(seen["levels"], [1.5, 2.5])
        self.assertEqual(seen["colours"], 2)

    def test_one_outline_in_the_middle_of_the_values_without_a_level(self):
        seen = self.render("--contour", self.tmp / "label.nii.gz")
        self.assertEqual(seen["levels"], [1.5])        # of 0 ... 3

    def test_the_crosshair_can_be_left_out(self):
        self.assertTrue(self.render()["crosshair"])
        self.assertFalse(self.render("--no-crosshair")["crosshair"])


class TestContourLevelsOnTheCommandLine(unittest.TestCase):
    def asked_for(self, *argv):
        from t1prep.gui.cat_vol_view import _contours_asked_for, _parse_args

        return _contours_asked_for(_parse_args(["image.nii.gz", *argv]))

    def test_every_volume_at_every_level(self):
        self.assertEqual(self.asked_for(), [])
        self.assertEqual(self.asked_for("--contour", "p1.nii"), [("p1.nii", None)])
        self.assertEqual(
            self.asked_for("--contour", "p0.nii", "--contour-level", "1.5", "2.5"),
            [("p0.nii", 1.5), ("p0.nii", 2.5)])
        self.assertEqual(
            self.asked_for("--contour", "p1.nii", "--contour", "p2.nii",
                           "--contour-level", "0.5"),
            [("p1.nii", 0.5), ("p2.nii", 0.5)])


@unittest.skipUnless(sys.platform == "darwin", "only macOS has a Dock to stay out of")
class TestABatchProcessStaysInTheBackground(unittest.TestCase):
    PROGRAM = '''
        import ctypes, ctypes.util, json
        from t1prep.gui.viewer_common import stay_in_the_background

        seen = {"first": stay_in_the_background(), "again": stay_in_the_background()}
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def message(restype, *argtypes):
            return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p,
                                    *argtypes)(("objc_msgSend", objc))

        application = message(ctypes.c_void_p)(
            objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication"))
        policy = lambda: message(ctypes.c_long)(                       # noqa: E731
            application, objc.sel_registerName(b"activationPolicy"))
        seen["policy"] = policy()
        # What VTK does when it creates a window of its own: ask for a
        # regular application (0), one with an icon in the Dock
        message(ctypes.c_bool, ctypes.c_long)(
            application, objc.sel_registerName(b"setActivationPolicy:"), 0)
        seen["policy_after_vtk_asked"] = policy()
        print("RESULT " + json.dumps(seen))
    '''

    def test_it_is_an_agent_and_stays_one(self):
        done, seen = run_python(self.PROGRAM)
        self.assertIsNotNone(seen, done.stderr)
        self.assertTrue(seen["first"] and seen["again"])
        # 1 is NSApplicationActivationPolicyAccessory: no Dock icon, no menu bar
        self.assertEqual(seen["policy"], 1)
        self.assertEqual(seen["policy_after_vtk_asked"], 1)


class TestTheSheetsOfTheSurfaceViewer(unittest.TestCase):
    def test_six_views_are_what_they_were(self):
        from t1prep.gui.cat_surf_view import montage_layout

        order, x, y, turn_x, turn_z = montage_layout()
        self.assertEqual(order, (0, 1, 0, 1, 0, 1))
        self.assertEqual(x, [0, 360.0, 27.0, 333.0, 180.0, 180.0])
        self.assertEqual(y, [0, 0, 144.0, 144.0, 108.0, 108.0])
        self.assertEqual(turn_x, [270, 270, 270, 270, 0, 0])
        self.assertEqual(turn_z, [90, -90, -90, 90, 0, 0])

    def test_the_smaller_sheets_leave_views_out(self):
        from t1prep.gui.cat_surf_view import MONTAGE_VIEWS, montage_layout

        self.assertEqual(sorted(MONTAGE_VIEWS), [2, 4, 6])
        six = montage_layout(6)
        for views in (4, 2):
            with self.subTest(views=views):
                order, x, y, turn_x, turn_z = montage_layout(views)
                self.assertEqual(len(order), views)
                # The same hemispheres turned the same way as in the six-view
                # sheet: lateral first, then medial; the dorsal pair is gone
                self.assertEqual(order, six[0][:views])
                self.assertEqual(turn_x, six[3][:views])
                self.assertEqual(turn_z, six[4][:views])
                self.assertEqual(len({len(x), len(y), len(turn_x), len(turn_z)}), 1)
        with self.assertRaises(ValueError):
            montage_layout(3)

    def test_the_option(self):
        from t1prep.gui.cat_surf_view import parse_args

        self.assertEqual(parse_args([str(_TEMPLATE)]).views, 6)
        self.assertEqual(parse_args([str(_TEMPLATE), "--views", "4"]).views, 4)
        with self.assertRaises(SystemExit):
            parse_args([str(_TEMPLATE), "--views", "5"])


@unittest.skipUnless(_TEMPLATE.is_file(), "surface templates not installed")
class TestReadingASurface(unittest.TestCase):
    def test_the_mesh_is_the_one_in_the_file(self):
        # The reader hands the arrays over whole; it used to copy them vertex
        # by vertex and triangle by triangle
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from t1prep.gui.cat_surf_view import read_gifti_mesh

        poly = read_gifti_mesh(str(_TEMPLATE))
        arrays = nib.load(str(_TEMPLATE)).darrays
        points = next(a.data for a in arrays if a.data.dtype.kind == "f")
        triangles = next(a.data for a in arrays if a.data.dtype.kind in "iu")
        read = vtk_to_numpy(poly.GetPoints().GetData())
        self.assertEqual(read.shape, points.shape)
        np.testing.assert_array_equal(read, points.astype(np.float32))
        self.assertEqual(poly.GetNumberOfPolys(), len(triangles))
        cells = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        np.testing.assert_array_equal(cells, triangles)


def _has_a_window_system() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@unittest.skipUnless(_TEMPLATE.is_file(), "surface templates not installed")
@unittest.skipUnless(_has_a_window_system(), "no window system for an OpenGL viewer")
class TestTheSurfaceViewerWritesItsPictureOffScreen(_WithATemporaryFolder):
    PROGRAM = '''
        import json, os, sys
        from t1prep.gui import cat_surf_view as viewer

        seen = {}
        show = viewer.Viewer.show

        def watched_show(self):
            seen["kept_off_the_screen"] = bool(self.testAttribute(
                viewer.Qt.WidgetAttribute.WA_DontShowOnScreen))
            seen["views"] = len(self._montage_bkg)
            return show(self)

        viewer.Viewer.show = watched_show
        try:
            viewer.main(sys.argv[1:])
        except SystemExit as done:
            seen["status"] = done.code or 0
        seen["written"] = os.path.isfile(sys.argv[sys.argv.index("--output") + 1])
        print("RESULT " + json.dumps(seen))
    '''

    def test_output_writes_the_picture_and_shows_no_window(self):
        picture = self.tmp / "sheet.png"
        done, seen = run_python(self.PROGRAM, _TEMPLATE, "--views", "4",
                                "--size", "360", "270", "--output", picture)
        self.assertIsNotNone(seen, done.stderr[-600:])
        self.assertEqual(seen["status"], 0, done.stderr[-600:])
        self.assertTrue(seen["written"])
        self.assertTrue(seen["kept_off_the_screen"])
        self.assertEqual(seen["views"], 4)
        from PySide6 import QtGui

        image = QtGui.QImage(str(picture))
        self.assertFalse(image.isNull())
        # A surface was drawn into it: not one colour all over
        colours = {image.pixel(x, y) for x in range(0, image.width(), 9)
                   for y in range(0, image.height(), 9)}
        self.assertGreater(len(colours), 20)


if __name__ == "__main__":
    unittest.main()
