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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import nibabel as nib
    from t1prep.gui.cat_vol_view import CatImageViewer, VolumeViewerWindow
except Exception as exc:  # pragma: no cover - depends on optional deps
    raise unittest.SkipTest(f"cat_vol_view unavailable: {exc}")


def _write_volume(path: Path, affine: np.ndarray, shape=(20, 24, 28)):
    """Write a volume whose voxel values encode their own file index."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _file_voxel_of(value: float, shape=(20, 24, 28)):
    """Recover the file voxel index a value from _write_volume came from.

    VTK may store the image with flipped axes relative to the file (and
    compensates in the sform it reports), so the voxel a VTK index refers to is
    identified through the value rather than assumed.
    """
    return tuple(int(v) for v in np.unravel_index(int(round(value)), shape))


class TestWorldSpace(unittest.TestCase):
    """The viewer must work in the millimetre space of the NIfTI header.

    The slices are linked to a surface viewer through world coordinates, so a
    voxel-only mapping (VTK reads NIfTI into voxel space) would put the cursor
    of the two windows in different places.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _viewer(self, affine, shape=(20, 24, 28)):
        name = f"vol_{abs(hash(affine.tobytes())) % 10 ** 8}.nii.gz"
        path = _write_volume(self.tmp / name, affine, shape)
        viewer = CatImageViewer(percentile_range=None)
        viewer.load_image(str(path))
        return viewer

    def test_header_transform_is_used(self):
        """World coordinates must agree with the NIfTI affine, voxel by voxel."""
        affine = np.array([[-1.0, 0, 0, 80.0],
                           [0, 1.0, 0, -116.0],
                           [0, 0, 1.0, -72.0],
                           [0, 0, 0, 1.0]])
        viewer = self._viewer(affine)
        self.assertTrue(viewer._world_from_header)
        for index in ((0, 0, 0), (3, 4, 5), (19, 23, 27)):
            value = viewer.get_value_at_index(index)
            expected = affine @ np.array([*_file_voxel_of(value), 1.0])
            self.assertEqual(
                tuple(round(v, 3) for v in viewer._world_from_index(index)),
                tuple(round(float(v), 3) for v in expected[:3]),
                index,
            )

    def test_anisotropic_voxels(self):
        """Voxel size must be part of the index-to-world transform.

        VTK reports the sform normalized (it maps index*spacing to world), so
        using it directly on voxel indices scales every coordinate by 1/spacing
        — an offset in all three axes for any image that is not 1 mm.
        """
        affine = np.array([[-2.0, 0, 0, 50.0],
                           [0, 3.0, 0, -60.0],
                           [0, 0, 4.0, -30.0],
                           [0, 0, 0, 1.0]])
        viewer = self._viewer(affine, shape=(10, 12, 14))
        for index in ((0, 0, 0), (1, 1, 1), (9, 11, 13)):
            value = viewer.get_value_at_index(index)
            expected = affine @ np.array([*_file_voxel_of(value, (10, 12, 14)), 1.0])
            self.assertEqual(
                tuple(round(v, 3) for v in viewer._world_from_index(index)),
                tuple(round(float(v), 3) for v in expected[:3]),
                index,
            )
        # Physical extent must come out in millimetres, not voxels
        self.assertEqual(
            [round(v, 3) for v in viewer._voxel_axis_lengths()],
            [20.0, 36.0, 56.0],
        )

    def test_world_index_round_trip(self):
        affine = np.array([[0.0, 0, -1.2, 109.0],
                           [0, -1.05, 0, 121.0],
                           [-1.1, 0, 0, 116.0],
                           [0, 0, 0, 1.0]])
        viewer = self._viewer(affine)
        for ijk in ((0, 0, 0), (5, 7, 11), (19, 23, 27)):
            world = viewer._world_from_index(ijk)
            back = viewer._index_from_world(world)
            self.assertEqual(tuple(int(round(v)) for v in back), ijk)

    def test_pane_axes_follow_anatomy(self):
        """Each pane slices the voxel axis matching its anatomical plane."""
        ras = np.diag([1.0, 1.0, 1.0, 1.0])
        # axial cuts k, sagittal i, coronal j for an RAS-ordered image
        self.assertEqual(self._viewer(ras)._pane_axis, [2, 0, 1])

        # Sagittal-first storage (as written by many scanners): i runs along
        # -z (superior-inferior), j along -y, k along -x
        permuted = np.array([[0.0, 0, -1.0, 109.0],
                             [0, -1.0, 0, 121.0],
                             [-1.0, 0, 0, 116.0],
                             [0, 0, 0, 1.0]])
        self.assertEqual(self._viewer(permuted)._pane_axis, [0, 2, 1])

    def test_falls_back_without_header_transform(self):
        """A zeroed sform/qform must not produce a degenerate world space."""
        path = _write_volume(self.tmp / "noform.nii.gz", np.eye(4))
        img = nib.load(str(path))
        img.header.set_sform(None, code=0)
        img.header.set_qform(None, code=0)
        nib.save(nib.Nifti1Image(img.get_fdata(), None, img.header), str(path))
        viewer = CatImageViewer(percentile_range=None)
        viewer.load_image(str(path))
        self.assertIsNotNone(viewer._vox2world)
        self.assertEqual(len(viewer._pane_axis), 3)
        self.assertEqual(sorted(viewer._pane_axis), [0, 1, 2])


class TestCursorApi(unittest.TestCase):
    """Public cursor API used to link the viewer to the surface window."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        affine = np.array([[-1.0, 0, 0, 80.0],
                           [0, 1.0, 0, -116.0],
                           [0, 0, 1.0, -72.0],
                           [0, 0, 0, 1.0]])
        path = _write_volume(Path(self._tmp.name) / "vol.nii.gz", affine)
        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_set_and_get_world_position(self):
        """A position sent in comes back unchanged (it is on the voxel grid)."""
        self.viewer.set_world_position((70.0, -106.0, -62.0))
        self.assertEqual(self.viewer.get_world_position(), (70.0, -106.0, -62.0))

    def test_position_is_clamped_to_the_volume(self):
        self.viewer.set_world_position((1e6, 1e6, 1e6))
        i, j, k = self.viewer.get_index()
        ext = self.viewer._image.GetExtent()
        self.assertTrue(ext[0] <= i <= ext[1])
        self.assertTrue(ext[2] <= j <= ext[3])
        self.assertTrue(ext[4] <= k <= ext[5])

    def test_callback_only_on_request(self):
        seen = []
        self.viewer.on_position_changed = seen.append
        # Moving the cursor from a linked window must not echo back
        self.viewer.set_world_position((70.0, -106.0, -62.0))
        self.assertEqual(seen, [])
        # A local change (click, scroll) reports the new position
        self.viewer.set_world_position((70.0, -106.0, -62.0), notify=True)
        self.assertEqual(seen, [(70.0, -106.0, -62.0)])

    def test_zoom_to_bounding_box(self):
        """Zoom sets an mm bounding box centred on the cursor (SPM style)."""
        self.viewer.set_world_position((70.0, -106.0, -62.0))
        full = [r.GetActiveCamera().GetParallelScale() for r in self.viewer.renderers]

        self.viewer.set_field_of_view(20.0)
        self.assertEqual(self.viewer.get_field_of_view(), 20.0)
        for ren in self.viewer.renderers:
            cam = ren.GetActiveCamera()
            self.assertAlmostEqual(cam.GetParallelScale(), 10.0)
            self.assertEqual(tuple(cam.GetFocalPoint()), (70.0, -106.0, -62.0))

        # Moving the cursor keeps the zoomed view centred on it
        self.viewer.set_world_position((72.0, -104.0, -60.0))
        for ren in self.viewer.renderers:
            self.assertEqual(tuple(ren.GetActiveCamera().GetFocalPoint()),
                             (72.0, -104.0, -60.0))

        self.viewer.set_field_of_view(None)
        self.assertIsNone(self.viewer.get_field_of_view())
        for ren, scale in zip(self.viewer.renderers, full):
            self.assertAlmostEqual(ren.GetActiveCamera().GetParallelScale(), scale)

    def test_value_at_cursor(self):
        self.viewer.set_index(3, 4, 5)
        value = self.viewer.get_value_at_index()
        self.assertEqual(self.viewer.get_index(), (3, 4, 5))
        # The value identifies a voxel of the written volume
        self.assertEqual(len(_file_voxel_of(value)), 3)
        self.assertEqual(value, self.viewer.get_value_at_index((3, 4, 5)))


class TestNeurologicalOrientation(unittest.TestCase):
    """Slices are shown "left is left" (neurological), not mirrored."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        affine = np.array([[-1.0, 0, 0, 80.0],
                           [0, 1.0, 0, -116.0],
                           [0, 0, 1.0, -72.0],
                           [0, 0, 0, 1.0]])
        path = _write_volume(Path(self._tmp.name) / "vol.nii.gz", affine)
        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _screen_right(camera):
        """World direction that points right on screen: view direction x up."""
        pos = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())
        return np.cross(focal - pos, np.array(camera.GetViewUp()))

    def test_patient_right_is_on_screen_right(self):
        for pane in (CatImageViewer.VIEW_AXIAL, CatImageViewer.VIEW_CORONAL):
            right = self._screen_right(self.viewer.renderers[pane].GetActiveCamera())
            # +x is the patient's right in the RAS world of the NIfTI header
            self.assertGreater(right[0], 0.0, f"pane {pane} is mirrored")

    def test_superior_is_up_and_anterior_left_in_sagittal(self):
        cam = self.viewer.renderers[CatImageViewer.VIEW_SAGITTAL].GetActiveCamera()
        self.assertEqual(tuple(cam.GetViewUp()), (0.0, 0.0, 1.0))
        # anterior (+y) must point left on screen
        self.assertLess(self._screen_right(cam)[1], 0.0)

    def test_axial_has_anterior_up(self):
        cam = self.viewer.renderers[CatImageViewer.VIEW_AXIAL].GetActiveCamera()
        self.assertEqual(tuple(cam.GetViewUp()), (0.0, 1.0, 0.0))


class TestWindowContract(unittest.TestCase):
    """The window both tools use exposes the zoom menu levels."""

    def test_zoom_levels(self):
        labels = [label for label, _ in VolumeViewerWindow.ZOOM_LEVELS]
        self.assertIn("Full volume", labels)
        values = [mm for _, mm in VolumeViewerWindow.ZOOM_LEVELS]
        self.assertIn(None, values)
        self.assertIn(20.0, values)
        # descending order, whole volume first
        numbers = [v for v in values if v]
        self.assertEqual(numbers, sorted(numbers, reverse=True))

    def test_window_is_shared_with_the_surface_viewer(self):
        from t1prep.gui import cat_surf_view
        self.assertIs(cat_surf_view.VolumeViewerWindow, VolumeViewerWindow)


if __name__ == "__main__":
    unittest.main()
