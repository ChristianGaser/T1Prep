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
    from t1prep.gui.cat_vol_view import (
        CatImageViewer,
        VolumeViewerWindow,
        _split_inputs,
        link_windows,
        MAX_VOLUMES,
    )
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
        """A position sent in comes back unchanged."""
        self.viewer.set_world_position((70.0, -106.0, -62.0))
        self.assertEqual(self.viewer.get_world_position(), (70.0, -106.0, -62.0))

    def test_position_is_not_snapped_to_the_voxel_grid(self):
        """The cursor keeps the picked position; only the slices are voxel-wise."""
        self.viewer.set_world_position((70.4, -105.7, -62.2))
        world = self.viewer.get_world_position()
        self.assertEqual(tuple(round(v, 3) for v in world), (70.4, -105.7, -62.2))
        # the voxel used for slices and intensity is the one it falls into
        self.assertEqual(self.viewer.get_index(),
                         tuple(int(round(v)) for v in self.viewer.get_index_exact()))

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

        # …including a position between voxels
        self.viewer.set_world_position((72.4, -104.3, -60.8))
        for ren in self.viewer.renderers:
            focal = tuple(round(v, 3) for v in ren.GetActiveCamera().GetFocalPoint())
            self.assertEqual(focal, (72.4, -104.3, -60.8))

        self.viewer.set_field_of_view(None)
        self.assertIsNone(self.viewer.get_field_of_view())
        for ren, scale in zip(self.viewer.renderers, full):
            self.assertAlmostEqual(ren.GetActiveCamera().GetParallelScale(), scale)

    def test_recentering_can_be_switched_off(self):
        """A zoomed view can be pinned instead of following the cursor."""
        self.viewer.set_world_position((70.0, -106.0, -62.0))
        self.viewer.set_field_of_view(20.0)
        self.viewer.set_recenter(False)

        self.viewer.set_world_position((74.0, -102.0, -58.0))
        for ren in self.viewer.renderers:
            self.assertEqual(tuple(ren.GetActiveCamera().GetFocalPoint()),
                             (70.0, -106.0, -62.0))

        # Picking a zoom level is an explicit request and still centres
        self.viewer.set_field_of_view(40.0)
        for ren in self.viewer.renderers:
            self.assertEqual(tuple(ren.GetActiveCamera().GetFocalPoint()),
                             (74.0, -102.0, -58.0))

        # Switching it back on catches up with the cursor
        self.viewer.set_world_position((76.0, -100.0, -56.0))
        self.viewer.set_recenter(True)
        for ren in self.viewer.renderers:
            self.assertEqual(tuple(ren.GetActiveCamera().GetFocalPoint()),
                             (76.0, -100.0, -56.0))

    def test_value_at_cursor(self):
        self.viewer.set_index(3, 4, 5)
        value = self.viewer.get_value_at_index()
        self.assertEqual(self.viewer.get_index(), (3, 4, 5))
        # The value identifies a voxel of the written volume
        self.assertEqual(len(_file_voxel_of(value)), 3)
        self.assertEqual(value, self.viewer.get_value_at_index((3, 4, 5)))


class TestRegionNames(unittest.TestCase):
    """Region lists ship with different column layouts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _read(self, text):
        path = self.tmp / "atlas.csv"
        path.write_text(text, encoding="utf-8")
        return CatImageViewer._read_region_names(str(path))

    def test_with_abbreviation_column(self):
        names = self._read("ROIid;ROIabbr;ROIname;ROIcolor\n"
                           "1;lPreCG;Left Precentral gyrus;203 142 203\n"
                           "2;rPreCG;Right Precentral gyrus;203 142 203\n")
        self.assertEqual(names[1], "Left Precentral gyrus")
        self.assertEqual(names[2], "Right Precentral gyrus")

    def test_without_abbreviation_column(self):
        """Hammers puts the name in the second column."""
        names = self._read("ROIid;ROIname;Vgm;Vwm;Vcsf;ROIcolor\n"
                           "1;TL hippocampus R;1;0;0;0 204 0\n")
        self.assertEqual(names[1], "TL hippocampus R")

    def test_unreadable_file_is_not_fatal(self):
        self.assertEqual(CatImageViewer._read_region_names(str(self.tmp / "nope.csv")), {})


class TestAtlasLookup(unittest.TestCase):
    """The atlas is sampled in world space, at the user's choice."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        affine = np.array([[-1.0, 0, 0, 10.0],
                           [0, 1.0, 0, -12.0],
                           [0, 0, 1.0, -14.0],
                           [0, 0, 0, 1.0]])
        _write_volume(tmp / "image.nii.gz", affine)
        # Atlas on the same grid: label 1 in one half, 2 in the other
        labels = np.zeros((20, 24, 28), dtype=np.int16)
        labels[:10] = 1
        labels[10:] = 2
        nib.save(nib.Nifti1Image(labels, affine), str(tmp / "atlas.nii.gz"))
        (tmp / "atlas.csv").write_text(
            "ROIid;ROIabbr;ROIname;ROIcolor\n1;a;Region A;0 0 0\n2;b;Region B;0 0 0\n",
            encoding="utf-8")
        self.atlas = str(tmp / "atlas.nii.gz")

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(tmp / "image.nii.gz"))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def _region_at(self, world):
        self.viewer.set_world_position(world)
        return self.viewer._atlas_region()

    def test_no_atlas_by_default(self):
        self.assertIsNone(self.viewer.atlas_path)
        self.assertIsNone(self.viewer._atlas_region())

    def test_region_follows_world_position(self):
        self.viewer.set_atlas(self.atlas)
        self.assertEqual(self.viewer.atlas_path, self.atlas)
        # world x = 10 - i, so i < 10 (label 1) is x > 0
        self.assertEqual(self._region_at((8.0, 0.0, 0.0)), "Region A")
        self.assertEqual(self._region_at((-8.0, 0.0, 0.0)), "Region B")

    def test_atlas_can_be_switched_off(self):
        self.viewer.set_atlas(self.atlas)
        self.viewer.set_atlas(None)
        self.assertIsNone(self.viewer.atlas_path)
        self.assertIsNone(self.viewer._atlas_region())

    def test_broken_atlas_is_ignored(self):
        self.viewer.set_atlas(str(Path(self._tmp.name) / "missing.nii.gz"))
        self.assertIsNone(self.viewer.atlas_path)


class TestInfoPanel(unittest.TestCase):
    """The free quadrant reports the image properties and the cursor."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        affine = np.array([[-2.0, 0, 0, 10.0],
                           [0, 3.0, 0, -12.0],
                           [0, 0, 4.0, -14.0],
                           [0, 0, 0, 1.0]])
        path = _write_volume(Path(self._tmp.name) / "image.nii.gz", affine,
                             shape=(10, 12, 14))
        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_static_lines(self):
        text = "\n".join(self.viewer._static_info_lines())
        self.assertIn("image.nii.gz", text)
        self.assertIn("10 x 12 x 14", text)
        self.assertIn("2 x 3 x 4 mm", text)
        self.assertIn("orientation", text)

    def test_cursor_lines(self):
        self.viewer.set_world_position((4.0, 0.0, 2.0))
        text = "\n".join(self.viewer._cursor_info_lines())
        self.assertIn("voxel", text)
        self.assertIn("mm", text)
        self.assertIn("value", text)
        self.assertNotIn("atlas", text)

    def test_can_be_hidden(self):
        self.viewer.set_info_visible(False)
        self.assertFalse(self.viewer.show_info)
        self.assertFalse(self.viewer._info_actor.GetVisibility())
        self.viewer.set_info_visible(True)
        self.assertTrue(self.viewer._info_actor.GetVisibility())

    def test_interpolation_can_be_switched_to_nearest(self):
        """Raw voxels on demand, e.g. to judge segmentation edges."""
        self.assertTrue(self.viewer.interpolate)
        self.assertTrue(all(a.GetInterpolate() for a in self.viewer._image_actors))

        self.viewer.set_interpolation(False)
        self.assertFalse(self.viewer.interpolate)
        for actor in self.viewer._image_actors:
            self.assertFalse(actor.GetInterpolate())
            self.assertEqual(actor.GetProperty().GetInterpolationTypeAsString(),
                             "Nearest")

        self.viewer.set_interpolation(True)
        self.assertTrue(all(a.GetInterpolate() for a in self.viewer._image_actors))


class TestSampling(unittest.TestCase):
    """The reported intensity follows how the slices are drawn."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Voxel values equal their own world x, so the exact answer is known
        shape = (12, 12, 12)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-10.0, -10.0, -10.0]
        i = np.arange(shape[0])[:, None, None] * np.ones(shape)
        data = (affine[0, 0] * i + affine[0, 3]).astype(np.float32)
        path = Path(self._tmp.name) / "ramp.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_interpolated_value_matches_the_cursor(self):
        self.viewer.set_interpolation(True)
        for x in (-4.0, -3.0, -2.5, -1.0, 3.7):
            self.viewer.set_world_position((x, 0.0, 0.0))
            self.assertAlmostEqual(self.viewer.get_value(), x, places=3, msg=f"x={x}")

    def test_raw_value_is_the_voxel_the_cursor_is_in(self):
        self.viewer.set_interpolation(False)
        for x in (-3.0, -2.5, -1.0):
            self.viewer.set_world_position((x, 0.0, 0.0))
            self.assertEqual(self.viewer.get_value(), self.viewer.get_value_at_index())

    def test_switching_mode_changes_the_reported_value(self):
        self.viewer.set_world_position((-3.0, 0.0, 0.0))
        self.viewer.set_interpolation(True)
        interpolated = self.viewer.get_value()
        self.viewer.set_interpolation(False)
        self.assertNotAlmostEqual(interpolated, self.viewer.get_value())
        # between voxels the raw value is one of the neighbours
        self.assertEqual(self.viewer.get_value(), self.viewer.get_value_at_index())


class TestOverlayVolume(unittest.TestCase):
    """A second volume drawn in colour on top of the displayed one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.affine = np.array([[-1.0, 0, 0, 10.0],
                                [0, 1.0, 0, -12.0],
                                [0, 0, 1.0, -14.0],
                                [0, 0, 0, 1.0]])
        self.image = str(_write_volume(tmp / "image.nii.gz", self.affine))
        # Same grid, values 10x the image's, so the two are told apart
        data = 10.0 * np.arange(20 * 24 * 28, dtype=np.float32).reshape(20, 24, 28)
        nib.save(nib.Nifti1Image(data, self.affine), str(tmp / "overlay.nii.gz"))
        self.overlay = str(tmp / "overlay.nii.gz")
        # Different grid
        nib.save(nib.Nifti1Image(np.zeros((10, 12, 14), np.float32),
                                 self.affine @ np.diag([2, 2, 2, 1])),
                 str(tmp / "coarse.nii.gz"))
        self.coarse = str(tmp / "coarse.nii.gz")
        # Same dimensions, different voxel size
        stretched = self.affine.copy()
        stretched[:, :3] *= 2
        nib.save(nib.Nifti1Image(np.zeros((20, 24, 28), np.float32), stretched),
                 str(tmp / "stretched.nii.gz"))
        self.stretched = str(tmp / "stretched.nii.gz")

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(self.image)
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_none_by_default(self):
        self.assertIsNone(self.viewer.overlay_path)
        self.assertIsNone(self.viewer.get_overlay_value())

    def test_reported_intensity_comes_from_the_overlay(self):
        self.viewer.set_world_position((4.0, -6.0, -8.0))
        background = self.viewer.get_value()
        self.viewer.set_overlay(self.overlay)
        self.assertEqual(self.viewer.overlay_path, self.overlay)
        self.assertAlmostEqual(self.viewer.get_value(), 10.0 * background, places=3)
        # the image underneath is still readable
        self.assertAlmostEqual(self.viewer.get_background_value(), background, places=3)

    def test_removing_it_restores_the_image_value(self):
        self.viewer.set_world_position((4.0, -6.0, -8.0))
        background = self.viewer.get_value()
        self.viewer.set_overlay(self.overlay)
        self.viewer.set_overlay(None)
        self.assertIsNone(self.viewer.overlay_path)
        self.assertAlmostEqual(self.viewer.get_value(), background, places=3)

    def test_grid_must_match(self):
        """Resampling is out of scope, so a different grid is refused."""
        with self.assertRaises(ValueError) as caught:
            self.viewer.set_overlay(self.coarse)
        self.assertIn("dimensions", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.viewer.set_overlay(self.stretched)
        self.assertIn("voxel size", str(caught.exception))
        self.assertIsNone(self.viewer.overlay_path)

    def test_actors_follow_the_slices(self):
        self.viewer.set_overlay(self.overlay)
        self.assertEqual(len([a for a in self.viewer._overlay_actors if a]), 3)
        self.viewer.set_index(5, 6, 7)
        for pane, actor in enumerate(self.viewer._overlay_actors):
            self.assertEqual(actor.GetDisplayExtent(),
                             self.viewer._image_actors[pane].GetDisplayExtent())

    def test_settings_reach_the_lookup_table(self):
        from t1prep.gui.colormaps import FIRE
        self.viewer.set_overlay(self.overlay)
        self.viewer.overlay_colormap = FIRE
        self.viewer.overlay_range = [0.0, 100.0]
        self.viewer.overlay_clip = (-10.0, 10.0)
        lut = self.viewer._overlay_lut()
        self.assertEqual(tuple(round(v, 3) for v in lut.GetTableRange()), (0.0, 100.0))
        # values inside the clip window are transparent, so the background shows
        self.assertEqual(lut.GetTableValue(0)[3], 0.0)
        self.assertGreater(lut.GetTableValue(int(0.8 * lut.GetNumberOfTableValues()))[3], 0.0)

    def test_image_stays_visible_under_the_overlay(self):
        self.viewer.set_overlay(self.overlay)
        for actor in self.viewer._image_actors:
            self.assertTrue(actor.GetVisibility())

    def test_voxels_without_a_value_are_not_painted(self):
        """NaN outside a statistic mask must not show VTK's dark red."""
        lut = self.viewer._overlay_lut()
        self.assertEqual(tuple(lut.GetNanColor()), (0.0, 0.0, 0.0, 0.0))

    def test_overlay_is_never_smoothed(self):
        """Interpolating a thresholded map would invent values at its edges."""
        self.viewer.set_overlay(self.overlay)
        for interpolate in (True, False):
            self.viewer.set_interpolation(interpolate)
            for actor in self.viewer._overlay_actors:
                self.assertFalse(actor.GetInterpolate())
            for actor in self.viewer._image_actors:
                self.assertEqual(bool(actor.GetInterpolate()), interpolate)


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


class TestSeveralVolumes(unittest.TestCase):
    """Several volumes open one window each, with linked cursors."""

    class _Stub:
        """Stands in for a window: records what it was told to show."""

        def __init__(self):
            self.on_position_changed = None
            self.seen = []

        def set_world_position(self, world):
            self.seen.append(tuple(world))

    def test_inputs_are_split_by_type(self):
        volumes, surfaces = _split_inputs(
            ["a.nii.gz", "lh.central.gii", "b.nii", "c.mnc", "x.vtp"])
        self.assertEqual(volumes, ["a.nii.gz", "b.nii", "c.mnc"])
        self.assertEqual(surfaces, ["lh.central.gii", "x.vtp"])

    def test_at_most_three_volumes(self):
        self.assertEqual(MAX_VOLUMES, 3)

    def test_a_pick_moves_the_other_windows(self):
        windows = [self._Stub() for _ in range(3)]
        link_windows(windows)
        windows[0].on_position_changed((1.0, 2.0, 3.0), windows[0])
        self.assertEqual(windows[0].seen, [])          # never itself
        self.assertEqual(windows[1].seen, [(1.0, 2.0, 3.0)])
        self.assertEqual(windows[2].seen, [(1.0, 2.0, 3.0)])

        windows[2].on_position_changed((4.0, 5.0, 6.0), windows[2])
        self.assertEqual(windows[0].seen, [(4.0, 5.0, 6.0)])
        self.assertEqual(windows[2].seen, [(1.0, 2.0, 3.0)])

    def test_moving_a_window_does_not_echo_back(self):
        """set_world_position must not report, or the windows would loop."""
        windows = [self._Stub() for _ in range(2)]
        link_windows(windows)
        windows[0].on_position_changed((1.0, 2.0, 3.0), windows[0])
        self.assertEqual(len(windows[1].seen), 1)


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
