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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import nibabel as nib
    from vtkmodules.vtkRenderingCore import vtkRenderWindow
    from t1prep.gui.cat_vol_view import (
        CatImageViewer,
        Montage,
        MontageWindow,
        VolumeViewerWindow,
        parse_slices,
        render_montage,
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

    def test_the_same_grid_is_used_untouched(self):
        self.viewer.set_overlay(self.overlay)
        self.assertFalse(self.viewer.overlay_resampled)

    def test_another_grid_is_resampled_instead_of_refused(self):
        for path in (self.coarse, self.stretched):
            self.viewer.set_overlay(path)
            self.assertEqual(self.viewer.overlay_path, path)
            self.assertTrue(self.viewer.overlay_resampled)
            self.assertEqual(self.viewer._overlay_image.GetDimensions(),
                             self.viewer._image.GetDimensions())

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


class TestOverlayResampling(unittest.TestCase):
    """An overlay only has to be registered, not stored on the same grid.

    Atlases, templates and statistical maps rarely share the voxel grid of the
    image they belong to, so the two are lined up through the millimetre space
    of their headers.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.image_affine = np.array([[-2.0, 0, 0, 40.0],
                                      [0, 2.0, 0, -50.0],
                                      [0, 0, 2.0, -30.0],
                                      [0, 0, 0, 1.0]])
        nib.save(nib.Nifti1Image(np.zeros((40, 40, 40), np.float32), self.image_affine),
                 str(tmp / "image.nii.gz"))
        # 1 mm, half the size, and its voxel axes in a different order
        self.overlay_affine = np.array([[0, 0, 1.0, -20.0],
                                        [1.0, 0, 0, -25.0],
                                        [0, 1.0, 0, -15.0],
                                        [0, 0, 0, 1.0]])
        rng = np.random.default_rng(0)
        self.data = rng.integers(0, 9, size=(40, 40, 40)).astype(np.float32)
        nib.save(nib.Nifti1Image(self.data, self.overlay_affine),
                 str(tmp / "overlay.nii.gz"))
        self.overlay = str(tmp / "overlay.nii.gz")

        # No sform/qform at all, so there is no space to resample through
        headerless = nib.Nifti1Image(np.zeros((7, 8, 9), np.float32), np.eye(4))
        headerless.set_sform(None, code=0)
        headerless.set_qform(None, code=0)
        nib.save(headerless, str(tmp / "headerless.nii.gz"))
        self.headerless = str(tmp / "headerless.nii.gz")

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(tmp / "image.nii.gz"))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")
        self.viewer.set_overlay(self.overlay)

    def tearDown(self):
        self._tmp.cleanup()

    def _expected(self):
        """Overlay value at the cursor, straight from the file's affine."""
        centre = np.array(self.viewer._world_from_index(self.viewer.get_index()))
        index = np.round(np.linalg.inv(self.overlay_affine)
                         @ np.array([*centre, 1.0]))[:3].astype(int)
        if np.any(index < 0) or np.any(index >= np.array(self.data.shape)):
            return None
        return float(self.data[tuple(index)])

    def test_values_land_where_the_header_says(self):
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(60):
            world = (rng.uniform(-30, 30), rng.uniform(-40, 20), rng.uniform(-20, 40))
            self.viewer.set_world_position(world)
            expected = self._expected()
            if expected is None:
                continue
            checked += 1
            self.assertAlmostEqual(self.viewer.get_overlay_value(), expected, places=5)
        self.assertGreater(checked, 10)   # the positions did cover the overlay

    def test_outside_the_overlay_there_is_no_value(self):
        self.viewer.set_world_position((300.0, 300.0, 300.0))
        value = self.viewer.get_overlay_value()
        self.assertTrue(value != value)   # NaN, so the colour table skips it
        # and the panel does not print "nan"
        self.assertIn("value       -", "\n".join(self.viewer._cursor_info_lines()))

    def test_the_range_comes_from_the_file_not_from_the_padding(self):
        self.assertEqual(self.viewer.overlay_range,
                         [float(self.data.min()), float(self.data.max())])

    def test_the_panel_says_that_it_was_resampled(self):
        text = "\n".join(self.viewer._cursor_info_lines())
        self.assertIn("(resampled)", text)

    def test_without_a_header_it_is_refused_with_a_reason(self):
        with self.assertRaises(ValueError) as caught:
            self.viewer.set_overlay(self.headerless)
        message = str(caught.exception)
        self.assertIn("different voxel grid", message)
        self.assertIn("sform", message)


class TestContours(unittest.TestCase):
    """Outlines of other volumes drawn over the slices (CheckReg style)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        affine = np.array([[-1.0, 0, 0, 32.0],
                           [0, 1.0, 0, -32.0],
                           [0, 0, 1.0, -32.0],
                           [0, 0, 0, 1.0]])
        i, j, k = np.mgrid[0:64, 0:64, 0:64]
        radius = np.sqrt((i - 32) ** 2 + (j - 32) ** 2 + (k - 32) ** 2)
        nib.save(nib.Nifti1Image(np.clip(120 - 2 * radius, 0, None).astype(np.float32),
                                 affine), str(tmp / "image.nii.gz"))
        # A mask of radius 20 mm around the origin, on a 2 mm grid of its own
        i, j, k = np.mgrid[0:32, 0:32, 0:32]
        radius = np.sqrt((i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2)
        nib.save(nib.Nifti1Image((radius < 10).astype(np.float32),
                                 affine @ np.diag([2, 2, 2, 1])),
                 str(tmp / "mask.nii.gz"))
        self.mask = str(tmp / "mask.nii.gz")
        self.tmp = tmp
        self.affine = affine

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(tmp / "image.nii.gz"))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_outline_is_drawn_in_every_pane(self):
        entry = self.viewer.add_contour(self.mask)
        self.assertEqual(len(self.viewer.contours), 1)
        for pane in range(3):
            self.assertIsNotNone(entry['actors'][pane])
            entry['filters'][pane].Update()
            self.assertGreater(entry['filters'][pane].GetOutput().GetNumberOfLines(), 0)

    def test_it_lands_where_the_mask_is(self):
        """A 20 mm mask around the origin must be outlined there, not elsewhere."""
        entry = self.viewer.add_contour(self.mask)
        bounds = [0.0] * 6
        entry['actors'][CatImageViewer.VIEW_AXIAL].GetBounds(bounds)
        for low, high in ((bounds[0], bounds[1]), (bounds[2], bounds[3])):
            self.assertAlmostEqual(low, -19.0, delta=2.0)
            self.assertAlmostEqual(high, 19.0, delta=2.0)

    def test_the_level_defaults_to_the_middle_of_the_range(self):
        entry = self.viewer.add_contour(self.mask)
        self.assertAlmostEqual(entry['level'], 0.5)      # a 0..1 mask
        self.viewer.set_contour_level(self.mask, 0.9)
        self.assertAlmostEqual(self.viewer.contours[0]['level'], 0.9)
        for squares in self.viewer.contours[0]['filters']:
            self.assertAlmostEqual(squares.GetValue(0), 0.9)

    def test_the_outline_follows_the_displayed_slice(self):
        entry = self.viewer.add_contour(self.mask)
        pane = CatImageViewer.VIEW_AXIAL
        axis = self.viewer._pane_axis[pane]
        before = entry['filters'][pane].GetImageRange()
        self.viewer.step_slice(5, pane)
        after = entry['filters'][pane].GetImageRange()
        self.assertNotEqual(before, after)
        self.assertEqual(after[2 * axis], self.viewer.get_index()[axis])

    def test_colours_are_handed_out_in_turn(self):
        first = self.viewer.add_contour(self.mask)
        second = self.viewer.add_contour(self.mask)
        self.assertEqual(first['color'], CatImageViewer.CONTOUR_COLORS[0])
        self.assertEqual(second['color'], CatImageViewer.CONTOUR_COLORS[1])

    def test_a_volume_that_lands_elsewhere_is_reported(self):
        """A silent outline is what "the contour does not work" looks like.

        The two headers decide where a volume ends up; when they put it
        somewhere else entirely, nothing is drawn and nothing used to be said.
        """
        far = self.affine.copy()
        far[:3, 3] += 1000.0            # a metre away, in millimetre space
        i, j, k = np.mgrid[0:32, 0:32, 0:32]
        radius = np.sqrt((i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2)
        elsewhere = str(self.tmp / "elsewhere.nii.gz")
        nib.save(nib.Nifti1Image((radius < 10).astype(np.float32), far), elsewhere)
        with self.assertRaises(ValueError) as caught:
            self.viewer.add_contour(elsewhere)
        self.assertIn("header", str(caught.exception))
        self.assertEqual(self.viewer.contours, [])

    def test_an_outline_that_misses_these_slices_can_be_told_apart(self):
        """Drawn, but not where the cursor is: the window says so, not raises."""
        entry = self.viewer.add_contour(self.mask)
        self.assertGreater(self.viewer.contour_lines_shown(entry), 0)
        self.viewer.set_contour_level(self.mask, 5.0)     # above a 0..1 mask
        self.assertEqual(self.viewer.contour_lines_shown(entry), 0)

    def test_they_can_be_removed(self):
        self.viewer.add_contour(self.mask)
        self.viewer.add_contour(self.mask, level=0.2)
        self.viewer.remove_contour(self.mask)
        self.assertEqual(self.viewer.contours, [])
        self.viewer.add_contour(self.mask)
        self.viewer.clear_contours()
        self.assertEqual(self.viewer.contours, [])
        for renderer in self.viewer.renderers:
            actors = renderer.GetActors()
            actors.InitTraversal()
            # only the crosshair lines are left behind
            self.assertLessEqual(actors.GetNumberOfItems(), 2)


class TestSurfaceOutlines(unittest.TestCase):
    """Surfaces cut by the slice planes and drawn as coloured outlines."""

    def setUp(self):
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.vtkCommonCore import vtkFloatArray
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        affine = np.array([[-1.0, 0, 0, 32.0],
                           [0, 1.0, 0, -32.0],
                           [0, 0, 1.0, -32.0],
                           [0, 0, 0, 1.0]])
        self.image = str(tmp / "image.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), affine),
                 self.image)

        sphere = vtkSphereSource()
        sphere.SetRadius(20.0)
        sphere.SetThetaResolution(32)
        sphere.SetPhiResolution(32)
        sphere.Update()
        self.poly = sphere.GetOutput()
        # Per-vertex values, as an overlay puts them on a surface
        values = vtkFloatArray()
        values.SetName("overlay")
        values.SetNumberOfTuples(self.poly.GetNumberOfPoints())
        for i in range(self.poly.GetNumberOfPoints()):
            values.SetValue(i, float(i % 10))
        self.poly.GetPointData().SetScalars(values)

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(self.image)

    def tearDown(self):
        self._tmp.cleanup()

    def _setup(self):
        try:
            self.viewer.setup(window_title="test")
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def test_values_do_not_colour_the_outline_by_themselves(self):
        """A mapper paints by scalars unless told not to.

        The per-vertex values of a surface then decided the colour of its
        outline through VTK's default blue-to-red table, which lost the one
        colour that said which surface the outline belonged to.
        """
        self.viewer.add_surface(self.poly, (1.0, 0.0, 0.0))
        self._setup()
        for pane in range(3):
            entry = self.viewer._surface_contours[pane][0]
            self.assertEqual(len(entry['actors']), 1)
            actor = entry['actors'][0]
            self.assertFalse(actor.GetMapper().GetScalarVisibility())
            self.assertEqual(actor.GetProperty().GetColor(), (1.0, 0.0, 0.0))

    def test_each_surface_keeps_a_colour_of_its_own(self):
        self.viewer.add_surface(self.poly, VolumeViewerWindow.SURFACE_COLORS[0])
        self.viewer.add_surface(self.poly, VolumeViewerWindow.SURFACE_COLORS[1])
        self._setup()
        colors = [entry['actors'][0].GetProperty().GetColor()
                  for entry in self.viewer._surface_contours[0]]
        self.assertEqual(colors, [VolumeViewerWindow.SURFACE_COLORS[0],
                                  VolumeViewerWindow.SURFACE_COLORS[1]])
        self.assertEqual(len(set(colors)), 2)

    def test_an_overlay_colours_the_outline_the_way_it_colours_the_surface(self):
        from t1prep.gui.colormaps import JET, build_overlay_lut
        lut = build_overlay_lut(JET, 1.0)
        self.viewer.add_surface(self.poly, (1.0, 0.0, 0.0), lut=lut,
                                scalar_range=(0.0, 9.0))
        self._setup()
        for pane in range(3):
            plain, colored = self.viewer._surface_contours[pane][0]['actors']
            # The surface colour stays underneath, for the values the table
            # leaves transparent
            self.assertFalse(plain.GetMapper().GetScalarVisibility())
            self.assertEqual(plain.GetProperty().GetColor(), (1.0, 0.0, 0.0))
            self.assertTrue(colored.GetMapper().GetScalarVisibility())
            self.assertIs(colored.GetMapper().GetLookupTable(), lut)
            self.assertEqual(colored.GetMapper().GetScalarRange(), (0.0, 9.0))

    def test_the_window_takes_the_colours_the_surface_viewer_gives_it(self):
        """CAT_SurfView hands its hemispheres over with their overlay colours."""
        from t1prep.gui.cat_vol_view import _surface_display
        from t1prep.gui.colormaps import JET, build_overlay_lut
        lut = build_overlay_lut(JET, 1.0)

        given = _surface_display({'poly': self.poly, 'lut': lut,
                                  'range': (0.5, 5.0)}, (1.0, 0.0, 0.0))
        self.assertIs(given['surface'], self.poly)
        self.assertIs(given['lut'], lut)
        self.assertEqual(given['scalar_range'], (0.5, 5.0))
        # A hemisphere without an overlay keeps the colour that tells it apart
        plain = _surface_display({'poly': self.poly}, (0.0, 1.0, 0.0))
        self.assertEqual(plain['color'], (0.0, 1.0, 0.0))
        self.assertIsNone(plain['lut'])
        # And a file name is still just a file name
        self.assertEqual(_surface_display("lh.central.gii", (0.0, 0.6, 1.0)),
                         {'surface': "lh.central.gii", 'color': (0.0, 0.6, 1.0)})


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


class TestZoomLock(unittest.TestCase):
    """The mouse must not change the zoom, which the menu owns.

    The interactor style zooms on a right-drag and on the wheel.  A trackpad
    makes both easy to trigger by accident, and worse: the context menu opens
    on the same button, so it takes the release the style waits for and the
    view keeps zooming on every later mouse move.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        self.path = _write_volume(Path(self._tmp.name) / "vol.nii.gz", affine)

    def tearDown(self):
        self._tmp.cleanup()

    def _viewer(self, **kwargs):
        viewer = CatImageViewer(percentile_range=None, **kwargs)
        viewer.load_image(str(self.path))
        try:
            viewer.setup(window_title="test")
            viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")
        return viewer

    @staticmethod
    def _zoom(viewer):
        return [r.GetActiveCamera().GetParallelScale() for r in viewer.renderers]

    def test_locked_by_default(self):
        self.assertTrue(self._viewer().lock_zoom)

    def test_the_wheel_steps_slices_without_zooming(self):
        viewer = self._viewer()
        before = self._zoom(viewer)
        viewer.interactor.SetEventPosition(200, 200)
        viewer.interactor.MouseWheelForwardEvent()
        first = viewer.get_index()
        viewer.interactor.MouseWheelForwardEvent()
        self.assertEqual(self._zoom(viewer), before)
        self.assertNotEqual(viewer.get_index(), first)   # slices still step

    def test_a_right_drag_does_not_zoom(self):
        viewer = self._viewer()
        before = self._zoom(viewer)
        viewer.interactor.SetEventPosition(200, 200)
        viewer.interactor.RightButtonPressEvent()
        viewer.interactor.SetEventPosition(200, 300)
        viewer.interactor.MouseMoveEvent()
        self.assertEqual(self._zoom(viewer), before)

    def test_a_swallowed_release_does_not_leave_the_view_zooming(self):
        """The context menu keeps the release, so no drag may have started."""
        viewer = self._viewer()
        before = self._zoom(viewer)
        viewer.interactor.SetEventPosition(200, 200)
        viewer.interactor.RightButtonPressEvent()          # menu opens, no release
        for y in (300, 320, 340):
            viewer.interactor.SetEventPosition(220, y)
            viewer.interactor.MouseMoveEvent()
        self.assertEqual(self._zoom(viewer), before)

    def test_the_menu_zoom_still_works(self):
        viewer = self._viewer()
        viewer.set_field_of_view(20.0)
        self.assertEqual(self._zoom(viewer), [10.0, 10.0, 10.0])
        viewer.set_field_of_view(None)
        self.assertNotEqual(self._zoom(viewer), [10.0, 10.0, 10.0])

    def test_unlocking_gives_the_mouse_its_zoom_back(self):
        viewer = self._viewer(lock_zoom=False)
        before = self._zoom(viewer)
        viewer.interactor.SetEventPosition(200, 200)
        viewer.interactor.RightButtonPressEvent()
        viewer.interactor.SetEventPosition(200, 300)
        viewer.interactor.MouseMoveEvent()
        self.assertNotEqual(self._zoom(viewer), before)

    def test_locking_again_repairs_a_messed_up_zoom(self):
        viewer = self._viewer(lock_zoom=False)
        viewer.set_field_of_view(20.0)
        wanted = self._zoom(viewer)
        viewer.interactor.SetEventPosition(200, 200)
        viewer.interactor.RightButtonPressEvent()
        viewer.interactor.SetEventPosition(200, 320)
        viewer.interactor.MouseMoveEvent()
        self.assertNotEqual(self._zoom(viewer), wanted)

        viewer.set_lock_zoom(True)
        self.assertEqual(self._zoom(viewer), wanted)
        # ... and the interrupted drag is over, so moving on changes nothing
        viewer.interactor.SetEventPosition(260, 360)
        viewer.interactor.MouseMoveEvent()
        self.assertEqual(self._zoom(viewer), wanted)


class TestOrientationLetters(unittest.TestCase):
    """Each pane says which way it is turned, read off its camera."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # LAS on disk, as most NIfTI files are
        affine = np.array([[-1.0, 0, 0, 80.0],
                           [0, 1.0, 0, -116.0],
                           [0, 0, 1.0, -72.0],
                           [0, 0, 0, 1.0]])
        self.path = _write_volume(Path(self._tmp.name) / "vol.nii.gz", affine)

    def tearDown(self):
        self._tmp.cleanup()

    def _viewer(self, **kwargs):
        viewer = CatImageViewer(percentile_range=None, **kwargs)
        viewer.load_image(str(self.path))
        try:
            viewer.setup(window_title="test")
            viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")
        return viewer

    @staticmethod
    def _letters(viewer, pane):
        """left, right, top, bottom letter of a pane."""
        return tuple(a.GetInput() for a in viewer._orientation_actors[pane])

    def test_the_letters_match_the_neurological_layout(self):
        viewer = self._viewer()
        self.assertEqual(self._letters(viewer, CatImageViewer.VIEW_AXIAL),
                         ("L", "R", "A", "P"))
        self.assertEqual(self._letters(viewer, CatImageViewer.VIEW_CORONAL),
                         ("L", "R", "S", "I"))
        self.assertEqual(self._letters(viewer, CatImageViewer.VIEW_SAGITTAL),
                         ("A", "P", "S", "I"))

    def test_a_direction_becomes_the_letter_it_points_at(self):
        letter = CatImageViewer._direction_letter
        self.assertEqual(letter((1.0, 0.0, 0.0)), "R")
        self.assertEqual(letter((-1.0, 0.0, 0.0)), "L")
        self.assertEqual(letter((0.0, 0.9, -0.2)), "A")
        self.assertEqual(letter((0.0, -0.9, 0.2)), "P")
        self.assertEqual(letter((0.0, 0.0, 1.0)), "S")
        self.assertEqual(letter((0.0, 0.0, -1.0)), "I")
        self.assertEqual(letter((0.0, 0.0, 0.0)), "")

    def test_they_can_be_switched_off(self):
        viewer = self._viewer()
        viewer.set_orientation_labels(False)
        for pane in range(3):
            for actor in viewer._orientation_actors[pane]:
                self.assertFalse(actor.GetVisibility())
        viewer.set_orientation_labels(True)
        self.assertTrue(viewer._orientation_actors[0][0].GetVisibility())

    def test_nothing_is_claimed_without_an_anatomical_space(self):
        """No sform means the world is not RAS, so no letter is truthful."""
        viewer = self._viewer()
        viewer._world_from_header = False
        viewer._update_orientation_labels()
        for actor in viewer._orientation_actors[0]:
            self.assertFalse(actor.GetVisibility())


class TestKeyboardAndScreenshot(unittest.TestCase):
    """Slice stepping, zoom stepping and saving a PNG."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _write_volume(Path(self._tmp.name) / "vol.nii.gz",
                                  np.diag([1.0, 1.0, 1.0, 1.0]))
        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(self.path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_stepping_moves_along_the_axis_of_the_named_pane(self):
        for pane in range(3):
            axis = self.viewer._pane_axis[pane]
            before = list(self.viewer.get_index())
            self.viewer.step_slice(1, pane)
            after = list(self.viewer.get_index())
            self.assertEqual(after[axis], before[axis] + 1)
            after[axis] = before[axis]
            self.assertEqual(after, before)      # the other two do not move

    def test_without_a_pane_the_last_one_used_is_stepped(self):
        self.viewer.step_slice(1, CatImageViewer.VIEW_SAGITTAL)
        self.assertEqual(self.viewer.last_pane, CatImageViewer.VIEW_SAGITTAL)
        axis = self.viewer._pane_axis[CatImageViewer.VIEW_SAGITTAL]
        before = self.viewer.get_index()[axis]
        self.viewer.step_slice(3)
        self.assertEqual(self.viewer.get_index()[axis], before + 3)

    def test_stepping_stops_at_the_edge(self):
        self.viewer.step_slice(10 ** 6, CatImageViewer.VIEW_AXIAL)
        axis = self.viewer._pane_axis[CatImageViewer.VIEW_AXIAL]
        extent = self.viewer._image.GetExtent()
        self.assertEqual(self.viewer.get_index()[axis], extent[2 * axis + 1])

    def test_the_crosshair_can_be_hidden(self):
        self.viewer.set_crosshair_visible(False)
        self.assertFalse(self.viewer.show_crosshair)
        for pane in self.viewer._line_act:
            for actor in pane:
                self.assertFalse(actor.GetVisibility())
        self.viewer.set_crosshair_visible(True)
        self.assertTrue(self.viewer._line_act[0][0].GetVisibility())

    def test_a_screenshot_is_written(self):
        target = Path(self._tmp.name) / "shot"
        written = self.viewer.save_screenshot(str(target), scale=1)
        self.assertTrue(written.endswith(".png"))   # the suffix is added
        self.assertGreater(os.path.getsize(written), 0)
        with open(written, "rb") as fh:
            self.assertEqual(fh.read(4), b"\x89PNG")


class TestZoomStepping(unittest.TestCase):
    """The +/- keys walk through the zoom levels of the menu."""

    class _Window:
        ZOOM_LEVELS = VolumeViewerWindow.ZOOM_LEVELS
        _step_zoom = VolumeViewerWindow._step_zoom

        def __init__(self, current=None):
            self.current = current
            self.viewer = self
            self.asked = []

        def get_field_of_view(self):
            return self.current

        def set_zoom(self, mm):
            self.asked.append(mm)
            self.current = mm

    def test_zooming_in_and_out(self):
        window = self._Window()          # starts at "Full volume"
        window._step_zoom(1)
        self.assertEqual(window.current, 160.0)
        window._step_zoom(1)
        self.assertEqual(window.current, 80.0)
        window._step_zoom(-1)
        self.assertEqual(window.current, 160.0)

    def test_it_stops_at_both_ends(self):
        window = self._Window()
        window._step_zoom(-1)
        self.assertIsNone(window.current)
        for _ in range(len(VolumeViewerWindow.ZOOM_LEVELS) + 3):
            window._step_zoom(1)
        self.assertEqual(window.current, VolumeViewerWindow.ZOOM_LEVELS[-1][1])

    def test_a_zoom_that_is_not_a_level_starts_over(self):
        window = self._Window(current=33.0)
        window._step_zoom(1)
        self.assertEqual(window.current, 160.0)

    def test_every_shortcut_is_documented(self):
        keys = [row[0] for row in VolumeViewerWindow.SHORTCUTS]
        self.assertIn("s", keys)                     # screenshot
        self.assertIn("Up, Right", keys)             # slices
        for row in VolumeViewerWindow.SHORTCUTS:
            self.assertEqual(len(row), 3)
            self.assertTrue(row[1])                  # every key says what it does


class TestIntensityScaling(unittest.TestCase):
    """NIfTI stores value * scl_slope + scl_inter, and VTK hands out the value.

    Scanners and statistical maps are routinely written as integers with a
    slope, so without applying it every number the viewer shows — cursor value,
    intensity range, display window, overlay range, contour level — would be in
    storage units.
    """

    SLOPE, INTER = 2.0, 10.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.affine = np.array([[-1.0, 0, 0, 4.0],
                                [0, 1.0, 0, -4.0],
                                [0, 0, 1.0, -4.0],
                                [0, 0, 0, 1.0]])
        raw = (np.arange(8 * 8 * 8).reshape(8, 8, 8) % 100).astype(np.int16)
        image = nib.Nifti1Image(raw, self.affine)
        image.header.set_slope_inter(self.SLOPE, self.INTER)
        nib.save(image, str(tmp / "scaled.nii.gz"))
        self.path = str(tmp / "scaled.nii.gz")
        self.truth = nib.load(self.path).get_fdata()
        # The same data without scaling, to be sure nothing is touched then
        nib.save(nib.Nifti1Image(raw, self.affine), str(tmp / "plain.nii.gz"))
        self.plain = str(tmp / "plain.nii.gz")

        self.viewer = self._viewer(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _viewer(self, path):
        viewer = CatImageViewer(percentile_range=None)
        viewer.load_image(path)
        try:
            viewer.setup(window_title="test")
            viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")
        return viewer

    def _world_of(self, ijk):
        return tuple((self.affine @ np.array([*ijk, 1.0]))[:3])

    def test_the_value_is_what_nibabel_reads(self):
        for ijk in ((0, 0, 0), (1, 2, 3), (4, 5, 6), (7, 7, 7)):
            self.viewer.set_world_position(self._world_of(ijk))
            self.assertAlmostEqual(self.viewer.get_value(),
                                   float(self.truth[ijk]), places=5)

    def test_the_reported_range_is_the_real_one(self):
        self.assertEqual(tuple(self.viewer._image.GetScalarRange()),
                         (float(self.truth.min()), float(self.truth.max())))

    def test_the_panel_reports_the_file_type_and_the_scaling(self):
        text = "\n".join(self.viewer._static_info_lines())
        self.assertIn("data type   short", text)       # not the float it became
        self.assertIn("scaling     x 2 + 10", text)
        self.assertIn(f"intensity   {self.truth.min():g} .. {self.truth.max():g}", text)

    def test_a_negative_offset_reads_well(self):
        image = nib.load(self.path)
        image.header.set_slope_inter(2.0, -10.0)
        nib.save(image, self.path)
        viewer = self._viewer(self.path)
        self.assertIn("scaling     x 2 - 10", "\n".join(viewer._static_info_lines()))

    def test_the_display_window_follows_the_real_values(self):
        viewer = CatImageViewer(percentile_range=(3.0, 97.0))
        viewer.load_image(self.path)
        window, level = viewer.get_window_level()
        low, high = level - 0.5 * window, level + 0.5 * window
        self.assertGreaterEqual(low, self.truth.min() - 1e-6)
        self.assertLessEqual(high, self.truth.max() + 1e-6)
        # a window taken from the stored shorts would sit far below this
        self.assertGreater(high, self.truth.max() / 2.0)

    def test_the_overlay_is_scaled_too(self):
        self.viewer.set_overlay(self.path)
        self.assertEqual(self.viewer.overlay_range,
                         [float(self.truth.min()), float(self.truth.max())])
        self.viewer.set_world_position(self._world_of((1, 2, 3)))
        self.assertAlmostEqual(self.viewer.get_overlay_value(),
                               float(self.truth[1, 2, 3]), places=5)

    def test_a_contour_level_is_in_real_values(self):
        entry = self.viewer.add_contour(self.path)
        self.assertAlmostEqual(entry['range'][0], float(self.truth.min()), places=5)
        self.assertAlmostEqual(entry['level'],
                               0.5 * (self.truth.min() + self.truth.max()), places=5)

    def test_an_unscaled_file_is_left_alone(self):
        """No slope means no float copy of the whole volume."""
        viewer = self._viewer(self.plain)
        self.assertEqual(viewer._rescale, (1.0, 0.0))
        self.assertEqual(viewer._image.GetScalarTypeAsString(), "short")
        self.assertNotIn("scaling", "\n".join(viewer._static_info_lines()))

    def test_a_slope_of_zero_means_no_scaling(self):
        """NIfTI spec: scl_slope = 0 switches the scaling off."""
        image = nib.load(self.plain)
        image.header['scl_slope'] = 0.0
        image.header['scl_inter'] = 7.0
        nib.save(image, str(Path(self._tmp.name) / "zero.nii.gz"))
        viewer = self._viewer(str(Path(self._tmp.name) / "zero.nii.gz"))
        self.assertEqual(viewer._rescale, (1.0, 0.0))


class TestGoTo(unittest.TestCase):
    """Jumping to the origin and to the strongest voxel."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.affine = np.array([[-1.0, 0, 0, 10.0],
                                [0, 1.0, 0, -12.0],
                                [0, 0, 1.0, -14.0],
                                [0, 0, 0, 1.0]])
        data = np.zeros((20, 24, 28), dtype=np.float32)
        data[3, 4, 5] = 99.0                      # the one peak
        nib.save(nib.Nifti1Image(data, self.affine), str(tmp / "image.nii.gz"))
        self.peak_voxel = (3, 4, 5)
        overlay = np.zeros((20, 24, 28), dtype=np.float32)
        overlay[11, 12, 13] = 5.0                 # a different peak
        nib.save(nib.Nifti1Image(overlay, self.affine), str(tmp / "overlay.nii.gz"))
        self.overlay = str(tmp / "overlay.nii.gz")

        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(tmp / "image.nii.gz"))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_origin_is_the_origin_of_the_millimetre_space(self):
        self.viewer.set_index(1, 2, 3)
        self.viewer.go_to_origin()
        for value in self.viewer.get_world_position():
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_the_maximum_is_found(self):
        index = self.viewer.go_to_maximum()
        self.assertEqual(self.viewer.get_value_at_index(index), 99.0)
        self.assertEqual(self.viewer.get_index(), index)

    def test_with_an_overlay_its_maximum_wins(self):
        """A statistical map is what you want the peak of, not the anatomy."""
        self.viewer.set_overlay(self.overlay)
        self.viewer.go_to_maximum()
        self.assertEqual(self.viewer.get_overlay_value(), 5.0)

    def test_linked_windows_are_told(self):
        seen = []
        self.viewer.on_position_changed = seen.append
        self.viewer.go_to_maximum()
        self.viewer.go_to_origin()
        self.assertEqual(len(seen), 2)


class TestMontageLayout(unittest.TestCase):
    """The sheet of slices: which millimetres, in what grid, called what.

    Slices are given as start, step and stop in millimetres, the way they are
    written down in cat_vol_slice_overlay.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # 20 x 24 x 28 voxels of 1 mm; z runs from -14 to 13 mm
        affine = np.array([[-1.0, 0, 0, 10.0],
                           [0, 1.0, 0, -12.0],
                           [0, 0, 1.0, -14.0],
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

    def _montage(self, pane=CatImageViewer.VIEW_AXIAL, **kwargs):
        """A montage on its own render window — no Qt window involved."""
        return Montage(self.viewer, vtkRenderWindow(), pane=pane, **kwargs)

    # ---- millimetres ----
    def test_the_axis_is_the_one_the_slices_step_along(self):
        for pane, letter in ((CatImageViewer.VIEW_AXIAL, "z"),
                             (CatImageViewer.VIEW_CORONAL, "y"),
                             (CatImageViewer.VIEW_SAGITTAL, "x")):
            self.assertEqual(self._montage(pane=pane).axis_letter(), letter)

    def test_start_step_and_stop_are_taken_literally(self):
        montage = self._montage(slices_mm=(-10.0, 5.0, 10.0))
        self.assertEqual(montage.slice_positions(), [-10.0, -5.0, 0.0, 5.0, 10.0])
        self.assertEqual([montage._slice_label(i) for i in montage._slice_indices()],
                         ["z = -10", "z = -5", "z = 0", "z = 5", "z = 10"])

    def test_a_stop_that_is_not_a_whole_number_of_steps(self):
        montage = self._montage(slices_mm=(-10.0, 7.0, 6.0))
        self.assertEqual(montage.slice_positions(), [-10.0, -3.0, 4.0])

    def test_positions_outside_the_volume_are_left_out(self):
        low, high = self._montage()._extent_mm()
        montage = self._montage(slices_mm=(low - 100.0, 10.0, high + 100.0))
        for position in montage.slice_positions():
            self.assertGreaterEqual(position, low - 1e-6)
            self.assertLessEqual(position, high + 1e-6)
        self.assertTrue(montage.slice_positions())

    def test_a_step_of_zero_shows_nothing_rather_than_hanging(self):
        self.assertEqual(self._montage(slices_mm=(-10.0, 0.0, 10.0)).slice_positions(),
                         [])

    def test_a_reversed_range_still_runs_upwards(self):
        montage = self._montage(slices_mm=(10.0, 5.0, -10.0))
        self.assertEqual(montage.slice_positions(), [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_a_tiny_step_is_capped(self):
        montage = self._montage(slices_mm=(-14.0, 0.01, 13.0))
        self.assertEqual(len(montage.slice_positions()), MontageWindow.MAX_SLICES)

    def test_the_default_range_covers_the_volume(self):
        montage = self._montage()
        low, high = montage._extent_mm()
        self.assertGreaterEqual(montage.start_mm, low)
        self.assertLessEqual(montage.stop_mm, high)
        self.assertGreater(montage.step_mm, 0)
        self.assertGreaterEqual(len(montage._slice_indices()), 8)

    def test_millimetres_and_slices_agree(self):
        montage = self._montage()
        for position in (-10.0, -3.0, 0.0, 7.0):
            index = montage._index_for_mm(position)
            self.assertAlmostEqual(montage._mm_for_index(index), position, places=6)

    # ---- the grid ----
    def test_an_explicit_list_is_used_as_it_is(self):
        """The slices of a figure are usually hand-picked, not a series."""
        montage = self._montage(slices=[-10.0, 0.0, 4.0, 11.0])
        self.assertEqual(montage.slice_positions(), [-10.0, 0.0, 4.0, 11.0])
        self.assertEqual([montage._slice_label(i)
                          for i in montage._slice_indices()],
                         ["z = -10", "z = 0", "z = 4", "z = 11"])

    def test_a_list_beats_a_range(self):
        montage = self._montage(slices=[0.0, 5.0], slices_mm=(-10.0, 1.0, 10.0))
        self.assertEqual(montage.slice_positions(), [0.0, 5.0])
        montage.set_slices(None)          # and going back works
        self.assertEqual(len(montage.slice_positions()), 21)

    def test_a_list_is_filtered_by_the_volume_too(self):
        low, high = self._montage()._extent_mm()
        montage = self._montage(slices=[low - 50.0, 0.0, high + 50.0])
        self.assertEqual(montage.slice_positions(), [0.0])

    def test_the_grid_stays_roughly_square_when_not_told(self):
        for count, expected in ((1, (1, 1)), (4, (2, 2)), (9, (3, 3)),
                                (12, (4, 3)), (16, (4, 4))):
            self.assertEqual(self._montage()._grid(count), expected)

    def test_columns_fix_the_width_and_rows_follow(self):
        montage = self._montage(columns=5)
        self.assertEqual(montage._grid(12), (5, 3))
        self.assertEqual(montage._grid(5), (5, 1))

    def test_rows_fix_the_height_and_columns_follow(self):
        self.assertEqual(self._montage(rows=2)._grid(9), (5, 2))

    def test_both_given_is_taken_as_given(self):
        """Even when it cannot hold every slice — _build says so."""
        montage = self._montage(columns=3, rows=2)
        self.assertEqual(montage._grid(20), (3, 2))

    def test_every_slice_gets_a_tile_when_the_grid_is_free(self):
        for slices in ((-10.0, 1.0, 10.0), (-14.0, 3.0, 13.0), (0.0, 5.0, 0.0)):
            montage = self._montage(slices_mm=slices)
            count = len(montage._slice_indices())
            columns, rows = montage._grid(count)
            self.assertGreaterEqual(columns * rows, count)

    def test_the_label_says_where_the_slice_really_is(self):
        montage = self._montage()
        index = montage._slice_indices()[2]
        axis = self.viewer._pane_axis[montage.pane]
        position = list(self.viewer.get_index())
        position[axis] = index
        expected = self.viewer._world_from_index(tuple(position))[2]
        self.assertEqual(montage._slice_label(index), f"z = {expected:.0f}")


class TestSliceSpecification(unittest.TestCase):
    """--slices takes a list or a start:step:stop range, both in mm."""

    def test_a_list(self):
        for text in ("25 30 40 80", "25,30,40,80", " 25, 30 40,80 "):
            self.assertEqual(parse_slices(text), ([25.0, 30.0, 40.0, 80.0], None))

    def test_negative_and_fractional_positions(self):
        self.assertEqual(parse_slices("-30 -7.5 0")[0], [-30.0, -7.5, 0.0])

    def test_a_range(self):
        self.assertEqual(parse_slices("-40:10:60"), (None, (-40.0, 10.0, 60.0)))

    def test_a_range_without_a_step_goes_in_millimetre_steps(self):
        self.assertEqual(parse_slices("-4:4"), (None, (-4.0, 1.0, 4.0)))

    def test_a_single_slice(self):
        self.assertEqual(parse_slices("12"), ([12.0], None))

    def test_nonsense_is_refused_with_a_reason(self):
        for text in ("", "   ", "abc", "1:2:3:4", "10:x:20"):
            with self.assertRaises(ValueError):
                parse_slices(text)


class TestMontageCommandLine(unittest.TestCase):
    """What the montage options add up to before anything is drawn."""

    def _args(self, *argv):
        from t1prep.gui.cat_vol_view import _parse_args
        return _parse_args(["image.nii.gz", *argv])

    def test_slice_values_may_start_with_a_minus(self):
        """mm coordinates are negative half the time; argparse hates that."""
        from t1prep.gui.cat_vol_view import _attach_minus_values
        self.assertEqual(_attach_minus_values(["--slices", "-30:10:30"]),
                         ["--slices=-30:10:30"])
        self.assertEqual(_attach_minus_values(["--slices", "-30 -15 0"]),
                         ["--slices=-30 -15 0"])
        # anything else is passed through untouched
        self.assertEqual(_attach_minus_values(["--slices", "10 20", "--colorbar"]),
                         ["--slices", "10 20", "--colorbar"])
        self.assertEqual(self._args("--slices", "-30:10:30").slices, "-30:10:30")
        self.assertEqual(self._args("--slices", "-30 -15 0").slices, "-30 -15 0")

    def test_an_unquoted_list_is_explained(self):
        with self.assertRaises(SystemExit):
            self._args("--slices", "25", "30", "40")

    def test_the_options_reach_the_montage(self):
        from t1prep.gui.cat_vol_view import _montage_options
        options = _montage_options(self._args(
            "--montage", "--slices", "25 30 40 80", "--orientation", "coronal",
            "--columns", "2", "--rows", "3", "--colorbar", "--no-labels"))
        self.assertEqual(options['slices'], [25.0, 30.0, 40.0, 80.0])
        self.assertIsNone(options['slices_mm'])
        self.assertEqual(options['pane'], CatImageViewer.VIEW_CORONAL)
        self.assertEqual((options['columns'], options['rows']), (2, 3))
        self.assertTrue(options['colorbar'])
        self.assertFalse(options['labels'])

    def test_a_range_becomes_start_step_stop(self):
        from t1prep.gui.cat_vol_view import _montage_options
        options = _montage_options(self._args("--slices", "-40:10:60"))
        self.assertEqual(options['slices_mm'], (-40.0, 10.0, 60.0))
        self.assertIsNone(options['slices'])

    def test_defaults_leave_the_slices_to_the_volume(self):
        from t1prep.gui.cat_vol_view import _montage_options
        options = _montage_options(self._args("--montage"))
        self.assertIsNone(options['slices'])
        self.assertIsNone(options['slices_mm'])
        self.assertEqual(options['pane'], CatImageViewer.VIEW_AXIAL)
        self.assertTrue(options['labels'])

    def test_a_bad_specification_is_refused(self):
        from t1prep.gui.cat_vol_view import _montage_options
        with self.assertRaises(ValueError):
            _montage_options(self._args("--slices", "1:2:3:4"))


class TestOverlayCommandLine(unittest.TestCase):
    """--range, --clip, --threshold and friends, applied to a viewer."""

    class _Viewer:
        overlay_path = "spmT_logP.nii.gz"
        overlay_range = [0.0, 5.0]
        overlay_clip = (0.0, 0.0)
        overlay_colormap = None
        overlay_opacity = 0.8
        overlay_inverse = False
        overlay_discrete = 0

        def __init__(self):
            self.window_level = None
            self.refreshed = 0

        def set_window_level(self, window, level):
            self.window_level = (window, level)

        def refresh_overlay(self):
            self.refreshed += 1

    def _apply(self, *argv):
        from t1prep.gui.cat_vol_view import _apply_overlay_options, _parse_args
        viewer = self._Viewer()
        _apply_overlay_options(viewer, _parse_args(["image.nii.gz", *argv]))
        return viewer

    def test_range_and_clip(self):
        viewer = self._apply("--range", "0", "8", "--clip", "-2", "2")
        self.assertEqual(viewer.overlay_range, [0.0, 8.0])
        self.assertEqual(viewer.overlay_clip, (-2.0, 2.0))
        self.assertEqual(viewer.refreshed, 1)

    def test_a_p_value_threshold_becomes_a_clip_window(self):
        viewer = self._apply("--threshold", "0.05")
        edge = -math.log10(0.05)
        self.assertAlmostEqual(viewer.overlay_clip[0], -edge)
        self.assertAlmostEqual(viewer.overlay_clip[1], edge)
        self.assertAlmostEqual(self._apply("--threshold", "0.001").overlay_clip[1],
                               3.0)

    def test_colours(self):
        from t1prep.gui.colormaps import FIRE
        viewer = self._apply("--colormap", "FIRE", "--opacity", "0.5",
                             "--inverse", "--discrete", "8")
        self.assertEqual(viewer.overlay_colormap, FIRE)
        self.assertEqual(viewer.overlay_opacity, 0.5)
        self.assertTrue(viewer.overlay_inverse)
        self.assertEqual(viewer.overlay_discrete, 8)

    def test_the_image_range_is_a_window_and_a_level(self):
        viewer = self._apply("--range-bkg", "20", "120")
        self.assertEqual(viewer.window_level, (100.0, 70.0))

    def test_nothing_given_changes_nothing(self):
        viewer = self._apply()
        self.assertEqual(viewer.overlay_range, [0.0, 5.0])
        self.assertEqual(viewer.overlay_clip, (0.0, 0.0))
        self.assertIsNone(viewer.window_level)


class TestBatchMontage(unittest.TestCase):
    """The montage as a script: no window, a PNG on disk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        affine = np.array([[-1.0, 0, 0, 16.0],
                           [0, 1.0, 0, -16.0],
                           [0, 0, 1.0, -16.0],
                           [0, 0, 0, 1.0]])
        i, j, k = np.mgrid[0:32, 0:32, 0:32]
        radius = np.sqrt((i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2)
        nib.save(nib.Nifti1Image(np.clip(100 - 3 * radius, 0, None).astype(np.float32),
                                 affine), str(tmp / "image.nii.gz"))
        self.image = str(tmp / "image.nii.gz")
        # 'log' in the name marks a -log10(p) map, as in CAT12
        nib.save(nib.Nifti1Image((6.0 * np.exp(-radius / 4.0)).astype(np.float32),
                                 affine), str(tmp / "spm_logP.nii.gz"))
        self.overlay = str(tmp / "spm_logP.nii.gz")
        self.out = tmp / "montage.png"

    def tearDown(self):
        self._tmp.cleanup()

    def _viewer(self, overlay=None):
        viewer = CatImageViewer(percentile_range=None)
        viewer.load_image(self.image)
        try:
            viewer.setup(window_title="test")
            viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")
        if overlay:
            viewer.set_overlay(overlay)
        return viewer

    def test_a_png_is_written_without_a_window(self):
        written = render_montage(self._viewer(), str(self.out),
                                 size=(400, 300), slices=[-10.0, 0.0, 10.0])
        self.assertTrue(os.path.exists(written))
        with open(written, "rb") as fh:
            self.assertEqual(fh.read(4), b"\x89PNG")

    def test_the_size_is_the_one_asked_for(self):
        render_montage(self._viewer(), str(self.out), size=(640, 480),
                       slices=[0.0])
        from PySide6 import QtGui
        image = QtGui.QImage(str(self.out))
        self.assertEqual((image.width(), image.height()), (640, 480))

    def test_a_colorbar_adds_a_renderer_of_its_own(self):
        viewer = self._viewer(self.overlay)
        window = vtkRenderWindow()
        montage = Montage(viewer, window, slices=[0.0], colorbar=True)
        montage.build()
        self.assertIsNotNone(montage._colorbar_renderer)
        # and the tiles make room for it
        self.assertGreater(montage._renderers[0].GetViewport()[1], 0.0)

    def test_a_logp_overlay_gets_p_value_ticks(self):
        viewer = self._viewer(self.overlay)
        viewer.overlay_clip = (-1.3, 1.3)
        montage = Montage(viewer, vtkRenderWindow(), slices=[0.0], colorbar=True)
        montage.build()
        lut = montage.colorbar_actor.GetLookupTable()
        labels = [lut.GetAnnotation(i) for i in range(lut.GetNumberOfAnnotatedValues())]
        self.assertIn("0.05", labels)     # as in cat_surf_results
        self.assertIn("0.01", labels)

    def test_labels_can_be_left_out(self):
        for labels, expected in ((True, 2), (False, 1)):
            montage = Montage(self._viewer(), vtkRenderWindow(), slices=[0.0],
                              labels=labels)
            montage.build()
            props = montage._renderers[0].GetViewProps()
            self.assertEqual(props.GetNumberOfItems(), expected)

    def test_the_message_says_what_was_drawn(self):
        seen = []
        render_montage(self._viewer(), str(self.out), size=(300, 200),
                       slices=[-10.0, 0.0, 10.0], on_message=seen.append)
        self.assertEqual(seen, ["3 slices"])

    def test_slices_outside_the_volume_are_reported(self):
        seen = []
        render_montage(self._viewer(), str(self.out), size=(300, 200),
                       slices=[0.0, 500.0], on_message=seen.append)
        self.assertIn("outside the volume", seen[0])


class TestDropTargets(unittest.TestCase):
    """Which dropped files the window takes, and what it does with them."""

    def test_volumes_and_surfaces_are_accepted(self):
        from PySide6 import QtCore
        for name in ("T1.nii.gz", "T1.nii", "brain.mnc", "lh.central.gii",
                     "mesh.vtk"):
            url = QtCore.QUrl.fromLocalFile(f"/data/{name}")
            self.assertTrue(VolumeViewerWindow._droppable(url), name)

    def test_anything_else_is_ignored(self):
        from PySide6 import QtCore
        for name in ("notes.txt", "report.pdf", "table.csv"):
            url = QtCore.QUrl.fromLocalFile(f"/data/{name}")
            self.assertFalse(VolumeViewerWindow._droppable(url), name)
        self.assertFalse(VolumeViewerWindow._droppable(
            QtCore.QUrl("https://example.org/T1.nii.gz")))

    def test_a_new_window_inherits_the_display_settings(self):
        class _Viewer:
            show_info = False
            interpolate = False
            recenter = False
            lock_zoom = True
            show_orientation = False

        class _Window:
            _viewer_options = VolumeViewerWindow._viewer_options
            viewer = _Viewer()

        options = _Window()._viewer_options()
        self.assertEqual(options, {'show_info': False, 'interpolate': False,
                                   'recenter': False, 'lock_zoom': True,
                                   'show_orientation': False})


class TestHistogram(unittest.TestCase):
    """The histogram sets the display range by dragging its two handles."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _widget(self, values=None):
        from t1prep.gui.controls import HistogramWidget
        widget = HistogramWidget()
        widget.resize(200, 80)
        rng = np.random.default_rng(0)
        widget.set_values(values if values is not None
                          else rng.normal(50, 10, 5000))
        return widget

    def test_it_asks_for_a_size(self):
        """A form layout hands a field its size hint — a widget without one
        ends up 0 px wide and never paints."""
        widget = self._widget()
        self.assertGreater(widget.sizeHint().width(), 0)
        self.assertGreater(widget.sizeHint().height(), 0)
        self.assertGreater(widget.minimumSizeHint().width(), 0)

    def test_the_panel_shows_it_for_a_volume(self):
        from t1prep.gui.controls import ControlPanel
        panel = ControlPanel()
        self.assertFalse(panel.histogram.isVisibleTo(panel))   # surfaces: off
        panel.configure_for_volume()
        self.assertTrue(panel.histogram.isVisibleTo(panel))
        self.assertTrue(panel.histogram_label.isVisibleTo(panel))

    def test_the_data_becomes_bars(self):
        widget = self._widget()
        self.assertEqual(len(widget._counts), 128)
        self.assertGreater(max(widget._counts), 0)

    def test_empty_data_is_survived(self):
        widget = self._widget(values=np.array([]))
        self.assertEqual(widget._counts, [])

    def test_infinities_are_left_out(self):
        values = np.array([0.0, 1.0, np.nan, np.inf, 2.0])
        widget = self._widget(values=values)
        self.assertEqual((widget._low, widget._high), (0.0, 2.0))

    def test_dragging_a_handle_reports_the_new_window(self):
        widget = self._widget()
        seen = []
        widget.windowChanged.connect(lambda lo, hi: seen.append((lo, hi)))
        widget.set_window(widget._low, widget._high)
        widget._dragging = 'low'
        widget._drag_to(widget.width() / 2.0)
        self.assertEqual(len(seen), 1)
        low, high = seen[-1]
        self.assertAlmostEqual(low, widget._to_value(widget.width() / 2.0), places=6)
        self.assertAlmostEqual(high, widget._high, places=6)

    def test_the_handles_cannot_cross(self):
        widget = self._widget()
        widget.set_window(widget._low, widget._high)
        widget._dragging = 'low'
        widget._drag_to(10 * widget.width())     # far past the other handle
        low, high = widget.window()
        self.assertLessEqual(low, high)

    def test_a_double_click_goes_back_to_the_full_range(self):
        widget = self._widget()
        widget.set_window(10.0, 20.0)
        widget.mouseDoubleClickEvent(None)
        self.assertEqual(widget.window(), (widget._low, widget._high))


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

    def test_volume_limit(self):
        from t1prep.gui.cat_vol_view import WINDOWS_PER_ROW
        self.assertGreaterEqual(MAX_VOLUMES, WINDOWS_PER_ROW)
        # they are tiled in full rows
        self.assertEqual(MAX_VOLUMES % WINDOWS_PER_ROW, 0)

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


class TestAppLaunch(unittest.TestCase):
    """Starting from the macOS app bundle rather than a shell."""

    def test_app_flag(self):
        from t1prep.gui.cat_vol_view import APP_BUNDLE_ENV, running_as_app
        previous = os.environ.pop(APP_BUNDLE_ENV, None)
        try:
            self.assertFalse(running_as_app())
            os.environ[APP_BUNDLE_ENV] = "1"
            self.assertTrue(running_as_app())
        finally:
            os.environ.pop(APP_BUNDLE_ENV, None)
            if previous is not None:
                os.environ[APP_BUNDLE_ENV] = previous

    def test_files_from_finder_are_collected(self):
        """Finder passes documents as events, not on the command line."""
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui.cat_vol_view import files_opened_by_finder

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        QtCore.QTimer.singleShot(0, lambda: app.postEvent(
            app, QtGui.QFileOpenEvent(QtCore.QUrl.fromLocalFile("/tmp/a.nii.gz"))))
        self.assertEqual(files_opened_by_finder(app, timeout_ms=1500),
                         ["/tmp/a.nii.gz"])

    def test_no_files_when_none_arrive(self):
        try:
            from PySide6 import QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui.cat_vol_view import files_opened_by_finder

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.assertEqual(files_opened_by_finder(app, timeout_ms=100), [])

    def test_files_are_still_opened_once_the_viewer_runs(self):
        """A double-click while a window is open must not be dropped.

        The start-up listener used to be removed again, so every later
        open-document event went nowhere: double-clicking a volume did nothing
        at all as long as a viewer was running.
        """
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui.cat_vol_view import finder_open_files

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        router = finder_open_files(app)
        opened = []
        router.set_handler(opened.extend)
        try:
            app.sendEvent(app, QtGui.QFileOpenEvent(
                QtCore.QUrl.fromLocalFile("/tmp/later.nii.gz")))
            self.assertEqual(opened, ["/tmp/later.nii.gz"])
        finally:
            router.set_handler(None)

    def test_files_wait_for_the_window_they_belong_to(self):
        """The event beats the window; what arrived first is handed over then."""
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui.cat_vol_view import finder_open_files

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        router = finder_open_files(app)
        app.sendEvent(app, QtGui.QFileOpenEvent(
            QtCore.QUrl.fromLocalFile("/tmp/early.nii.gz")))
        opened = []
        router.set_handler(opened.extend)
        try:
            self.assertEqual(opened, ["/tmp/early.nii.gz"])
        finally:
            router.set_handler(None)

    def test_double_click_reuses_the_application(self):
        """The file dialog and the window must share one QApplication.

        Creating a second one raises, which made the macOS app die on launch
        with no visible error.
        """
        try:
            from PySide6 import QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui import cat_vol_view

        first = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.assertIs(cat_vol_view.qt_application(), first)

        seen = {}
        original_ask, original_argv = cat_vol_view.ask_for_files, sys.argv
        previous = os.environ.get(cat_vol_view.APP_BUNDLE_ENV)
        os.environ[cat_vol_view.APP_BUNDLE_ENV] = "1"
        try:
            def _ask(app, caption, patterns):
                seen['app'] = app
                return []
            cat_vol_view.ask_for_files = _ask
            sys.argv = ['CAT_VolView']
            self.assertEqual(cat_vol_view.main(), 0)   # cancelled dialog, no crash
        finally:
            cat_vol_view.ask_for_files, sys.argv = original_ask, original_argv
            os.environ.pop(cat_vol_view.APP_BUNDLE_ENV, None)
            if previous is not None:
                os.environ[cat_vol_view.APP_BUNDLE_ENV] = previous
        self.assertIs(seen.get('app'), first)
        self.assertIs(QtWidgets.QApplication.instance(), first)


class TestWindowLayout(unittest.TestCase):
    """Several viewer windows are tiled, up to three per row."""

    class _Stub:
        """Only the geometry calls _place_windows uses."""

        def __init__(self, size=(200, 200)):
            self._w, self._h = size
            self._x = self._y = 0

        def width(self):
            return self._w

        def height(self):
            return self._h

        def resize(self, w, h):
            self._w, self._h = w, h

        def move(self, x, y):
            self._x, self._y = x, y

    def _positions(self, count):
        try:
            from PySide6 import QtWidgets
        except Exception as exc:  # pragma: no cover - needs Qt
            self.skipTest(f"Qt unavailable: {exc}")
        from t1prep.gui.cat_vol_view import _place_windows

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        screen = app.primaryScreen().availableGeometry()
        # small enough that three fit side by side on any test screen
        size = (max(60, screen.width() // 6), max(60, screen.height() // 6))
        windows = [self._Stub(size) for _ in range(count)]
        _place_windows(windows)
        return [(w._x, w._y) for w in windows]

    def test_three_go_in_one_row(self):
        rows = {y for _, y in self._positions(3)}
        columns = {x for x, _ in self._positions(3)}
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(columns), 3)

    def test_six_fill_two_rows_of_three(self):
        positions = self._positions(6)
        rows = sorted({y for _, y in positions})
        self.assertEqual(len(rows), 2)
        top = [x for x, y in positions if y == rows[0]]
        bottom = [x for x, y in positions if y == rows[1]]
        self.assertEqual(len(top), 3)
        self.assertEqual(len(bottom), 3)
        self.assertEqual(sorted(top), sorted(bottom))   # aligned columns

    def test_four_put_the_rest_below(self):
        positions = self._positions(4)
        rows = sorted({y for _, y in positions})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len([x for x, y in positions if y == rows[0]]), 3)
        self.assertEqual(len([x for x, y in positions if y == rows[1]]), 1)


class TestInfoFontSize(unittest.TestCase):
    """The panel text is sized from the render window, which grows on show."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = _write_volume(Path(self._tmp.name) / "vol.nii.gz", np.eye(4))
        self.viewer = CatImageViewer(percentile_range=None)
        self.viewer.load_image(str(path))
        try:
            self.viewer.setup(window_title="test")
            self.viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            self.skipTest(f"rendering unavailable: {exc}")

    def tearDown(self):
        self._tmp.cleanup()

    def test_font_follows_the_window_size(self):
        """A window that is still tiny gave a font that stayed tiny."""
        def font_for(width, height):
            self.viewer.render_window.SetSize(width, height)
            self.viewer._update_info_text()
            return self.viewer._info_actor.GetTextProperty().GetFontSize()

        small = font_for(200, 60)
        large = font_for(1600, 1200)
        self.assertLess(small, large)
        self.assertGreaterEqual(small, 7)
        self.assertLessEqual(large, 16)


class TestLinkedSettings(unittest.TestCase):
    """What the context menu changes applies to every linked window."""

    class _Viewer:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*args):
                self.calls.append((name, args))
            return record

    class _Window:
        # the real broadcast, on stand-in windows
        _for_each_window = VolumeViewerWindow._for_each_window

        def __init__(self):
            self.viewer = TestLinkedSettings._Viewer()
            self.peers = [self]
            self.labels = 0

        def _update_label(self):
            self.labels += 1

    def _group(self, count=3):
        windows = [self._Window() for _ in range(count)]
        for window in windows:
            window.peers = windows
        return windows

    def test_zoom_reaches_every_window(self):
        windows = self._group()
        VolumeViewerWindow.set_zoom(windows[0], 40.0)
        for window in windows:
            self.assertEqual(window.viewer.calls, [("set_field_of_view", (40.0,))])

    def test_atlas_and_interpolation_too(self):
        windows = self._group()
        VolumeViewerWindow.set_atlas(windows[1], "/data/atlas.nii.gz")
        VolumeViewerWindow.set_interpolation(windows[1], False)
        for window in windows:
            names = [name for name, _ in window.viewer.calls]
            self.assertEqual(names, ["set_atlas", "set_interpolation"])
            self.assertEqual(window.labels, 1)   # the reported value follows

    def test_a_lone_window_still_works(self):
        window = self._Window()
        VolumeViewerWindow.set_recenter(window, False)
        self.assertEqual(window.viewer.calls, [("set_recenter", (False,))])

    def test_the_zoom_lock_reaches_every_window(self):
        windows = self._group()
        VolumeViewerWindow.set_lock_zoom(windows[2], False)
        for window in windows:
            self.assertEqual(window.viewer.calls, [("set_lock_zoom", (False,))])

    def test_a_broken_window_does_not_stop_the_others(self):
        windows = self._group()

        class _Broken:
            peers = windows
            _for_each_window = VolumeViewerWindow._for_each_window

            @property
            def viewer(self):
                raise RuntimeError("closed")

        windows.insert(1, _Broken())
        VolumeViewerWindow.set_zoom(windows[0], 20.0)
        self.assertEqual(windows[-1].viewer.calls, [("set_field_of_view", (20.0,))])


class TestTitlePath(unittest.TestCase):
    """The title bar names the directory, since file names repeat."""

    def test_short_paths_are_kept(self):
        from t1prep.gui.cat_vol_view import shorten_path
        self.assertEqual(shorten_path("/data/sub-01/mri"), "/data/sub-01/mri")
        self.assertEqual(shorten_path("relative/sub-02/mri"), "relative/sub-02/mri")

    def test_long_paths_keep_their_end(self):
        """The end identifies the volume, and macOS would cut it off itself."""
        from t1prep.gui.cat_vol_view import shorten_path
        short = shorten_path("/var/folders/4x/40sqmh7x5gn1vspcrntxy9dh0000gn/T/"
                             "tmpu6fawg0h/sub-01/mri")
        self.assertEqual(short, "…/tmpu6fawg0h/sub-01/mri")

    def test_stays_short_enough_for_a_title_bar(self):
        """Whatever is shown is the end of the real path, and it fits."""
        from t1prep.gui.cat_vol_view import shorten_path
        for path in ("/very/deep/study/derivatives/T1Prep/sub-01/anat/extra",
                     "/" + "a" * 80 + "/" + "b" * 80 + "/cc",
                     "/data/" + "x" * 200):
            short = shorten_path(path)
            self.assertLessEqual(len(short), 40, path)
            self.assertTrue(path.endswith(short.lstrip("…")), path)

    def test_a_single_component_is_left_alone(self):
        from t1prep.gui.cat_vol_view import shorten_path
        self.assertEqual(shorten_path("/data"), "/data")


class TestSharedViewerHelpers(unittest.TestCase):
    """The pieces both viewers use, in one place (viewer_common)."""

    def test_drop_targets_are_decided_by_suffix(self):
        from PySide6 import QtCore
        from t1prep.gui.viewer_common import (VOLUME_SUFFIXES, SURFACE_SUFFIXES,
                                              droppable_url)
        for name in ("T1.nii.gz", "brain.mnc", "lh.central.gii", "lh.aparc.annot"):
            url = QtCore.QUrl.fromLocalFile(f"/data/{name}")
            self.assertTrue(droppable_url(url, VOLUME_SUFFIXES + SURFACE_SUFFIXES))
        self.assertFalse(droppable_url(QtCore.QUrl.fromLocalFile("/data/notes.txt"),
                                       VOLUME_SUFFIXES))
        self.assertFalse(droppable_url(QtCore.QUrl("https://example.org/T1.nii.gz"),
                                       VOLUME_SUFFIXES))

    def test_both_viewers_claim_the_same_zoom_events(self):
        from t1prep.gui.viewer_common import ZOOM_EVENTS
        for event in ("RightButtonPressEvent", "MouseWheelForwardEvent",
                      "PinchEvent"):
            self.assertIn(event, ZOOM_EVENTS)

    def test_notes_are_quiet_unless_asked_for(self):
        import io
        from contextlib import redirect_stderr
        from t1prep.gui import viewer_common
        was = viewer_common._verbose
        try:
            viewer_common.set_verbose(False)
            quiet = io.StringIO()
            with redirect_stderr(quiet):
                viewer_common.note("a fallback happened")
            self.assertEqual(quiet.getvalue(), "")

            viewer_common.set_verbose(True)
            loud = io.StringIO()
            with redirect_stderr(loud):
                viewer_common.note("a fallback happened")
            self.assertIn("a fallback happened", loud.getvalue())
        finally:
            viewer_common.set_verbose(was)

    def test_the_title_shortener_lives_in_the_shared_module(self):
        from t1prep.gui.viewer_common import shorten_path
        from t1prep.gui.cat_vol_view import shorten_path as imported
        self.assertIs(imported, shorten_path)


class TestFinderOpenEvents(unittest.TestCase):
    """macOS sends an open-document event for the command-line files as well.

    Opening them again gave every volume a second window: two files on the
    command line, four windows on screen.
    """

    class _Window:
        """Stands in for a viewer window, recording what it was asked to open."""

        def __init__(self, image_path, visible=True):
            self.image_path = image_path
            self.peers = [self]
            self.surface_paths = []
            self._visible = visible
            self.opened = []
            self.raised = 0

        def isVisible(self):
            return self._visible

        def open_volume(self, path):
            self.opened.append(path)

        def add_surface(self, path):
            self.surface_paths.append(path)

        def raise_(self):
            self.raised += 1

        def activateWindow(self):
            pass

    def setUp(self):
        from t1prep.gui.cat_vol_view import _open_from_finder
        self._open = _open_from_finder
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        affine = np.eye(4)
        self.first = str(_write_volume(tmp / "a.nii.gz", affine))
        self.second = str(_write_volume(tmp / "b.nii.gz", affine))
        self.third = str(_write_volume(tmp / "c.nii.gz", affine))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_file_already_open_is_raised_not_opened_again(self):
        windows = [self._Window(self.first), self._Window(self.second)]
        self._open(windows, [self.first, self.second])
        self.assertEqual([w.opened for w in windows], [[], []])
        self.assertEqual(windows[0].raised, 1)
        self.assertEqual(windows[1].raised, 1)

    def test_a_new_file_still_opens(self):
        windows = [self._Window(self.first)]
        self._open(windows, [self.third])
        self.assertEqual(windows[0].opened, [self.third])

    def test_the_same_file_by_another_path(self):
        """A relative path or a symlink names the same volume."""
        windows = [self._Window(self.first)]
        awkward = str(Path(self.first).parent / "." / Path(self.first).name)
        self._open(windows, [awkward])
        self.assertEqual(windows[0].opened, [])
        self.assertEqual(windows[0].raised, 1)

    def test_windows_the_user_closed_are_skipped(self):
        closed = self._Window(self.first, visible=False)
        alive = self._Window(self.second)
        self._open([closed, alive], [self.third])
        self.assertEqual(alive.opened, [self.third])
        self.assertEqual(closed.opened, [])

    def test_a_surface_is_not_outlined_twice(self):
        window = self._Window(self.first)
        window.surface_paths = ["/data/lh.central.gii"]
        self._open([window], ["/data/lh.central.gii"])
        self.assertEqual(window.surface_paths, ["/data/lh.central.gii"])

    def test_nothing_open_at_all(self):
        closed = self._Window(self.first, visible=False)
        self._open([closed], [self.third])       # must not raise
        self.assertEqual(closed.opened, [])
