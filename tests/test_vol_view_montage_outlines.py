"""Contours and surface outlines on the tiles of a montage.

The panes of CAT_VolView outline other volumes (``--contour``) and cut surfaces
with their slice planes.  A montage is a sheet of slices made from such a
viewer, and it used to draw the image and the overlay only — so the QC figure
of a segmentation or of the central surfaces came out as bare slices, without a
word about what was left out.

Everything is drawn into an off-screen render window on synthetic data: a
volume with a bright blob, a ball-shaped mask and two spheres, all around the
same point, so it is known which slices cut which of them.
"""

import contextlib
import io
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
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkFloatArray
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    from vtkmodules.vtkIOImage import vtkPNGReader
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    from vtkmodules.vtkRenderingCore import vtkRenderWindow
    from t1prep.gui.cat_vol_view import (
        CatImageViewer,
        Montage,
        MontageWindow,
        VolumeViewerWindow,
        main,
        render_montage,
    )
    from t1prep.gui.colormaps import JET, build_overlay_lut
except Exception as exc:  # pragma: no cover - depends on optional deps
    raise unittest.SkipTest(f"cat_vol_view unavailable: {exc}")


SHAPE = (40, 44, 48)
#: 1 mm voxels; voxel (20, 22, 24) is the origin of the millimetre space
AFFINE = np.array([[-1.0, 0, 0, 20.0],
                   [0, 1.0, 0, -22.0],
                   [0, 0, 1.0, -24.0],
                   [0, 0, 0, 1.0]])
#: Radius in mm of the mask, of the plain sphere and of the one with values
MASK_RADIUS = 10.0
SPHERE_RADIUS = 12.0
INNER_RADIUS = 7.0
#: Slices of the sheet, in voxels from the common centre: the outer two miss
#: everything, the next two cut the plain sphere only, the inner three cut all
OFFSETS = (-16, -11, -6, 0, 6, 11, 16)

YELLOW = CatImageViewer.CONTOUR_COLORS[0]
RED, GREEN = VolumeViewerWindow.SURFACE_COLORS[:2]


def _without_header(path: Path) -> Path:
    """A copy of *path* that says nothing about where its voxels are."""
    image = nib.load(str(path))
    image.header.set_sform(None, code=0)
    image.header.set_qform(None, code=0)
    bare = path.with_name("bare_" + path.name)
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), None, image.header),
             str(bare))
    return bare


def _write_volumes(tmp: Path, header: bool):
    """Write the image and the mask; returns their file names."""
    i, j, k = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]]
    radius = np.sqrt((i - 20) ** 2 + (j - 22) ** 2 + (k - 24) ** 2)
    image = tmp / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.clip(100 - 3 * radius, 0, None).astype(np.float32),
                             AFFINE), str(image))
    mask = tmp / "mask.nii.gz"
    if header:
        # On a 2 mm grid of its own, so the outline is one that was resampled
        i, j, k = np.mgrid[0:20, 0:22, 0:24]
        coarse = 2.0 * np.sqrt((i - 10) ** 2 + (j - 11) ** 2 + (k - 12) ** 2)
        nib.save(nib.Nifti1Image((coarse < MASK_RADIUS).astype(np.float32),
                                 AFFINE @ np.diag([2.0, 2.0, 2.0, 1.0])), str(mask))
        return str(image), str(mask)
    # Without a transform nothing can be resampled: same grid, voxel for voxel
    nib.save(nib.Nifti1Image((radius < MASK_RADIUS).astype(np.float32), AFFINE),
             str(mask))
    return str(_without_header(image)), str(_without_header(mask))


def _sphere(centre, radius: float = SPHERE_RADIUS, values: bool = False):
    """A sphere around *centre*, with per-vertex values when asked for."""
    source = vtkSphereSource()
    source.SetCenter(*centre)
    source.SetRadius(radius)
    source.SetThetaResolution(24)
    source.SetPhiResolution(24)
    source.Update()
    poly = source.GetOutput()
    if values:
        array = vtkFloatArray()
        array.SetName("overlay")
        array.SetNumberOfTuples(poly.GetNumberOfPoints())
        for point in range(poly.GetNumberOfPoints()):
            array.SetValue(point, float(point % 10))
        poly.GetPointData().SetScalars(array)
    return poly


def _centre(viewer) -> tuple:
    """Voxel of the brightest point, where mask and spheres are centred too.

    Looked up rather than assumed: VTK may store the volume with flipped axes
    relative to the file.
    """
    values = vtk_to_numpy(viewer._image.GetPointData().GetScalars())
    return tuple(int(v) for v in np.unravel_index(
        int(np.argmax(values)), viewer._image.GetDimensions(), order="F"))


def _positions(viewer, pane, offsets=OFFSETS):
    """Millimetre positions of the slices *offsets* voxels off the centre."""
    axis = viewer._pane_axis[pane]
    along = int(np.argmax(np.abs(viewer._voxel_axis_directions()[axis])))
    positions = []
    for offset in offsets:
        index = list(_centre(viewer))
        index[axis] += offset
        positions.append(viewer._world_from_index(tuple(index))[along])
    return positions


def _montage(viewer, pane, offsets=OFFSETS, **kwargs) -> Montage:
    """A built montage on a window of its own — no Qt window involved."""
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(420, 300)
    montage = Montage(viewer, window, pane=pane,
                      slices=_positions(viewer, pane, offsets), **kwargs)
    montage.build()
    return montage


def _add_surface(viewer, color, **sphere):
    """Outline a sphere, the way the window adds a surface dropped on it."""
    lut = build_overlay_lut(JET, 1.0) if sphere.get('values') else None
    viewer.add_surface(
        _sphere(viewer._world_from_index(_centre(viewer)), **sphere), color,
        lut=lut, scalar_range=(0.0, 9.0) if lut is not None else None)
    viewer._build_surface_contours()
    viewer._set_slices_from_index()


def _outlines(renderer):
    """The line actors of a tile; image and overlay are not ``vtkActor``."""
    actors = renderer.GetActors()
    actors.InitTraversal()
    return [actors.GetNextActor() for _ in range(actors.GetNumberOfItems())]


def _lines(actor) -> int:
    """How many line cells *actor* draws."""
    mapper = actor.GetMapper()
    mapper.Update()
    return mapper.GetInput().GetNumberOfLines()


def _pixels(png, color, tolerance: int = 40) -> int:
    """How many pixels of the PNG have (nearly) the colour of an outline."""
    reader = vtkPNGReader()
    reader.SetFileName(str(png))
    reader.Update()
    scalars = reader.GetOutput().GetPointData().GetScalars()
    rgb = vtk_to_numpy(scalars)[:, :3].astype(int)
    wanted = np.round(255 * np.asarray(color, dtype=float)).astype(int)
    return int(np.all(np.abs(rgb - wanted) <= tolerance, axis=1).sum())


class _Scene(unittest.TestCase):
    """The files of the scene, and rendered viewers on its image."""

    #: Whether the volumes carry an sform; without one the viewer falls back to
    #: the geometry VTK gives the image and places its actors differently
    HEADER = True

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.image, cls.mask = _write_volumes(cls.tmp, cls.HEADER)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _viewer(cls) -> CatImageViewer:
        viewer = CatImageViewer(percentile_range=None)
        viewer.load_image(cls.image)
        try:
            viewer.setup(window_title="test")
            viewer.render(headless=True)
        except Exception as exc:  # pragma: no cover - no GL in this environment
            raise unittest.SkipTest(f"rendering unavailable: {exc}")
        if viewer._world_from_header != cls.HEADER:
            raise AssertionError("the image is not placed the way this test needs")
        return viewer


class TestTilesCarryTheOutlines(_Scene):
    """Every tile draws what the pane draws on that slice, and nothing else."""

    #: Radius of what each line actor of a tile outlines, in the order they are
    #: added: the contour, the plain sphere, the inner sphere and its values.
    #: The resampled mask is crossed about a millimetre inside its radius.
    RADII = (MASK_RADIUS - 1.0, SPHERE_RADIUS, INNER_RADIUS, INNER_RADIUS)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.viewer = cls._viewer()
        cls.contour = cls.viewer.add_contour(cls.mask)
        _add_surface(cls.viewer, RED)
        _add_surface(cls.viewer, GREEN, radius=INNER_RADIUS, values=True)
        cls.montages = {pane: _montage(cls.viewer, pane)
                        for _, pane in Montage.AXES}

    def _in_pane(self, pane):
        """The line actors of a pane of the viewer, in the order of a tile."""
        actors = [self.contour['actors'][pane]]
        for entry in self.viewer._surface_contours[pane]:
            actors.extend(entry['actors'])
        return actors

    def _tiles(self):
        """(pane, voxel offset of the slice, line actors) for every tile."""
        for _, pane in Montage.AXES:
            renderers = self.montages[pane]._renderers
            self.assertEqual(len(renderers), len(OFFSETS))
            for offset, renderer in zip(OFFSETS, renderers):
                actors = _outlines(renderer)
                self.assertEqual(len(actors), len(self.RADII), (pane, offset))
                yield pane, offset, actors

    def test_lines_are_drawn_where_the_slice_cuts_and_only_there(self):
        """A tile beside the mask or the surface must not borrow their outline."""
        for pane, offset, actors in self._tiles():
            for actor, radius in zip(actors, self.RADII):
                if abs(offset) < radius:
                    self.assertGreater(_lines(actor), 20, (pane, offset, radius))
                else:
                    self.assertEqual(_lines(actor), 0, (pane, offset, radius))

    def test_colours_and_widths_are_those_of_the_panes(self):
        for pane, offset, actors in self._tiles():
            for tile, shown in zip(actors, self._in_pane(pane)):
                self.assertIsNot(tile, shown)       # an actor of its own
                self.assertEqual(tile.GetProperty().GetColor(),
                                 shown.GetProperty().GetColor())
                self.assertEqual(tile.GetProperty().GetLineWidth(),
                                 shown.GetProperty().GetLineWidth())
            self.assertEqual(
                [actor.GetProperty().GetColor() for actor in actors[:3]],
                [YELLOW, RED, GREEN])

    def test_the_values_of_a_surface_colour_its_outline(self):
        """Through the table of the surface, over the line in its own colour."""
        lut = self.viewer.surfaces[1]['lut']
        self.assertIsNotNone(lut)
        for pane, offset, actors in self._tiles():
            _, plain, under, coloured = actors
            for flat in (plain, under):
                self.assertFalse(flat.GetMapper().GetScalarVisibility())
            self.assertTrue(coloured.GetMapper().GetScalarVisibility())
            self.assertIs(coloured.GetMapper().GetLookupTable(), lut)
            self.assertEqual(coloured.GetMapper().GetScalarRange(), (0.0, 9.0))
            self.assertEqual(_lines(coloured), _lines(under))

    def test_a_tile_shows_what_the_pane_shows_on_that_slice(self):
        """Same lines in the same place: the pane is moved to every tile."""
        viewer = self.viewer
        for _, pane in Montage.AXES:
            montage = self.montages[pane]
            axis = viewer._pane_axis[pane]
            for index, renderer in zip(montage._slice_indices(),
                                       montage._renderers):
                position = list(viewer.get_index())
                position[axis] = index
                viewer.set_index(*position)
                tiles, shown_in_pane = _outlines(renderer), self._in_pane(pane)
                self.assertEqual(len(tiles), len(shown_in_pane), (pane, index))
                for tile, shown in zip(tiles, shown_in_pane):
                    self.assertEqual(_lines(tile), _lines(shown), (pane, index))
                    if not _lines(tile):
                        continue
                    for mine, theirs in zip(tile.GetBounds(), shown.GetBounds()):
                        self.assertAlmostEqual(mine, theirs, places=5,
                                               msg=(pane, index))

    def test_the_lines_lie_between_the_slice_and_the_camera(self):
        """In the slice plane itself the image wins the depth test."""
        for _, pane in Montage.AXES:
            renderer = self.montages[pane]._renderers[OFFSETS.index(0)]
            camera = renderer.GetActiveCamera()
            towards = (np.array(camera.GetPosition())
                       - np.array(camera.GetFocalPoint()))
            towards /= np.linalg.norm(towards)

            def depth(prop):
                bounds = np.array(prop.GetBounds())
                return float(towards @ (bounds[0::2] + bounds[1::2])) / 2.0

            props = renderer.GetViewProps()
            props.InitTraversal()
            image = depth(props.GetNextProp())
            lifts = [depth(actor) - image for actor in _outlines(renderer)]
            # Above the overlay (0.25 mm), and the values above their surface
            for lift, expected in zip(lifts, (0.35, 0.5, 0.5, 0.6)):
                self.assertAlmostEqual(lift, expected, places=4, msg=pane)
            near, far = camera.GetClippingRange()
            self.assertLess(near, camera.GetDistance() - max(lifts))
            self.assertGreater(far, camera.GetDistance())

    def test_the_written_figure_shows_them(self):
        """What the defect looked like: a PNG without a single line in it."""
        bare = self._viewer()
        outlined = self._viewer()
        outlined.add_contour(self.mask)
        _add_surface(outlined, RED)
        for label, pane in Montage.AXES:
            options = dict(size=(240, 200), pane=pane, labels=False,
                           slices=_positions(bare, pane, (0,)))
            before = render_montage(bare, str(self.tmp / f"bare_{label}.png"),
                                    **options)
            after = render_montage(outlined, str(self.tmp / f"{label}.png"),
                                   **options)
            for color in (YELLOW, RED):
                self.assertEqual(_pixels(before, color), 0, label)
                self.assertGreater(_pixels(after, color), 40, label)


class TestTilesWithoutHeaderTransform(TestTilesCarryTheOutlines):
    """The same sheet for a volume the viewer has to place by VTK's geometry.

    Without an sform the image actors carry no transform and the outlines are
    lifted through their position instead of a matrix; the tiles have to follow
    the panes through that branch as well.
    """

    HEADER = False
    #: The mask is on the grid of the image here, so nothing is resampled
    RADII = (MASK_RADIUS, SPHERE_RADIUS, INNER_RADIUS, INNER_RADIUS)


class TestSheetFollowsTheViewer(_Scene):
    """What the montage takes from the viewer besides the lines themselves."""

    def setUp(self):
        self.viewer = self._viewer()
        self.pane = CatImageViewer.VIEW_CORONAL

    def _tile(self, pane=None, offset=0):
        """The line actors of a one-slice montage."""
        montage = _montage(self.viewer, self.pane if pane is None else pane,
                           (offset,))
        return _outlines(montage._renderers[0])

    def test_an_overlay_does_not_hide_them(self):
        """The overlay is lifted off the image too; the lines go above it."""
        magenta, white = (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)   # not in the overlay
        self.viewer.set_overlay(self.image)
        self.viewer.overlay_opacity = 1.0
        self.viewer.refresh_overlay()
        options = dict(size=(240, 200), pane=self.pane, labels=False,
                       slices=_positions(self.viewer, self.pane, (0,)))
        before = render_montage(self.viewer, str(self.tmp / "overlay.png"),
                                **options)
        self.viewer.add_contour(self.mask, color=magenta)
        _add_surface(self.viewer, white)
        after = render_montage(self.viewer, str(self.tmp / "outlined.png"),
                               **options)
        for color in (magenta, white):
            self.assertEqual(_pixels(before, color), 0)
            self.assertGreater(_pixels(after, color), 40)

    def test_what_is_hidden_in_the_pane_is_hidden_on_the_sheet(self):
        entry = self.viewer.add_contour(self.mask)
        _add_surface(self.viewer, RED)
        entry['actors'][self.pane].SetVisibility(0)
        contour, surface = self._tile()
        self.assertFalse(contour.GetVisibility())
        self.assertTrue(surface.GetVisibility())
        # Only that pane was touched
        contour, surface = self._tile(CatImageViewer.VIEW_AXIAL)
        self.assertTrue(contour.GetVisibility())
        self.assertTrue(surface.GetVisibility())

        for actor in self.viewer._surface_contours[self.pane][0]['actors']:
            actor.SetVisibility(0)
        contour, surface = self._tile()
        self.assertFalse(surface.GetVisibility())

    def test_the_image_frames_the_tile_not_the_surface(self):
        """A surface reaching beyond the image must not shrink its slice."""
        def cameras(montage):
            return [(renderer.GetActiveCamera().GetParallelScale(),
                     renderer.GetActiveCamera().GetFocalPoint())
                    for renderer in montage._renderers]

        before = cameras(_montage(self.viewer, self.pane))
        _add_surface(self.viewer, RED, radius=30.0)
        montage = _montage(self.viewer, self.pane)
        (surface,) = _outlines(montage._renderers[OFFSETS.index(0)])
        self.assertGreater(_lines(surface), 0)
        self.assertEqual(cameras(montage), before)

    def test_a_rebuild_takes_the_outlines_as_they_are_now(self):
        """The sheet is redrawn from the viewer, not added to."""
        self.viewer.add_contour(self.mask)
        montage = _montage(self.viewer, self.pane, (0, 16))
        self.assertEqual([len(_outlines(r)) for r in montage._renderers], [1, 1])

        self.viewer.set_contour_level(self.mask, 5.0)       # above a 0..1 mask
        _add_surface(self.viewer, RED)
        montage.build()
        (contour, surface), missed = (_outlines(r) for r in montage._renderers)
        self.assertEqual(_lines(contour), 0)
        self.assertGreater(_lines(surface), 0)
        self.assertEqual([_lines(actor) for actor in missed], [0, 0])

        self.viewer.clear_contours()
        montage.build()
        self.assertEqual([len(_outlines(r)) for r in montage._renderers], [1, 1])

    def test_the_window_redraws_them_with_its_sheet(self):
        """MontageWindow draws through this class, whatever its toolbar changes.

        The window itself needs a display; its handler is all that is called.
        """
        class _Spin:
            def __init__(self, value):
                self._value = value

            def value(self):
                return self._value

        class _Window:
            columns_spin = _Spin(1)
            rows_spin = _Spin(0)

        self.viewer.add_contour(self.mask)
        _add_surface(self.viewer, RED)
        window = _Window()
        window.montage = _montage(self.viewer, self.pane, (-6, 0, 6))
        MontageWindow._layout_changed(window)
        self.assertEqual(window.montage._grid(), (1, 3))
        for renderer in window.montage._renderers:
            self.assertEqual(renderer.GetViewport()[0::2], (0.0, 1.0))
            self.assertEqual([_lines(actor) > 0 for actor in _outlines(renderer)],
                             [True, True])


class TestBatchFigure(_Scene):
    """``CAT_VolView --headless --montage --screenshot``, as a script calls it."""

    def _figure(self, name, surfaces=(), options=()):
        path = self.tmp / name
        # The colours of the surfaces are reported on stdout
        with contextlib.redirect_stdout(io.StringIO()):
            status = main([self.image, *surfaces, "--headless", "--montage",
                           "--slices", "0", "--no-labels",
                           "--montage-size", "240", "200",
                           "--screenshot", str(path), *options])
        self.assertEqual(status, 0)
        return path

    def test_contours_and_surfaces_reach_the_written_montage(self):
        surface = str(self.tmp / "sphere.vtp")
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(surface)
        writer.SetInputData(_sphere((0.0, 0.0, 0.0)))
        writer.Write()

        bare = self._figure("bare.png")
        outlined = self._figure("outlined.png", surfaces=[surface],
                                options=["--contour", self.mask])
        for color in (YELLOW, RED):
            self.assertEqual(_pixels(bare, color), 0)
            self.assertGreater(_pixels(outlined, color), 40)


if __name__ == "__main__":
    unittest.main()
