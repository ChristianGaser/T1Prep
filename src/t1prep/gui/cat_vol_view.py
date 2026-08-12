"""
cat_vol_view.py

Single-window VTK image viewer with 3 orthogonal slices (axial, coronal,
sagittal) arranged in an SPM12-like layout, with optional surface overlays.

Layout (SPM12-like)::

    +----------+----------+
    | Coronal  | Sagittal |
    |  (top-L) |  (top-R) |
    +----------+----------+
    | Axial    |          |
    |  (bot-L) |  (empty) |
    +----------+----------+

Display intensities are scaled to the 3rd--97th percentile range by default.

Usage (CLI):
    CAT_VolView <image> [surf1] [surf2] [surf3] \
        --size 400 [--percentile 3 97]

    # source checkout
    python src/t1prep/gui/cat_vol_view.py <image> [surf1] [surf2] [surf3]

Notes:
- Tries to use vtkNIFTIImageReader for NIfTI, vtkMINCImageReader for MINC,
  and vtkImageReader2Factory otherwise.
- Surfaces: supports .gii, .vtp, .vtk, .obj, .stl via appropriate VTK readers.
- The viewer works in the millimetre space of the NIfTI sform/qform, so images
  and surfaces line up without adjustment and each pane shows the anatomical
  plane it is named after, whatever voxel order the image is stored in.
  ``--mirror`` is available for surfaces that need the x flip anyway.
- ``CatImageViewer`` can render into a render window supplied by a host
  application; CAT_SurfView embeds it as its volume window and links the cursor
  to the surface through :meth:`CatImageViewer.set_world_position` and the
  ``on_position_changed`` callback.
"""

from __future__ import annotations

import itertools
import os
import sys
import argparse
from typing import List, Optional, Sequence, Tuple

# Import minimal VTK modules explicitly (avoids large monolithic import)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPlane
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonMath import vtkMatrix3x3, vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkIOImage import (
    vtkNIFTIImageReader,
    vtkImageReader2Factory,
)
try:
    from vtkmodules.vtkIOImage import vtkMINCImageReader  # type: ignore
except Exception:  # pragma: no cover - optional
    vtkMINCImageReader = None  # type: ignore
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkOBJReader
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
try:
    # Available in VTK 9+ when GIFTI IO is enabled
    from vtkmodules.vtkIOGeometry import vtkGIFTIReader  # type: ignore
except Exception:  # pragma: no cover - optional
    vtkGIFTIReader = None  # type: ignore
try:
    import nibabel as nib  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    nib = None  # type: ignore
    np = None  # type: ignore
try:
    from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore
except Exception:  # pragma: no cover - optional
    vtk_to_numpy = None  # type: ignore

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkImageActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)
try:
    from vtkmodules.vtkImagingColor import (
        vtkImageMapToWindowLevelColors,
    )
except Exception:  # pragma: no cover
    vtkImageMapToWindowLevelColors = None  # type: ignore
from vtkmodules.vtkIOImage import vtkPNGWriter
# Ensure rendering backend and interaction styles are registered (VTK 9 modular)
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkFiltersSources import vtkLineSource

# Qt window + interactor.  QVTKRWIBase has to be chosen before the widget is
# imported; CAT_SurfView imports this module and relies on the same setting.
from PySide6 import QtCore, QtWidgets
import vtkmodules.qt as _vtk_qt
_vtk_qt.QVTKRWIBase = "QOpenGLWidget"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def _guess_image_reader(image_path: str):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".nii", ".gz") or image_path.lower().endswith(".nii.gz"):
        reader = vtkNIFTIImageReader()
        reader.SetFileName(image_path)
        return reader
    if ext == ".mnc":
        if vtkMINCImageReader is not None:
            reader = vtkMINCImageReader()
            reader.SetFileName(image_path)
            return reader
        # Fallback to factory if MINC reader is unavailable
        factory = vtkImageReader2Factory()
        reader = factory.CreateImageReader2(image_path)
        if reader is None:
            raise RuntimeError(
                "VTK build lacks MINC reader support and factory could not "
                "resolve a reader for this .mnc file."
            )
        reader.SetFileName(image_path)
        return reader
    # Fallback to factory (MHA/MHD/NRRD and others)
    factory = vtkImageReader2Factory()
    reader = factory.CreateImageReader2(image_path)
    if reader is None:
        raise RuntimeError(f"Unsupported image type: {image_path}")
    reader.SetFileName(image_path)
    return reader


def _load_surface(surface_path: str) -> vtkPolyData:
    """Load a surface file into vtkPolyData using appropriate reader by extension."""
    ext = os.path.splitext(surface_path)[1].lower()

    if ext == ".gii":
        # Prefer VTK reader if available
        if vtkGIFTIReader is not None:
            try:
                r = vtkGIFTIReader()
                r.SetFileName(surface_path)
                r.Update()
                out = r.GetOutput()
                if out is not None:
                    return out
            except Exception:
                pass
        # Fallback: use nibabel to parse GIFTI and convert to vtkPolyData
        if nib is None or np is None:
            raise RuntimeError(
                "GIFTI reader not available in this VTK build and nibabel is missing."
            )
        img = nib.load(surface_path)
        coords = None
        faces = None
        # Heuristically locate pointset and triangle arrays
        for da in getattr(img, 'darrays', []):
            arr = getattr(da, 'data', None)
            if arr is None:
                continue
            if getattr(arr, 'ndim', 0) == 2 and arr.shape[1] == 3:
                if np.issubdtype(arr.dtype, np.floating) and coords is None:
                    coords = np.asarray(arr, dtype=np.float32)
                elif np.issubdtype(arr.dtype, np.integer) and faces is None:
                    faces = np.asarray(arr, dtype=np.int64)
        if coords is None or faces is None:
            raise RuntimeError("Failed to parse coordinates/faces from GIFTI file: " + surface_path)

        points = vtkPoints()
        # SetData expects a VTK array; set points one-by-one for compatibility
        points.SetNumberOfPoints(int(coords.shape[0]))
        for idx in range(coords.shape[0]):
            x, y, z = float(coords[idx, 0]), float(coords[idx, 1]), float(coords[idx, 2])
            points.SetPoint(idx, x, y, z)

        polys = vtkCellArray()
        for tri in faces:
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        return poly

    if ext == ".vtp":
        r = vtkXMLPolyDataReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".vtk":
        r = vtkPolyDataReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".stl":
        r = vtkSTLReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    if ext == ".obj":
        r = vtkOBJReader()
        r.SetFileName(surface_path)
        r.Update()
        return r.GetOutput()

    raise RuntimeError(f"Unsupported surface type or reader unavailable: {surface_path}")


def _mirror_polydata_x(poly: vtkPolyData) -> vtkPolyData:
    t = vtkTransform()
    t.Scale(-1.0, 1.0, 1.0)
    f = vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(t)
    f.Update()
    return f.GetOutput()


class CatImageViewer:
    """Single-window orthogonal viewer with SPM12-like layout.

    The three views are rendered in a single window arranged as::

        +----------+----------+
        | Coronal  | Sagittal |
        |  (top-L) |  (top-R) |
        +----------+----------+
        | Axial    |          |
        |  (bot-L) |  (empty) |
        +----------+----------+

    Viewer index mapping:
        0 → XY  (Axial,     slicing along Z)
        1 → YZ  (Sagittal,  slicing along X)
        2 → XZ  (Coronal,   slicing along Y)
    """

    VIEW_AXIAL = 0
    VIEW_SAGITTAL = 1
    VIEW_CORONAL = 2

    def __init__(
        self,
        window_size: int = 400,
        mirror_surfaces: bool = False,
        verbose: bool = False,
        surface_convention: str = "auto",
        percentile_range: Optional[Tuple[float, float]] = (3.0, 97.0),
        render_window: Optional[vtkRenderWindow] = None,
        interactor: Optional[vtkRenderWindowInteractor] = None,
    ):
        """Create the viewer.

        Args:
            render_window: Render window to draw into.  Pass the one of a Qt
                ``QVTKRenderWindowInteractor`` to embed the viewer in another
                application (CAT_SurfView does this); a standalone window is
                created when omitted.
            interactor: Interactor belonging to *render_window*; required
                together with it.
        """
        self.window_size = int(window_size)
        self.verbose = bool(verbose)
        self.surface_convention = surface_convention.lower()
        self.percentile_range = percentile_range
        # Avoid double flipping: if a convention is explicitly provided,
        # do not mirror by default.
        if self.surface_convention in ("ras", "lps") and mirror_surfaces:
            self.mirror_surfaces = False
            if self.verbose:
                print("[cat_vol_view] Disabling mirroring due to "
                      "explicit --surface-convention")
        else:
            self.mirror_surfaces = bool(mirror_surfaces)
        self._image = None
        self._vox2world = None  # 4×4 matrix (list of lists): index -> world
        # Same transform in the form the image actors need (data coordinates,
        # i.e. index times spacing, to world)
        self._actor_matrix = None
        # True when the transform comes from the NIfTI sform/qform, i.e. the
        # world space is anatomical millimetres
        self._world_from_header = False
        # Voxel axis each pane slices along; recomputed per image
        self._pane_axis: List[int] = [2, 0, 1]
        # Edge length in mm the panes are zoomed to (None = whole volume)
        self._fov_mm: Optional[float] = None

        # Three renderers + image actors (one per orthogonal view)
        self.renderers: List[vtkRenderer] = [
            vtkRenderer(), vtkRenderer(), vtkRenderer(),
        ]
        self._image_actors: List[Optional[vtkImageActor]] = [
            None, None, None,
        ]
        self._wl_filters: List = [None, None, None]
        # Single render window and interactor.  When both are supplied the
        # viewer is embedded in a host application, which owns the window
        # geometry and the event loop.
        self.embedded = render_window is not None and interactor is not None
        self.render_window: vtkRenderWindow = render_window or vtkRenderWindow()
        self.interactor: vtkRenderWindowInteractor = (
            interactor or vtkRenderWindowInteractor()
        )
        # Viewport bounds computed in setup – default equal quadrants
        self._viewports: List[Tuple[float, float, float, float]] = [
            (0.0, 0.0, 0.5, 0.5),   # Axial    (bottom-left)
            (0.0, 0.5, 0.5, 1.0),   # Sagittal (top-left)
            (0.5, 0.5, 1.0, 1.0),   # Coronal  (top-right)
        ]

        # Surfaces to overlay: list of (polydata, colour)
        self.surfaces: List[Tuple[vtkPolyData, Tuple[float, float, float]]] = []
        # Crosshair / shared state
        self._ijk: Optional[List[int]] = None
        # Surface contour pipelines per view
        self._surface_contours: List[List] = [[], [], []]
        # Interaction callbacks (kept alive)
        self._event_cbs: List = []
        self._left_down = False
        # Called with the world (mm) position whenever the user moves the
        # cursor by clicking or scrolling.  Used to link the viewer to other
        # windows, e.g. the surface view of CAT_SurfView.
        self.on_position_changed = None

    # -------- Geometry helpers --------
    def _voxel_axis_directions(self) -> List[Tuple[float, float, float]]:
        """Unit world direction of each voxel axis."""
        m = self._vox2world
        dirs = []
        for a in range(3):
            v = (m[0][a], m[1][a], m[2][a])
            n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 or 1.0
            dirs.append((v[0] / n, v[1] / n, v[2] / n))
        return dirs

    def _voxel_axis_lengths(self) -> List[float]:
        """Physical length of the volume along each voxel axis (mm)."""
        m = self._vox2world
        ext = self._image.GetExtent()
        out = []
        for a in range(3):
            n = float(ext[2 * a + 1] - ext[2 * a] + 1)
            step = (m[0][a] ** 2 + m[1][a] ** 2 + m[2][a] ** 2) ** 0.5
            out.append(n * step)
        return out

    def _world_matrix(self) -> vtkMatrix4x4:
        """Transform for the image actors: data coordinates to world."""
        source = self._actor_matrix or self._vox2world
        m = vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(source[r][c]))
        return m

    def _assign_pane_axes(self):
        """Choose the voxel axis each pane slices along.

        The panes are anatomical (axial, sagittal, coronal) but the voxel axes
        are not — a scan may be stored sagittal-first.  Every pane therefore
        takes the voxel axis whose world direction is most parallel to the
        anatomical axis it cuts, so each pane shows the plane it is named
        after whatever the storage order.  For an image stored in RAS order
        this reproduces the classic k/i/j assignment.
        """
        dirs = self._voxel_axis_directions()
        wanted = (2, 0, 1)  # axial cuts world Z, sagittal X, coronal Y
        best, best_score = (2, 0, 1), -1.0
        for perm in itertools.permutations(range(3)):
            score = sum(abs(dirs[perm[p]][wanted[p]]) for p in range(3))
            if score > best_score:
                best, best_score = perm, score
        self._pane_axis = list(best)
        if self.verbose:
            names = ["Axial", "Sagittal", "Coronal"]
            print("[cat_vol_view] Pane -> voxel axis: "
                  + ", ".join(f"{names[p]}={'ijk'[self._pane_axis[p]]}"
                              for p in range(3)))

    def _world_bbox_lengths(self) -> Tuple[float, float, float]:
        """Size of the volume along the world axes (mm)."""
        ext = self._image.GetExtent()
        corners = [
            self._world_from_index((ext[0 + (i & 1)], ext[2 + ((i >> 1) & 1)],
                                    ext[4 + ((i >> 2) & 1)]))
            for i in range(8)
        ]
        return tuple(
            max(c[ax] for c in corners) - min(c[ax] for c in corners)
            for ax in range(3)
        )

    # -------- Layout helpers --------
    def _compute_viewports(self) -> Tuple[int, int]:
        """Compute SPM12-like viewport fractions from image physical dims.

        Returns the pixel *(width, height)* for the render window.
        """
        # World extents, so the layout does not depend on the storage order
        px, py, pz = self._world_bbox_lengths()

        # Prevent division by zero for degenerate images
        if px + py == 0 or py + pz == 0:
            px = py = pz = 1.0

        col = px / (px + py)   # left-column width fraction
        row = py / (py + pz)   # bottom-row height fraction

        # Panes are placed so their aspect matches the pane they sit in: the
        # left column is as wide as the volume is in x (coronal, axial), the
        # right column as wide as it is in y (sagittal); the bottom row is as
        # tall as the volume is in y, the top row as tall as it is in z.
        gap = 0.003  # thin black border between viewports
        self._viewports = [
            (0.0,        0.0,        col - gap, row - gap),  # Axial    (bot-L)
            (col + gap,  row + gap,  1.0,       1.0),        # Sagittal (top-R)
            (0.0,        row + gap,  col - gap, 1.0),        # Coronal  (top-L)
        ]

        # Window pixel dimensions preserving physical proportions
        total_w = px + py
        total_h = py + pz
        max_phys = max(total_w, total_h)
        scale = (2 * self.window_size) / max_phys
        win_w = max(int(total_w * scale), 200)
        win_h = max(int(total_h * scale), 200)
        return win_w, win_h

    # -------- Viewport query --------
    def _get_active_view(self, x: int, y: int) -> int:
        """Return the viewer index whose viewport contains *(x, y)*.

        Returns ``-1`` when the point is outside every viewport (e.g. the
        empty bottom-right quadrant).
        """
        w, h = self.render_window.GetSize()
        if w == 0 or h == 0:
            return -1
        nx, ny = x / w, y / h
        for i, (xmin, ymin, xmax, ymax) in enumerate(self._viewports):
            if xmin <= nx <= xmax and ymin <= ny <= ymax:
                return i
        return -1

    def _get_view_from_renderer(self, renderer: vtkRenderer) -> int:
        """Map a VTK renderer instance to the internal view index."""
        for i, ren in enumerate(self.renderers):
            if ren == renderer:
                return i
        return -1

    # -------- Scroll handler --------
    def _on_scroll(self, view_idx: int, delta: int):
        """Advance the slice in *view_idx* by *delta* steps."""
        if self._ijk is None or not (0 <= view_idx < 3):
            return
        axis = self._pane_axis[view_idx]
        ext = self._image.GetExtent()
        lo, hi = ext[2 * axis], ext[2 * axis + 1]
        self._ijk[axis] = max(lo, min(hi, self._ijk[axis] + delta))
        self._set_slices_from_index()
        self._update_crosshair_lines()
        self._notify_position()

    def _dispatch_pointer(self, x: int, y: int):
        """Route a pointer position to the appropriate view and update slices."""
        ren = self.interactor.FindPokedRenderer(x, y)
        view = self._get_view_from_renderer(ren)
        if view < 0:
            view = self._get_active_view(x, y)
        if view >= 0:
            self._on_click(view, x, y)

    def _bind_interaction_events(self):
        """Install robust interactor observers for click/scroll only."""
        self._event_cbs = []

        def _left_down_cb(obj, evt):
            x, y = obj.GetEventPosition()
            self._dispatch_pointer(x, y)

        def _left_up_cb(obj, evt):
            return None

        def _wheel_fwd_cb(obj, evt):
            x, y = obj.GetEventPosition()
            ren = self.interactor.FindPokedRenderer(x, y)
            view = self._get_view_from_renderer(ren)
            if view < 0:
                view = self._get_active_view(x, y)
            if view >= 0:
                self._on_scroll(view, 1)

        def _wheel_back_cb(obj, evt):
            x, y = obj.GetEventPosition()
            ren = self.interactor.FindPokedRenderer(x, y)
            view = self._get_view_from_renderer(ren)
            if view < 0:
                view = self._get_active_view(x, y)
            if view >= 0:
                self._on_scroll(view, -1)

        self._event_cbs.extend([
            _left_down_cb,
            _left_up_cb,
            _wheel_fwd_cb,
            _wheel_back_cb,
        ])

        self.interactor.AddObserver("LeftButtonPressEvent", _left_down_cb)
        self.interactor.AddObserver("LeftButtonReleaseEvent", _left_up_cb)
        self.interactor.AddObserver("MouseWheelForwardEvent", _wheel_fwd_cb)
        self.interactor.AddObserver("MouseWheelBackwardEvent", _wheel_back_cb)

    # -------- Camera setup --------
    def _setup_cameras_spm12(self):
        """Set camera orientation per view, neurological convention.

        Axial   : camera from superior (+Z), view-up = +Y → anterior up,
                   patient-left on screen-left (neurological).
        Sagittal: camera from left (−X), view-up = +Z → superior up,
                   anterior on screen-left.
        Coronal : camera from posterior (−Y), view-up = +Z → superior up,
                   patient-left on screen-left (neurological).

        With the camera looking along *d* and *up* on screen, screen-right is
        ``d × up``; both in-plane views put +x (the patient's right) there,
        which is what "left is left" means.

        These directions are anatomical, which only holds because the world
        space comes from the NIfTI sform/qform (see :meth:`load_image`); the
        slices are then oriented the same way for every image, whatever voxel
        order it is stored in.
        """
        # (camera offset from the focal point, view-up)
        placement = [
            ((0.0, 0.0, 100.0), (0.0, 1.0, 0.0)),   # Axial, from superior
            ((-100.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # Sagittal, from the left
            ((0.0, -100.0, 0.0), (0.0, 0.0, 1.0)),  # Coronal, from posterior
        ]
        for vi in range(3):
            ren = self.renderers[vi]
            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            fp = list(cam.GetFocalPoint())
            pscale = cam.GetParallelScale()

            offset, view_up = placement[vi]
            cam.SetPosition(fp[0] + offset[0], fp[1] + offset[1], fp[2] + offset[2])
            cam.SetViewUp(*view_up)

            cam.SetParallelScale(pscale)
            ren.ResetCameraClippingRange()

    def _setup_fixed_fov(self):
        """Fix camera focal point and FOV from full-volume dimensions.

        This keeps each view static in its pane while slices change.
        """
        self._apply_field_of_view()

    def get_field_of_view(self) -> Optional[float]:
        """Edge length in mm the panes are zoomed to, None when showing all."""
        return self._fov_mm

    def set_field_of_view(self, mm: Optional[float]):
        """Zoom every pane to an *mm* bounding box around the cursor.

        Follows the zoom of the SPM ortho viewer: the value is the edge length
        of the box in millimetres (e.g. 20 for 20x20 mm), and None shows the
        whole volume again.
        """
        self._fov_mm = float(mm) if mm else None
        self._apply_field_of_view()
        self.render_window.Render()

    def _apply_field_of_view(self):
        """Point every camera at the region the current zoom asks for."""
        if self._image is None:
            return
        ext = self._image.GetExtent()

        if self._fov_mm:
            # Zoomed: centre on the cursor, same box in every pane
            focus = self.get_world_position()
            scales = [0.5 * self._fov_mm] * 3
        else:
            focus = None
            # Largest in-plane dimension per view, with a small margin
            lengths = self._voxel_axis_lengths()
            margin = 1.05
            scales = []
            for pane in range(3):
                in_plane = [a for a in range(3) if a != self._pane_axis[pane]]
                scales.append(0.5 * max(lengths[a] for a in in_plane) * margin)
        if focus is None:
            focus = self._world_from_index((
                0.5 * (ext[0] + ext[1]),
                0.5 * (ext[2] + ext[3]),
                0.5 * (ext[4] + ext[5]),
            ))

        for vi, ren in enumerate(self.renderers):
            cam = ren.GetActiveCamera()
            old_f = cam.GetFocalPoint()
            old_p = cam.GetPosition()
            offset = [old_p[i] - old_f[i] for i in range(3)]
            cam.SetFocalPoint(*focus)
            cam.SetPosition(*[focus[i] + offset[i] for i in range(3)])
            cam.SetParallelScale(scales[vi])
            ren.ResetCameraClippingRange()

    # -------- Crosshair helpers --------
    def _init_crosshair(self):
        """Create crosshair line actors in each viewer renderer."""
        extent = self._image.GetExtent()

        # Two line actors per viewer (horizontal and vertical in-plane)
        self._line_src = []
        self._line_act = []
        for vi in range(3):
            lr = []
            la = []
            for _ in range(2):
                ls = vtkLineSource()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(ls.GetOutputPort())
                act = vtkActor()
                act.SetMapper(mapper)
                act.GetProperty().SetColor(1.0, 1.0, 0.0)
                act.GetProperty().SetLineWidth(0.8)
                self.renderers[vi].AddActor(act)
                lr.append(ls)
                la.append(act)
            self._line_src.append(lr)
            self._line_act.append(la)

        # Initial crosshair position: center index
        cx = int(0.5 * (extent[0] + extent[1]))
        cy = int(0.5 * (extent[2] + extent[3]))
        cz = int(0.5 * (extent[4] + extent[5]))
        self._ijk = [cx, cy, cz]
        self._update_crosshair_lines()

    def _world_from_index(self, ijk: Tuple[int, int, int]):
        i, j, k = ijk
        # Prefer full 4x4 voxel-to-world
        if self._vox2world is not None:
            m = self._vox2world
            vx, vy, vz = float(i), float(j), float(k)
            wx = m[0][0]*vx + m[0][1]*vy + m[0][2]*vz + m[0][3]
            wy = m[1][0]*vx + m[1][1]*vy + m[1][2]*vz + m[1][3]
            wz = m[2][0]*vx + m[2][1]*vy + m[2][2]*vz + m[2][3]
            return (wx, wy, wz)
        # Fallback to origin/spacing/direction
        ox, oy, oz = self._image.GetOrigin()
        sx, sy, sz = self._image.GetSpacing()
        try:
            dm = self._image.GetDirectionMatrix()
            d = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
        except Exception:
            d = None
        vx = i * sx; vy = j * sy; vz = k * sz
        if d is not None:
            wx = ox + d[0][0]*vx + d[0][1]*vy + d[0][2]*vz
            wy = oy + d[1][0]*vx + d[1][1]*vy + d[1][2]*vz
            wz = oz + d[2][0]*vx + d[2][1]*vy + d[2][2]*vz
            return (wx, wy, wz)
        return (ox + vx, oy + vy, oz + vz)

    def _world_from_index_center(self, ijk: Tuple[float, float, float]):
        i, j, k = ijk
        return self._world_from_index((i + 0.5, j + 0.5, k + 0.5))

    def _index_from_world(self, world: Tuple[float, float, float]):
        wx, wy, wz = world
        # Use inverse of voxel->world when available
        if self._vox2world is not None:
            m = self._vox2world
            R = [[m[0][0], m[0][1], m[0][2]],
                 [m[1][0], m[1][1], m[1][2]],
                 [m[2][0], m[2][1], m[2][2]]]
            t = [m[0][3], m[1][3], m[2][3]]
            v = [wx - t[0], wy - t[1], wz - t[2]]
            if np is not None:
                Rm = np.array(R, dtype=float)
                vm = np.array(v, dtype=float)
                try:
                    ijk = np.linalg.solve(Rm, vm)
                    return float(ijk[0]), float(ijk[1]), float(ijk[2])
                except Exception:
                    pass
            # Manual 3x3 inverse fallback
            det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
                   - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
                   + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
            if det != 0:
                inv = [[0.0]*3 for _ in range(3)]
                inv[0][0] = (R[1][1]*R[2][2]-R[1][2]*R[2][1])/det
                inv[0][1] = (R[0][2]*R[2][1]-R[0][1]*R[2][2])/det
                inv[0][2] = (R[0][1]*R[1][2]-R[0][2]*R[1][1])/det
                inv[1][0] = (R[1][2]*R[2][0]-R[1][0]*R[2][2])/det
                inv[1][1] = (R[0][0]*R[2][2]-R[0][2]*R[2][0])/det
                inv[1][2] = (R[0][2]*R[1][0]-R[0][0]*R[1][2])/det
                inv[2][0] = (R[1][0]*R[2][1]-R[1][1]*R[2][0])/det
                inv[2][1] = (R[0][1]*R[2][0]-R[0][0]*R[2][1])/det
                inv[2][2] = (R[0][0]*R[1][1]-R[0][1]*R[1][0])/det
                i = inv[0][0]*v[0] + inv[0][1]*v[1] + inv[0][2]*v[2]
                j = inv[1][0]*v[0] + inv[1][1]*v[1] + inv[1][2]*v[2]
                k = inv[2][0]*v[0] + inv[2][1]*v[1] + inv[2][2]*v[2]
                return i, j, k
        # Fallback using origin/spacing/direction (approximate inverse assuming orthonormal)
        ox, oy, oz = self._image.GetOrigin()
        sx, sy, sz = self._image.GetSpacing()
        try:
            dm = self._image.GetDirectionMatrix()
            d = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
            # transpose for inverse if orthonormal
            vx = wx - ox; vy = wy - oy; vz = wz - oz
            i = (d[0][0]*vx + d[1][0]*vy + d[2][0]*vz) / sx
            j = (d[0][1]*vx + d[1][1]*vy + d[2][1]*vz) / sy
            k = (d[0][2]*vx + d[1][2]*vy + d[2][2]*vz) / sz
            return i, j, k
        except Exception:
            # axis-aligned fallback
            return ( (wx - ox)/sx, (wy - oy)/sy, (wz - oz)/sz )

    def _camera_offset(self, view_idx: int, distance: float = 0.5):
        """Return a small world-space shift towards the camera of *view_idx*.

        The crosshair lies exactly in the slice plane, where the depth buffer
        lets the image win.  Offsetting it towards the camera keeps it visible
        for any image orientation, whereas a fixed index-space offset points
        away from the camera as soon as the corresponding voxel axis is
        flipped — which is what hid the crosshair in the coronal view.
        """
        cam = self.renderers[view_idx].GetActiveCamera()
        px, py, pz = cam.GetPosition()
        fx, fy, fz = cam.GetFocalPoint()
        dx, dy, dz = px - fx, py - fy, pz - fz
        norm = (dx * dx + dy * dy + dz * dz) ** 0.5
        if norm < 1e-12:
            return (0.0, 0.0, 0.0)
        s = distance / norm
        return (dx * s, dy * s, dz * s)

    def _update_crosshair_lines(self):
        extent = self._image.GetExtent()

        def w(ijk, off):
            wx, wy, wz = self._world_from_index(tuple(ijk))
            return (wx + off[0], wy + off[1], wz + off[2])

        # Each pane gets one line along either in-plane voxel axis, both
        # crossing at the cursor and spanning the whole image.
        for pane in range(3):
            fixed = self._pane_axis[pane]
            in_plane = [a for a in range(3) if a != fixed]
            off = self._camera_offset(pane)
            for line, axis in zip(self._line_src[pane], in_plane):
                p1 = list(self._ijk); p2 = list(self._ijk)
                p1[axis] = extent[2 * axis]
                p2[axis] = extent[2 * axis + 1]
                line.SetPoint1(*w(p1, off))
                line.SetPoint2(*w(p2, off))
                line.Modified()

        # Trigger re-render
        self.render_window.Render()

    def _build_surface_contours(self):
        # Remove existing contour actors
        for vi in range(3):
            ren = self.renderers[vi]
            for entry in self._surface_contours[vi]:
                try:
                    ren.RemoveActor(entry.get('actor'))
                except Exception:
                    pass
            self._surface_contours[vi] = []

        # World-space normal of the voxel axis each pane slices along
        dirs = self._voxel_axis_directions()
        normals = [dirs[self._pane_axis[vi]] for vi in range(3)]
        # Origins must match the actual slice location used by the image actors,
        # which slices at integer voxel indices. Do NOT add the +0.5 voxel-center shift here.
        origins = [self._world_from_index(tuple(self._ijk))] * 3

        for (poly, color) in self.surfaces:
            for vi in range(3):
                plane = vtkPlane()
                plane.SetNormal(*normals[vi])
                plane.SetOrigin(*origins[vi])
                cutter = vtkCutter()
                cutter.SetCutFunction(plane)
                cutter.SetInputData(poly)
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(cutter.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)
                actor.GetProperty().SetLineWidth(1.2)
                actor.GetProperty().LightingOff()
                # The contour lies exactly in the slice plane, where the image
                # would win the depth test; nudge it towards the camera
                actor.SetPosition(*self._camera_offset(vi))
                self.renderers[vi].AddActor(actor)
                self._surface_contours[vi].append({'plane': plane, 'cutter': cutter, 'actor': actor})

    def _update_surface_planes(self):
        if not self.surfaces:
            return
        i, j, k = self._ijk
        # Keep plane origins in sync with viewer slice indices (integer index positions)
        origins = [
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
        ]
        for vi in range(3):
            for entry in self._surface_contours[vi]:
                plane = entry.get('plane')
                if plane is not None:
                    plane.SetOrigin(*origins[vi])
                    plane.Modified()
        # trigger rerender
        self.render_window.Render()

    def _set_slices_from_index(self):
        """Update each image actor's display extent to show the current
        slice, then re-render."""
        ext = list(self._image.GetExtent())

        for pane in range(3):
            axis = self._pane_axis[pane]
            display = list(ext)
            display[2 * axis] = display[2 * axis + 1] = self._ijk[axis]
            self._image_actors[pane].SetDisplayExtent(*display)

        # A zoomed view stays centred on the cursor
        if self._fov_mm:
            self._apply_field_of_view()

        # Keep clipping valid but do not move camera/focal point.
        for ren in self.renderers:
            ren.ResetCameraClippingRange()

        self._update_surface_planes()
        self.render_window.Render()

    def _on_click(self, view_idx: int, x: int, y: int):
        """Handle a click/drag at window pixel *(x, y)* in *view_idx*."""
        if self._ijk is None:
            return
        ren = self.renderers[view_idx]

        # --- Build a world-space ray from the click position ----------
        def _display_to_world(dx, dy, dz):
            ren.SetDisplayPoint(dx, dy, dz)
            ren.DisplayToWorld()
            wp = ren.GetWorldPoint()
            ww = wp[3] if wp[3] != 0 else 1.0
            return (wp[0] / ww, wp[1] / ww, wp[2] / ww)

        p1 = _display_to_world(x, y, 0)
        p2 = _display_to_world(x, y, 1)

        # --- Intersect ray with the current slice plane ---------------
        # The plane normal is the world direction of the voxel axis this pane
        # slices along
        n = self._voxel_axis_directions()[self._pane_axis[view_idx]]
        p0 = self._world_from_index(tuple(self._ijk))

        # Line-plane intersection:  p = p1 + t * (p2 - p1)
        v = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        w = (p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])
        denom = n[0] * v[0] + n[1] * v[1] + n[2] * v[2]
        if abs(denom) < 1e-12:
            return  # ray nearly parallel to slice plane
        t = (n[0] * w[0] + n[1] * w[1] + n[2] * w[2]) / denom
        wx = p1[0] + t * v[0]
        wy = p1[1] + t * v[1]
        wz = p1[2] + t * v[2]

        # --- Convert world position to voxel indices ------------------
        ii, jj, kk = self._index_from_world((wx, wy, wz))
        i = int(round(ii))
        j = int(round(jj))
        k = int(round(kk))
        ext = self._image.GetExtent()
        i = max(ext[0], min(ext[1], i))
        j = max(ext[2], min(ext[3], j))
        k = max(ext[4], min(ext[5], k))

        self._ijk = [i, j, k]
        self._set_slices_from_index()
        self._update_crosshair_lines()
        self._notify_position()
        if self.verbose:
            print(f"[cat_vol_view] Click -> ijk=({i},{j},{k})")

    # ---------- Cursor position ----------
    def _notify_position(self):
        """Report the current cursor to a linked viewer, if one is attached."""
        cb = self.on_position_changed
        if cb is None:
            return
        world = self.get_world_position()
        if world is None:
            return
        try:
            cb(world)
        except Exception:
            pass

    def get_index(self) -> Optional[Tuple[int, int, int]]:
        """Current cursor as voxel indices, or None before setup()."""
        if self._ijk is None:
            return None
        return (int(self._ijk[0]), int(self._ijk[1]), int(self._ijk[2]))

    def get_world_position(self) -> Optional[Tuple[float, float, float]]:
        """Current cursor in world (mm) coordinates, or None before setup()."""
        if self._ijk is None or self._image is None:
            return None
        return self._world_from_index(tuple(self._ijk))

    def get_value_at_index(self, ijk: Optional[Tuple[int, int, int]] = None):
        """Image intensity at *ijk* (default: the cursor), None if unavailable."""
        if self._image is None:
            return None
        if ijk is None:
            ijk = self.get_index()
        if ijk is None:
            return None
        try:
            return float(self._image.GetScalarComponentAsDouble(
                int(ijk[0]), int(ijk[1]), int(ijk[2]), 0))
        except Exception:
            return None

    def set_index(self, i: int, j: int, k: int, notify: bool = False):
        """Move the cursor to a voxel index, clamped to the image extent."""
        if self._image is None:
            return
        ext = self._image.GetExtent()
        self._ijk = [
            max(ext[0], min(ext[1], int(round(i)))),
            max(ext[2], min(ext[3], int(round(j)))),
            max(ext[4], min(ext[5], int(round(k)))),
        ]
        self._set_slices_from_index()
        self._update_crosshair_lines()
        if notify:
            self._notify_position()

    def set_world_position(self, world: Tuple[float, float, float],
                           notify: bool = False):
        """Move the cursor to a world (mm) position.

        This is the entry point used to link the viewer to another window:
        picking a point on the surface in CAT_SurfView moves the slices here.
        """
        if self._image is None:
            return
        i, j, k = self._index_from_world(tuple(world))
        self.set_index(i, j, k, notify=notify)

    # ---------- Public API ----------
    def load_image(self, image_path: str):
        if self.verbose:
            print(f"[cat_vol_view] Loading image: {image_path}")
        reader = _guess_image_reader(image_path)
        reader.Update()
        self._image = reader.GetOutput()
        if self._image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        dims = self._image.GetDimensions()
        if self.verbose:
            print(f"[cat_vol_view] Image dimensions: {dims}")
            try:
                print("[cat_vol_view] VTK image origin:", self._image.GetOrigin())
                print("[cat_vol_view] VTK image spacing:", self._image.GetSpacing())
                dm = self._image.GetDirectionMatrix()
                if dm is not None:
                    D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
                    print("[cat_vol_view] VTK image direction:")
                    for r in D:
                        print("    ", r)
            except Exception:
                pass
        if dims == (0, 0, 0):
            raise RuntimeError(
                f"Image appears empty or unreadable (dims={dims}): {image_path}"
            )
        # Voxel-to-world: NIfTI keeps the anatomical (mm) mapping in the
        # sform/qform, which VTK reports separately instead of baking it into
        # the image — the image itself stays in voxel space.  Taking it from
        # the header is what puts the slices, the crosshair and any surface
        # into the same millimetre space, which in turn is what makes the
        # cursor linking with a surface viewer meaningful.
        self._vox2world = None
        self._actor_matrix = None
        self._world_from_header = False
        for getter in ("GetSFormMatrix", "GetQFormMatrix"):
            try:
                m = getattr(reader, getter)()
            except Exception:
                m = None
            if m is None:
                continue
            M = [[float(m.GetElement(r, c)) for c in range(4)] for r in range(4)]
            if all(abs(M[r][c]) < 1e-12 for r in range(3) for c in range(3)):
                continue  # unset form
            # VTK reports this transform normalized: it maps *data* coordinates
            # (voxel index times spacing) to world.  The image actors apply the
            # spacing themselves, so they take it as it is, while index-based
            # coordinate maths has to fold the spacing in — without that every
            # position is off by the voxel size in each axis.
            self._actor_matrix = M
            spacing = self._image.GetSpacing()
            self._vox2world = [
                [M[r][c] * spacing[c] for c in range(3)] + [M[r][3]]
                for r in range(4)
            ]
            self._world_from_header = True
            if self.verbose:
                print(f"[cat_vol_view] Using voxel->world from {getter[3:-6]}:")
                for row in self._vox2world:
                    print("    ", row)
            break
        if self._vox2world is None:
            # No usable header transform: fall back to the geometry VTK
            # applied to the image itself (origin/spacing/direction).
            ox, oy, oz = self._image.GetOrigin()
            sx, sy, sz = self._image.GetSpacing()
            try:
                dm = self._image.GetDirectionMatrix()
                D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
            except Exception:
                D = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            self._vox2world = [
                [D[0][0]*sx, D[0][1]*sy, D[0][2]*sz, ox],
                [D[1][0]*sx, D[1][1]*sy, D[1][2]*sz, oy],
                [D[2][0]*sx, D[2][1]*sy, D[2][2]*sz, oz],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if self.verbose:
                print("[cat_vol_view] Using voxel->world from VTK image properties:")
                for row in self._vox2world:
                    print("    ", row)
        else:
            # Origin and orientation come from the header transform the actors
            # carry (see setup()); the image keeps its spacing, which that
            # transform expects the actors to have applied already.
            self._image.SetOrigin(0.0, 0.0, 0.0)
            try:
                ident = vtkMatrix3x3()
                ident.Identity()
                self._image.SetDirectionMatrix(ident)
            except Exception:
                pass
        self._assign_pane_axes()

        # Window/level – prefer percentile-based scaling
        rng = self._image.GetScalarRange()
        self._wl = (float(rng[1] - rng[0]),
                     float(0.5 * (rng[1] + rng[0])))
        if (self.percentile_range is not None
                and np is not None and vtk_to_numpy is not None):
            try:
                scalars = self._image.GetPointData().GetScalars()
                if scalars is not None:
                    arr = vtk_to_numpy(scalars).ravel().astype(float)
                    lo_pct, hi_pct = np.percentile(
                        arr, list(self.percentile_range))
                    if hi_pct > lo_pct:
                        self._wl = (float(hi_pct - lo_pct),
                                     float(0.5 * (hi_pct + lo_pct)))
                        if self.verbose:
                            print(
                                f"[cat_vol_view] Percentile "
                                f"{self.percentile_range}: "
                                f"window={self._wl[0]:.1f}, "
                                f"level={self._wl[1]:.1f}")
            except Exception as exc:
                if self.verbose:
                    print(f"[cat_vol_view] Percentile scaling failed: {exc}")
        return self

    def add_surface(self, surface: "str | vtkPolyData", color: Tuple[float, float, float]):
        if isinstance(surface, vtkPolyData):
            poly = surface
        else:
            poly = _load_surface(surface)
        # Optionally convert surface convention (LPS<->RAS)
        poly = self._apply_surface_convention(poly)
        if self.mirror_surfaces:
            poly = _mirror_polydata_x(poly)
            if self.verbose:
                print("[cat_vol_view] Applied surface mirroring (scale -1,1,1)")
        # Heuristic: if the surface center is far from the image center in world coords,
        # translate it so centers coincide (common when surfaces are in 0..FOV vs centered at 0).
        # Skipped once the image carries its own millimetre transform: surface
        # and image then share one space and must not be nudged apart.
        try:
            if (self._image is not None and self._vox2world is not None
                    and not self._world_from_header):
                extent = self._image.GetExtent()
                cx = 0.5 * (extent[0] + extent[1])
                cy = 0.5 * (extent[2] + extent[3])
                cz = 0.5 * (extent[4] + extent[5])
                img_center = self._world_from_index_center((cx, cy, cz))
                # Surface center from bounds
                b = [0.0]*6
                poly.GetBounds(b)
                surf_center = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5)
                # Compute image world bounding box diagonal length as scale reference
                w000 = self._world_from_index((extent[0], extent[2], extent[4]))
                w111 = self._world_from_index((extent[1], extent[3], extent[5]))
                diag = ((w111[0]-w000[0])**2 + (w111[1]-w000[1])**2 + (w111[2]-w000[2])**2) ** 0.5
                # If centers differ by more than 20% of diag (or 10mm minimum), recenter
                dx = img_center[0] - surf_center[0]
                dy = img_center[1] - surf_center[1]
                dz = img_center[2] - surf_center[2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                if dist > max(10.0, 0.2 * diag):
                    t = vtkTransform()
                    t.Translate(dx, dy, dz)
                    f = vtkTransformPolyDataFilter()
                    f.SetInputData(poly)
                    f.SetTransform(t)
                    f.Update()
                    poly = f.GetOutput()
                    if self.verbose:
                        print(f"[cat_vol_view] Recentered surface by translation (dx,dy,dz)=({dx:.3f},{dy:.3f},{dz:.3f})")
        except Exception:
            pass
        self.surfaces.append((poly, color))
        return self

    def _apply_surface_convention(self, poly: vtkPolyData) -> vtkPolyData:
        conv = self.surface_convention
        if conv == "none" or conv == "auto":
            return poly
        # We treat image voxel-to-world as RAS (NIfTI q/sform standard). Convert surfaces to RAS if needed.
        if conv == "lps":
            t = vtkTransform()
            # LPS -> RAS: flip X and Y
            t.Scale(-1.0, -1.0, 1.0)
            f = vtkTransformPolyDataFilter()
            f.SetInputData(poly)
            f.SetTransform(t)
            f.Update()
            if self.verbose:
                print("[cat_vol_view] Converted surface from LPS to RAS (flip X,Y)")
            return f.GetOutput()
        # conv == "ras" -> assume already RAS
        return poly

    def setup(self, window_title: Optional[str] = None):
        """Set up the single-window viewer with SPM12-like viewport layout.

        Parameters
        ----------
        window_title : str, optional
            Title for the render window (defaults to ``"Ortho Viewer"``).
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        # Compute proportional viewports & pixel window size.  An embedded
        # viewer only takes the viewport layout; size and title belong to the
        # host widget.
        win_w, win_h = self._compute_viewports()
        # Kept so an embedding host can size its widget with the same
        # proportions the standalone window would use
        self.window_pixel_size = (win_w, win_h)
        if not self.embedded:
            self.render_window.SetSize(win_w, win_h)
            self.render_window.SetWindowName(window_title or "Ortho Viewer")

        ext = self._image.GetExtent()

        for i in range(3):
            # Create image actor with window/level ---
            actor = vtkImageActor()
            if vtkImageMapToWindowLevelColors is not None:
                wlf = vtkImageMapToWindowLevelColors()
                wlf.SetInputData(self._image)
                wlf.SetWindow(self._wl[0])
                wlf.SetLevel(self._wl[1])
                actor.GetMapper().SetInputConnection(wlf.GetOutputPort())
                self._wl_filters[i] = wlf
            else:
                actor.SetInputData(self._image)
                try:
                    prop = actor.GetProperty()
                    prop.SetColorWindow(self._wl[0])
                    prop.SetColorLevel(self._wl[1])
                except Exception:
                    pass
            if self._world_from_header:
                # The image was reset to voxel geometry in load_image(), so the
                # actor carries the millimetre transform from the NIfTI header
                actor.SetUserMatrix(self._world_matrix())
            self._image_actors[i] = actor

            ren = self.renderers[i]
            ren.AddActor(actor)
            ren.SetViewport(*self._viewports[i])
            ren.SetBackground(0, 0, 0)
            self.render_window.AddRenderer(ren)

            if self.verbose:
                names = ["Axial", "Sagittal", "Coronal"]
                print(f"[cat_vol_view] Configured {names[i]} "
                      f"viewport {self._viewports[i]}")

        # Background renderer for the empty bottom-right quadrant
        bg_ren = vtkRenderer()
        col_edge = self._viewports[0][2] + 0.003
        row_edge = self._viewports[0][3] + 0.003
        bg_ren.SetViewport(col_edge, 0.0, 1.0, row_edge)
        bg_ren.SetBackground(0, 0, 0)
        self.render_window.AddRenderer(bg_ren)

        # Set initial crosshair to the centre of the volume
        self._ijk = [
            (ext[0] + ext[1]) // 2,
            (ext[2] + ext[3]) // 2,
            (ext[4] + ext[5]) // 2,
        ]

        # Set display extents so actors have valid bounds for ResetCamera
        self._set_slices_from_index()

        # Reset cameras to fill each viewport, then enforce SPM12
        for ren in self.renderers:
            ren.ResetCamera()
        self._setup_cameras_spm12()
        self._setup_fixed_fov()

        # Connect interactor AFTER creating actors to avoid any
        # internal event-observer interference
        if not self.embedded:
            self.interactor.SetRenderWindow(self.render_window)
        style = _OrthoStyle(self)
        self.interactor.SetInteractorStyle(style)
        self._ortho_style = style  # prevent garbage collection
        try:
            self.interactor.Initialize()
        except Exception:
            pass
        self._bind_interaction_events()

        # Crosshair overlays & surface contours
        self._init_crosshair()
        if self.surfaces:
            self._build_surface_contours()
        self._set_slices_from_index()

        return self

    def render(self, *, screenshot: Optional[str] = None,
               headless: bool = False):
        """Render the viewer and optionally save a screenshot.

        Parameters
        ----------
        screenshot : str, optional
            Path for a single combined PNG screenshot of all three views.
        headless : bool
            If ``True``, enable off-screen rendering and do not start the
            event loop.
        """
        use_offscreen = bool(headless or screenshot)
        if use_offscreen:
            try:
                self.render_window.SetOffScreenRendering(1)
            except Exception:
                pass
            try:
                self.render_window.SetMultiSamples(0)
            except Exception:
                pass

        self.render_window.Render()

        if screenshot:
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            try:
                w2i.ReadFrontBufferOff()
            except Exception:
                pass
            try:
                w2i.SetInputBufferTypeToRGBA()
            except Exception:
                pass
            w2i.Update()
            writer = vtkPNGWriter()
            if not screenshot.lower().endswith(".png"):
                screenshot += ".png"
            writer.SetFileName(screenshot)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            if self.verbose:
                print(f"[cat_vol_view] Wrote screenshot: {screenshot}")

        if not headless:
            if self.verbose:
                print("[cat_vol_view] Starting interactor…")
            self.interactor.Start()


# ------------------------------------------------------------------ #
#  Interaction style                                                  #
# ------------------------------------------------------------------ #

class _OrthoStyle(vtkInteractorStyleImage):
    """Custom interactor style for the combined orthogonal viewer.

    * Left-click: handled by explicit interactor observers in CatImageViewer.
    * Mouse-wheel: handled by explicit interactor observers in CatImageViewer.
    * Middle / right button: default pan / zoom from parent class.

    We intentionally do **not** call ``super().OnLeftButtonDown()`` because
    the base ``vtkInteractorStyleImage`` would enter window/level adjust
    mode and swallow all subsequent events.
    """

    def __init__(self, parent: CatImageViewer):
        super().__init__()
        self._parent = parent

    # -- left button: no-op here; handled by observers --
    def OnLeftButtonDown(self):
        super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        super().OnLeftButtonUp()

    # -- right button: the host shows a context menu when embedded, so the
    #    inherited drag-to-zoom must not start underneath it --
    def OnRightButtonDown(self):
        if getattr(self._parent, 'embedded', False):
            return
        super().OnRightButtonDown()

    def OnRightButtonUp(self):
        if getattr(self._parent, 'embedded', False):
            return
        super().OnRightButtonUp()

    def OnMouseMove(self):
        super().OnMouseMove()


# ------------------------------------------------------------------ #
#  Qt window                                                          #
# ------------------------------------------------------------------ #

class VolumeViewerWindow(QtWidgets.QMainWindow):
    """Window around :class:`CatImageViewer`.

    Used by ``CAT_VolView`` and, with *on_position_changed* wired up, as the
    volume window of CAT_SurfView, so both offer the same slices, status line
    and context menu.

    Args:
        image_path: Volume to display.
        surfaces: Surface files or ``vtkPolyData`` drawn as outlines on the
            slices; they must be in the millimetre space of the image.
        on_position_changed: Called as ``callback(world_xyz, window)`` when the
            user moves the cursor by clicking or scrolling.
        viewer_kwargs: Passed on to :class:`CatImageViewer`.
    """

    #: Zoom levels of the context menu: label and bounding-box edge length in
    #: mm, following the ortho viewer of SPM (None = whole volume).
    ZOOM_LEVELS = (
        ("Full volume", None),
        ("160 x 160 mm", 160.0),
        ("80 x 80 mm", 80.0),
        ("40 x 40 mm", 40.0),
        ("20 x 20 mm", 20.0),
        ("10 x 10 mm", 10.0),
    )

    SURFACE_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 1.0))

    def __init__(self, image_path: str, parent=None, on_position_changed=None,
                 surfaces: Sequence = (), **viewer_kwargs):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(f"Volume: {os.path.basename(image_path)}")
        self.on_position_changed = on_position_changed

        central = QtWidgets.QWidget(self)
        box = QtWidgets.QVBoxLayout(central)
        box.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        box.addWidget(self.vtk_widget, 1)
        try:
            self.vtk_widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow, True)
        except Exception:
            pass

        render_window = self.vtk_widget.GetRenderWindow()
        self.viewer = CatImageViewer(
            render_window=render_window,
            interactor=render_window.GetInteractor(),
            **viewer_kwargs,
        )
        self.viewer.load_image(image_path)
        for i, surface in enumerate(surfaces):
            self.viewer.add_surface(
                surface, self.SURFACE_COLORS[i % len(self.SURFACE_COLORS)])
        self.viewer.setup(window_title=os.path.basename(image_path))
        self.viewer.on_position_changed = self._position_changed

        self._label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self._label)
        self._update_label()

        width, height = getattr(self.viewer, 'window_pixel_size', (700, 700))
        self.resize(int(width), int(height) + 24)

        try:
            policy = QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        except AttributeError:
            policy = QtCore.Qt.CustomContextMenu
        self.vtk_widget.setContextMenuPolicy(policy)
        self.vtk_widget.customContextMenuRequested.connect(self._show_context_menu)

        # The interactor may only be initialised once the widget has a window
        QtCore.QTimer.singleShot(0, self._post_show)

    def _post_show(self):
        try:
            self.vtk_widget.Initialize()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    # -------- context menu --------
    def _show_context_menu(self, pos):
        """Right-click menu of the slice views."""
        menu = QtWidgets.QMenu(self)
        current = self.viewer.get_field_of_view()

        zoom_menu = menu.addMenu("Zoom")
        for label, mm in self.ZOOM_LEVELS:
            action = zoom_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current == mm)
            action.triggered.connect(lambda _checked=False, v=mm: self.set_zoom(v))
        # Further sections (window/level, overlays, …) go here

        menu.exec(self.vtk_widget.mapToGlobal(pos))

    def set_zoom(self, mm: Optional[float]):
        """Zoom the slices to an mm bounding box around the cursor."""
        try:
            self.viewer.set_field_of_view(mm)
        except Exception:
            pass

    # -------- cursor --------
    def _update_label(self):
        ijk = self.viewer.get_index()
        world = self.viewer.get_world_position()
        if ijk is None or world is None:
            self._label.setText("")
            return
        value = self.viewer.get_value_at_index(ijk)
        text = (f"voxel [{ijk[0]}, {ijk[1]}, {ijk[2]}]    "
                f"mm ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")
        if value is not None:
            text += f"    value {value:g}"
        self._label.setText(text)

    def _position_changed(self, world_xyz: Tuple[float, float, float]):
        """Report a position picked in a slice to whoever is linked."""
        self._update_label()
        if self.on_position_changed is not None:
            try:
                self.on_position_changed(world_xyz, self)
            except Exception:
                pass

    def set_world_position(self, world_xyz: Tuple[float, float, float]):
        """Move the slices to a world (mm) position picked elsewhere."""
        try:
            self.viewer.set_world_position(world_xyz)
            self._update_label()
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description=(
            "Single-window orthogonal VTK image viewer (SPM12-like "
            "layout) with optional surface overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "image",
        help="Input image: .nii(.gz), .mnc, .mha/.mhd, .nrrd, …",
    )
    p.add_argument(
        "surfaces", nargs="*",
        help="0-3 surface files (.gii, .vtk, .vtp, .obj, .stl)",
    )
    p.add_argument(
        "--size", type=int, default=400,
        help="Approximate half-width of the window in pixels",
    )
    p.add_argument(
        "--mirror", action="store_true",
        help="Mirror surfaces along x (scale -1,1,1)",
    )
    p.add_argument(
        "--no-mirror", action="store_true",
        help=argparse.SUPPRESS,  # now the default; kept for compatibility
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Do not start interactor (no window)",
    )
    p.add_argument(
        "--screenshot", type=str, default=None,
        help="Path (.png) to save a combined screenshot and exit",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic information",
    )
    p.add_argument(
        "--surface-convention",
        choices=["auto", "ras", "lps", "none"],
        default="auto",
        help=(
            "Coordinate convention of input surfaces; converted to "
            "match the image world (RAS). Use 'none' to skip."
        ),
    )
    p.add_argument(
        "--percentile", nargs=2, type=float, default=[3, 97],
        metavar=("LOW", "HIGH"),
        help="Percentile range for display intensity scaling",
    )
    p.add_argument(
        "--no-percentile", action="store_true",
        help="Use the full intensity range instead of percentile scaling",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point."""
    args = _parse_args(argv)

    pct = None if args.no_percentile else tuple(args.percentile)
    options = dict(
        window_size=args.size,
        # Surfaces and image share the millimetre space of the NIfTI header,
        # so mirroring is opt-in
        mirror_surfaces=bool(args.mirror) and not args.no_mirror,
        verbose=args.verbose,
        surface_convention=args.surface_convention,
        percentile_range=pct,
    )
    surfaces = list(args.surfaces[:3])

    if args.screenshot or args.headless:
        # Batch mode renders without a window, so no Qt application is needed
        viewer = CatImageViewer(**options)
        viewer.load_image(args.image)
        for i, surf in enumerate(surfaces):
            viewer.add_surface(surf, VolumeViewerWindow.SURFACE_COLORS[i])
        viewer.setup(window_title=os.path.basename(args.image))
        viewer.render(screenshot=args.screenshot, headless=True)
        return 0

    app = QtWidgets.QApplication(sys.argv)
    window = VolumeViewerWindow(args.image, surfaces=surfaces, **options)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
