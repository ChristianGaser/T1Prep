"""
cat_viewimage.py

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
    python src/t1prep/gui/cat_viewimage.py <image> [surf1] [surf2] [surf3] \
        --size 400 [--no-mirror] [--percentile 3 97]

Notes:
- Tries to use vtkNIFTIImageReader for NIfTI, vtkMINCImageReader for MINC,
  and vtkImageReader2Factory otherwise.
- Surfaces: supports .gii, .vtp, .vtk, .obj, .stl via appropriate VTK readers.
- Mirroring (scale -1,1,1) is applied by default to match C++ example.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Sequence, Tuple

# Import minimal VTK modules explicitly (avoids large monolithic import)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPlane
from vtkmodules.vtkCommonCore import vtkPoints
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
        mirror_surfaces: bool = True,
        verbose: bool = False,
        surface_convention: str = "auto",
        percentile_range: Optional[Tuple[float, float]] = (3.0, 97.0),
    ):
        self.window_size = int(window_size)
        self.verbose = bool(verbose)
        self.surface_convention = surface_convention.lower()
        self.percentile_range = percentile_range
        # Avoid double flipping: if a convention is explicitly provided,
        # do not mirror by default.
        if self.surface_convention in ("ras", "lps") and mirror_surfaces:
            self.mirror_surfaces = False
            if self.verbose:
                print("[cat_viewimage] Disabling mirroring due to "
                      "explicit --surface-convention")
        else:
            self.mirror_surfaces = bool(mirror_surfaces)
        self._image = None
        self._vox2world = None  # 4×4 matrix (list of lists)

        # Three renderers + image actors (one per orthogonal view)
        self.renderers: List[vtkRenderer] = [
            vtkRenderer(), vtkRenderer(), vtkRenderer(),
        ]
        self._image_actors: List[Optional[vtkImageActor]] = [
            None, None, None,
        ]
        self._wl_filters: List = [None, None, None]
        # Single render window and interactor
        self.render_window: vtkRenderWindow = vtkRenderWindow()
        self.interactor: vtkRenderWindowInteractor = (
            vtkRenderWindowInteractor()
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

    # -------- Layout helpers --------
    def _compute_viewports(self) -> Tuple[int, int]:
        """Compute SPM12-like viewport fractions from image physical dims.

        Returns the pixel *(width, height)* for the render window.
        """
        dims = self._image.GetDimensions()
        spacing = self._image.GetSpacing()

        px = dims[0] * abs(spacing[0])
        py = dims[1] * abs(spacing[1])
        pz = dims[2] * abs(spacing[2])

        # Prevent division by zero for degenerate images
        if px + py == 0 or py + pz == 0:
            px = py = pz = 1.0

        col = px / (px + py)   # left-column width fraction
        row = py / (py + pz)   # bottom-row height fraction

        gap = 0.003  # thin black border between viewports
        self._viewports = [
            (0.0,        0.0,        col - gap, row - gap),  # Axial  (bot-L)
            (0.0,        row + gap,  col - gap, 1.0),        # Sagit  (top-L)
            (col + gap,  row + gap,  1.0,       1.0),        # Coron  (top-R)
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
        # view 0 (XY/Axial) → k (axis 2)
        # view 1 (YZ/Sagittal) → i (axis 0)
        # view 2 (XZ/Coronal) → j (axis 1)
        axis_map = {0: 2, 1: 0, 2: 1}
        axis = axis_map.get(view_idx)
        if axis is None:
            return
        ext = self._image.GetExtent()
        lo, hi = ext[2 * axis], ext[2 * axis + 1]
        self._ijk[axis] = max(lo, min(hi, self._ijk[axis] + delta))
        self._set_slices_from_index()
        self._update_crosshair_lines()

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
        """Set camera orientation per view for SPM12 radiological convention.

        Axial   : camera from inferior (−Z), view-up = +Y → anterior up,
                   patient-left on screen-right (radiological).
        Sagittal: camera from left (−X), view-up = +Z → superior up,
                   anterior on screen-left.
        Coronal : camera from anterior (+Y), view-up = +Z → superior up,
                   patient-left on screen-right (radiological).
        """
        for vi in range(3):
            ren = self.renderers[vi]
            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            fp = list(cam.GetFocalPoint())
            pscale = cam.GetParallelScale()

            if vi == 0:  # Axial (XY)
                cam.SetPosition(fp[0], fp[1], fp[2] - 100)
                cam.SetViewUp(0, 1, 0)
            elif vi == 1:  # Sagittal (YZ)
                cam.SetPosition(fp[0] - 100, fp[1], fp[2])
                cam.SetViewUp(0, 0, 1)
            elif vi == 2:  # Coronal (XZ)
                cam.SetPosition(fp[0], fp[1] + 100, fp[2])
                cam.SetViewUp(0, 0, 1)

            # Apply requested display transform:
            # 1) rotate 90° clockwise in-plane
            # 2) switch left/right display (horizontal mirror)
            cam.Roll(90.0)
            px, py, pz = cam.GetPosition()
            cam.SetPosition(2.0 * fp[0] - px, 2.0 * fp[1] - py, 2.0 * fp[2] - pz)

            cam.SetParallelScale(pscale)
            ren.ResetCameraClippingRange()

    def _setup_fixed_fov(self):
        """Fix camera focal point and FOV from full-volume dimensions.

        This keeps each view static in its pane while slices change.
        """
        ext = self._image.GetExtent()

        ni = float(ext[1] - ext[0] + 1)
        nj = float(ext[3] - ext[2] + 1)
        nk = float(ext[5] - ext[4] + 1)

        # Physical axis lengths in world coordinates
        if self._vox2world is not None:
            m = self._vox2world
            li = ni * ((m[0][0] ** 2 + m[1][0] ** 2 + m[2][0] ** 2) ** 0.5)
            lj = nj * ((m[0][1] ** 2 + m[1][1] ** 2 + m[2][1] ** 2) ** 0.5)
            lk = nk * ((m[0][2] ** 2 + m[1][2] ** 2 + m[2][2] ** 2) ** 0.5)
        else:
            sx, sy, sz = self._image.GetSpacing()
            li, lj, lk = ni * abs(sx), nj * abs(sy), nk * abs(sz)

        # Use largest in-plane dimension for each view with small margin.
        margin = 1.05
        scales = [
            0.5 * max(li, lj) * margin,  # Axial (XY)
            0.5 * max(lj, lk) * margin,  # Sagittal (YZ)
            0.5 * max(li, lk) * margin,  # Coronal (XZ)
        ]

        cx = 0.5 * (ext[0] + ext[1])
        cy = 0.5 * (ext[2] + ext[3])
        cz = 0.5 * (ext[4] + ext[5])
        fx, fy, fz = self._world_from_index((cx, cy, cz))

        for vi, ren in enumerate(self.renderers):
            cam = ren.GetActiveCamera()
            old_fx, old_fy, old_fz = cam.GetFocalPoint()
            old_px, old_py, old_pz = cam.GetPosition()
            dx = old_px - old_fx
            dy = old_py - old_fy
            dz = old_pz - old_fz
            cam.SetFocalPoint(fx, fy, fz)
            cam.SetPosition(fx + dx, fy + dy, fz + dz)
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

    def _update_crosshair_lines(self):
        extent = self._image.GetExtent()
        i, j, k = self._ijk
        eps = 0.01

        def w(ii, jj, kk):
            return self._world_from_index((ii, jj, kk))

        # For each viewer, update two lines spanning the in-plane axes
        # Viewer 0: XY plane at Z=k
        #   - horizontal line: x in [ix0..ix1] at y=j, z=k
        #   - vertical line:   y in [iy0..iy1] at x=i, z=k
        # Viewer 1: YZ plane at X=i
        # Viewer 2: XZ plane at Y=j
        # Compute endpoints in world coordinates

        # XY
        ls_h, ls_v = self._line_src[0]
        p1 = w(extent[0], j, k + eps); p2 = w(extent[1], j, k + eps)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i, extent[2], k + eps); p2 = w(i, extent[3], k + eps)
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

        # YZ
        ls_h, ls_v = self._line_src[1]
        p1 = w(i + eps, extent[2], k); p2 = w(i + eps, extent[3], k)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i + eps, j, extent[4]); p2 = w(i + eps, j, extent[5])
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

        # XZ
        ls_h, ls_v = self._line_src[2]
        p1 = w(extent[0], j + eps, k); p2 = w(extent[1], j + eps, k)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i, j + eps, extent[4]); p2 = w(i, j + eps, extent[5])
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

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

        i, j, k = self._ijk
        # Compute world-space normals using voxel-to-world matrix if available
        normals = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        if self._vox2world is not None:
            m = self._vox2world
            # columns of linear part are world directions of i,j,k axes
            normals = [
                (m[0][2], m[1][2], m[2][2]),  # k axis for XY
                (m[0][0], m[1][0], m[2][0]),  # i axis for YZ
                (m[0][1], m[1][1], m[2][1]),  # j axis for XZ
            ]
        # Origins must match the actual slice location used by the image actors,
        # which slices at integer voxel indices. Do NOT add the +0.5 voxel-center shift here.
        origins = [
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
            self._world_from_index((i, j, k)),
        ]

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
        i, j, k = self._ijk
        ext = self._image.GetExtent()

        # Axial (view 0): XY plane at Z = k
        self._image_actors[0].SetDisplayExtent(
            ext[0], ext[1], ext[2], ext[3], k, k)
        # Sagittal (view 1): YZ plane at X = i
        self._image_actors[1].SetDisplayExtent(
            i, i, ext[2], ext[3], ext[4], ext[5])
        # Coronal (view 2): XZ plane at Y = j
        self._image_actors[2].SetDisplayExtent(
            ext[0], ext[1], j, j, ext[4], ext[5])

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
        i0, j0, k0 = self._ijk
        # Plane normals per view (k-axis for Axial, i for Sagittal,
        # j for Coronal)
        if self._vox2world is not None:
            m = self._vox2world
            normals = [
                (m[0][2], m[1][2], m[2][2]),
                (m[0][0], m[1][0], m[2][0]),
                (m[0][1], m[1][1], m[2][1]),
            ]
        else:
            normals = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        n = normals[view_idx]
        p0 = self._world_from_index((i0, j0, k0))

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
        if self.verbose:
            print(f"[cat_viewimage] Click -> ijk=({i},{j},{k})")

    # ---------- Public API ----------
    def load_image(self, image_path: str):
        if self.verbose:
            print(f"[cat_viewimage] Loading image: {image_path}")
        reader = _guess_image_reader(image_path)
        reader.Update()
        self._image = reader.GetOutput()
        if self._image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        dims = self._image.GetDimensions()
        if self.verbose:
            print(f"[cat_viewimage] Image dimensions: {dims}")
            try:
                print("[cat_viewimage] VTK image origin:", self._image.GetOrigin())
                print("[cat_viewimage] VTK image spacing:", self._image.GetSpacing())
                dm = self._image.GetDirectionMatrix()
                if dm is not None:
                    D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
                    print("[cat_viewimage] VTK image direction:")
                    for r in D:
                        print("    ", r)
            except Exception:
                pass
        if dims == (0, 0, 0):
            raise RuntimeError(
                f"Image appears empty or unreadable (dims={dims}): {image_path}"
            )
        # Build voxel->world from VTK image properties (origin/spacing/direction)
        try:
            ox, oy, oz = self._image.GetOrigin()
            sx, sy, sz = self._image.GetSpacing()
            try:
                dm = self._image.GetDirectionMatrix()
                D = [[dm.GetElement(r, c) for c in range(3)] for r in range(3)]
            except Exception:
                D = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            M = [
                [D[0][0]*sx, D[0][1]*sy, D[0][2]*sz, ox],
                [D[1][0]*sx, D[1][1]*sy, D[1][2]*sz, oy],
                [D[2][0]*sx, D[2][1]*sy, D[2][2]*sz, oz],
                [0.0, 0.0, 0.0, 1.0],
            ]
            self._vox2world = M
            if self.verbose:
                print("[cat_viewimage] Using voxel->world from VTK image properties:")
                for r in self._vox2world:
                    print("    ", r)
        except Exception:
            # Fallback to qform/sform via nibabel
            self._vox2world = None
            try:
                if nib is not None:
                    img = nib.load(image_path)
                    aff = None
                    if hasattr(img, 'header'):
                        s = img.header.get_sform(coded=True)
                        q = img.header.get_qform(coded=True)
                        if s is not None and s[1] > 0:
                            aff = s[0]
                        elif q is not None and q[1] > 0:
                            aff = q[0]
                    if aff is None and hasattr(img, 'affine'):
                        aff = img.affine
                    if aff is not None:
                        self._vox2world = [[float(aff[r, c]) for c in range(4)] for r in range(4)]
                        if self.verbose:
                            print("[cat_viewimage] Using voxel->world from qform/sform (fallback):")
                            for r in self._vox2world:
                                print("    ", r)
            except Exception as e:
                if self.verbose:
                    print(f"[cat_viewimage] qform/sform read failed: {e}")
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
                                f"[cat_viewimage] Percentile "
                                f"{self.percentile_range}: "
                                f"window={self._wl[0]:.1f}, "
                                f"level={self._wl[1]:.1f}")
            except Exception as exc:
                if self.verbose:
                    print(f"[cat_viewimage] Percentile scaling failed: {exc}")
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
                print("[cat_viewimage] Applied surface mirroring (scale -1,1,1)")
        # Heuristic: if the surface center is far from the image center in world coords,
        # translate it so centers coincide (common when surfaces are in 0..FOV vs centered at 0).
        try:
            if self._image is not None and self._vox2world is not None:
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
                        print(f"[cat_viewimage] Recentered surface by translation (dx,dy,dz)=({dx:.3f},{dy:.3f},{dz:.3f})")
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
                print("[cat_viewimage] Converted surface from LPS to RAS (flip X,Y)")
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

        # Compute proportional viewports & pixel window size
        win_w, win_h = self._compute_viewports()
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
            self._image_actors[i] = actor

            ren = self.renderers[i]
            ren.AddActor(actor)
            ren.SetViewport(*self._viewports[i])
            ren.SetBackground(0, 0, 0)
            self.render_window.AddRenderer(ren)

            if self.verbose:
                names = ["Axial", "Sagittal", "Coronal"]
                print(f"[cat_viewimage] Configured {names[i]} "
                      f"viewport {self._viewports[i]}")

        # Background renderer for the empty bottom-right quadrant
        bg_ren = vtkRenderer()
        col_edge = self._viewports[0][2] + 0.003
        row_edge = self._viewports[0][3] + 0.003
        bg_ren.SetViewport(col_edge, 0.0, 1.0, row_edge)
        bg_ren.SetBackground(0, 0, 0)
        self.render_window.AddRenderer(bg_ren)

        # Set initial crosshair to the centre of the volume
        cx = (ext[0] + ext[1]) // 2
        cy = (ext[2] + ext[3]) // 2
        cz = (ext[4] + ext[5]) // 2
        self._ijk = [cx, cy, cz]

        # Set display extents so actors have valid bounds for ResetCamera
        self._image_actors[0].SetDisplayExtent(
            ext[0], ext[1], ext[2], ext[3], cz, cz)
        self._image_actors[1].SetDisplayExtent(
            cx, cx, ext[2], ext[3], ext[4], ext[5])
        self._image_actors[2].SetDisplayExtent(
            ext[0], ext[1], cy, cy, ext[4], ext[5])

        # Reset cameras to fill each viewport, then enforce SPM12
        for ren in self.renderers:
            ren.ResetCamera()
        self._setup_cameras_spm12()
        self._setup_fixed_fov()

        # Connect interactor AFTER creating actors to avoid any
        # internal event-observer interference
        self.interactor.SetRenderWindow(self.render_window)
        style = _OrthoStyle(self)
        self.interactor.SetInteractorStyle(style)
        self._ortho_style = style  # prevent garbage collection
        self.interactor.Initialize()
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
                print(f"[cat_viewimage] Wrote screenshot: {screenshot}")

        if not headless:
            if self.verbose:
                print("[cat_viewimage] Starting interactor…")
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

    def OnMouseMove(self):
        super().OnMouseMove()


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
        "--no-mirror", action="store_true",
        help="Disable mirroring (scale -1,1,1)",
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

    viewer = CatImageViewer(
        window_size=args.size,
        mirror_surfaces=not args.no_mirror,
        verbose=args.verbose,
        surface_convention=args.surface_convention,
        percentile_range=pct,
    )
    viewer.load_image(args.image)

    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for i, surf in enumerate(args.surfaces[:3]):
        viewer.add_surface(surf, colors[i])

    viewer.setup(window_title=os.path.basename(args.image))
    viewer.render(
        screenshot=args.screenshot,
        headless=args.headless or bool(args.screenshot),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
