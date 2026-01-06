"""
cat_viewimage.py

Modular VTK image viewer providing 3 orthogonal slice views (axial, coronal, sagittal)
with optional surface overlays. Designed to be embedded or used as a CLI tool.

Intended as a Python port of deprecated/CAT_Image.cxx, and to be integrated
with cat_viewsurf-like surface tooling.

Usage (CLI):
    python src/t1prep/gui/cat_viewimage.py <image.(nii|nii.gz|mnc|mha|mhd|nrrd)> [surf1] [surf2] [surf3] \
                                                                                --size 400 [--no-mirror]

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

from vtkmodules.vtkInteractionImage import vtkResliceImageViewer
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
    vtkPropPicker,
    vtkCellPicker,
)
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
    """Three-pane orthogonal viewer with optional surface overlays."""

    def __init__(self, window_size: int = 400, mirror_surfaces: bool = True, verbose: bool = False, surface_convention: str = "auto"):
        self.window_size = int(window_size)
        self.verbose = bool(verbose)
        self.surface_convention = surface_convention.lower()
        # Avoid double flipping: if a convention is explicitly provided, do not mirror by default
        if self.surface_convention in ("ras", "lps") and mirror_surfaces:
            self.mirror_surfaces = False
            if self.verbose:
                print("[cat_viewimage] Disabling mirroring due to explicit --surface-convention")
        else:
            self.mirror_surfaces = bool(mirror_surfaces)
        self._image = None
        self._vox2world = None  # 4x4 matrix (list of lists)

        # Three reslice viewers with their own render windows/interactors
        self.viewers: List[vtkResliceImageViewer] = [
            vtkResliceImageViewer(), vtkResliceImageViewer(), vtkResliceImageViewer()
        ]
        self.render_windows: List[vtkRenderWindow] = [
            vtkRenderWindow(), vtkRenderWindow(), vtkRenderWindow()
        ]
        self.interactors: List[vtkRenderWindowInteractor] = [
            vtkRenderWindowInteractor(), vtkRenderWindowInteractor(), vtkRenderWindowInteractor()
        ]
        # Surfaces to overlay: list of (polydata, color)
        self.surfaces: List[Tuple[vtkPolyData, Tuple[float, float, float]]] = []
        # Crosshair/shared state
        self._crosshair = None  # type: ignore
        self._ijk = None  # type: ignore
        # pickers per view (set in setup)
        self._pickers = [None, None, None]
        # keep custom styles alive
        self._styles = [None, None, None]
        # keep click callbacks alive
        self._click_cbs = [None, None, None]
        # surface contour pipelines per view: list of dicts with plane/cutter/actor
        self._surface_contours = [[], [], []]

    # -------- Crosshair helpers --------
    def _init_crosshair(self):
        image = self._image
        extent = image.GetExtent()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        # Map: viewer index -> constant axis for slicing
        # 0: XY (slice along Z), 1: YZ (slice along X), 2: XZ (slice along Y)
        self._plane_normals = [2, 0, 1]

        # Two line actors per viewer (horizontal and vertical in plane)
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
                self.viewers[vi].GetRenderer().AddActor(act)
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
        p1 = w(extent[0], j, k); p2 = w(extent[1], j, k)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i, extent[2], k); p2 = w(i, extent[3], k)
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

        # YZ
        ls_h, ls_v = self._line_src[1]
        p1 = w(i, extent[2], k); p2 = w(i, extent[3], k)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i, j, extent[4]); p2 = w(i, j, extent[5])
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

        # XZ
        ls_h, ls_v = self._line_src[2]
        p1 = w(extent[0], j, k); p2 = w(extent[1], j, k)
        ls_h.SetPoint1(*p1); ls_h.SetPoint2(*p2)
        p1 = w(i, j, extent[4]); p2 = w(i, j, extent[5])
        ls_v.SetPoint1(*p1); ls_v.SetPoint2(*p2)
        ls_h.Modified(); ls_v.Modified()

        # Trigger re-render
        for rw in self.render_windows:
            rw.Render()

    def _build_surface_contours(self):
        # Remove existing contour actors
        for vi in range(3):
            ren = self.viewers[vi].GetRenderer()
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
        # Origins must match the actual slice location used by vtkResliceImageViewer,
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
                self.viewers[vi].GetRenderer().AddActor(actor)
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
        for rw in self.render_windows:
            rw.Render()

    def _set_slices_from_index(self):
        i, j, k = self._ijk
        # match orientations mapping earlier
        self.viewers[0].SetSlice(k)  # XY: slice along Z
        self.viewers[1].SetSlice(i)  # YZ: slice along X
        self.viewers[2].SetSlice(j)  # XZ: slice along Y
        # update surface cutting planes to follow new slice
        self._update_surface_planes()
        for rw in self.render_windows:
            rw.Render()

    def _on_click(self, view_idx: int, x: int, y: int):
        if self.verbose:
            print(f"[cat_viewimage] Click event in view {view_idx} at display (x,y)=({x},{y})")
        # Pick world position under (x,y) in the clicked view. Prefer a CellPicker tied to the image actor.
        ren = self.viewers[view_idx].GetRenderer()
        picker = self._pickers[view_idx]
        picked = False
        wx = wy = wz = 0.0
        if picker is not None:
            picked = bool(picker.Pick(x, y, 0, ren))
            if picked:
                wx, wy, wz = picker.GetPickPosition()
        if not picked:
            # Fallback: intersect click ray with current slice plane
            # Compute display-to-world for near/far points
            ren.SetDisplayPoint(x, y, 0)
            ren.DisplayToWorld()
            p1 = ren.GetWorldPoint()[:3]
            ren.SetDisplayPoint(x, y, 1)
            ren.DisplayToWorld()
            p2 = ren.GetWorldPoint()[:3]
            # Plane normal and point depend on view; use voxel->world for accuracy
            i0, j0, k0 = self._ijk
            # normals for views: XY uses k-axis, YZ uses i-axis, XZ uses j-axis
            if self._vox2world is not None:
                m = self._vox2world
                normals = [
                    (m[0][2], m[1][2], m[2][2]),
                    (m[0][0], m[1][0], m[2][0]),
                    (m[0][1], m[1][1], m[2][1]),
                ]
            else:
                normals = [ (0.0,0.0,1.0), (1.0,0.0,0.0), (0.0,1.0,0.0) ]
            n = normals[view_idx]
            p0 = self._world_from_index_center((i0, j0, k0))
            # Line-plane intersection p = p1 + t*(p2-p1)
            v = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
            w = (p0[0]-p1[0], p0[1]-p1[1], p0[2]-p1[2])
            denom = n[0]*v[0] + n[1]*v[1] + n[2]*v[2]
            t = 0.0 if denom == 0 else (n[0]*w[0] + n[1]*w[1] + n[2]*w[2]) / denom
            wx = p1[0] + t * v[0]
            wy = p1[1] + t * v[1]
            wz = p1[2] + t * v[2]

        # Convert world to index using inverse transform
        ii, jj, kk = self._index_from_world((wx, wy, wz))
        i = int(round(ii))
        j = int(round(jj))
        k = int(round(kk))
        # Clamp to extent
        ext = self._image.GetExtent()
        i = min(max(i, ext[0]), ext[1])
        j = min(max(j, ext[2]), ext[3])
        k = min(max(k, ext[4]), ext[5])
        self._ijk = [i, j, k]
        self._set_slices_from_index()
        self._update_crosshair_lines()
        if self.verbose:
            print(f"[cat_viewimage] Click -> index (i,j,k)=({i},{j},{k})")

    def _make_click_cb(self, view_idx: int):
        def _cb(obj, evt):
            try:
                x, y = obj.GetEventPosition()
            except Exception:
                try:
                    x, y = self.interactors[view_idx].GetEventPosition()
                except Exception:
                    x, y = (0, 0)
            self._on_click(view_idx, x, y)
        return _cb

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
        # Basic window/level from scalar range
        rng = self._image.GetScalarRange()
        self._wl = (float(rng[1] - rng[0]), float(0.5 * (rng[1] + rng[0])))
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

    def setup(self, window_titles: Optional[Sequence[str]] = None):
        if self._image is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        # Configure viewers
        orientations = (
            (self.viewers[0].SetSliceOrientationToXY, "Axial"),
            (self.viewers[1].SetSliceOrientationToYZ, "Coronal"),
            (self.viewers[2].SetSliceOrientationToXZ, "Sagittal"),
        )

        for i, viewer in enumerate(self.viewers):
            rw = self.render_windows[i]
            iren = self.interactors[i]
            viewer.SetRenderWindow(rw)
            iren.SetRenderWindow(rw)
            # Link viewer to interactor so it installs the right style/observers
            viewer.SetupInteractor(iren)
            # Configure a picker tied to the image actor for reliable picking
            cp = vtkCellPicker()
            cp.SetTolerance(0.0005)
            try:
                img_act = viewer.GetImageActor()  # type: ignore[attr-defined]
                if img_act is not None:
                    cp.AddPickList(img_act)
                    cp.PickFromListOn()
            except Exception:
                # Fallback: use general picking without list
                pass
            iren.SetPicker(cp)
            self._pickers[i] = cp
            # Install custom click style for crosshair sync
            style = _ClickStyle(self, i)
            style.SetInteractor(iren)
            iren.SetInteractorStyle(style)
            self._styles[i] = style
            viewer.SetInputData(self._image)
            orientations[i][0]()  # set orientation
            # Window/level
            viewer.SetColorWindow(self._wl[0])
            viewer.SetColorLevel(self._wl[1])
            # Size and placement
            rw.SetSize(self.window_size, self.window_size)
            if i == 0:
                rw.SetPosition(0, 40)
            elif i == 1:
                rw.SetPosition(0, 60 + self.window_size)
            else:
                rw.SetPosition(self.window_size, 60 + self.window_size)

            title = (
                window_titles[i]
                if window_titles and i < len(window_titles)
                else orientations[i][1]
            )
            rw.SetWindowName(title)

            viewer.GetRenderer().ResetCamera()
            if self.verbose:
                print(f"[cat_viewimage] Configured {title} view window at position {rw.GetPosition()}")

        # Initialize all interactors; enable for all (one Start will run the loop)
        for iren in self.interactors:
            iren.Initialize()
            try:
                iren.Enable()
            except Exception:
                pass

        # Install click observers on all interactors so clicks are captured regardless of style
        for i, iren in enumerate(self.interactors):
            cb = self._make_click_cb(i)
            iren.AddObserver("LeftButtonPressEvent", cb)
            self._click_cbs[i] = cb

        # Place slices to middle
        for viewer in self.viewers:
            extent = viewer.GetInput().GetExtent()
            axis = viewer.GetSliceOrientation()
            # axis: 0=XY (Z slicing), 1=YZ (X), 2=ZX (Y)
            if axis == 0:
                zmin, zmax = extent[4], extent[5]
                viewer.SetSlice((zmin + zmax) // 2)
            elif axis == 1:
                xmin, xmax = extent[0], extent[1]
                viewer.SetSlice((xmin + xmax) // 2)
            else:  # axis == 2
                ymin, ymax = extent[2], extent[3]
                viewer.SetSlice((ymin + ymax) // 2)

        # Initialize crosshair overlays and sync slices to index
        self._init_crosshair()
        # Build surface contours aligned with current slices
        if self.surfaces:
            self._build_surface_contours()
        self._set_slices_from_index()

        return self

    def render(self, *, screenshot: Optional[str] = None, headless: bool = False):
        # Configure offscreen if needed (macOS often segfaults when capturing without this)
        use_offscreen = bool(headless or screenshot)
        for i, rw in enumerate(self.render_windows):
            if use_offscreen:
                try:
                    rw.SetOffScreenRendering(1)
                except Exception:
                    pass
                try:
                    rw.SetMultiSamples(0)
                except Exception:
                    pass
            # Render each window to ensure a valid back buffer
            rw.Render()

        if screenshot:
            # Save three screenshots with suffices
            suffices = ["axial", "coronal", "sagittal"]
            for idx, rw in enumerate(self.render_windows):
                w2i = vtkWindowToImageFilter()
                w2i.SetInput(rw)
                # Prefer back buffer and RGBA when available
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
                # If a single filename is provided, add suffixes; else use as-is.
                if screenshot.lower().endswith(".png"):
                    base = screenshot[:-4]
                    out_path = f"{base}_{suffices[idx]}.png"
                else:
                    out_path = f"{screenshot}_{suffices[idx]}.png"
                writer.SetFileName(out_path)
                writer.SetInputConnection(w2i.GetOutputPort())
                writer.Write()
                if self.verbose:
                    print(f"[cat_viewimage] Wrote screenshot: {out_path}")

        # Start one interactor to drive all windows unless headless
        if not headless and self.interactors:
            if self.verbose:
                print("[cat_viewimage] Starting interactor…")
            # Try the primary interactor first
            self.interactors[0].Start()
            # If it returned immediately (seen on some macOS setups), try others
            if self.verbose:
                print("[cat_viewimage] Interactor returned; checking fallback…")
            for i, iren in enumerate(self.interactors[1:], start=1):
                if iren is not None and hasattr(iren, "Start"):
                    if self.verbose:
                        print(f"[cat_viewimage] Starting interactor {i}…")
                    iren.Start()

class _ClickStyle(vtkInteractorStyleImage):
    def __init__(self, parent, view_idx):
        super().__init__()
        self._parent = parent
        self._view_idx = view_idx

    def OnLeftButtonDown(self):
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        try:
            self._parent._on_click(self._view_idx, x, y)
        except Exception as e:
            if getattr(self._parent, 'verbose', False):
                print(f"[cat_viewimage] Click handling error: {e}")
        # call base to keep default image interactions
        super().OnLeftButtonDown()


def _parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Orthogonal VTK image viewer with optional surface overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", help="Input image: .nii(.gz), .mnc, .mha/.mhd, .nrrd, ...")
    p.add_argument("surfaces", nargs="*", help="0-3 surface files (.gii, .vtk, .vtp, .obj, .stl)")
    p.add_argument("--size", type=int, default=400, help="Window size (pixels)")
    p.add_argument("--no-mirror", action="store_true", help="Disable mirroring (scale -1,1,1)")
    p.add_argument("--headless", action="store_true", help="Do not start interactor (no windows)")
    p.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Base path or filename (.png) to save screenshots and exit",
    )
    p.add_argument("--verbose", action="store_true", help="Print diagnostic information")
    p.add_argument(
        "--surface-convention",
        choices=["auto", "ras", "lps", "none"],
        default="auto",
        help=(
            "Coordinate convention of input surfaces; converted to match the image world (RAS). "
            "Use 'none' to skip any conversion."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    viewer = CatImageViewer(window_size=args.size, mirror_surfaces=not args.no_mirror, verbose=args.verbose, surface_convention=args.surface_convention)
    viewer.load_image(args.image)

    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for i, surf in enumerate(args.surfaces[:3]):
        viewer.add_surface(surf, colors[i])

    viewer.setup(window_titles=[os.path.basename(args.image)] * 3)
    viewer.render(screenshot=args.screenshot, headless=args.headless or bool(args.screenshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
