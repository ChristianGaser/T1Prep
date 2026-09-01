"""Registration of native cortical surfaces to fsLR space.

fMRIPrep expects two spheres per hemisphere in fsLR space and skips its own
work when it finds them:

``space-fsLR_desc-reg_sphere``
    The baseline registration.  sMRIPrep builds it with ``wb_command
    -surface-sphere-project-unproject``, which carries a sphere already
    registered to *fsaverage* through the fixed fsaverage-to-fsLR deformation.
    No optimisation is involved, and :func:`project_unproject` reproduces the
    Workbench result to well under a micrometre.

``space-fsLR_desc-msmsulc_sphere``
    sMRIPrep refines the baseline with MSMSulc, a folding-driven spherical
    registration onto the fsLR group average.  :func:`write_msm_sphere`
    replaces it with CAT-Surface's Spherical Demons, run against the fsLR
    average midthickness.

See :mod:`t1prep.data.templates_surfaces_fsLR` for the provenance of the
template surfaces both functions rely on.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "project_unproject",
    "rigid_align_sphere",
    "spherical_barycentric",
    "write_reg_sphere",
    "write_msm_sphere",
]


# ---------------------------------------------------------------------------
# Spherical barycentric resampling
# ---------------------------------------------------------------------------


def _vertex_to_faces(faces: np.ndarray, n_vertices: int):
    """Build a CSR-style vertex-to-incident-face index.

    Args:
        faces: ``(n_faces, 3)`` triangle vertex indices.
        n_vertices: Number of vertices in the mesh.

    Returns:
        Tuple ``(face_of, starts)`` where the faces incident to vertex ``v``
        are ``face_of[starts[v]:starts[v + 1]]``.
    """
    flat = faces.ravel(order="C")
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(n_vertices + 1))
    return order // 3, starts


def spherical_barycentric(points, sphere, faces, neighbours: int = 6):
    """Locate ``points`` within the triangles of a spherical mesh.

    Each point is treated as a ray from the origin; the ray is intersected with
    the triangles incident to the point's nearest mesh vertices and the
    triangle whose intersection has non-negative barycentric coordinates wins.
    Points that fall in no candidate triangle — which happens only for meshes
    with irregular valence — keep the least-negative candidate, so the result
    is always a valid interpolation over three neighbouring vertices.

    Args:
        points: ``(n, 3)`` query positions on (or near) the sphere.
        sphere: ``(m, 3)`` vertices of the spherical mesh to search.
        faces: ``(k, 3)`` triangles of that mesh.
        neighbours: Nearest vertices per point whose faces are considered.

    Returns:
        Tuple ``(indices, weights)``, both ``(n, 3)``: the enclosing triangle's
        vertex indices and their barycentric weights, which sum to one.
    """
    points = np.asarray(points, dtype=np.float64)
    sphere = np.asarray(sphere, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_of, starts = _vertex_to_faces(faces, len(sphere))
    max_valence = int((starts[1:] - starts[:-1]).max())

    nearest = np.atleast_2d(cKDTree(sphere).query(points, k=neighbours)[1])

    best_index = np.zeros((len(points), 3), dtype=np.int64)
    best_weight = np.zeros((len(points), 3), dtype=np.float64)
    best_error = np.full(len(points), np.inf)
    pending = np.ones(len(points), dtype=bool)

    for column in range(nearest.shape[1]):
        active = np.flatnonzero(pending)
        if active.size == 0:
            break
        vertex = nearest[active, column]
        counts = starts[vertex + 1] - starts[vertex]
        for slot in range(max_valence):
            usable = slot < counts
            if not usable.any():
                continue
            subset = active[usable]
            triangle = faces[face_of[starts[vertex[usable]] + slot]]
            # Solve point = w0*a + w1*b + w2*c for the ray through the origin;
            # normalising the weights projects the hit back onto the sphere.
            corners = np.stack(
                [sphere[triangle[:, 0]], sphere[triangle[:, 1]], sphere[triangle[:, 2]]],
                axis=-1,
            )
            try:
                weight = np.linalg.solve(corners, points[subset][..., None])[..., 0]
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate mesh
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                weight = weight / weight.sum(axis=1, keepdims=True)
            error = np.maximum(0.0, -weight).sum(axis=1)

            improved = error < best_error[subset]
            target = subset[improved]
            best_error[target] = error[improved]
            best_index[target] = triangle[improved]
            best_weight[target] = weight[improved]
            pending[target[error[improved] <= 1e-9]] = False

    return best_index, best_weight


def project_unproject(sphere_in, project_to, unproject_from, faces):
    """Carry a sphere through a fixed deformation between two spherical frames.

    Equivalent to ``wb_command -surface-sphere-project-unproject``: every
    vertex of ``sphere_in`` is located in ``project_to`` and re-expressed with
    the same barycentric weights in ``unproject_from``, which shares
    ``project_to``'s topology but lives in the destination frame.

    Args:
        sphere_in: ``(n, 3)`` sphere to carry over, in ``project_to``'s frame.
        project_to: ``(m, 3)`` source-frame vertices of the deformation mesh.
        unproject_from: ``(m, 3)`` destination-frame vertices of that mesh.
        faces: ``(k, 3)`` triangles shared by both deformation meshes.

    Returns:
        ``(n, 3)`` vertices in the destination frame, on a sphere of the same
        radius as ``sphere_in``.
    """
    index, weight = spherical_barycentric(sphere_in, project_to, faces)
    moved = (np.asarray(unproject_from, dtype=np.float64)[index] * weight[:, :, None]).sum(axis=1)
    radius = np.linalg.norm(np.asarray(sphere_in, dtype=np.float64), axis=1).mean()
    return moved / np.linalg.norm(moved, axis=1, keepdims=True) * radius


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------


def rigid_align_sphere(sphere, target):
    """Rotate a sphere onto a target parameterisation of the same mesh.

    The rotation that best maps ``sphere`` onto ``target`` in the least-squares
    sense, from the orthogonal Procrustes solution.  This is the pre-alignment
    MSMSulc performs with ``wb_command -surface-affine-regression`` followed by
    ``-surface-modify-sphere``; restricting it to a rotation makes the
    reprojection unnecessary, since a rotation already maps the sphere onto
    itself.

    Args:
        sphere: ``(n, 3)`` vertices to rotate.
        target: ``(n, 3)`` vertices of the same mesh in the destination frame.

    Returns:
        ``(n, 3)`` rotated vertices, on ``sphere``'s own radius.
    """
    sphere = np.asarray(sphere, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    u, _, vt = np.linalg.svd(sphere.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # forbid a reflection
        u[:, -1] *= -1
        rotation = u @ vt
    return sphere @ rotation


def _template(templates_dir: str, fshemi: str, name: str) -> str:
    path = os.path.join(templates_dir, f"{fshemi}.{name}.gii")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"fsLR template surface not found: {path}. It ships in "
            "t1prep/data/templates_surfaces_fsLR."
        )
    return path


def write_reg_sphere(
    sphere_reg_file: str,
    out_file: str,
    fslr_templates_dir: str,
    fsaverage_sphere_file: str,
    fshemi: str,
) -> None:
    """Write the ``space-fsLR_desc-reg`` sphere for one hemisphere.

    Args:
        sphere_reg_file: The subject's sphere registered to fsaverage
            (T1Prep's ``Spherereg_surface``).
        out_file: Destination ``.surf.gii``.
        fslr_templates_dir: Directory holding the fsLR template surfaces.
        fsaverage_sphere_file: The fsaverage sphere the deformation is defined
            on — T1Prep's own surface template, which shares its mesh with
            ``<fshemi>.sphere.fsLR_deformed.gii``.
        fshemi: ``"lh"`` or ``"rh"``.
    """
    import cat_surf

    sphere_reg, sphere_faces = cat_surf.read_surface(sphere_reg_file)
    fsaverage, faces = cat_surf.read_surface(fsaverage_sphere_file)
    deformed, _ = cat_surf.read_surface(
        _template(fslr_templates_dir, fshemi, "sphere.fsLR_deformed")
    )

    moved = project_unproject(sphere_reg, fsaverage, deformed, faces)
    cat_surf.write_surface(out_file, moved.astype(np.float32), sphere_faces)


def write_msm_sphere(
    mid_surface_file: str,
    reg_sphere_file: str,
    out_file: str,
    fslr_templates_dir: str,
    fshemi: str,
    sphere_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Write the ``space-fsLR_desc-msmsulc`` sphere for one hemisphere.

    Stands in for sMRIPrep's MSMSulc step.  Spherical Demons registers the
    subject onto the fsLR average midthickness, driven by the depth-potential
    features CAT-Surface derives from the two anatomies — which play the role
    MSMSulc gives to sulcal depth.

    Following MSMSulc, the registration starts from the subject's *native*
    sphere rigidly rotated onto the fsLR frame, not from the non-linearly
    aligned ``desc-reg`` sphere.  Starting from ``desc-reg`` would compose this
    refinement on top of the fsaverage registration and inherit its areal
    distortion; from a rigid start the whole non-linear deformation is this
    one registration's, which is what keeps distortion comparable to MSMSulc's.

    Args:
        mid_surface_file: The subject's central (midthickness) surface.
        reg_sphere_file: The ``desc-reg`` sphere from :func:`write_reg_sphere`.
            Used as the target of the rigid pre-alignment, and as the starting
            point itself when ``sphere_file`` is not given.
        out_file: Destination ``.surf.gii``.
        fslr_templates_dir: Directory holding the fsLR template surfaces.
        fshemi: ``"lh"`` or ``"rh"``.
        sphere_file: The subject's native sphere.  Omit to refine
            ``reg_sphere_file`` directly instead of restarting from a rigid
            alignment.
        verbose: Forwarded to CAT-Surface.
    """
    import cat_surf
    from cat_surf import cli as cs_cli

    start_file = reg_sphere_file
    tmp_start = None
    if sphere_file:
        sphere, faces = cat_surf.read_surface(sphere_file)
        target, _ = cat_surf.read_surface(reg_sphere_file)
        rotated = rigid_align_sphere(sphere, target)
        # CAT-Surface works on files, so the rotated sphere needs one; it is
        # an intermediate and is removed once the registration has read it.
        tmp_start = f"{out_file}.rigid-init.gii"
        cat_surf.write_surface(tmp_start, rotated.astype(np.float32), faces)
        start_file = tmp_start

    try:
        cs_cli.surf_spherical_demon(
            source_file=mid_surface_file,
            source_sphere_file=start_file,
            target_file=_template(fslr_templates_dir, fshemi, "midthickness.fsLR"),
            target_sphere_file=_template(fslr_templates_dir, fshemi, "sphere.fsLR"),
            output_sphere_file=out_file,
            verbose=verbose,
        )
    finally:
        if tmp_start and os.path.exists(tmp_start):
            os.remove(tmp_start)
