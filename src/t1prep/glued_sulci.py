"""Detection of glued (buried) sulci on a cortical surface.

A glued sulcus is a place where the two banks of a sulcus were never pulled
apart, so the surface touches itself: two patches sit a fraction of a
millimetre from each other, facing one another, while lying far apart across
the mesh.

That is deliberately *not* the same defect as a self-intersection.  The
triangles of a glued sulcus need not cross -- the banks merely come into
contact -- which is why ``cat_surf.fix_self_intersect`` leaves the condition
untouched (measured: it returns a glued pial surface bit-for-bit unchanged)
and why a dedicated measure is needed.  Surface area cannot see it either:
across a pair of hemispheres whose glued-vertex counts differed 26-fold the
areas differed by 2.5%.

The measure counts vertices that take part in such a contact, which makes it
comparable between hemispheres of different mesh density when expressed as a
fraction.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = ["glued_sulci", "DEFAULT_RADIUS", "DEFAULT_NORMAL_DOT", "DEFAULT_HOPS"]

#: Contact distance in mm.  Two banks nearer than this are touching rather
#: than merely facing each other across a thin sulcus.
DEFAULT_RADIUS = 0.75

#: Upper bound on the dot product of the two vertex normals.  Opposing banks
#: point at one another, so their normals are close to anti-parallel; a pair
#: on the same sheet has a dot product near +1.
DEFAULT_NORMAL_DOT = -0.5

#: Mesh-graph radius that still counts as "the same patch".  Neighbours within
#: this many hops are excluded so ordinary local geometry is not reported.
DEFAULT_HOPS = 2


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return unit vertex normals, area-weighted by the incident faces."""
    e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_normals = np.cross(e1, e2)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0] = 1.0
    return normals / norm[:, None]


def glued_sulci(
    vertices: Any,
    faces: Any,
    radius: float = DEFAULT_RADIUS,
    normal_dot: float = DEFAULT_NORMAL_DOT,
    hops: int = DEFAULT_HOPS,
) -> Dict[str, float]:
    """Measure how much of a surface is glued to itself.

    Parameters
    ----------
    vertices : array_like, shape (V, 3)
        Vertex coordinates in mm.
    faces : array_like, shape (F, 3)
        Triangle indices.
    radius : float, optional
        Contact distance in mm (default :data:`DEFAULT_RADIUS`).
    normal_dot : float, optional
        Keep only pairs whose unit normals have a dot product below this
        (default :data:`DEFAULT_NORMAL_DOT`), i.e. patches that face
        each other.
    hops : int, optional
        Exclude pairs within this many mesh-graph hops
        (default :data:`DEFAULT_HOPS`).

    Returns
    -------
    dict
        ``n_vertices``, ``glued_vertices``, ``glued_fraction`` (of all
        vertices), ``contact_pairs`` and ``area`` (mm^2).
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    n = len(vertices)

    e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area = float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())
    if n == 0 or len(faces) == 0:
        return {"n_vertices": n, "glued_vertices": 0, "glued_fraction": 0.0,
                "contact_pairs": 0, "area": area}

    normals = _vertex_normals(vertices, faces)

    pairs = cKDTree(vertices).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        facing = np.einsum(
            "ij,ij->i", normals[pairs[:, 0]], normals[pairs[:, 1]]
        ) < normal_dot
        pairs = pairs[facing]
    if len(pairs):
        # Edges of the mesh, symmetrised, then grown to `hops` so that
        # everything reachable locally can be subtracted in one lookup.
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        adj = coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(n, n),
        ).tocsr()
        adj = ((adj + adj.T) > 0).astype(np.int8)
        reach = adj
        for _ in range(max(hops - 1, 0)):
            reach = ((reach @ adj + reach) > 0).astype(np.int8)
        local = np.asarray(reach[pairs[:, 0], pairs[:, 1]]).ravel().astype(bool)
        pairs = pairs[~local]

    glued = np.unique(pairs) if len(pairs) else np.empty(0, dtype=int)
    return {
        "n_vertices": int(n),
        "glued_vertices": int(len(glued)),
        "glued_fraction": float(len(glued) / n),
        "contact_pairs": int(len(pairs)),
        "area": area,
    }
