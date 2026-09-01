"""Tests for :mod:`t1prep.fslr`.

The reference for :func:`project_unproject` is a rotation applied in closed
form: when the destination mesh is the source mesh rigidly rotated, carrying a
sphere through the deformation has to reproduce that same rotation exactly, so
the test does not depend on the interpolation it is checking.
"""

import numpy as np
import pytest

from t1prep.fslr import project_unproject, rigid_align_sphere, spherical_barycentric


def _icosphere(subdivisions=3, radius=100.0):
    """A geodesic sphere, refined by repeated triangle subdivision."""
    t = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}
        new_faces = []
        vertices = list(vertices)

        def middle(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                vertices.append((vertices[a] + vertices[b]) / 2.0)
                midpoint[key] = len(vertices) - 1
            return midpoint[key]

        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        vertices = np.array(vertices)
        faces = np.array(new_faces, dtype=np.int64)

    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius, faces


def _rotation(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (
        np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    )
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        @ np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        @ np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


@pytest.fixture(scope="module")
def sphere():
    return _icosphere(subdivisions=4)


def test_weights_are_a_convex_combination(sphere):
    vertices, faces = sphere
    query, _ = _icosphere(subdivisions=3)
    index, weight = spherical_barycentric(query, vertices, faces)

    np.testing.assert_allclose(weight.sum(axis=1), 1.0, atol=1e-9)
    assert (weight >= -1e-9).all(), "an enclosing triangle must give non-negative weights"
    assert index.max() < len(vertices)


def test_projecting_a_mesh_onto_itself_is_the_identity(sphere):
    vertices, faces = sphere
    moved = project_unproject(vertices, vertices, vertices, faces)
    np.testing.assert_allclose(moved, vertices, atol=1e-8)


def test_deformation_is_carried_exactly(sphere):
    """A rigidly rotated destination mesh must reproduce that rotation."""
    vertices, faces = sphere
    rotation = _rotation(0.31, -0.17, 0.44)
    query, _ = _icosphere(subdivisions=3)

    moved = project_unproject(query, vertices, vertices @ rotation.T, faces)
    np.testing.assert_allclose(moved, query @ rotation.T, atol=1e-6)


def test_radius_is_preserved(sphere):
    vertices, faces = sphere
    query, _ = _icosphere(subdivisions=3, radius=100.0)
    moved = project_unproject(query, vertices, vertices * 0.5, faces)
    # unproject_from is scaled, but the output stays on the input's sphere
    np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 100.0, atol=1e-8)


def test_query_points_need_not_be_mesh_vertices(sphere):
    """Interpolating inside a face is exact for a linear destination map."""
    vertices, faces = sphere
    face = faces[0]
    inside = (vertices[face] * np.array([0.5, 0.3, 0.2])[:, None]).sum(axis=0)
    inside = inside / np.linalg.norm(inside) * 100.0

    rotation = _rotation(0.2, 0.1, -0.3)
    moved = project_unproject(inside[None], vertices, vertices @ rotation.T, faces)
    np.testing.assert_allclose(moved[0], inside @ rotation.T, atol=1e-6)


def test_rigid_alignment_recovers_a_known_rotation(sphere):
    """The Procrustes fit must undo exactly the rotation that was applied."""
    vertices, _ = sphere
    rotation = _rotation(0.7, -0.25, 0.9)
    target = vertices @ rotation.T

    aligned = rigid_align_sphere(vertices, target)
    np.testing.assert_allclose(aligned, target, atol=1e-8)


def test_rigid_alignment_stays_on_the_sphere(sphere):
    """No scale or shear, so the radius is untouched -- unlike an affine fit."""
    vertices, _ = sphere
    rng = np.random.default_rng(0)
    target = vertices @ _rotation(0.4, 0.2, -0.1).T * 1.7    # rotated *and* scaled
    target += rng.normal(scale=2.0, size=target.shape)       # and noisy

    aligned = rigid_align_sphere(vertices, target)
    np.testing.assert_allclose(
        np.linalg.norm(aligned, axis=1), np.linalg.norm(vertices, axis=1), atol=1e-8
    )


def test_rigid_alignment_never_reflects(sphere):
    """A reflected target must not induce a determinant -1 'rotation'."""
    vertices, _ = sphere
    aligned = rigid_align_sphere(vertices, vertices * np.array([1.0, 1.0, -1.0]))
    # Recover the operator from the fit and check it is a proper rotation.
    operator = np.linalg.lstsq(vertices, aligned, rcond=None)[0]
    assert np.linalg.det(operator) > 0
