"""Tests for the ANTs/ITK composite writer in :mod:`t1prep.segment`.

The oracle for the point mapping is ``torch``'s own ``grid_sample`` /
``affine_grid``, because that is the convention deepmriprep's ``warp_xy`` /
``warp_yx`` and ``affine`` are expressed in.  The oracle for the file layout is
a plain-NumPy reader written from the ITK ``CompositeTransform`` spec rather
than from the writer under test.
"""

import numpy as np
import nibabel as nib
import pytest
import torch
import torch.nn.functional as F

h5py = pytest.importorskip("h5py")

from t1prep.segment import save_deformation_h5

RAS2LPS = np.diag([-1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Independent ITK reader
# ---------------------------------------------------------------------------
def _read_composite(path):
    """Return ``[(type, fixed, params), ...]`` in ITK queue order."""
    out = []
    with h5py.File(path, "r") as hf:
        group = hf["TransformGroup"]
        for key in sorted(group, key=int):
            ttype = group[key]["TransformType"][0]
            ttype = ttype.decode() if isinstance(ttype, bytes) else ttype
            if ttype.startswith("Composite"):
                continue
            out.append(
                (
                    ttype,
                    np.asarray(group[key]["TransformFixedParameters"], dtype=np.float64),
                    np.asarray(group[key]["TransformParameters"], dtype=np.float64),
                )
            )
    return out


def _apply_composite(path, points_ras):
    """Map RAS points through the composite, following the ITK spec."""
    from scipy.ndimage import map_coordinates

    stages = []
    for ttype, fixed, params in _read_composite(path):
        if "Affine" in ttype:
            matrix, translation, centre = params[:9].reshape(3, 3), params[9:12], fixed[:3]

            def stage(p, m=matrix, t=translation, c=centre):
                return (p - c) @ m.T + c + t

        else:
            size = fixed[:3].astype(int)
            origin, spacing = fixed[3:6], fixed[6:9]
            direction = fixed[9:18].reshape(3, 3)
            # ITK buffers vector images component-fastest, then x, then y, then z.
            field = params.reshape(size[2], size[1], size[0], 3).transpose(2, 1, 0, 3)
            inverse = np.linalg.inv(direction * spacing)

            def stage(p, f=field, o=origin, inv=inverse):
                index = (p - o) @ inv.T
                shift = np.stack(
                    [map_coordinates(f[..., c], index.T, order=1, mode="nearest")
                     for c in range(3)],
                    axis=-1,
                )
                return p + shift

        stages.append(stage)

    points = points_ras @ RAS2LPS.T
    for stage in reversed(stages):          # ITK applies the queue back to front
        points = stage(points)
    return points @ RAS2LPS.T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NATIVE_SHAPE = (20, 22, 24)
NATIVE_AFFINE = np.array(
    [[1.1, 0.0, 0.0, -11.0], [0.0, 1.2, 0.0, -13.0], [0.0, 0.0, 0.9, -10.0], [0, 0, 0, 1.0]]
)
WARP_SHAPE = (9, 11, 13)
WARP_AFFINE = np.array(
    [[2.0, 0.0, 0.0, -8.0], [0.0, 2.0, 0.0, -10.0], [0.0, 0.0, 2.0, -12.0], [0, 0, 0, 1.0]]
)
REF_SHAPE = (11, 12, 13)
REF_AFFINE = np.array(
    [[1.5, 0.0, 0.0, -7.5], [0.0, 1.5, 0.0, -8.0], [0.0, 0.0, 1.5, -9.0], [0, 0, 0, 1.0]]
)

THETA = np.array(
    [
        [0.92, 0.05, -0.03, 0.06],
        [-0.04, 0.88, 0.07, -0.05],
        [0.02, -0.06, 0.95, 0.04],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _identity_grid(shape):
    """Absolute ``align_corners=True`` sampling grid that samples in place."""
    theta = torch.eye(3, 4)[None]
    return F.affine_grid(theta, [1, 1, *shape], align_corners=True)[0].numpy()


def _native_img():
    return nib.Nifti1Image(np.zeros(NATIVE_SHAPE, dtype=np.float32), NATIVE_AFFINE)


def _warp_img(grid):
    return nib.Nifti1Image(np.asarray(grid, dtype=np.float32), WARP_AFFINE)


def _oracle_native_mm(grid_coords):
    """Native RAS mm for normalised coordinates, via torch's own conventions.

    Builds a native volume whose three channels hold each voxel's RAS
    coordinate and samples it, so the answer comes from ``grid_sample`` rather
    than from any coordinate algebra in the code under test.

    Returns the sampled coordinates together with a mask of the samples that
    landed strictly inside the volume; ``grid_sample`` pads with zeros rather
    than extrapolating, so points outside it carry no information.
    """
    index = np.indices(NATIVE_SHAPE, dtype=np.float64)
    coords = np.einsum("ij,jxyz->xyzi", NATIVE_AFFINE[:3, :3], index) + NATIVE_AFFINE[:3, 3]
    volume = torch.from_numpy(coords.transpose(3, 0, 1, 2)[None].copy()).double()
    theta = torch.from_numpy(THETA[None, :3]).double()
    grid = torch.from_numpy(np.asarray(grid_coords, dtype=np.float64))
    # Apply THETA to the sampling grid, exactly as the pipeline composes them.
    flat = grid.reshape(-1, 3)
    moved = flat @ theta[0, :, :3].T + theta[0, :, 3]
    sampled = F.grid_sample(
        volume, moved.reshape(1, *grid.shape[:-1], 3), align_corners=True, mode="bilinear"
    )
    inside = (moved.abs() <= 1.0).all(-1).reshape(grid.shape[:-1]).numpy()
    return sampled[0].permute(1, 2, 3, 0).numpy(), inside


def _ref_points():
    """Interior points of the reference grid, in RAS mm."""
    index = np.indices(REF_SHAPE, dtype=np.float64)
    ras = np.einsum("ij,jxyz->xyzi", REF_AFFINE[:3, :3], index) + REF_AFFINE[:3, 3]
    return ras[2:-2, 2:-2, 2:-2].reshape(-1, 3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_forward_matches_torch_oracle(tmp_path):
    """The composite reproduces what grid_sample+affine_grid would produce."""
    rng = np.random.default_rng(0)
    grid = _identity_grid(WARP_SHAPE) + 0.05 * rng.standard_normal((*WARP_SHAPE, 3))
    path = tmp_path / "fwd.h5"
    save_deformation_h5(
        _warp_img(grid), THETA, _native_img(), str(path),
        ref_shape=WARP_SHAPE, ref_affine=WARP_AFFINE,
    )

    # Evaluate at the warp grid's own voxel centres, where no interpolation of
    # the stored field is involved.
    index = np.indices(WARP_SHAPE, dtype=np.float64)
    points = (np.einsum("ij,jxyz->xyzi", WARP_AFFINE[:3, :3], index)
              + WARP_AFFINE[:3, 3]).reshape(-1, 3)

    got = _apply_composite(str(path), points)
    expected, inside = _oracle_native_mm(grid)
    inside = inside.reshape(-1)
    assert inside.sum() > 0.5 * inside.size
    np.testing.assert_allclose(got[inside], expected.reshape(-1, 3)[inside], atol=1e-3)


def test_identity_warp_gives_pure_affine(tmp_path):
    """With an identity warp the composite collapses to the linear stage."""
    grid = _identity_grid(WARP_SHAPE)
    path = tmp_path / "aff.h5"
    save_deformation_h5(
        _warp_img(grid), THETA, _native_img(), str(path),
        ref_shape=REF_SHAPE, ref_affine=REF_AFFINE,
    )
    points = _ref_points()
    got = _apply_composite(str(path), points)

    # Same points expressed as normalised warp-grid coordinates.
    n0, n1, n2 = WARP_SHAPE
    voxel = nib.affines.apply_affine(np.linalg.inv(WARP_AFFINE), points)
    grid_coords = np.stack(
        [2 * voxel[:, 2] / (n2 - 1) - 1,
         2 * voxel[:, 1] / (n1 - 1) - 1,
         2 * voxel[:, 0] / (n0 - 1) - 1],
        axis=-1,
    )
    expected, inside = _oracle_native_mm(grid_coords[None, None])
    inside = inside.reshape(-1)
    assert inside.sum() > 0.5 * inside.size
    np.testing.assert_allclose(got[inside], expected.reshape(-1, 3)[inside], atol=1e-3)


def test_inverse_round_trip(tmp_path):
    """forward then inverse returns the original points.

    A constant shift in normalised coordinates is used, because its exact
    inverse is the opposite shift — so the pair really is invertible and any
    error in the queue order or in inverting the linear stage shows up.
    """
    shift = np.array([0.08, -0.05, 0.11])
    forward = _identity_grid(WARP_SHAPE) + shift
    backward = _identity_grid(WARP_SHAPE) - shift
    fwd, inv = tmp_path / "f.h5", tmp_path / "i.h5"
    save_deformation_h5(_warp_img(forward), THETA, _native_img(), str(fwd),
                        ref_shape=REF_SHAPE, ref_affine=REF_AFFINE)
    save_deformation_h5(_warp_img(backward), THETA, _native_img(), str(inv),
                        ref_shape=REF_SHAPE, ref_affine=REF_AFFINE, inverse=True)

    points = _ref_points()
    np.testing.assert_allclose(
        _apply_composite(str(inv), _apply_composite(str(fwd), points)), points, atol=1e-3
    )


def test_queue_order(tmp_path):
    """ANTs writes affine-then-field forward and field-then-affine inverse."""
    grid = _identity_grid(WARP_SHAPE)
    kwargs = dict(ref_shape=REF_SHAPE, ref_affine=REF_AFFINE)
    fwd, inv = tmp_path / "f.h5", tmp_path / "i.h5"
    save_deformation_h5(_warp_img(grid), THETA, _native_img(), str(fwd), **kwargs)
    save_deformation_h5(_warp_img(grid), THETA, _native_img(), str(inv), inverse=True, **kwargs)

    assert [t[0] for t in _read_composite(str(fwd))] == [
        "AffineTransform_float_3_3", "DisplacementFieldTransform_float_3_3"
    ]
    assert [t[0] for t in _read_composite(str(inv))] == [
        "DisplacementFieldTransform_float_3_3", "AffineTransform_float_3_3"
    ]


def test_file_layout_matches_itk(tmp_path):
    """Datasets, dtypes and string types match what ITK 5.x writes."""
    grid = _identity_grid(WARP_SHAPE)
    path = tmp_path / "x.h5"
    save_deformation_h5(_warp_img(grid), THETA, _native_img(), str(path),
                        ref_shape=REF_SHAPE, ref_affine=REF_AFFINE)

    with h5py.File(path, "r") as hf:
        assert set(hf) == {"HDFVersion", "ITKVersion", "OSName", "OSVersion", "TransformGroup"}
        # The composite header carries only its type, as ITK writes it.
        assert list(hf["TransformGroup/0"]) == ["TransformType"]
        assert hf["TransformGroup/0/TransformType"].id.get_type().is_variable_str()
        for key in ("1", "2"):
            group = hf[f"TransformGroup/{key}"]
            assert group["TransformType"].id.get_type().is_variable_str()
            assert group["TransformFixedParameters"].dtype == np.float64
            assert group["TransformParameters"].dtype == np.float32

        fixed = np.asarray(hf["TransformGroup/2/TransformFixedParameters"])
        np.testing.assert_allclose(fixed[:3], REF_SHAPE)
        # Origin and direction are stored in LPS, so x and y are negated.
        np.testing.assert_allclose(fixed[3:6], RAS2LPS @ REF_AFFINE[:3, 3])
        np.testing.assert_allclose(fixed[6:9], np.diag(REF_AFFINE[:3, :3]))
        np.testing.assert_allclose(fixed[9:18], np.diag([-1.0, -1.0, 1.0]).ravel())


def test_accepts_5d_warp_and_dataframe_affine(tmp_path):
    """``warp_xy`` may arrive as (x, y, z, 1, 3) and ``affine`` as a DataFrame."""
    pd = pytest.importorskip("pandas")
    grid = _identity_grid(WARP_SHAPE)
    plain, fancy = tmp_path / "a.h5", tmp_path / "b.h5"
    kwargs = dict(ref_shape=REF_SHAPE, ref_affine=REF_AFFINE)
    save_deformation_h5(_warp_img(grid), THETA, _native_img(), str(plain), **kwargs)
    save_deformation_h5(
        nib.Nifti1Image(grid[:, :, :, None, :].astype(np.float32), WARP_AFFINE),
        pd.DataFrame(THETA), _native_img(), str(fancy), **kwargs,
    )
    points = _ref_points()
    np.testing.assert_allclose(
        _apply_composite(str(fancy), points), _apply_composite(str(plain), points), atol=1e-6
    )
