"""Reading and writing registration transforms.

- ITK/ANTs plain-text affines (``#Insight Transform File V1.0``), the format
  fMRIPrep uses for its rigid and affine transforms;
- ITK ``CompositeTransform`` HDF5 files (``*_xfm.h5``) and SPM12 deformation
  fields (``y_*.nii``), built from deepmriprep's normalised sampling grids.

Kept apart from :mod:`t1prep.segment`: the module itself needs only numpy,
nibabel and scipy (plus h5py for the composite writer), not the segmentation
models.
"""

from __future__ import annotations

import platform

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

__all__ = [
    "save_affine_itk_txt",
    "load_affine_itk_txt",
    "save_deformation_spm",
    "save_deformation_h5",
]


# ---------------------------------------------------------------------------
# Plain-text affines
# ---------------------------------------------------------------------------


def save_affine_itk_txt(affine_ras: np.ndarray, out_path: str) -> None:
    """Save a 4x4 RAS affine matrix as an ITK/ANTs plain-text transform file.

    Writes the ``#Insight Transform File V1.0`` format used by ANTs, ITK,
    and fMRIPrep.  Handles the RAS→LPS coordinate-system conversion: the
    3x3 rotation/scaling block and the translation vector are both negated
    on their x and y components before writing.

    Args:
        affine_ras: 4x4 affine matrix in RAS coordinates (e.g. the T1w-to-MNI
            registration matrix returned by deepmriprep).
        out_path: Output file path (should end with ``.txt``).
    """
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    M_lps = ras2lps @ affine_ras[:3, :3] @ ras2lps
    T_lps = ras2lps @ affine_ras[:3, 3]
    params = np.concatenate([M_lps.ravel(order="C"), T_lps])
    with open(out_path, "w") as fh:
        fh.write("#Insight Transform File V1.0\n")
        fh.write("#Transform 0\n")
        fh.write("Transform: AffineTransform_float_3_3\n")
        fh.write("Parameters: " + " ".join(f"{v:.10g}" for v in params) + "\n")
        fh.write("FixedParameters: 0 0 0\n")


def load_affine_itk_txt(path: str) -> np.ndarray:
    """Read an ITK plain-text affine back into a 4x4 RAS matrix.

    Args:
        path: An ``#Insight Transform File V1.0`` file holding a single
            affine or rigid transform.

    Returns:
        The 4x4 transform in RAS coordinates.

    Raises:
        ValueError: If the file carries no ``Parameters:`` line.
    """
    parameters = center = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("Parameters:"):
                parameters = np.array(line.split(":", 1)[1].split(), dtype=np.float64)
            elif line.startswith("FixedParameters:"):
                center = np.array(line.split(":", 1)[1].split(), dtype=np.float64)
    if parameters is None:
        raise ValueError(f"No 'Parameters:' line in {path}")

    matrix, translation = parameters[:9].reshape(3, 3), parameters[9:12]
    center = np.zeros(3) if center is None or center.size < 3 else center[:3]
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    out = np.eye(4)
    out[:3, :3] = ras2lps @ matrix @ ras2lps
    # ITK rotates about `center`; fold that back into a plain translation.
    out[:3, 3] = ras2lps @ (center + translation - matrix @ center)
    return out


# ---------------------------------------------------------------------------
# Deformation fields
# ---------------------------------------------------------------------------


#: Reference grid the displacement fields are written on: the grid of
#: ``tpl-MNI152NLin2009cAsym_res-01``, which is what ANTs uses for the warp
#: it stores in fMRIPrep's ``*_mode-image_xfm.h5``.  T1Prep's own warp lives on
#: the smaller 1.5 mm ``Template_4_GS`` grid; resampling it onto the full MNI
#: field of view matters because ITK returns a *zero* displacement for points
#: outside a displacement field's buffer, which would tear the transform at the
#: template edge when fMRIPrep normalises whole-head BOLD data.
MNI152NLIN2009CASYM_SHAPE = (193, 229, 193)
MNI152NLIN2009CASYM_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -96.0],
        [0.0, 1.0, 0.0, -132.0],
        [0.0, 0.0, 1.0, -78.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

#: RAS (NIfTI) to LPS (ITK/ANTs) axis flip.
_RAS2LPS = np.diag([-1.0, -1.0, 1.0])


def _normalized_grid_to_ras(img: nib.Nifti1Image) -> np.ndarray:
    """Map ``torch`` sampling-grid coordinates of ``img`` to RAS millimetres.

    deepmriprep expresses every spatial mapping as a :func:`torch.nn.functional.
    grid_sample` grid: coordinates are normalised to ``[-1, 1]`` with
    ``align_corners=True``, and the last axis is ordered ``(x, y, z)`` where
    ``x`` indexes the *fastest* tensor dimension.  Because ``nifti_to_tensor``
    feeds the RAS-canonical array to torch unpermuted, that ``x`` is array axis
    2 and ``z`` is array axis 0 — the reverse of the NIfTI axis order.

    Args:
        img: Image defining the grid (only its shape and affine are used).

    Returns:
        A 4×4 matrix mapping homogeneous ``(x, y, z)`` grid coordinates to RAS
        millimetres.
    """
    img = nib.as_closest_canonical(img)
    n0, n1, n2 = img.shape[:3]
    grid_to_voxel = np.array(
        [
            [0.0, 0.0, (n0 - 1) / 2, (n0 - 1) / 2],
            [0.0, (n1 - 1) / 2, 0.0, (n1 - 1) / 2],
            [(n2 - 1) / 2, 0.0, 0.0, (n2 - 1) / 2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return img.affine @ grid_to_voxel


def _warp_to_displacement(warp_nii: nib.Nifti1Image):
    """Convert a deepmriprep sampling grid into an RAS displacement field.

    ``warp_xy`` / ``warp_yx`` hold *absolute* normalised sampling coordinates,
    not displacements.  ITK wants a displacement in millimetres, so subtract
    each voxel's own position after converting the grid to RAS.

    Args:
        warp_nii: ``warp_xy`` or ``warp_yx`` as returned by ``run_warp_register``.

    Returns:
        Tuple ``(displacement, img)`` where ``displacement`` has shape
        ``(x, y, z, 3)`` in RAS millimetres and ``img`` is the RAS-canonical
        warp image defining its grid.
    """
    img = nib.as_closest_canonical(warp_nii)
    grid = np.asarray(img.dataobj, dtype=np.float64)
    if grid.ndim == 5:
        grid = grid[:, :, :, 0, :]

    grid_to_ras = _normalized_grid_to_ras(img)
    target = grid @ grid_to_ras[:3, :3].T + grid_to_ras[:3, 3]

    index = np.indices(grid.shape[:3], dtype=np.float64)
    source = np.einsum("ij,jxyz->xyzi", img.affine[:3, :3], index) + img.affine[:3, 3]
    return target - source, img


def save_deformation_spm(
    warp_nii: nib.Nifti1Image,
    affine_norm: np.ndarray,
    native_img: nib.Nifti1Image,
    out_path: str,
) -> None:
    """Save the T1w-to-MNI mapping as an SPM12-compatible deformation field.

    ``warp_xy`` on its own is only the non-linear half of the registration, in
    normalised ``[-1, 1]`` sampling coordinates, stored 4-D.  SPM needs all
    three of those things different:

    * the **linear stage composed in** -- without it the field lands on the
      wrong anatomy and roughly half the brain falls outside the native image;
    * **millimetres** in the native image's world space, not normalised
      coordinates;
    * a **5-D** ``[X, Y, Z, 1, 3]`` array.  ``spm_deformations:get_def`` reads
      ``Nii.dat(:,:,:,1,:)`` and then indexes ``d(4)`` and ``d(5)``; given a 4-D
      field that read collapses to 3-D and SPM fails with "Index exceeds the
      number of array elements. Index must not exceed 3."

    The grid stays the warp's own -- MNI space -- because SPM's
    ``Normalise: Write`` derives its default output bounding box from the
    deformation's geometry, and the values are the native millimetres each of
    those output voxels pulls from.

    Args:
        warp_nii: ``warp_xy`` as returned by ``run_warp_register``.
        affine_norm: The 4x4 normalised affine from ``run_affine_register``,
            mapping template-space grid coordinates to native ones.
        native_img: Image defining the native grid (``mask``).
        out_path: Output ``.nii`` path.
    """
    img = nib.as_closest_canonical(warp_nii)
    grid = np.asarray(img.dataobj, dtype=np.float64)
    if grid.ndim == 5:
        grid = grid[:, :, :, 0, :]

    affine_norm = np.asarray(
        affine_norm.values if hasattr(affine_norm, "values") else affine_norm,
        dtype=np.float64,
    )
    # grid holds template-normalised sampling coordinates; affine_norm carries
    # those to native-normalised ones and the native grid-to-RAS matrix to
    # millimetres.  The warp's own grid-to-RAS cancels out of the composition
    # that save_deformation_h5 uses, which is why it does not appear here.
    to_mm = _normalized_grid_to_ras(nib.as_closest_canonical(native_img)) @ affine_norm
    mm = grid @ to_mm[:3, :3].T + to_mm[:3, 3]

    out = np.ascontiguousarray(mm[:, :, :, None, :], dtype=np.float32)
    nib.save(nib.Nifti1Image(out, img.affine), out_path)


def _resample_displacement(
    displacement: np.ndarray,
    warp_img: nib.Nifti1Image,
    out_shape,
    out_affine: np.ndarray,
    pre_displacement=None,
    post_displacement=None,
) -> np.ndarray:
    """Interpolate an RAS displacement field onto another grid.

    Extrapolates by edge replication rather than with zeros, so the field stays
    continuous where the output grid reaches past T1Prep's template.

    Args:
        displacement: ``(x, y, z, 3)`` RAS displacement on ``warp_img``'s grid.
        warp_img: RAS-canonical image defining ``displacement``'s grid.
        out_shape: Target grid shape.
        out_affine: Target grid RAS affine.
        pre_displacement: RAS displacement field applied to each output point
            *before* T1Prep's warp is looked up.  Used to retarget the
            composite at another template by stepping through the fixed
            template-to-template warp first.
        post_displacement: RAS displacement field applied to the mapped point
            *after* T1Prep's warp, for the opposite direction.

    Returns:
        The displacement resampled to ``(*out_shape, 3)``.
    """
    ras_to_warp = np.linalg.inv(warp_img.affine)
    out = np.empty((*out_shape, 3), dtype=np.float32)

    def sample(field_img, ras):
        """Look an RAS displacement field up at RAS points."""
        field = nib.as_closest_canonical(field_img)
        data = np.asarray(field.dataobj, dtype=np.float64)
        if data.ndim == 5:
            data = data[:, :, :, 0, :]
        inverse = np.linalg.inv(field.affine)
        vox = np.einsum("ij,jxyz->ixyz", inverse[:3, :3], ras) + inverse[:3, 3, None, None, None]
        return np.stack(
            [map_coordinates(data[..., c], vox, order=1, mode="nearest") for c in range(3)]
        )

    # One slab at a time: the coordinate arrays for the full 193x229x193 grid
    # would be several hundred megabytes on their own.
    for i in range(out_shape[0]):
        index = np.indices((1, *out_shape[1:]), dtype=np.float64)
        index[0] += i
        ras = np.einsum("ij,jxyz->ixyz", out_affine[:3, :3], index) + out_affine[:3, 3, None, None, None]
        if pre_displacement is None and post_displacement is None:
            # Same result as the general path below, but without forming
            # (ras + d) - ras, whose cancellation perturbs the stored float32
            # by ~1e-6 mm.  Keeps the single-template output bit-exact.
            vox = np.einsum("ij,jxyz->ixyz", ras_to_warp[:3, :3], ras) + ras_to_warp[:3, 3, None, None, None]
            for c in range(3):
                out[i, ..., c] = map_coordinates(
                    displacement[..., c], vox, order=1, mode="nearest"
                )[0]
            continue

        source = ras
        if pre_displacement is not None:
            # Step into the space T1Prep's warp is defined on before looking
            # it up, so the stored displacement spans both hops.
            source = ras + sample(pre_displacement, ras)
        vox = np.einsum("ij,jxyz->ixyz", ras_to_warp[:3, :3], source) + ras_to_warp[:3, 3, None, None, None]
        target = source + np.stack(
            [map_coordinates(displacement[..., c], vox, order=1, mode="nearest") for c in range(3)]
        )
        if post_displacement is not None:
            target = target + sample(post_displacement, target)
        out[i] = np.moveaxis(target - ras, 0, -1)[0]
    return out


def _itk_affine(affine_ras: np.ndarray):
    """Build the ITK datasets for an ``AffineTransform_float_3_3``.

    Args:
        affine_ras: 4×4 affine in RAS millimetres.

    Returns:
        Tuple ``(transform_type, fixed_parameters, parameters)``.
    """
    matrix = _RAS2LPS @ affine_ras[:3, :3] @ _RAS2LPS
    translation = _RAS2LPS @ affine_ras[:3, 3]
    return (
        "AffineTransform_float_3_3",
        np.zeros(3),  # centre of rotation
        np.concatenate([matrix.ravel(order="C"), translation]),
    )


def _itk_displacement_field(displacement: np.ndarray, affine: np.ndarray):
    """Build the ITK datasets for a ``DisplacementFieldTransform_float_3_3``.

    Args:
        displacement: ``(x, y, z, 3)`` RAS displacement field.
        affine: RAS affine of the field's grid.

    Returns:
        Tuple ``(transform_type, fixed_parameters, parameters)``.
    """
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    direction = (_RAS2LPS @ affine[:3, :3]) / spacing
    origin = _RAS2LPS @ affine[:3, 3]
    fixed = np.concatenate(
        [
            np.array(displacement.shape[:3], dtype=np.float64),
            origin,
            spacing,
            direction.ravel(order="C"),
        ]
    )
    # ITK buffers vector images component-fastest, then x, then y, then z.
    # ``ravel`` already copies, so flip x and y to LPS on its result rather than
    # on a second full-size copy of the field.
    parameters = np.moveaxis(displacement, -1, 0).ravel(order="F")
    parameters[0::3] *= -1.0
    parameters[1::3] *= -1.0
    return "DisplacementFieldTransform_float_3_3", fixed, parameters


def _write_itk_composite(out_path: str, transforms) -> None:
    """Write an ITK ``CompositeTransform`` HDF5 file.

    Mirrors the layout ITK 5.x writes from ANTs: variable-length ASCII
    type strings, ``float64`` fixed parameters, gzip-compressed ``float32``
    parameters, and a group ``0`` that carries only the composite's type.

    Args:
        out_path: Destination ``.h5`` path.
        transforms: ``(type, fixed_parameters, parameters)`` tuples in ITK queue
            order.  ITK applies the queue back to front, so the transform that
            acts *first* is written last.

    Raises:
        ImportError: If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to save deformation fields as HDF5. "
            "Install it with: pip install h5py"
        ) from exc

    string_dtype = h5py.string_dtype(encoding="ascii")

    def write_string(parent, name, value):
        dataset = parent.create_dataset(name, (1,), dtype=string_dtype)
        dataset[0] = value

    with h5py.File(out_path, "w") as hf:
        write_string(hf, "HDFVersion", h5py.version.hdf5_version)
        # Declares the format generation, not a library T1Prep links against;
        # ITK and nitransforms record it but do not check it on read.
        write_string(hf, "ITKVersion", "5.4.0")
        write_string(hf, "OSName", platform.system())
        write_string(hf, "OSVersion", platform.release())
        group = hf.create_group("TransformGroup")
        write_string(group.create_group("0"), "TransformType",
                     "CompositeTransform_float_3_3")
        for i, (transform_type, fixed, parameters) in enumerate(transforms, start=1):
            sub = group.create_group(str(i))
            write_string(sub, "TransformType", transform_type)
            sub.create_dataset(
                "TransformFixedParameters", data=np.asarray(fixed, dtype=np.float64)
            )
            parameters = np.asarray(parameters, dtype=np.float32)
            sub.create_dataset(
                "TransformParameters",
                data=parameters,
                chunks=(min(parameters.size, 1048576),),
                compression="gzip",
                compression_opts=5,
            )


def save_deformation_h5(
    warp_nii: nib.Nifti1Image,
    affine_norm: np.ndarray,
    native_img: nib.Nifti1Image,
    out_path: str,
    inverse: bool = False,
    ref_shape=MNI152NLIN2009CASYM_SHAPE,
    ref_affine: np.ndarray = MNI152NLIN2009CASYM_AFFINE,
    template_displacement=None,
) -> None:
    """Save T1Prep's registration as an ANTs/ITK-compatible composite HDF5 file.

    T1Prep splits the T1w-to-MNI mapping the same way ANTs does — a linear stage
    followed by a non-linear one — but expresses both as ``torch`` sampling
    grids in normalised ``[-1, 1]`` coordinates rather than as millimetre
    displacements.  This writes the pair out as the two-element ITK
    ``CompositeTransform`` that ``antsApplyTransforms`` and fMRIPrep consume.

    Following ITK, the stored transforms map points from the *output* image's
    space back to the *input* image's space, which is the opposite of the
    direction the BIDS ``from-``/``to-`` filename entities name.  So the default
    (``inverse=False``) writes the ``from-T1w_to-MNI152NLin2009cAsym`` file — the
    one that resamples T1w *images* into MNI — from ``warp_xy``, whose point
    mapping runs MNI to T1w.

    Args:
        warp_nii: ``warp_xy`` when ``inverse`` is False, ``warp_yx`` otherwise.
        affine_norm: The 4×4 normalised affine from ``run_affine_register``
            (deepmriprep's ``affine``), mapping template-space grid coordinates
            to native ones.
        native_img: Image defining the native grid, i.e. the one T1Prep writes
            its native-space outputs on (``mask``).
        out_path: Output file path (should end with ``.h5``).
        inverse: Write the ``from-MNI152NLin2009cAsym_to-T1w`` composite.
        ref_shape: Grid shape for the stored displacement field.
        ref_affine: RAS affine of the grid for the stored displacement field.
        template_displacement: A fixed template-to-template RAS displacement
            field, to retarget the composite at a second template.  It is
            applied before T1Prep's warp in the forward direction and after it
            in the inverse one, so the same field serves both -- pass the one
            whose direction matches ``inverse``.
    """
    affine_norm = np.asarray(
        affine_norm.values if hasattr(affine_norm, "values") else affine_norm,
        dtype=np.float64,
    )
    displacement, warp_img = _warp_to_displacement(warp_nii)

    # The linear stage in millimetres.  The intermediate space it maps from is
    # the warp template's physical space, which is also what the field above
    # maps into, so the two compose exactly.
    affine_ras = (
        _normalized_grid_to_ras(native_img)
        @ affine_norm
        @ np.linalg.inv(_normalized_grid_to_ras(warp_img))
    )

    field = _itk_displacement_field(
        _resample_displacement(
            displacement,
            warp_img,
            ref_shape,
            ref_affine,
            pre_displacement=None if inverse else template_displacement,
            post_displacement=template_displacement if inverse else None,
        ),
        np.asarray(ref_affine, dtype=np.float64),
    )
    if inverse:
        # Points run native -> linear -> non-linear, so the affine is applied
        # first and therefore written last.
        _write_itk_composite(out_path, [field, _itk_affine(np.linalg.inv(affine_ras))])
    else:
        _write_itk_composite(out_path, [_itk_affine(affine_ras), field])
