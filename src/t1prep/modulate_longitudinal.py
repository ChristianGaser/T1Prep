"""Longitudinal Jacobian modulation: CAT12's ageing model, minus the average segmentation.

:mod:`t1prep.warp_longitudinal` produces one small diffeomorphism per time point
towards an unbiased subject average.  This module turns those into the maps a
longitudinal VBM analysis is actually run on.

What makes longitudinal VBM more sensitive than running cross-sectional VBM
twice is not the registration on its own -- it is that everything *except* the
between-scan volume change is held fixed.  CAT12 does that by sharing two
things across the time points of a subject:

1. **One tissue map.**  CAT12 segments the subject average once and reuses that
   segmentation for every time point, so segmentation differences cannot leak
   into the longitudinal contrast.
2. **One spatial normalisation.**  The average's warp to MNI is applied to every
   time point, so normalisation differences cannot leak in either.

Each time point then differs *only* by its longitudinal Jacobian.  That is the
whole design, and dropping either half gives back most of the noise the
pipeline exists to remove: two independently estimated cross-sectional warps of
the same subject differ by more than the atrophy between the scans.

This module reproduces both properties without segmenting the average, which is
the one deviation from CAT12:

* the shared tissue map is the mean of the time points' own segmentations
  carried into average space, ``pbar = mean_i p_i(phi_i(x))``, standing in for
  CAT12's segmentation of the average image;
* the shared normalisation is the mean of the per-time-point SPM ``y_`` fields,
  standing in for the average's warp to MNI.

Both are averages rather than a chosen reference, which keeps the unbiasedness
that the rigid stage's SE(3) barycentre and the velocity re-centring establish.

The output for time point ``i`` is, for every MNI voxel ``x``::

    mwmwp_i(x) = pbar(v) * det J_phi_i(v) * det J_y(x),    v = A_work^-1 y(x)

Two Jacobians, hence the doubled ``mw`` in the output name -- the same
convention CAT12's ageing model uses.

The modulation moves the volume change out of the shape and into the
intensity, where a voxel-wise test can see it.  With a shared tissue map the
integral reproduces that time point's own native tissue volume to the extent
that the registration explains the between-scan difference -- measured within
0.1 % on a synthetic series where it does.

Caveat worth carrying into any analysis: because the tissue map is shared, this
measures volume change *as located by the registration*.  It cannot represent a
change the deformation model does not express, and the membrane prior in
``warp_longitudinal`` shrinks the estimate, so the maps read as a spatial
pattern rather than a calibrated absolute rate.

CLI usage
---------
Run after T1Prep has processed every time point::

    python -m t1prep.modulate_longitudinal \\
        --tissue tp1/mri/p1tp1.nii tp2/mri/p1tp2.nii \\
        --displacement tp1/mri/tp1_desc-longDisplacement.nii.gz ... \\
        --log-jacobian tp1/mri/tp1_desc-longLogJacobian.nii.gz ... \\
        --deformation tp1/mri/y_tp1.nii tp2/mri/y_tp2.nii \\
        --out-dir DIR

References
----------
Ashburner J, Ridgway GR (2013). Symmetric diffeomorphic modeling of
longitudinal structural MRI. *Front Neurosci* 6:197.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from scipy.ndimage import map_coordinates

from .realign_longitudinal import _split_nifti_name
from .utils import get_filenames

__all__ = [
    "ModulationOutputs",
    "modulate_longitudinal",
    "output_name",
    "resolve_inputs",
    "run_cli",
]


@dataclass
class ModulationOutputs:
    """Results of longitudinal Jacobian modulation.

    Attributes:
        modulated: One modulated tissue map per time point, on the MNI grid the
            ``y_`` fields are defined on.
        shared_tissue: The shared tissue map in average space, i.e. this
            pipeline's stand-in for CAT12's segmentation of the average.
        mni_affine: Grid-to-RAS affine of the MNI grid.
        native_volumes_mm3: Per time point, the integral of the modulated map.
            Modulation is volume preserving, so with ``tissue_source='timepoint'``
            these reproduce each time point's native tissue volume exactly; with
            the shared map they do so insofar as the registration explains the
            between-scan difference.
    """

    modulated: List[np.ndarray]
    shared_tissue: np.ndarray
    mni_affine: np.ndarray
    native_volumes_mm3: List[float]


def _load_deformation(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read an SPM ``y_`` field.

    ``save_deformation_spm`` writes a 5-D ``[X, Y, Z, 1, 3]`` array on the
    *template* grid whose values are the native world millimetres each MNI
    voxel pulls from.

    Args:
        path: Path to the ``y_*.nii`` file.

    Returns:
        ``(field, affine)`` where ``field`` is ``(X, Y, Z, 3)`` in native
        millimetres and ``affine`` is the MNI grid's grid-to-RAS matrix.

    Raises:
        ValueError: If the file is not a 4-D or 5-D deformation field.
    """
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim == 5:
        data = data[:, :, :, 0, :]
    elif data.ndim != 4:
        raise ValueError(
            f"{path}: expected a 4-D or 5-D SPM deformation field, got shape {data.shape}"
        )
    if data.shape[-1] != 3:
        raise ValueError(f"{path}: deformation must have 3 components, got {data.shape[-1]}")
    return data, np.asarray(img.affine, dtype=np.float64)


def _jacobian_determinant(field: np.ndarray, grid_affine: np.ndarray) -> np.ndarray:
    """Volume ratio of a pull-back deformation field.

    Args:
        field: ``(X, Y, Z, 3)`` giving, per grid voxel, the source point in
            world millimetres.
        grid_affine: The field's own grid-to-RAS affine.

    Returns:
        ``(X, Y, Z)`` of ``|det d(source mm) / d(target mm)|``.  Differentiating
        against voxel indices and dividing by the grid's own determinant is what
        turns a per-voxel derivative into the millimetre-to-millimetre ratio
        modulation needs.
    """
    jac = np.empty(field.shape[:3] + (3, 3), dtype=np.float64)
    for axis in range(3):
        jac[..., axis] = np.gradient(field, axis=axis)
    det = np.linalg.det(jac)
    return np.abs(det / np.linalg.det(grid_affine[:3, :3]))


def _fit_affine_determinant(field: np.ndarray, grid_affine: np.ndarray) -> float:
    """Volume factor of the affine part of a deformation field.

    CAT12 lets modulation use the full transform or only its non-linear part;
    the latter divides out overall head size.  Separating them means fitting the
    affine that best explains the field and taking its determinant.

    Args:
        field: ``(X, Y, Z, 3)`` in world millimetres.
        grid_affine: The field's grid-to-RAS affine.

    Returns:
        ``|det|`` of the fitted millimetre-to-millimetre linear part.
    """
    shape = field.shape[:3]
    # A regular subsample is plenty: the fit has 12 parameters.
    step = tuple(max(1, n // 24) for n in shape)
    idx = np.stack(
        np.meshgrid(
            *[np.arange(0, n, s, dtype=np.float64) for n, s in zip(shape, step)],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    target_mm = idx @ grid_affine[:3, :3].T + grid_affine[:3, 3]
    source_mm = field[:: step[0], :: step[1], :: step[2]].reshape(-1, 3)
    design = np.concatenate([target_mm, np.ones((target_mm.shape[0], 1))], axis=1)
    solution, *_ = np.linalg.lstsq(design, source_mm, rcond=None)
    return float(abs(np.linalg.det(solution[:3, :])))


def _sample_world(img: nib.Nifti1Image, world_mm: np.ndarray, order: int = 1) -> np.ndarray:
    """Sample ``img`` at world-millimetre points.

    Args:
        img: Image to sample.
        world_mm: ``(..., 3)`` RAS millimetre coordinates.
        order: Spline order for :func:`scipy.ndimage.map_coordinates`.

    Returns:
        Sampled values with ``world_mm``'s leading shape.  Points outside the
        image return 0, which is the right answer for a probability map.
    """
    inv = np.linalg.inv(np.asarray(img.affine, dtype=np.float64))
    flat = world_mm.reshape(-1, 3)
    vox = flat @ inv[:3, :3].T + inv[:3, 3]
    data = np.asarray(img.dataobj, dtype=np.float32)
    sampled = map_coordinates(data, vox.T, order=order, mode="constant", cval=0.0)
    return sampled.reshape(world_mm.shape[:-1])


def _world_grid(shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    """Return ``(X, Y, Z, 3)`` world millimetres for every voxel of a grid."""
    idx = np.stack(
        np.meshgrid(*[np.arange(int(n), dtype=np.float64) for n in shape], indexing="ij"),
        axis=-1,
    )
    return idx @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]



#: Legacy tissue-map prefixes, in T1Prep's (SPM's) numbering.
_LEGACY_CLASSES = {"1": "GM", "2": "WM", "3": "CSF"}


def _template_affine() -> Optional[np.ndarray]:
    """Affine of the grid T1Prep writes its other MNI-space volumes on.

    ``mwp1``/``mwp2`` land on deepmriprep's ``Template_4_GS`` grid, whose first
    axis is stored reversed relative to the RAS-canonical ``y_`` field this
    stage computes on (``save_deformation_spm`` canonicalises it).  Both
    describe the same physical lattice, so writing on the ``y_`` grid is
    geometrically correct -- but the array would be stored mirrored with
    respect to every other volume in the folder, and a naive voxel-wise
    comparison against ``mwp1`` would then silently compare mirrored brains.

    Returns:
        The 4x4 affine, or ``None`` when the template cannot be located, in
        which case the caller keeps the ``y_`` grid.
    """
    try:
        from deepmriprep.utils import DATA_PATH

        path = os.path.join(str(DATA_PATH), "templates", "Template_4_GS.nii.gz")
        return np.asarray(nib.load(path).affine, dtype=np.float64)
    except Exception:
        return None


def _reorient(volumes: List[np.ndarray], affine: np.ndarray, target: np.ndarray):
    """Re-store ``volumes`` in ``target``'s axis order.

    This is a pure permutation and flip of the array -- no interpolation, no
    resampling -- because the two grids are the same lattice written with
    different axis directions.

    Args:
        volumes: Arrays on the grid described by ``affine``.
        affine: The volumes' current grid-to-RAS affine.
        target: The affine to store them under.

    Returns:
        ``(volumes, affine)``, unchanged when the orientations already agree or
        when the grids are not the same shape.
    """
    transform = ornt_transform(io_orientation(affine), io_orientation(target))
    if np.array_equal(transform, io_orientation(np.eye(4))):
        return volumes, affine
    moved = [apply_orientation(v, transform) for v in volumes]
    if moved and moved[0].shape != volumes[0].shape:
        # A permutation would change the shape; only same-shape grids are safe
        # to relabel with the target affine.
        return volumes, affine
    return moved, target


def _parse_tissue_name(path: str):
    """Recover the naming scheme, base name and tissue class from a tissue map.

    Args:
        path: A native-space tissue probability map written by T1Prep.

    Returns:
        ``(use_bids, bname, tissue_class, ext)``, or ``None`` when the filename
        matches neither scheme -- in which case the caller has no basis for
        constructing a table-derived name and should fall back.
    """
    stem, ext = _split_nifti_name(path)
    ext = ext.lstrip(".")
    match = re.match(r"^(?P<bname>.+)_label-(?P<cls>GM|WM|CSF)_probseg$", stem)
    if match:
        return True, match.group("bname"), match.group("cls"), ext
    match = re.match(r"^p(?P<n>[123])(?P<bname>.+)$", stem)
    if match:
        return False, match.group("bname"), _LEGACY_CLASSES[match.group("n")], ext
    return None


def output_name(tissue_path: str) -> str:
    """Build the output filename for a modulated longitudinal tissue map.

    The name comes from T1Prep's own naming table, with the modulation marker
    **doubled** -- ``mwmwp1<name>.nii`` -- which is what CAT12's ageing model
    writes.  The doubling is not decoration: these maps really are modulated
    twice, once by the longitudinal Jacobian and once by the spatial
    normalisation, and a single ``mw`` would both understate that and collide
    with the cross-sectional ``mwp1`` T1Prep writes into the same folder.

    In the BIDS scheme the marker inside the space token is ``-modulated``, so
    that is what gets doubled, keeping the two schemes consistent.

    Args:
        tissue_path: The native tissue map this output was derived from.

    Returns:
        A filename such as ``mwmwp1<name>.nii`` (legacy) or
        ``<name>_space-..-modulated-modulated_label-GM_probseg.nii`` (BIDS).
        Falls back to ``<stem>_desc-longModulated<ext>`` for a tissue map whose
        name matches neither scheme.
    """
    parsed = _parse_tissue_name(tissue_path)
    if parsed is None:
        stem, ext = _split_nifti_name(tissue_path)
        return f"{stem}_desc-longModulated{ext}"
    use_bids, bname, tissue_class, ext = parsed
    space = get_filenames(use_bids, bname, "", "", "", ext).get("Warp_modulated_space", "")
    if space:
        # Legacy's whole token is the marker ("mw"); BIDS carries it as a
        # trailing "-modulated" inside a longer space name.
        space = space + "-modulated" if use_bids else space + space
    name = get_filenames(use_bids, bname, "", "", space, ext).get(f"{tissue_class}_volume", "")
    if not name:
        stem, raw_ext = _split_nifti_name(tissue_path)
        return f"{stem}_desc-longModulated{raw_ext}"
    return name


def resolve_inputs(
    mri_dirs: Sequence[str],
    basenames: Sequence[str],
    tissue_class: str = "GM",
    long_dirs: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Find the four input sets from each time point's output directory.

    T1Prep names its outputs by either the legacy SPM scheme (``p1<name>.nii``,
    ``y_<name>.nii``) or the BIDS one, chosen from the output path, and the
    longitudinal fields carry a third set of suffixes.  Resolving that in the
    shell would mean reimplementing :func:`t1prep.utils.get_filenames`, so the
    lookup lives here and the pipeline only has to pass directories and names.

    Args:
        mri_dirs: One directory per time point holding T1Prep's outputs -- the
            native tissue maps and the ``y_`` deformation.
        basenames: The matching input basenames, without a NIfTI extension.
        tissue_class: ``"GM"``, ``"WM"`` or ``"CSF"``.
        long_dirs: Where ``warp_longitudinal`` wrote its displacement and log
            Jacobian, when that is not ``mri_dirs``.  The pipeline runs the
            warp stage before T1Prep, so its outputs sit beside the realigned
            volumes rather than beside T1Prep's; defaults to ``mri_dirs``.

    Returns:
        ``(tissue, displacement, log_jacobian, deformation)`` path lists.

    Raises:
        ValueError: If the two lists differ in length, or listing what is
            missing when a file cannot be found.
    """
    if len(mri_dirs) != len(basenames):
        raise ValueError(
            f"Need one basename per directory ({len(basenames)} vs {len(mri_dirs)})"
        )
    if long_dirs is None:
        long_dirs = list(mri_dirs)
    elif len(long_dirs) != len(mri_dirs):
        raise ValueError(
            f"Need one longitudinal directory per time point "
            f"({len(long_dirs)} vs {len(mri_dirs)})"
        )

    def _first_existing(directory: str, candidates: Sequence[str]) -> Optional[str]:
        for name in candidates:
            if not name:
                continue
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return path
        return None

    tissue, displacement, log_jacobian, deformation = [], [], [], []
    missing: List[str] = []
    for directory, long_dir, bname in zip(mri_dirs, long_dirs, basenames):
        # Try both naming schemes and both extensions rather than guessing which
        # one this run used.
        names = {"tissue": [], "deformation": []}
        for use_bids in (False, True):
            for ext in ("nii.gz", "nii"):
                table = get_filenames(use_bids, bname, "", "", "", ext)
                names["tissue"].append(table.get(f"{tissue_class}_volume", ""))
                names["deformation"].append(table.get("Def_volume", ""))

        found = {
            "tissue": _first_existing(directory, names["tissue"]),
            "deformation": _first_existing(directory, names["deformation"]),
            "displacement": _first_existing(
                long_dir,
                [f"{bname}_desc-longDisplacement.{e}" for e in ("nii.gz", "nii")],
            ),
            "log_jacobian": _first_existing(
                long_dir,
                [f"{bname}_desc-longLogJacobian.{e}" for e in ("nii.gz", "nii")],
            ),
        }
        for key, value in found.items():
            where = long_dir if key in ("displacement", "log_jacobian") else directory
            if value is None:
                missing.append(f"{key} for '{bname}' in {where}")
        tissue.append(found["tissue"])
        displacement.append(found["displacement"])
        log_jacobian.append(found["log_jacobian"])
        deformation.append(found["deformation"])

    if missing:
        raise ValueError(
            "Could not find the longitudinal modulation inputs:\n  "
            + "\n  ".join(missing)
            + "\n\nThe native segmentation needs '--t1prep-arg --p', and the "
            "longitudinal fields need warp_longitudinal to have run with "
            "--save-displacement."
        )
    return tissue, displacement, log_jacobian, deformation


def modulate_longitudinal(
    tissue_paths: Sequence[str],
    displacement_paths: Sequence[str],
    log_jacobian_paths: Sequence[str],
    deformation_paths: Sequence[str],
    *,
    modulation: str = "full",
    tissue_source: str = "shared",
    verbose: bool = False,
) -> ModulationOutputs:
    """Modulate each time point's tissue map by its longitudinal Jacobian.

    Args:
        tissue_paths: Native-space tissue probability maps, one per time point,
            in the same order the longitudinal registration used.  These are
            T1Prep's own ``p1``/``p2`` outputs for the realigned volumes.
        displacement_paths: ``*_desc-longDisplacement`` fields from
            ``warp_longitudinal --save-displacement``, in RAS millimetres.
        log_jacobian_paths: ``*_desc-longLogJacobian`` maps from
            ``warp_longitudinal``.
        deformation_paths: SPM ``y_*`` fields, one per time point.  Their mean
            becomes the single normalisation shared by every time point.
        modulation: ``"full"`` modulates by the whole transform, as CAT12's
            default does; ``"nonlinear"`` divides the affine part out, which
            controls for overall head size.
        tissue_source: ``"shared"`` (default, and what CAT12 does) uses one
            tissue map for every time point so the contrast carries only the
            Jacobian; ``"timepoint"`` keeps each time point's own segmentation,
            which reintroduces segmentation differences into the contrast.
        verbose: Print per-time-point volumes.

    Returns:
        A :class:`ModulationOutputs`.

    Raises:
        ValueError: If the input lists have different lengths, if fewer than two
            time points are given, or if an option value is unknown.
    """
    counts = {
        "tissue": len(tissue_paths),
        "displacement": len(displacement_paths),
        "log-jacobian": len(log_jacobian_paths),
        "deformation": len(deformation_paths),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"One entry per time point is required for each input: {counts}")
    n_tp = len(tissue_paths)
    if n_tp < 2:
        raise ValueError("Longitudinal modulation needs at least two time points")
    if modulation not in ("full", "nonlinear"):
        raise ValueError(f"--modulation must be 'full' or 'nonlinear' (got: {modulation})")
    if tissue_source not in ("shared", "timepoint"):
        raise ValueError(
            f"--tissue-source must be 'shared' or 'timepoint' (got: {tissue_source})"
        )

    # ---- the shared normalisation ----------------------------------------
    fields, mni_affine = [], None
    for path in deformation_paths:
        field, affine = _load_deformation(path)
        if mni_affine is None:
            mni_affine, reference_shape = affine, field.shape[:3]
        elif field.shape[:3] != reference_shape or not np.allclose(affine, mni_affine):
            raise ValueError(
                f"{path}: deformation fields must share one grid; "
                f"got {field.shape[:3]} vs {reference_shape}"
            )
        fields.append(field)
    # Averaging absolute target coordinates is a first-order Frechet mean, which
    # is all that is needed: the time points of one subject differ by far less
    # than the curvature of the space of deformations.
    shared_y = np.mean(np.stack(fields, axis=0), axis=0)
    det_cross = _jacobian_determinant(shared_y, mni_affine)
    if modulation == "nonlinear":
        det_cross = det_cross / max(_fit_affine_determinant(shared_y, mni_affine), 1e-12)

    # ---- average space ----------------------------------------------------
    # The longitudinal fields all live on warp_longitudinal's working grid, and
    # the rigid stage already put every time point in one world frame, so that
    # grid *is* the average space.
    logjac_imgs = [nib.load(p) for p in log_jacobian_paths]
    disp_imgs = [nib.load(p) for p in displacement_paths]
    work_affine = np.asarray(logjac_imgs[0].affine, dtype=np.float64)
    work_shape = tuple(int(n) for n in logjac_imgs[0].shape[:3])
    for img in logjac_imgs + disp_imgs:
        if tuple(img.shape[:3]) != work_shape:
            raise ValueError(
                "Longitudinal displacement and log-Jacobian files must share one grid "
                f"({tuple(img.shape[:3])} vs {work_shape})"
            )

    tissue_imgs = [nib.load(p) for p in tissue_paths]
    avg_world = _world_grid(work_shape, work_affine)

    # phi_i carries an average-space point into time point i, so sampling each
    # time point there is what puts them all on comparable anatomy.
    warped_tissue = []
    for img, disp_img in zip(tissue_imgs, disp_imgs):
        disp_mm = np.asarray(disp_img.dataobj, dtype=np.float64)
        if disp_mm.ndim == 5:
            disp_mm = disp_mm[:, :, :, 0, :]
        warped_tissue.append(_sample_world(img, avg_world + disp_mm))
    shared_tissue = np.mean(np.stack(warped_tissue, axis=0), axis=0)

    # ---- push to MNI through the shared normalisation ---------------------
    inv_work = np.linalg.inv(work_affine)
    avg_vox = shared_y @ inv_work[:3, :3].T + inv_work[:3, 3]
    coords = avg_vox.reshape(-1, 3).T

    tissue_on_mni = {}
    if tissue_source == "shared":
        tissue_on_mni["shared"] = map_coordinates(
            shared_tissue, coords, order=1, mode="constant", cval=0.0
        ).reshape(reference_shape)

    voxel_mm3 = float(abs(np.linalg.det(mni_affine[:3, :3])))
    modulated, volumes = [], []
    for idx in range(n_tp):
        if tissue_source == "shared":
            tissue = tissue_on_mni["shared"]
        else:
            tissue = map_coordinates(
                warped_tissue[idx], coords, order=1, mode="constant", cval=0.0
            ).reshape(reference_shape)
        det_long = map_coordinates(
            np.exp(np.asarray(logjac_imgs[idx].dataobj, dtype=np.float64)),
            coords,
            order=1,
            mode="constant",
            cval=1.0,
        ).reshape(reference_shape)
        out = (tissue * det_long * det_cross).astype(np.float32)
        modulated.append(out)
        volumes.append(float(out.sum()) * voxel_mm3)
        if verbose:
            print(
                f"  time point {idx + 1}: modulated volume = {volumes[-1] / 1000.0:.2f} ml"
            )

    # Store the result the way T1Prep stores its other MNI-space volumes, so the
    # longitudinal maps sit in the folder array-comparable with mwp1/mwp2.
    target = _template_affine()
    if target is not None and target.shape == (4, 4):
        modulated, mni_affine = _reorient(modulated, mni_affine, target)

    return ModulationOutputs(
        modulated=modulated,
        shared_tissue=shared_tissue.astype(np.float32),
        mni_affine=mni_affine,
        native_volumes_mm3=volumes,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Modulate longitudinal tissue maps by their Jacobian (CAT12's ageing "
            "model, without segmenting the average)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tissue",
        nargs="+",
        help="Native-space tissue maps (p1/p2), one per time point, in order",
    )
    p.add_argument(
        "--displacement",
        nargs="+",
        help="'*_desc-longDisplacement' fields from warp_longitudinal",
    )
    p.add_argument(
        "--log-jacobian",
        nargs="+",
        help="'*_desc-longLogJacobian' maps from warp_longitudinal",
    )
    p.add_argument(
        "--deformation",
        nargs="+",
        help="SPM 'y_*' fields, one per time point; their mean is shared by all",
    )
    p.add_argument(
        "--mri-dirs",
        nargs="+",
        help=(
            "Instead of the four lists above: one output directory per time point. "
            "Filenames are resolved with T1Prep's own naming table, so this works "
            "for both the legacy and BIDS schemes."
        ),
    )
    p.add_argument(
        "--long-dirs",
        nargs="+",
        help=(
            "Where warp_longitudinal wrote its displacement and log-Jacobian, if "
            "that is not --mri-dirs. The pipeline runs the warp stage before "
            "T1Prep, so those land beside the realigned volumes."
        ),
    )
    p.add_argument(
        "--names",
        nargs="+",
        help="Input basenames without extension, matching --mri-dirs",
    )
    p.add_argument(
        "--tissue-class",
        choices=("GM", "WM", "CSF"),
        default="GM",
        help="Which tissue class to modulate when using --mri-dirs",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help=(
            "Directory for outputs. Names follow T1Prep's table with the "
            "modulation marker doubled, as CAT12's ageing model does: "
            "mwmwp1<name>.nii, or the BIDS equivalent."
        ),
    )
    p.add_argument(
        "--out-subfolders",
        nargs="+",
        help="Optional subfolder per time point, to avoid colliding basenames",
    )
    p.add_argument(
        "--modulation",
        choices=("full", "nonlinear"),
        default="full",
        help="Modulate by the whole transform, or divide out the affine (head size)",
    )
    p.add_argument(
        "--tissue-source",
        choices=("shared", "timepoint"),
        default="shared",
        help=(
            "'shared' reuses one tissue map for every time point so the contrast "
            "carries only the Jacobian, as CAT12 does; 'timepoint' keeps each "
            "time point's own segmentation"
        ),
    )
    p.add_argument(
        "--save-shared-tissue",
        action="store_true",
        help="Also write the shared tissue map in average space",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-time-point volumes")
    return p.parse_args(argv)


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m t1prep.modulate_longitudinal``."""
    args = _parse_args(argv)

    explicit = [args.tissue, args.displacement, args.log_jacobian, args.deformation]
    if args.mri_dirs is not None:
        if any(v is not None for v in explicit):
            raise SystemExit("--mri-dirs cannot be combined with the explicit path lists")
        if args.names is None:
            raise SystemExit("--mri-dirs also needs --names")
        try:
            (
                args.tissue,
                args.displacement,
                args.log_jacobian,
                args.deformation,
            ) = resolve_inputs(
                args.mri_dirs,
                args.names,
                str(args.tissue_class),
                long_dirs=args.long_dirs,
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
    elif args.long_dirs is not None:
        raise SystemExit("--long-dirs only applies together with --mri-dirs")
    elif any(v is None for v in explicit):
        raise SystemExit(
            "Give either --mri-dirs with --names, or all four of --tissue, "
            "--displacement, --log-jacobian and --deformation"
        )

    if args.out_subfolders is not None and len(args.out_subfolders) != len(args.tissue):
        raise SystemExit(
            f"--out-subfolders expects {len(args.tissue)} entries (one per time point); "
            f"got {len(args.out_subfolders)}"
        )

    try:
        outputs = modulate_longitudinal(
            args.tissue,
            args.displacement,
            args.log_jacobian,
            args.deformation,
            modulation=str(args.modulation),
            tissue_source=str(args.tissue_source),
            verbose=bool(args.verbose),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    for idx, path in enumerate(args.tissue):
        dest = args.out_dir
        if args.out_subfolders is not None:
            dest = os.path.join(dest, args.out_subfolders[idx])
        os.makedirs(dest, exist_ok=True)
        nib.save(
            nib.Nifti1Image(outputs.modulated[idx], outputs.mni_affine),
            os.path.join(dest, output_name(path)),
        )

    if args.save_shared_tissue:
        os.makedirs(args.out_dir, exist_ok=True)
        _, ext = _split_nifti_name(args.tissue[0])
        nib.save(
            nib.Nifti1Image(outputs.shared_tissue, nib.load(args.log_jacobian[0]).affine),
            os.path.join(args.out_dir, f"longitudinal_shared_tissue{ext}"),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
