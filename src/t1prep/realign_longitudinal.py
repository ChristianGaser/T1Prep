"""
Rigid realignment of a series of 3D NIfTI volumes to the first volume or 
optionally to the mean.

The implementation mirrors SPM's realign module conceptually. It optimizes a 
6-DOF rigid transform (per subject) directly in world coordinates and optionally 
writes resampled volumes or only updates the output headers and also allows for
an inverse-consistent rigid registration to an unbiased mid-space for N 
timepoints.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes as _voxel_sizes
from nibabel.processing import resample_from_to
from scipy.linalg import expm, logm
from scipy.ndimage import gaussian_filter, map_coordinates, center_of_mass
from scipy.optimize import least_squares


@dataclass
class RigidRealignOutputs:
    """Container for results produced by rigid realignment."""

    reference_img: nib.Nifti1Image
    transforms: List[np.ndarray]  # world transforms applied to original affines
    resampled_in_reference: Optional[List[np.ndarray]] = None


def _rigid_matrix(params: np.ndarray) -> np.ndarray:
    rx, ry, rz, tx, ty, tz = params
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def _choose_samples(volume: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.isfinite(volume) & (volume > 0)
    shape = volume.shape
    total_vox = int(np.prod(shape))
    if n_samples <= 0 or n_samples >= total_vox:
        coords = np.argwhere(mask if mask.any() else np.ones_like(volume, dtype=bool))
        return coords.astype(np.float64)

    step = max(1, int(np.ceil((total_vox / float(n_samples)) ** (1.0 / volume.ndim))))
    grid_axes = [np.arange(0, dim, step, dtype=np.float64) for dim in shape]
    mesh = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, volume.ndim)
    if mask.any():
        valid = mask[tuple(mesh.astype(np.int64).T)]
        coords = mesh[valid]
        if coords.size == 0:
            coords = np.argwhere(mask)
    else:
        coords = mesh

    if coords.shape[0] < n_samples and mask.any():
        all_mask = np.argwhere(mask)
        need = min(n_samples - coords.shape[0], all_mask.shape[0])
        if need > 0:
            idx = np.linspace(0, all_mask.shape[0] - 1, need, dtype=np.int64)
            coords = np.concatenate([coords, all_mask[idx].astype(np.float64)], axis=0)

    if coords.shape[0] > n_samples:
        idx = rng.choice(coords.shape[0], size=n_samples, replace=False)
        coords = coords[idx]
    return coords.astype(np.float64)


def _map_samples(volume: np.ndarray, vox_coords: np.ndarray) -> np.ndarray:
    axes = vox_coords.T
    return map_coordinates(volume, axes, order=1, mode="nearest")


def _center_of_mass_world(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    com_vox = center_of_mass(volume)
    if not np.all(np.isfinite(com_vox)):
        com_vox = 0.5 * (np.asarray(volume.shape, dtype=np.float64) - 1.0)
    return nib.affines.apply_affine(affine, np.asarray(com_vox, dtype=np.float64))


def _matrix_barycenter(mats: Sequence[np.ndarray], tol: float = 1e-20, max_iter: int = 1024) -> np.ndarray:
    if not mats:
        raise ValueError("Need at least one transform to compute a barycenter")
    M = np.eye(4, dtype=np.float64)
    for _ in range(max_iter):
        S = np.zeros_like(M)
        for mat in mats:
            delta = logm(np.linalg.solve(M, mat))
            S += np.real_if_close(delta)
        S /= float(len(mats))
        if float(np.sum(np.abs(S) ** 2)) < tol:
            break
        M = M @ expm(S)
    return M


def _apply_inverse_consistent(
    transforms: Sequence[np.ndarray],
    ref_data: np.ndarray,
    ref_affine: np.ndarray,
    ref_header: nib.Nifti1Header,
) -> Tuple[List[np.ndarray], nib.Nifti1Image]:
    mean_t = _matrix_barycenter(transforms)
    adjust = np.linalg.inv(mean_t)
    new_transforms = [adjust @ T for T in transforms]
    new_affine = adjust @ ref_affine
    header = ref_header.copy()
    header.set_sform(new_affine, code=1)
    header.set_qform(new_affine, code=1)
    template_img = nib.Nifti1Image(np.asarray(ref_data, dtype=np.float32), new_affine, header=header)
    return new_transforms, template_img


def _is_identity(matrix: np.ndarray, atol: float = 1e-5) -> bool:
    return np.allclose(matrix, np.eye(matrix.shape[0]), atol=atol)


def _split_nifti_name(path: str) -> Tuple[str, str]:
    basename = os.path.basename(path)
    lower = basename.lower()
    if lower.endswith(".nii.gz"):
        return basename[:-7], ".nii.gz"
    if lower.endswith(".nii"):
        return basename[:-4], ".nii"
    base, ext = os.path.splitext(basename)
    if ext.lower() == ".gz" and base.lower().endswith(".nii"):
        return base[:-4], ".nii.gz"
    if ext:
        return base, ext
    return base, ".nii"


def _build_output_path(inp_path: str, out_dir: str, naming: str, suffix: str = "") -> str:
    stem, ext = _split_nifti_name(inp_path)
    if naming == "bids":
        name = f"{stem}{suffix}{ext}"
    else:
        name = f"r{stem}{ext}"
    return os.path.join(out_dir, name)


def _resample_images_to_reference(
    images: Sequence[nib.Nifti1Image],
    transforms: Sequence[np.ndarray],
    reference_img: nib.Nifti1Image,
) -> List[np.ndarray]:
    ref_affine = reference_img.affine
    ref_header = reference_img.header.copy()
    resampled: List[np.ndarray] = []
    for img, T in zip(images, transforms):
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        if _is_identity(T) and np.allclose(T @ img.affine, ref_affine) and data.shape == reference_img.shape:
            resampled.append(data)
            continue
        aligned_img = nib.Nifti1Image(data, T @ img.affine, header=img.header)
        res = resample_from_to(aligned_img, reference_img, order=1)
        resampled.append(np.asarray(res.get_fdata(), dtype=np.float32))
    return resampled


def _align_against_reference(
    reference_img: nib.Nifti1Image,
    images: Sequence[nib.Nifti1Image],
    n_iter: int,
    n_samples: int,
    verbose: bool,
    skip_identity_for_first: bool,
) -> RigidRealignOutputs:
    ref_data = np.asarray(reference_img.get_fdata(), dtype=np.float32)
    ref_affine = reference_img.affine.astype(np.float64)
    ref_header = reference_img.header
    ref_com = _center_of_mass_world(ref_data, ref_affine)
    rng = np.random.default_rng(2024)
    sample_vox = _choose_samples(ref_data, n_samples, rng)
    sample_mm = nib.affines.apply_affine(ref_affine, sample_vox)
    sigmas = np.linspace(1.5, 0.0, max(1, int(n_iter)), dtype=np.float64)
    ref_samples_cache: Dict[float, np.ndarray] = {}
    transforms: List[np.ndarray] = []

    for idx, img in enumerate(images):
        if skip_identity_for_first and idx == 0 and img is reference_img:
            transforms.append(np.eye(4, dtype=np.float64))
            continue

        mov_data = np.asarray(img.get_fdata(), dtype=np.float32)
        params = np.zeros(6, dtype=np.float64)
        moving_affine = img.affine.astype(np.float64)
        mov_com = _center_of_mass_world(mov_data, moving_affine)
        params[3:] = ref_com - mov_com

        for sigma in sigmas:
            if sigma not in ref_samples_cache:
                ref_vol = gaussian_filter(ref_data, sigma, mode="nearest") if sigma > 0 else ref_data
                ref_vals = _map_samples(ref_vol, sample_vox)
                ref_samples_cache[float(sigma)] = ref_vals.astype(np.float32)
            ref_vals = ref_samples_cache[float(sigma)]
            mov_vol = gaussian_filter(mov_data, sigma, mode="nearest") if sigma > 0 else mov_data
            mov_vol = mov_vol.astype(np.float32, copy=False)
            cost_fn = _build_cost_function(sample_mm, ref_vals, moving_affine, mov_vol)
            result = least_squares(
                cost_fn,
                params,
                method="trf",
                loss="soft_l1",
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                max_nfev=200,
            )
            params = result.x
            if verbose:
                print(f"[{idx}] sigma={sigma:.2f} cost={result.cost:.4f}")

        if 0.0 not in ref_samples_cache:
            ref_vol = ref_data
            ref_vals = _map_samples(ref_vol, sample_vox)
            ref_samples_cache[0.0] = ref_vals.astype(np.float32)
        ref_vals = ref_samples_cache[0.0]
        mov_vol = mov_data.astype(np.float32, copy=False)
        final_cost = _build_cost_function(sample_mm, ref_vals, moving_affine, mov_vol)
        result = least_squares(
            final_cost,
            params,
            method="trf",
            loss="soft_l1",
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=300,
        )
        params = result.x
        if verbose:
            print(f"[{idx}] sigma=0.00 refine cost={result.cost:.4f}")

        T = _rigid_matrix(params)
        transforms.append(T)

    return RigidRealignOutputs(
        reference_img=nib.Nifti1Image(ref_data, ref_affine, header=ref_header),
        transforms=transforms,
    )


def _build_cost_function(
    ref_mm: np.ndarray,
    ref_samples: np.ndarray,
    moving_affine: np.ndarray,
    moving_volume: np.ndarray,
) -> callable:
    inv_map = np.linalg.inv(moving_affine)

    def cost(params: np.ndarray) -> np.ndarray:
        T = _rigid_matrix(params)
        world_to_moving = inv_map @ np.linalg.inv(T)
        moving_vox = nib.affines.apply_affine(world_to_moving, ref_mm)
        coords = moving_vox.T  # map_coordinates expects x,y,z ordering
        vals = map_coordinates(moving_volume, coords, order=1, mode="nearest")
        diff = vals - ref_samples
        return diff

    return cost


def rigid_realign_to_first(
    images: Sequence[nib.Nifti1Image],
    n_iter: int = 3,
    n_samples: int = 40000,
    verbose: bool = False,
    inverse_consistent: bool = False,
    register_to_mean: bool = True,
) -> RigidRealignOutputs:
    """Rigidly align all images either to the first volume or to a derived mean.

    Parameters
    ----------
    images:
        Iterable of NIfTI images to align. The first entry is used as the
        reference for the initial pass.
    n_iter:
        Number of multi-scale passes (mirrors SPM's behaviour). Each pass runs a
        coarse-to-fine cascade of Gaussian-smooth cost functions.
    n_samples:
        Number of voxels sampled from the reference volume for the cost
        function. Samples are drawn on a strided grid and topped up to ensure
        deterministic coverage.
    verbose:
        Print optimisation diagnostics for each sigma level when ``True``.
    inverse_consistent:
        When ``True`` (default ``False``) remove any bias towards the chosen
        reference by computing the SE(3) barycentre of the estimated transforms
        (akin to ``spm_meanm``) and re-centring them around this average.
    register_to_mean:
        When ``True`` (default) perform a second alignment pass using the
        rigidly-aligned mean image from the first pass as template. This mimics
        ``spm_realign``'s "register-to-mean" option and is more robust when the
        first volume is noisy or corrupted. Disable it to align strictly to the
        first scan.
    """

    assert len(images) >= 1, "Need at least one image"

    initial_outputs = _align_against_reference(
        reference_img=images[0],
        images=images,
        n_iter=n_iter,
        n_samples=n_samples,
        verbose=verbose,
        skip_identity_for_first=True,
    )

    final_outputs = initial_outputs

    if register_to_mean:
        resampled = _resample_images_to_reference(
            images,
            initial_outputs.transforms,
            initial_outputs.reference_img,
        )
        mean_data = np.mean(np.stack(resampled, axis=0), axis=0).astype(np.float32)
        mean_img = nib.Nifti1Image(mean_data, initial_outputs.reference_img.affine, initial_outputs.reference_img.header)

        final_outputs = _align_against_reference(
            reference_img=mean_img,
            images=images,
            n_iter=n_iter,
            n_samples=n_samples,
            verbose=verbose,
            skip_identity_for_first=False,
        )

    if inverse_consistent:
        ref_data = np.asarray(final_outputs.reference_img.get_fdata(), dtype=np.float32)
        ref_affine = final_outputs.reference_img.affine
        transforms, ref_img = _apply_inverse_consistent(
            final_outputs.transforms,
            ref_data,
            ref_affine,
            final_outputs.reference_img.header,
        )
        final_outputs = RigidRealignOutputs(reference_img=ref_img, transforms=transforms)

    return final_outputs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rigid realignment to the first input (SPM-like)")
    p.add_argument("--inputs", nargs="+", required=True, help="Input NIfTI images")
    p.add_argument("--out-dir", required=True, help="Directory for outputs")
    p.add_argument("--iterations", type=int, default=3, help="Multi-scale passes (default: 3)")
    p.add_argument("--device", default="cpu", help="Ignored placeholder for interface compatibility")
    p.add_argument("--save-template", action="store_true", help="Save the reference volume (for parity)")
    p.add_argument("--save-resampled", action="store_true", help="Write resampled copies in reference space")
    p.add_argument("--update-headers", action="store_true", help="Only update headers with new affines")
    p.add_argument("--set-zooms-from-sform", action="store_true", help="Copy pixdim from updated sform")
    p.add_argument("--force-template-zooms", action="store_true", help="Use reference zooms even when updating headers")
    p.add_argument(
        "--inverse-consistent",
        action="store_true",
        help="Re-center rigid transforms using a barycentric SE(3) mean (SPM-style)",
    )
    p.add_argument(
        "--register-to-first",
        action="store_true",
        help=(
            "Skip the mean-template refinement pass and keep the initial alignment "
            "to the first scan (mirrors spm_realign without register-to-mean)."
        ),
    )
    p.add_argument(
        "--output-naming",
        choices=("bids", "legacy"),
        default="bids",
        help=(
            "Output naming scheme for --save-resampled outputs: 'bids' keeps suffix-based names (default); "
            "'legacy' prefixes outputs with 'r' to mimic classic SPM naming."
        ),
    )
    p.add_argument(
        "--use-skullstrip",
        action="store_true",
        help=(
            "Estimate rigid transforms using skull-stripped copies of the inputs (via segment.py skull_strip), "
            "but apply transforms to the original images for output."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Print optimizer diagnostics")
    return p.parse_args(argv)


def _skullstrip_for_realign(
    images: Sequence[nib.Nifti1Image],
    verbose: bool,
) -> List[nib.Nifti1Image]:
    """Return skull-stripped copies for use during transform estimation.

    This keeps the original image list untouched so that transforms can still be
    applied to the original images for output.

    Notes
    -----
    We import from segment.py lazily to avoid pulling in heavy deep-learning
    dependencies unless requested.
    """

    # Ensure local imports used by segment.py (e.g., `from utils import ...`) resolve
    # regardless of how this module is executed (script vs package).
    this_dir = str(Path(__file__).resolve().parent)
    if this_dir not in os.sys.path:
        os.sys.path.insert(0, this_dir)

    from segment import CustomPreprocess, prepare_model_files, setup_device, skull_strip  # type: ignore

    prepare_model_files()
    _, no_gpu = setup_device()
    prep = CustomPreprocess(no_gpu)

    stripped: List[nib.Nifti1Image] = []
    for i, img in enumerate(images, start=1):
        if verbose:
            print(f"Skull-stripping for realignment: {i}/{len(images)}")
        brain, _mask, _ = skull_strip(prep, img, verbose=False, count=0, end_count=1)
        stripped.append(brain)
    return stripped


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    save_resampled_inputs = args.save_resampled
    if save_resampled_inputs and args.update_headers:
        raise SystemExit("--save-resampled and --update-headers are mutually exclusive")

    imgs = [nib.load(p) for p in args.inputs]

    align_imgs = imgs
    if args.use_skullstrip:
        align_imgs = _skullstrip_for_realign(imgs, verbose=args.verbose)

    outputs = rigid_realign_to_first(
        align_imgs,
        n_iter=args.iterations,
        verbose=args.verbose,
        inverse_consistent=args.inverse_consistent,
        register_to_mean=not args.register_to_first,
    )
    ref_img = outputs.reference_img

    if save_resampled_inputs:
        out_vols = _resample_images_to_reference(imgs, outputs.transforms, ref_img)
        for inp, vol in zip(args.inputs, out_vols):
            out_path = _build_output_path(inp, args.out_dir, args.output_naming, suffix="_desc-realigned")
            nib.save(nib.Nifti1Image(vol, ref_img.affine, ref_img.header), out_path)

    if args.update_headers:
        # Safety: never update headers in-place. Require an output directory that
        # differs from every input's directory.
        out_dir_abs = os.path.abspath(args.out_dir)
        for path in args.inputs:
            in_dir_abs = os.path.abspath(os.path.dirname(path))
            if out_dir_abs == in_dir_abs:
                raise SystemExit(
                    "--update-headers requires --out-dir to be different from the input folder "
                    f"(got out-dir={out_dir_abs})"
                )

        template_zooms = None
        if args.force_template_zooms:
            template_zooms = tuple(float(v) for v in _voxel_sizes(ref_img.affine))
        for path, T in zip(args.inputs, outputs.transforms):
            img = nib.load(path)
            data = np.asarray(img.get_fdata(), dtype=np.float32)
            header = img.header.copy()
            new_affine = T @ img.affine
            header.set_sform(new_affine, code=1)
            header.set_qform(new_affine, code=1)
            if args.set_zooms_from_sform:
                try:
                    header.set_zooms(tuple(float(v) for v in _voxel_sizes(new_affine)))
                except Exception:
                    pass
            if template_zooms is not None:
                try:
                    header.set_zooms(template_zooms)
                except Exception:
                    pass
            # Save original data with modified header into out-dir, preserving the
            # original input filename (no 'r' prefix and no suffix).
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            nib.save(nib.Nifti1Image(data, new_affine, header=header), out_path)

    if args.save_template:
        _, ext = _split_nifti_name(args.inputs[0])
        nib.save(ref_img, os.path.join(args.out_dir, f"reference{ext}"))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
