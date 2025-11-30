"""
Inverse-consistent rigid registration to an unbiased mid-space for N timepoints (PyTorch),
now powered by torchreg's AffineRegistration.

This module iteratively:
- Registers each timepoint rigidly to a current template using torchreg.affine.AffineRegistration
- Resamples into the template space and averages to update the template
- Returns the final template (as a NIfTI image) and per-timepoint rigid transforms

Notes
-----
- No ITK or DIPY required. Depends on numpy, torch, nibabel, and torchreg.
- Transforms are represented as a normalized affine (3x4) where translation is in
    grid-normalized coordinates (align_corners=True). We also provide a voxel-space
    4x4 matrix for convenience.
- For labels, prefer nearest-neighbor resampling; for intensities, trilinear.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from nibabel.affines import voxel_sizes as _voxel_sizes
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation
from torchreg.affine import AffineRegistration


@dataclass
class RigidICOutputs:
    template_img: nib.Nifti1Image
    transforms_to_template: List[Dict[str, np.ndarray]]  # each: {R(3x3), t_norm(3,), M_vox(4x4)}
    resampled_in_template: Optional[List[np.ndarray]] = None  # last-iteration float volumes


def _to_tensor(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(vol.astype(np.float32, copy=False))
    if t.ndim == 3:
        t = t[None, None]  # (1,1,D,H,W)
    return t.to(device)


def _gauss_downsample(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    # Simple pyramidal downsampling via average pooling (good enough for rigid alignment)
    return F.avg_pool3d(x, kernel_size=scale, stride=scale, padding=0, count_include_pad=False)


def _euler_zyx_to_R(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    # Rotations about x (roll), y (pitch), z (yaw). We apply Rz @ Ry @ Rx.
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)
    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)
    return Rz @ Ry @ Rx


def _make_normalized_grid(shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    D, H, W = shape
    z = torch.linspace(-1.0, 1.0, D, device=device)
    y = torch.linspace(-1.0, 1.0, H, device=device)
    x = torch.linspace(-1.0, 1.0, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)  # (D,H,W,3) in (x,y,z) order for grid_sample
    return grid[None]  # (1,D,H,W,3)


def _apply_rigid(vol: torch.Tensor, R: torch.Tensor, t_norm: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # vol: (1,1,D,H,W), grid: (1,D,H,W,3) in normalized coords
    # Apply s = R*g + t to get source coordinates in normalized space
    g = grid.reshape(1, -1, 3)  # (1,N,3)
    s = torch.einsum('bij,bnj->bni', R[None], g) + t_norm[None, None, :]
    s = s.reshape_as(grid)
    # grid_sample expects (N,C,D,H,W) with grid (N,D,H,W,3) in x,y,z order
    return F.grid_sample(vol, s, mode='bilinear', padding_mode='border', align_corners=True)


def _ncc_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # x,y: (1,1,D,H,W)
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    num = (xv * yv).mean()
    den = torch.sqrt((xv * xv).mean() * (yv * yv).mean() + eps)
    return 1.0 - (num / (den + eps))  # minimize 1 - NCC


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest pure rotation (SO(3)) using SVD."""

    U, _, Vt = np.linalg.svd(matrix)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _apply_reference_zooms(affine: np.ndarray, reference_zooms: Optional[Sequence[float]]) -> np.ndarray:
    """Replace the linear part with reference voxel sizes while keeping orientation."""

    if reference_zooms is None:
        return affine.astype(np.float64)
    linear = affine[:3, :3]
    R = _project_to_rotation(linear)
    result = affine.astype(np.float64)
    result[:3, :3] = R @ np.diag(np.asarray(reference_zooms, dtype=np.float64))
    return result


def _rigid_affine_mean(mats: Sequence[np.ndarray], reference_zooms: Optional[Sequence[float]]) -> np.ndarray:
    """Rigid-only average: mean translation + quaternion average of rotations."""

    if not mats:
        raise ValueError("Need at least one affine to average")
    translations = np.stack([m[:3, 3] for m in mats], axis=0)
    rotations = np.stack([_project_to_rotation(m[:3, :3]) for m in mats], axis=0)
    try:
        rot = Rotation.from_matrix(rotations)
        R_mean = rot.mean().as_matrix()
    except Exception:
        R_mean = _project_to_rotation(rotations.mean(axis=0))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = R_mean
    affine[:3, 3] = translations.mean(axis=0)
    return _apply_reference_zooms(affine, reference_zooms)


def _log_euclidean_affine_mean(
    mats: Sequence[np.ndarray], reference_zooms: Optional[Sequence[float]]
) -> np.ndarray:
    """Log-Euclidean mean for affine matrices with rigid fallback."""

    if not mats:
        raise ValueError("Need at least one affine to average")
    try:
        logs = [logm(m) for m in mats]
        mean_log = sum(logs) / float(len(logs))
        mean_affine = expm(mean_log)
        if np.iscomplexobj(mean_affine):
            max_imag = float(np.abs(mean_affine.imag).max())
            if max_imag > 1e-6:
                return _rigid_affine_mean(mats, reference_zooms)
            mean_affine = mean_affine.real
    except Exception:
        return _rigid_affine_mean(mats, reference_zooms)
    mean_affine[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return _apply_reference_zooms(mean_affine, reference_zooms)


def inverse_consistent_rigid_N(
    images: Sequence[np.ndarray],
    n_iter: int = 2,
    device: str = "cpu",
    verbose: bool = False,
) -> RigidICOutputs:
    """Inverse-consistent (unbiased mid-space) rigid registration for N timepoints
    using torchreg's AffineRegistration in rigid mode (rotation + translation only).

    Parameters
    ----------
    images : sequence of np.ndarray
        Input 3D volumes, same shape. Values should be bias-corrected T1s or similar.
    n_iter : int
        Template refinement iterations (2–3 typical).
    device : str
        'cpu' or 'cuda'.
    verbose : bool
        Whether to print progress from torchreg (tqdm bars).
    """
    assert len(images) >= 2, "Need at least two timepoints"
    dev = torch.device(device)

    # Prepare tensors (1,1,D,H,W) per timepoint
    vols: List[torch.Tensor] = [_to_tensor(v, dev) for v in images]
    D, H, W = vols[0].shape[-3:]

    # Initialize template as mean of inputs
    template = sum(vols) / float(len(vols))

    # Iterative template refinement
    last_warped_np: List[np.ndarray] = []
    affines_norm: List[np.ndarray] = []  # (3x4)
    for _ in range(max(1, int(n_iter))):
        warped_this_iter: List[torch.Tensor] = []
        affines_norm = []
        # Build static channels: intensity and simple mask from template (>0)
        template_int = template.detach()
        template_mask = (template_int > 0).float()
        static = torch.cat([template_int, template_mask], dim=1)  # (1,2,D,H,W)

        for v in vols:
            # Moving channels: normalized intensity and simple mask (>0)
            mv = torch.nan_to_num(v, nan=0.0)
            mv_mask = (mv > 0).float()
            # Robust normalization by 95th percentile within mask
            with torch.no_grad():
                mv_vals = mv[mv_mask.bool()]
                if mv_vals.numel() > 0:
                    q95 = torch.quantile(mv_vals, 0.95).clamp(min=1e-6)
                else:
                    q95 = torch.tensor(1.0, device=dev)
            mv_norm = (mv / q95).clamp(max=1.0)
            moving = torch.cat([mv_norm, mv_mask], dim=1)

            # Rigid-only registration akin to preprocess.run_affine_register
            reg = AffineRegistration(
                scales=(24, 12),
                iterations=(500, 100),
                is_3d=True,
                learning_rate=1e-3,
                verbose=verbose,
                dissimilarity_function=torch.nn.MSELoss(),
                with_translation=True,
                with_rotation=True,
                with_zoom=False,
                with_shear=False,
                align_corners=True,
                padding_mode="zeros",
            )
            moved = reg(moving.to(dev), static.to(dev), return_moved=True)  # (1,2,D,H,W)
            # Use first channel (intensity) from moved
            moved_int = moved[:, :1].detach()
            warped_this_iter.append(moved_int)
            A = reg.get_affine(with_grad=False)[0].detach().cpu().numpy()  # (3,4)
            affines_norm.append(A)

        # Update template as average of warped intensities
        template = sum(warped_this_iter) / float(len(warped_this_iter))
        last_warped_np = [w.cpu().numpy()[0, 0] for w in warped_this_iter]

    # Convert normalized 3x4 affine to voxel-space 4x4 (align_corners=True)
    # Normalized <-> voxel mapping:
    # v = S*n + o, with S = diag((W-1)/2, (H-1)/2, (D-1)/2), o = [(W-1)/2,(H-1)/2,(D-1)/2]
    # Homogeneous forms:
    #   N2V = [[S, o],[0,1]],  V2N = [[S^{-1}, -S^{-1} o],[0,1]] = [[S^{-1}, -1],[0,1]]
    sx, sy, sz = (W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0
    N2V = np.eye(4, dtype=np.float32)
    N2V[:3, :3] = np.diag([sx, sy, sz]).astype(np.float32)
    N2V[:3, 3] = np.array([sx, sy, sz], dtype=np.float32)
    V2N = np.eye(4, dtype=np.float32)
    V2N[:3, :3] = np.diag([1.0 / sx, 1.0 / sy, 1.0 / sz]).astype(np.float32)
    V2N[:3, 3] = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    transforms: List[Dict[str, np.ndarray]] = []
    for A in affines_norm:
        M_norm = np.eye(4, dtype=np.float32)
        M_norm[:3, :3] = A[:, :3]
        M_norm[:3, 3] = A[:, 3]
        M_vox = N2V @ M_norm @ V2N
        transforms.append({
            "R": A[:, :3].astype(np.float32),
            "t_norm": A[:, 3].astype(np.float32),
            "M_vox": M_vox.astype(np.float32),
        })

    template_img = nib.Nifti1Image(template.cpu().numpy()[0, 0].astype(np.float32), affine=np.eye(4))
    return RigidICOutputs(template_img=template_img, transforms_to_template=transforms, resampled_in_template=last_warped_np)


def save_outputs(
    outputs: RigidICOutputs,
    out_dir: str,
    input_paths: Optional[Sequence[str]] = None,
    template_name: str = "rigid_template_T1.nii.gz",
    transform_prefix: str = "tp",
    save_template: bool = False,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    if save_template:
        nib.save(outputs.template_img, os.path.join(out_dir, template_name))
    tpaths: List[str] = []
    for i, T in enumerate(outputs.transforms_to_template):
        base = f"{transform_prefix}{i}"
        if input_paths:
            b = os.path.basename(input_paths[i])
            base = os.path.splitext(os.path.splitext(b)[0])[0]
        tpath = os.path.join(out_dir, f"{base}_toTemplate_rigid.npz")
        np.savez_compressed(tpath, R=T["R"], t_norm=T["t_norm"], M_vox=T["M_vox"])
        tpaths.append(tpath)
    return tpaths


def _compute_template_affine(
    input_affines: Sequence[np.ndarray],
    transforms_to_template: Sequence[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Average template→world mapping following CAT12's midpoint strategy."""

    reference_zooms = tuple(float(v) for v in _voxel_sizes(input_affines[0]))
    template_to_world: List[np.ndarray] = []
    for affine, transform in zip(input_affines, transforms_to_template):
        M_vox = transform["M_vox"].astype(np.float64)
        template_to_world.append(affine.astype(np.float64) @ M_vox)
    if len(template_to_world) == 1:
        return _apply_reference_zooms(template_to_world[0], reference_zooms)
    return _log_euclidean_affine_mean(template_to_world, reference_zooms)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverse-consistent rigid registration (mid-space) for N timepoints (PyTorch)")
    p.add_argument("--inputs", nargs="+", required=True, help="Input T1 images (NIfTI, same shape)")
    p.add_argument("--out-dir", required=True, help="Output directory for template and transforms")
    p.add_argument("--iterations", type=int, default=2, help="Template refinement iterations (default: 2)")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    # Output control
    p.add_argument("--save-template", action="store_true", help="Save the resampled template (average) volume")
    p.add_argument("--save-resampled-inputs", action="store_true", help="Save last-iteration resampled inputs in template space")
    p.add_argument("--save-resampled", action="store_true", help="Alias for --save-resampled-inputs (deprecated)")
    p.add_argument(
        "--update-headers",
        action="store_true",
        help="Instead of resampling inputs, write copies with only header (affine) updated to template space",
    )
    p.add_argument(
        "--set-zooms-from-sform",
        action="store_true",
        help="When used with --update-headers, also set header pixdim (zooms) from the new sform; default keeps original pixdim",
    )
    p.add_argument(
        "--force-template-zooms",
        action="store_true",
        help="When used with --update-headers, set header pixdim to match the first input's pixdim (template reference)",
    )
    p.add_argument("--verbose", action="store_true", help="Be verbose")
    return p.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    # Map deprecated alias
    save_resampled_inputs = args.save_resampled_inputs or args.save_resampled
    if save_resampled_inputs and args.update_headers:
        raise SystemExit("--save-resampled-inputs and --update-headers are mutually exclusive")
    imgs = []
    affines = []
    for p in args.inputs:
        im = nib.load(p)
        imgs.append(np.asarray(im.get_fdata(), dtype=np.float32))
        affines.append(im.affine)
    out = inverse_consistent_rigid_N(
        imgs, n_iter=args.iterations, device=args.device, verbose=args.verbose)

    template_affine = _compute_template_affine(affines, out.transforms_to_template)
    outputs_img = nib.Nifti1Image(out.template_img.get_fdata().astype(np.float32), affine=template_affine)
    out = RigidICOutputs(template_img=outputs_img, transforms_to_template=out.transforms_to_template, resampled_in_template=out.resampled_in_template)

    # Save transforms and (optionally) the template
    tpaths = save_outputs(out, args.out_dir, input_paths=args.inputs, save_template=args.save_template)

    # Optionally save resampled inputs in template space
    if save_resampled_inputs and out.resampled_in_template:
        for i, vol in enumerate(out.resampled_in_template):
            base = os.path.splitext(os.path.splitext(os.path.basename(args.inputs[i]))[0])[0]
            out_path = os.path.join(args.out_dir, f"{base}_inTemplate.nii.gz")
            nib.save(nib.Nifti1Image(vol.astype(np.float32), template_affine), out_path)

    # Or update headers only (no resampling): write copies with adjusted affine
    if args.update_headers:
        A_template = template_affine
        template_zooms = None
        if args.force_template_zooms:
            template_zooms = tuple(float(v) for v in _voxel_sizes(A_template))
        for i, in_path in enumerate(args.inputs):
            im = nib.load(in_path)
            # M_vox maps template vox -> moving vox (align_corners=True conversion accounted for)
            # For header update we need moving->template => inverse
            M_vox = out.transforms_to_template[i]["M_vox"].astype(np.float32)
            try:
                M_vox_inv = np.linalg.inv(M_vox)
            except np.linalg.LinAlgError:
                print(f"Warning: transform for {in_path} not invertible; skipping header update")
                continue
            A_new = A_template @ M_vox_inv
            # Modify sform/qform on the image object directly to preserve other header fields
            im.set_sform(A_new, code=1)
            im.set_qform(A_new, code=1)
            if args.set_zooms_from_sform:
                try:
                    from nibabel.affines import voxel_sizes as _voxel_sizes
                    vs_new = _voxel_sizes(A_new)
                    im.header.set_zooms(tuple(float(v) for v in vs_new))
                except Exception:
                    pass
            if args.force_template_zooms and template_zooms is not None:
                try:
                    im.header.set_zooms(tuple(float(v) for v in template_zooms))
                except Exception:
                    pass
            base = os.path.splitext(os.path.splitext(os.path.basename(in_path))[0])[0]
            out_path = os.path.join(args.out_dir, f"{base}_headerAligned.nii.gz")
            nib.save(im, out_path)

    msg_parts = [f"{len(tpaths)} transforms"]
    if args.save_template:
        msg_parts.append("template")
    if save_resampled_inputs:
        msg_parts.append("resampled inputs")
    if args.update_headers:
        msg_parts.append("header-updated copies")
    print(f"Saved {', '.join(msg_parts)} to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
