"""Low-dimensional groupwise diffeomorphic registration for longitudinal series.

This is the non-linear half of CAT12's *ageing* longitudinal model.  It picks up
where :mod:`t1prep.realign_longitudinal` stops: that module removes the rigid
part of the between-scan difference and re-centres the transforms on their SE(3)
barycentre so no time point is privileged.  Everything that is left over — the
ventricular expansion and cortical thinning an ageing brain accumulates between
scans — is still in the data, and each time point is then segmented as if it
were an unrelated subject.

The model here estimates that residual as a *small, smooth* deformation towards
an unbiased subject average:

* **Stationary velocity field (SVF).**  Each time point gets one velocity
  ``v_i``; the deformation is ``exp(v_i)``, integrated by scaling and squaring.
  For the deformation magnitudes seen between serial scans the SVF and the true
  geodesic agree to high order, so the momentum bookkeeping of geodesic shooting
  buys nothing measurable here — see Ashburner & Ridgway (2013), which is what
  SPM's serial longitudinal registration and CAT12's ageing model actually use.
* **Low-dimensional.**  ``v_i`` is not a free field.  It is stored as
  coefficients on a coarse control grid (``--control-spacing``, 12 mm by
  default) in millimetres and trilinearly upsampled to the working grid.  The
  parameterisation *is* the regulariser: a 12 mm lattice cannot express the
  folding-scale detail that a cross-sectional warp needs, which is exactly the
  restriction wanted between two scans of the same brain.
* **Unbiased.**  After every update the coefficients are re-centred on their
  mean, ``c_i <- c_i - mean_j c_j``.  In the log-Euclidean setting this is the
  tangent-space Frechet-mean condition, and it is the direct non-linear
  analogue of the ``_matrix_barycenter`` re-centring the rigid stage already
  performs.  No time point is the reference; the average is.

The quantity of interest is the log Jacobian determinant of ``exp(v_i)``: it is
the per-voxel log volume ratio between that time point and the subject average,
which is the map longitudinal VBM runs statistics on.

CLI usage
---------
Run on the *realigned* volumes produced by ``realign_longitudinal``::

    python -m t1prep.warp_longitudinal \\
        --inputs tp1.nii.gz tp2.nii.gz --out-dir out

For convenience, the repository also provides a wrapper::

    scripts/warp_longitudinal.sh --help

References
----------
Ashburner J, Ridgway GR (2013). Symmetric diffeomorphic modeling of
longitudinal structural MRI. *Front Neurosci* 6:197.

Vercauteren T, Pennec X, Perchant A, Ayache N (2009). Diffeomorphic demons:
efficient non-parametric image registration. *NeuroImage* 45(1):S61-S72.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to

from .realign_longitudinal import _split_nifti_name

__all__ = [
    "LongitudinalWarpOutputs",
    "groupwise_svf",
    "run_cli",
]

#: Below this displacement (in voxels) a scaling-and-squaring step is exact
#: enough that adding another squaring only costs time.
_SQUARING_TARGET_VOX = 0.5

#: Hard cap on squaring steps.  Seven doublings cover a 64-voxel displacement,
#: far beyond anything a longitudinal series should produce; hitting the cap is
#: a sign the rigid stage failed rather than a reason to integrate harder.
_MAX_SQUARING_STEPS = 7


@dataclass
class LongitudinalWarpOutputs:
    """Results of a groupwise low-dimensional SVF fit.

    Attributes:
        template_img: The unbiased subject average, on the working grid.
        coefficients: Per time point control-grid velocity coefficients in
            millimetres, each ``(3, *control_shape)``.
        displacements: Per time point displacement of ``exp(v_i)`` on the
            working grid, in **voxels**, each ``(3, *working_shape)`` with
            channel ``k`` displacing along array axis ``k``.
        log_jacobians: Per time point ``log det J`` of ``exp(v_i)`` on the
            working grid, each ``(*working_shape,)``.
        working_affine: Grid-to-RAS affine of the working grid.
        working_shape: Shape of the working grid.
        control_shape: Shape of the control lattice.
    """

    template_img: nib.Nifti1Image
    coefficients: List[np.ndarray] = field(default_factory=list)
    displacements: List[np.ndarray] = field(default_factory=list)
    log_jacobians: List[np.ndarray] = field(default_factory=list)
    working_affine: Optional[np.ndarray] = None
    working_shape: Optional[Tuple[int, int, int]] = None
    control_shape: Optional[Tuple[int, int, int]] = None


# --------------------------------------------------------------------------- #
# Grid helpers
#
# Displacements live in voxel units of the working grid with channel ``k``
# displacing along array axis ``k``.  ``grid_sample`` wants normalised ``[-1, 1]``
# coordinates whose last axis is ordered ``(x, y, z)`` with ``x`` indexing the
# *fastest* tensor dimension, i.e. the reverse of the array axis order.  The
# conversion is confined to :func:`_warp` so nothing else has to think about it.
# --------------------------------------------------------------------------- #


def _identity_vox(shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the identity voxel-coordinate field, ``(1, 3, *shape)``."""
    axes = [torch.arange(int(n), device=device, dtype=dtype) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    return grid[None]


def _warp(volume: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Resample ``volume`` at ``identity + disp``.

    Args:
        volume: ``(N, C, *shape)`` tensor to sample.
        disp: ``(N, 3, *shape)`` displacement in voxels, channel ``k`` along
            array axis ``k``.

    Returns:
        ``(N, C, *shape)`` resampled volume.
    """
    shape = disp.shape[2:]
    pos = _identity_vox(shape, disp.device, disp.dtype) + disp
    sizes = torch.tensor([float(n) for n in shape], device=disp.device, dtype=disp.dtype)
    # A size-1 axis has no interior to normalise against; keep it pinned at 0.
    denom = torch.clamp(sizes - 1.0, min=1.0).view(1, 3, 1, 1, 1)
    norm = 2.0 * pos / denom - 1.0
    grid = torch.stack([norm[:, 2], norm[:, 1], norm[:, 0]], dim=-1)
    return F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _squaring_steps(velocity: torch.Tensor) -> int:
    """Choose the number of scaling-and-squaring steps for ``velocity``."""
    peak = float(velocity.abs().max())
    if not math.isfinite(peak) or peak <= _SQUARING_TARGET_VOX:
        return 0
    steps = int(math.ceil(math.log2(peak / _SQUARING_TARGET_VOX)))
    return max(0, min(steps, _MAX_SQUARING_STEPS))


def _exp_svf(velocity: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
    """Integrate a stationary velocity field by scaling and squaring.

    Args:
        velocity: ``(N, 3, *shape)`` velocity in voxels per unit time.
        steps: Number of squarings; chosen from the peak magnitude when
            ``None``.

    Returns:
        ``(N, 3, *shape)`` displacement of ``exp(velocity)``, in voxels.
    """
    if steps is None:
        steps = _squaring_steps(velocity)
    disp = velocity / float(2 ** steps)
    for _ in range(steps):
        disp = disp + _warp(disp, disp)
    return disp


def _gradient(volume: torch.Tensor) -> torch.Tensor:
    """Central-difference spatial gradient of ``(N, 1, *shape)`` in voxels.

    Returns:
        ``(N, 3, *shape)`` with channel ``k`` holding the derivative along
        array axis ``k``.  Edges use a one-sided difference.
    """
    grads = []
    for axis in range(3):
        dim = axis + 2
        if volume.shape[dim] < 2:
            grads.append(torch.zeros_like(volume[:, 0]))
            continue
        upper = torch.narrow(volume, dim, 1, volume.shape[dim] - 1)
        lower = torch.narrow(volume, dim, 0, volume.shape[dim] - 1)
        diff = upper - lower
        # Average neighbouring forward differences to get the central one, and
        # repeat the end differences so the result keeps the input shape.
        first = torch.narrow(diff, dim, 0, 1)
        last = torch.narrow(diff, dim, diff.shape[dim] - 1, 1)
        padded = torch.cat([first, diff, last], dim=dim)
        centred = 0.5 * (
            torch.narrow(padded, dim, 0, volume.shape[dim])
            + torch.narrow(padded, dim, 1, volume.shape[dim])
        )
        grads.append(centred[:, 0])
    return torch.stack(grads, dim=1)


def _log_jacobian(disp: torch.Tensor) -> torch.Tensor:
    """Log Jacobian determinant of ``identity + disp``.

    Args:
        disp: ``(N, 3, *shape)`` displacement in voxels.

    Returns:
        ``(N, *shape)`` log determinant.  Non-positive determinants are clamped
        to a small floor rather than producing ``nan``; a diffeomorphic SVF
        should not produce them, so the clamp is a guard, not a correction.
    """
    rows = [_gradient(disp[:, k : k + 1]) for k in range(3)]
    jac = torch.stack(rows, dim=1)  # (N, 3, 3, *shape): d disp_k / d x_l
    eye = torch.eye(3, device=disp.device, dtype=disp.dtype).view(1, 3, 3, 1, 1, 1)
    jac = jac + eye
    det = (
        jac[:, 0, 0] * (jac[:, 1, 1] * jac[:, 2, 2] - jac[:, 1, 2] * jac[:, 2, 1])
        - jac[:, 0, 1] * (jac[:, 1, 0] * jac[:, 2, 2] - jac[:, 1, 2] * jac[:, 2, 0])
        + jac[:, 0, 2] * (jac[:, 1, 0] * jac[:, 2, 1] - jac[:, 1, 1] * jac[:, 2, 0])
    )
    return torch.log(torch.clamp(det, min=1e-6))


def _gaussian_smooth(volume: torch.Tensor, sigma_vox: Sequence[float]) -> torch.Tensor:
    """Separable Gaussian blur of ``(N, C, *shape)`` with per-axis sigma."""
    out = volume
    for axis, sigma in enumerate(sigma_vox):
        if sigma <= 0:
            continue
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(
            -radius, radius + 1, device=volume.device, dtype=volume.dtype
        )
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        kernel = kernel.view(shape).repeat(out.shape[1], 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        # F.pad orders the axes last-to-first.
        pad[2 * (2 - axis)] = radius
        pad[2 * (2 - axis) + 1] = radius
        out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel, groups=out.shape[1])
    return out


def _lattice_diff(coeff: torch.Tensor, axis: int) -> Optional[torch.Tensor]:
    """First difference of ``coeff`` along control-lattice ``axis``."""
    dim = axis + 2
    if coeff.shape[dim] < 2:
        return None
    return torch.narrow(coeff, dim, 1, coeff.shape[dim] - 1) - torch.narrow(
        coeff, dim, 0, coeff.shape[dim] - 1
    )


def _membrane_energy(coeff: torch.Tensor, spacing_mm: float) -> float:
    """Mean squared velocity gradient on the control lattice.

    This is the prior that makes the fit well-posed.  The lattice already
    band-limits the deformation, but nothing in the data term prevents the
    coefficients from *growing*: with enough iterations the demons force keeps
    finding deformations that explain image noise, and because the Jacobian
    depends on the velocity gradient that shows up directly as spurious volume
    change.  Penalising the gradient is what gives the optimiser a reason to
    stop.

    Args:
        coeff: ``(N, 3, *control_shape)`` coefficients in millimetres.
        spacing_mm: Control-point spacing.

    Returns:
        Dimensionless mean squared gradient.
    """
    total, count = 0.0, 0
    for axis in range(3):
        diff = _lattice_diff(coeff, axis)
        if diff is None:
            continue
        total += float((diff ** 2).sum())
        count += diff.numel()
    return total / (max(count, 1) * max(spacing_mm, 1e-6) ** 2)


def _lattice_laplacian(coeff: torch.Tensor) -> torch.Tensor:
    """Discrete Laplacian on the control lattice, with Neumann edges.

    Up to a factor this is the negative gradient of :func:`_membrane_energy`,
    so adding it to the coefficients is a descent step on the prior.
    """
    out = torch.zeros_like(coeff)
    for axis in range(3):
        dim = axis + 2
        if coeff.shape[dim] < 3:
            continue
        lo = torch.narrow(coeff, dim, 0, coeff.shape[dim] - 2)
        mid = torch.narrow(coeff, dim, 1, coeff.shape[dim] - 2)
        hi = torch.narrow(coeff, dim, 2, coeff.shape[dim] - 2)
        interior = lo + hi - 2.0 * mid
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = 1
        pad[2 * (2 - axis) + 1] = 1
        out = out + F.pad(interior, pad, mode="constant", value=0.0)
    return out


def _resize(volume: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Trilinearly resample ``(N, C, *)`` onto ``shape``."""
    if tuple(volume.shape[2:]) == tuple(shape):
        return volume
    return F.interpolate(volume, size=tuple(int(s) for s in shape), mode="trilinear", align_corners=True)


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #


def _working_reference(
    image: nib.Nifti1Image, resolution: Optional[float]
) -> nib.Nifti1Image:
    """Build the common grid every time point is resampled onto.

    Args:
        image: Image whose world placement defines the grid.
        resolution: Isotropic voxel size in millimetres, or ``None`` to keep
            ``image``'s own grid.

    Returns:
        An empty image carrying the target shape and affine.
    """
    if resolution is None:
        return image
    affine = np.asarray(image.affine, dtype=np.float64)
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    scale = np.asarray(zooms, dtype=np.float64) / float(resolution)
    shape = tuple(int(max(1, round(n * s))) for n, s in zip(image.shape[:3], scale))
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] / scale[None, :]
    # Keep the field of view centred on the same world point.
    old_centre = affine @ np.append(0.5 * (np.asarray(image.shape[:3]) - 1), 1.0)
    new_centre = new_affine @ np.append(0.5 * (np.asarray(shape) - 1), 1.0)
    new_affine[:3, 3] += (old_centre - new_centre)[:3]
    return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), new_affine)


def _normalise_intensity(
    data: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Scale to a robust ``[0, 1]`` range so demons forces are comparable.

    The demons force is a ratio of intensity difference to gradient magnitude,
    so two time points acquired with different receiver gains would otherwise
    produce a spurious deformation.
    """
    finite = np.isfinite(data)
    sample = data[finite & (mask if mask is not None else finite)]
    if sample.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(sample, [1.0, 99.0])
    if not np.isfinite(hi - lo) or (hi - lo) <= 0:
        hi, lo = float(sample.max()), float(sample.min())
    if (hi - lo) <= 0:
        return np.zeros_like(data, dtype=np.float32)
    out = (np.nan_to_num(data, nan=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _control_shape(
    working_shape: Sequence[int], zooms: Sequence[float], spacing_mm: float
) -> Tuple[int, int, int]:
    """Number of control points spanning the working grid at ``spacing_mm``."""
    extent_mm = [max(1e-6, (n - 1) * z) for n, z in zip(working_shape, zooms)]
    return tuple(int(max(2, round(e / spacing_mm) + 1)) for e in extent_mm)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Core estimation
# --------------------------------------------------------------------------- #


def groupwise_svf(
    images: Sequence[nib.Nifti1Image],
    *,
    control_spacing: float = 12.0,
    resolution: Optional[float] = 1.5,
    scales: Sequence[int] = (2, 1),
    iterations: Sequence[int] = (40, 40),
    step_size: float = 1.0,
    smoothing_mm: float = 6.0,
    max_step_mm: float = 0.15,
    regularisation: float = 0.05,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> LongitudinalWarpOutputs:
    """Fit one low-dimensional SVF per time point towards an unbiased average.

    Args:
        images: Rigidly realigned time points, in acquisition order.  They need
            not share a voxel grid; all are resampled onto a common one.
        control_spacing: Control-point spacing in millimetres.  Larger values
            restrict the deformation further; 12 mm is a deliberately stiff
            default suited to serial scans of one brain.
        resolution: Isotropic working resolution in millimetres, or ``None`` to
            work on the first image's own grid.
        scales: Coarse-to-fine image downsampling factors.  A shallow pyramid
            is deliberate: the control lattice already band-limits the
            deformation, so on synthetic serial pairs a deeper pyramid measured
            no better than ``(2, 1)`` while costing more.
        iterations: Iterations per entry of ``scales``.  Each level stops early
            once the penalised objective stops improving, so a generous value
            costs little.
        step_size: Multiplier on the demons update.
        smoothing_mm: FWHM-like Gaussian applied to the update field before it
            is projected onto the control lattice; this is the fluid-like part
            of the regularisation.
        regularisation: Weight of the membrane prior relative to the image
            term.  Zero reproduces plain groupwise demons, which has no reason
            to stop and will keep converting image noise into volume change.
        max_step_mm: Cap on how far one iteration may move a voxel, in
            millimetres.  The demons force is bounded by ``sigma/2`` *voxels*,
            so the underlying sigma is derived per pyramid level; capping in
            millimetres instead is what keeps a coarse level from moving four
            times further per iteration than a fine one.
        device: Torch device; defaults to CPU.  See the note in the module
            docstring of :mod:`t1prep._device` — ``grid_sampler_3d`` has no MPS
            kernel before torch 2.9, and the CPU fallback is slower than simply
            staying on the CPU.
        verbose: Print the mean absolute velocity per level.

    Returns:
        A :class:`LongitudinalWarpOutputs` with one coefficient set,
        displacement and log Jacobian per input.

    Raises:
        ValueError: If fewer than two images are given, or if ``scales`` and
            ``iterations`` have different lengths.
    """
    if len(images) < 2:
        raise ValueError("Groupwise registration needs at least two time points")
    if len(scales) != len(iterations):
        raise ValueError(
            f"--scales and --iterations must have the same length "
            f"({len(scales)} vs {len(iterations)})"
        )

    device = device or torch.device("cpu")
    reference = _working_reference(images[0], resolution)
    working_shape = tuple(int(n) for n in reference.shape[:3])
    working_affine = np.asarray(reference.affine, dtype=np.float64)
    zooms = np.sqrt((working_affine[:3, :3] ** 2).sum(axis=0))

    # ---- resample every time point onto the common grid -------------------
    volumes = []
    for img in images:
        if img.shape[:3] == working_shape and np.allclose(img.affine, working_affine):
            data = np.asarray(img.dataobj, dtype=np.float32)
        else:
            data = np.asarray(
                resample_from_to(img, reference, order=1).get_fdata(), dtype=np.float32
            )
        volumes.append(data)

    # The demons force divides by the image gradient, so in a flat region it is
    # a ratio of two noise terms: a direction with no anatomy behind it.  Left
    # unmasked those add up over the iterations into a random walk that shows up
    # as spurious volume change.  Restrict the force to voxels that carry
    # signal in every time point.
    mean_vol = np.stack(volumes, axis=0).mean(axis=0)
    rough_mask = mean_vol > (0.1 * float(np.percentile(mean_vol, 99.0)))
    volumes = [_normalise_intensity(v, rough_mask) for v in volumes]

    full = torch.from_numpy(np.stack(volumes, axis=0)[:, None]).to(device=device, dtype=torch.float32)
    mask_full = torch.from_numpy(rough_mask[None, None].astype(np.float32)).to(device=device)
    n_tp = full.shape[0]

    control_shape = _control_shape(working_shape, zooms, control_spacing)
    # Coefficients are in millimetres so the same lattice serves every scale.
    coeff = torch.zeros((n_tp, 3, *control_shape), device=device, dtype=torch.float32)

    if verbose:
        print(
            f"working grid {working_shape} @ {tuple(round(float(z), 2) for z in zooms)} mm, "
            f"control lattice {control_shape} @ {control_spacing} mm, {n_tp} time points"
        )

    for scale, n_iter in zip(scales, iterations):
        level_shape = tuple(max(2, int(round(n / scale))) for n in working_shape)
        level_zooms = [
            float(z) * float(n) / float(m)
            for z, n, m in zip(zooms, working_shape, level_shape)
        ]
        level = _resize(full, level_shape)
        level_mask = (_resize(mask_full, level_shape) > 0.5).to(level.dtype)
        # Blur proportionally to the downsampling so each level sees a matched
        # amount of detail.
        if scale > 1:
            level = _gaussian_smooth(level, [0.5 * scale] * 3)
        n_masked = float(level_mask.sum()) * n_tp
        # mm -> voxels at this level, per axis.
        mm_to_vox = torch.tensor(
            [1.0 / z for z in level_zooms], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1, 1)
        vox_to_mm = torch.tensor(
            level_zooms, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1, 1)
        # The update field is point-sampled onto the control lattice below, so
        # it has to be band-limited to that lattice first or the projection
        # aliases.  Gaussians compose in quadrature, which lets the requested
        # smoothing and the anti-alias kernel share one convolution.
        smooth_sigma = [
            math.sqrt(
                (max(0.0, smoothing_mm) / (2.355 * z)) ** 2
                + (control_spacing / (2.0 * z)) ** 2
            )
            for z in level_zooms
        ]
        # The demons force is bounded by ``sigma / 2`` voxels of *this* level.
        # Deriving sigma from a millimetre cap keeps every level taking the same
        # physical step; a fixed sigma would let the coarsest level move
        # ``scale`` times further per iteration and overshoot past anything the
        # finer levels could then undo.
        level_sigma = 2.0 * max(max_step_mm, 1e-6) / float(np.mean(level_zooms))

        # Groupwise objective: the across-time-point variance of the warped
        # images inside the mask.  It is what the update is trying to reduce,
        # so it is also the only honest thing to test convergence against.
        def _objective(c: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
            vel = _resize(c, level_shape) * mm_to_vox
            d = _exp_svf(vel)
            w = _warp(level, d)
            t = w.mean(dim=0, keepdim=True)
            sse = float((((w - t) ** 2) * level_mask).sum()) / max(n_masked, 1.0)
            energy = sse + regularisation * _membrane_energy(c, control_spacing)
            return energy, w, t

        step = float(step_size)
        best_energy, warped, template = _objective(coeff)
        best_coeff = coeff.clone()
        stalled = 0

        for it in range(int(n_iter)):
            # Symmetric demons force towards the current average.  Averaging the
            # two gradients is what makes the force symmetric in the sense of
            # Vercauteren et al.; the template broadcasts over time points.
            diff = template - warped
            grad = 0.5 * (_gradient(warped) + _gradient(template))
            grad_sq = (grad ** 2).sum(dim=1, keepdim=True)
            denom = grad_sq + (diff ** 2) / max(level_sigma ** 2, 1e-12)
            force = (diff / torch.clamp(denom, min=1e-8)) * grad * level_mask

            force = _gaussian_smooth(force, smooth_sigma)
            # Project onto the control lattice, back in millimetres.
            update = _resize(force * vox_to_mm, control_shape)
            # Descend on the image term and the prior together: the Laplacian
            # is (up to a constant) the negative gradient of the membrane
            # energy the objective now charges for.
            trial = coeff + step * update
            if regularisation > 0:
                trial = trial + regularisation * _lattice_laplacian(trial)
            # Unbiased: no time point is the reference, the average is.
            trial = trial - trial.mean(dim=0, keepdim=True)

            energy, warped_try, template_try = _objective(trial)
            if energy < best_energy:
                coeff, warped, template = trial, warped_try, template_try
                best_energy, best_coeff = energy, trial.clone()
                stalled = 0
            else:
                # Overshot: back off rather than walking further downhill on a
                # force that no longer describes the residual.
                stalled += 1
                step *= 0.5
                if stalled >= 4 or step < 1e-3 * step_size:
                    if verbose:
                        print(f"  scale {scale}: converged after {it + 1} iterations")
                    break

            if verbose and (it + 1) % 10 == 0:
                print(
                    f"  scale {scale} iter {it + 1:3d}/{n_iter}: "
                    f"objective = {best_energy:.6f}, step = {step:.3f}, "
                    f"mean |v| = {float(coeff.abs().mean()):.4f} mm"
                )

        coeff = best_coeff

    # ---- final fields on the working grid ---------------------------------
    mm_to_vox_full = torch.tensor(
        [1.0 / float(z) for z in zooms], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1, 1)
    velocity = _resize(coeff, working_shape) * mm_to_vox_full
    disp = _exp_svf(velocity)
    warped = _warp(full, disp)
    logjac = _log_jacobian(disp)

    template_data = warped.mean(dim=0)[0].cpu().numpy().astype(np.float32)
    template_img = nib.Nifti1Image(template_data, working_affine)

    return LongitudinalWarpOutputs(
        template_img=template_img,
        coefficients=[c.cpu().numpy() for c in coeff],
        displacements=[d.cpu().numpy() for d in disp],
        log_jacobians=[j.cpu().numpy() for j in logjac],
        working_affine=working_affine,
        working_shape=working_shape,
        control_shape=control_shape,
    )


def displacement_to_mm(disp_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert a voxel displacement field to RAS millimetres.

    Args:
        disp_vox: ``(3, X, Y, Z)`` displacement in voxels, channel ``k`` along
            array axis ``k``.
        affine: The grid's 4x4 grid-to-RAS affine.

    Returns:
        ``(X, Y, Z, 3)`` displacement in millimetres, ready to save as a 4-D
        NIfTI.
    """
    ras = np.einsum("ij,jxyz->ixyz", np.asarray(affine)[:3, :3], disp_vox)
    return np.ascontiguousarray(np.moveaxis(ras, 0, -1).astype(np.float32))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Low-dimensional groupwise diffeomorphic registration of a "
            "longitudinal series (CAT12-style ageing model)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Rigidly realigned NIfTI time points, in acquisition order",
    )
    p.add_argument("--out-dir", required=True, help="Directory for outputs")
    p.add_argument(
        "--out-subfolders",
        nargs="+",
        help=(
            "Optional subfolder names, one per input. Outputs for each input go to "
            "<out-dir>/<subfolder_i>, which avoids collisions when inputs share basenames."
        ),
    )
    p.add_argument(
        "--control-spacing",
        type=float,
        default=12.0,
        help="Control-point spacing in mm; larger is stiffer",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=1.5,
        help="Isotropic working resolution in mm (0 keeps the input grid)",
    )
    p.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[2, 1],
        help="Coarse-to-fine image downsampling factors",
    )
    p.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[40, 40],
        help="Maximum iterations per scale (each level stops early on convergence)",
    )
    p.add_argument("--step-size", type=float, default=1.0, help="Demons update step")
    p.add_argument(
        "--smoothing-mm",
        type=float,
        default=6.0,
        help="Gaussian FWHM applied to the update field",
    )
    p.add_argument(
        "--regularisation",
        type=float,
        default=0.05,
        help="Weight of the membrane prior; 0 disables it",
    )
    p.add_argument(
        "--max-step-mm",
        type=float,
        default=0.15,
        help="Cap on how far one iteration may move a voxel, in mm",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device. CPU is the default on purpose: grid_sampler_3d has no "
            "MPS kernel before torch 2.9 and the fallback is slower than the CPU."
        ),
    )
    p.add_argument(
        "--save-displacement",
        action="store_true",
        help="Also write the 4-D RAS displacement field of exp(v) in mm",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Also write each time point warped onto the subject average. Off by "
            "default: the per-time-point outputs of the rest of the pipeline "
            "describe that time point's own anatomy, and warping changes that."
        ),
    )
    p.add_argument(
        "--save-template",
        action="store_true",
        help="Write the unbiased subject average",
    )
    p.add_argument("--verbose", action="store_true", help="Print convergence diagnostics")
    return p.parse_args(argv)


def _dest_dir(base: str, subfolders: Optional[Sequence[str]], idx: int) -> str:
    if subfolders is None:
        os.makedirs(base, exist_ok=True)
        return base
    dest = os.path.join(base, subfolders[idx])
    os.makedirs(dest, exist_ok=True)
    return dest


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m t1prep.warp_longitudinal``."""
    args = _parse_args(argv)

    if args.out_subfolders is not None and len(args.out_subfolders) != len(args.inputs):
        raise SystemExit(
            f"--out-subfolders expects {len(args.inputs)} entries (one per input); "
            f"got {len(args.out_subfolders)}"
        )
    if len(args.scales) != len(args.iterations):
        raise SystemExit(
            f"--scales and --iterations must have the same length "
            f"({len(args.scales)} vs {len(args.iterations)})"
        )

    images = [nib.load(p) for p in args.inputs]
    outputs = groupwise_svf(
        images,
        control_spacing=float(args.control_spacing),
        resolution=float(args.resolution) if args.resolution and args.resolution > 0 else None,
        scales=tuple(int(s) for s in args.scales),
        iterations=tuple(int(i) for i in args.iterations),
        step_size=float(args.step_size),
        smoothing_mm=float(args.smoothing_mm),
        regularisation=float(args.regularisation),
        max_step_mm=float(args.max_step_mm),
        device=torch.device(str(args.device)),
        verbose=bool(args.verbose),
    )

    affine = outputs.working_affine
    for idx, path in enumerate(args.inputs):
        stem, ext = _split_nifti_name(path)
        dest = _dest_dir(args.out_dir, args.out_subfolders, idx)

        nib.save(
            nib.Nifti1Image(outputs.log_jacobians[idx].astype(np.float32), affine),
            os.path.join(dest, f"{stem}_desc-longLogJacobian{ext}"),
        )
        if args.save_displacement:
            nib.save(
                nib.Nifti1Image(displacement_to_mm(outputs.displacements[idx], affine), affine),
                os.path.join(dest, f"{stem}_desc-longDisplacement{ext}"),
            )
        if args.apply:
            src = images[idx]
            if src.shape[:3] != outputs.working_shape or not np.allclose(src.affine, affine):
                src = resample_from_to(
                    src, nib.Nifti1Image(np.zeros(outputs.working_shape, np.float32), affine), order=1
                )
            data = torch.from_numpy(np.asarray(src.get_fdata(), dtype=np.float32))[None, None]
            disp = torch.from_numpy(outputs.displacements[idx])[None]
            warped = _warp(data, disp)[0, 0].numpy().astype(np.float32)
            nib.save(
                nib.Nifti1Image(warped, affine),
                os.path.join(dest, f"{stem}_desc-longWarped{ext}"),
            )

    if args.save_template:
        _, ext = _split_nifti_name(args.inputs[0])
        os.makedirs(args.out_dir, exist_ok=True)
        nib.save(outputs.template_img, os.path.join(args.out_dir, f"longitudinal_average{ext}"))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
