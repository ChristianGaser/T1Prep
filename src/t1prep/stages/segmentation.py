"""Brain segmentation with deepmriprep's networks, run by T1Prep.

deepmriprep's ``BrainSegmentation`` (Fisch et al., MIT licence) on T1Prep's
own networks.  The coarse Unet3d sees the brain at 1/1.5 resolution, the 18
patch models refine 27 overlapping 128^3 patches (the last nine mirror the
first nine sagittally and reuse their models), and Gaussian weights blend
them into ``p0``.  Two things differ from upstream:

- ``p0`` is resized linearly, since B-spline interpolation made the label
  image negative in places and the segmentation less accurate;
- the networks run on the backend ``T1PREP_NN_BACKEND`` selects, and on the
  device T1Prep selected rather than the one deepmriprep pinned itself to.
  Going through ``resolve_device`` (rather than hard-coding MPS) is what keeps
  ``T1PREP_DEVICE=cpu`` meaningful for a CPU reference run.

With the torch backend ``p0`` is that of deepmriprep's archives, bit for
bit.  With tinygrad the torch operations around the networks stay on the
CPU, so this stage leaves the GPU to tinygrad.
"""

from __future__ import annotations

from pathlib import Path

import fill_voids
import numpy as np
import torch
import torch.nn.functional as F
from torchreg.utils import INTERP_KWARGS

from .._conv_chunk import chunked_conv3d
from .._device import interpolate, resolve_device
from .._intensity import scale_intensity
from ..nn.backends import backend_name
from ..nn.models import load_network

__all__ = ["BrainSegmentation"]

#: Lower corners of the 27 patches the patch models refine.
PATCHES_CSV = Path(__file__).resolve().parents[1] / "data" / "patches.csv"

#: The networks see the 339x411x339 working grid cropped to 336x384x336
#: (x 1:-2, y 15:-12, z :-3 of the RAS-ordered grid); their result is padded
#: back (``F.pad`` lists the last axis first).
_CROP = (slice(None), slice(None), slice(1, -2), slice(15, -12), slice(None, -3))
_PAD = (0, 3, 15, 12, 1, 2)

#: The patch models cover the first 18 patches; the rest mirror patches 0-8.
_N_MODELS = 18


def _smooth_kernel(kernel_size, sigma):
    """torchreg's ``smooth_kernel`` (MIT): a normalised separable Gaussian.

    Copied rather than imported for the explicit ``indexing="ij"``: torchreg
    leaves it to ``torch.meshgrid``'s default, which torch is about to require.
    """
    grids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32, device=sigma.device)
            for size in kernel_size
        ],
        indexing="ij",
    )
    kernel = 1
    for size, std, grid in zip(kernel_size, sigma, grids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * (2 * torch.pi) ** 0.5)
            * torch.exp(-(((grid - mean) / std) ** 2) / 2)
        )
    return kernel / kernel.sum()


def _patch_weights(patch_slices, shape, patch_shape, sigma):
    """deepmriprep's blending weights: a Gaussian per patch, normalised by their sum."""
    kernel = _smooth_kernel(patch_shape, sigma)[None, None]
    weight_sum = torch.zeros(shape, device=kernel.device)[None, None]
    for patch in patch_slices:
        weight_sum[patch] += kernel
    return [kernel / weight_sum[patch] for patch in patch_slices]


def _fill_holes(p0):
    """Label 1 for the holes inside the brain (``p0 > 0.9``), in place."""
    inside = p0[0, 0].detach().cpu().numpy() > 0.9
    holes = fill_voids.fill(inside) & ~inside
    if np.any(holes):
        p0[0, 0][torch.from_numpy(holes).to(p0.device)] = 1.0


class BrainSegmentation:
    """deepmriprep's brain segmentation on T1Prep's networks.

    Args:
        no_gpu: Run on the CPU even when an accelerator is available.
        shape: The cropped grid the patches tile.
        patch_shape: Shape of one patch.
        sigma: Width of the Gaussian blending weights, in voxels.
        fill_holes: Close holes inside the brain with label 1.
        chunk_conv: Run torch's CPU convolutions in slabs (no effect elsewhere):
            unsplit, the widest layer asks for a single 20.7 GB im2col buffer
            at this grid.
    """

    def __init__(
        self,
        no_gpu=False,
        shape=(336, 384, 336),
        patch_shape=(128, 128, 128),
        sigma=30.0,
        fill_holes=True,
        chunk_conv=True,
    ):
        backend = backend_name()
        device = torch.device("cpu") if no_gpu else resolve_device()[0]
        # tinygrad runs the networks; the torch operations around them stay on the CPU
        self.device = device if backend == "torch" else torch.device("cpu")
        self.coarse_model = load_network("segmentation_model", backend, device)
        self.patch_models = [
            load_network(f"segmentation_patch_{i}_model", backend, device)
            for i in range(_N_MODELS)
        ]
        corners = np.loadtxt(PATCHES_CSV, delimiter=",", skiprows=1, dtype=int)
        self.patch_slices = [
            (slice(None), slice(None))
            + tuple(slice(int(c), int(c) + s) for c, s in zip(corner, patch_shape))
            for corner in corners
        ]
        self.patch_weights = _patch_weights(
            self.patch_slices, shape, patch_shape, sigma * torch.ones(3)
        )
        self.fill_holes = fill_holes
        self.chunk_conv = chunk_conv

    def __call__(self, x):
        """``p0`` for the brain *x* on the working grid, ``(1, 1, 339, 411, 339)``.

        The result is on *x*'s grid and device.
        """
        device = x.device
        x = scale_intensity(x.to(self.device)[_CROP])
        with chunked_conv3d(enabled=self.chunk_conv):
            p0 = self.refine(x, self.coarse(x))
        if self.fill_holes:
            _fill_holes(p0)
        return F.pad(p0, _PAD).to(device)

    def coarse(self, x, scale_factor=1.5):
        """The coarse model's ``p0``, computed at 1/*scale_factor* resolution and resized back."""
        p0 = self.coarse_model(
            interpolate(x, scale_factor=1 / scale_factor, **INTERP_KWARGS)
        )
        return interpolate(p0, scale_factor=scale_factor, **INTERP_KWARGS)

    def refine(self, x, p0):
        """``p0`` refined patch by patch from the image and the coarse ``p0``."""
        refined = torch.zeros(x.shape, device=self.device)
        for i, (patch, weight) in enumerate(zip(self.patch_slices, self.patch_weights)):
            inp = torch.cat([x[patch], p0[patch]], dim=1)
            mirrored = i >= _N_MODELS
            out = self.patch_models[i % _N_MODELS](inp.flip(2) if mirrored else inp)
            refined[patch] += (out.flip(2) if mirrored else out) * weight.to(
                self.device
            )
        return refined

    def release(self) -> None:
        """Return the backend's cached memory; the weights stay loaded."""
        self.coarse_model.backend.release()
