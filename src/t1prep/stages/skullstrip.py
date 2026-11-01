"""Skull-stripping with deepbet's networks, run by T1Prep.

A port of ``deepbet.BrainExtraction`` (Fisch et al., MIT licence) whose two
LinkNet3d models -- a bounding-box model at 128^3 and the brain model at
256^3 -- are T1Prep's own networks.  The operations around them are
deepbet's, in deepbet's order, so with the torch backend the mask is
deepbet's to the bit.  Only the connected-component labelling moved from
``cc3d`` to ``scipy.ndimage.label``, with the same 26-connectivity.

With the tinygrad backend the torch operations around the networks stay on
the CPU, so this stage leaves the GPU to tinygrad.
"""

from __future__ import annotations

import fill_voids
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from .._device import interpolate, resolve_device
from ..nn.backends import backend_name
from ..nn.models import load_network
from ..utils import reoriented_nifti

__all__ = ["BrainExtraction"]

#: 26-connectivity, cc3d's default for 3-D labels.
_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def normalize(x, low, high):
    """deepbet's input scaling: clip to [low, high], standardise, map to ~ImageNet range."""
    x = (x - low) / (high - low)
    x = x.clamp(min=0, max=1)
    x = (x - x.mean()) / x.std()
    return 0.226 * x + 0.449


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Zero everything but the largest 26-connected component of *mask* (in place)."""
    labels, _ = ndimage.label(mask, structure=_CONNECTIVITY)
    largest = np.bincount(labels[labels != 0]).argmax()
    mask[labels != largest] = 0
    return mask


def _bounding_box(mask_small, shape, margin, threshold=0.02):
    """deepbet's head box on a grid of *shape*.

    The slab where *mask_small*'s projections onto each axis exceed
    *threshold*, scaled from *mask_small*'s grid to *shape* and widened by
    *margin* (a fraction of its size) on each side.
    """
    rs = [
        torch.where(mask_small.mean(dims) > threshold)[0]
        for dims in [(1, 2), (0, 2), (0, 1)]
    ]
    assert all(
        r.numel() > 0 for r in rs
    ), "Not enough foreground to calculate bounding box"
    center = torch.tensor([(r.max() + 1 + r.min()) / 2 for r in rs])
    size = torch.tensor([(r.max() + 1) - r.min() for r in rs])
    margin = margin * torch.ones(3)
    scale_factor = torch.tensor(shape) / torch.tensor(mask_small.shape)
    center, size = scale_factor * center, scale_factor * size
    size = (1 + 2 * margin) * size
    center, size = center.round(), size.round()
    bbox = [[int(c - s / 2), int(c + s / 2)] for c, s in zip(center, size)]
    return tuple(slice(max(0, b[0]), min(s, b[1]), 1) for b, s in zip(bbox, shape))


class BrainExtraction:
    """deepbet's skull-stripping on T1Prep's networks.

    Args:
        no_gpu: Run on the CPU even when an accelerator is available.
    """

    def __init__(self, no_gpu: bool = False):
        backend = backend_name()
        device = torch.device("cpu") if no_gpu else resolve_device()[0]
        # tinygrad runs the networks; the torch operations around them stay on the CPU
        self.device = device if backend == "torch" else torch.device("cpu")
        self.bbox_model = load_network("brain_extraction_bbox_model", backend, device)
        self.model = load_network("brain_extraction_model", backend, device)

    def __call__(self, img: nib.Nifti1Image, threshold: float = 0.5):
        """Skull-strip *img*: ``(brain, mask)``, the masked image and its uint8 mask."""
        x = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
        x = x[..., 0] if len(img.shape) == 4 else x
        probability, bbox = self.probability(x.copy())
        mask = probability > threshold
        mask[bbox] = keep_largest_connected_component(mask[bbox])
        mask[bbox] = fill_voids.fill(mask[bbox])
        # deepbet's apply_mask: an F-ordered copy, so the caller's image
        # (get_fdata may hand back its own buffer) stays as it was, and the
        # brain has deepbet's memory layout for whatever sums over it later
        x = x.flatten("F")
        x[~mask.flatten("F")] = 0.0
        x = x.reshape(mask.shape, order="F")
        if len(img.shape) == 4:
            x, mask = x[..., None], mask[..., None]
        brain = reoriented_nifti(x, img.affine, img.header)
        mask = reoriented_nifti(mask, brain.affine, brain.header)
        mask.header.set_data_dtype(np.uint8)
        return brain, mask

    def probability(
        self,
        x: np.ndarray,
        small_shape=(128, 128, 128),
        shape=(256, 256, 256),
        margin=0.1,
    ):
        """Brain probability on *x*'s grid, and the box it was computed in.

        The bounding-box model finds the head at *small_shape*; the brain model
        then sees that box, widened by *margin*, at *shape*.
        """
        x = torch.nan_to_num(torch.from_numpy(x).to(self.device))
        probability = torch.zeros_like(x)
        x_small = interpolate(x[None, None], small_shape, mode="nearest-exact")[0, 0]
        low, high = x_small.quantile(0.005), x_small.quantile(0.995)
        head = self.bbox_model(normalize(x_small, low, high)[None, None])[0, 1]
        head = keep_largest_connected_component((head > 0.5).float().cpu().numpy())
        bbox = _bounding_box(torch.from_numpy(head).to(self.device), x.shape, margin)
        x_box = interpolate(x[bbox][None, None], shape, mode="nearest-exact")[0, 0]
        brain = self.model(normalize(x_box, low, high)[None, None])[0, 1]
        probability[bbox] = interpolate(
            brain[None, None], probability[bbox].shape, mode="nearest-exact"
        )[0, 0]
        return probability.cpu().numpy(), bbox

    def release(self) -> None:
        """Return the backend's cached memory; the weights stay loaded."""
        self.model.backend.release()
