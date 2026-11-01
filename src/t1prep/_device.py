"""Compute-device selection.

Two devices matter in a run:

- :func:`resolve_device` -- the accelerator T1Prep selected (CUDA, else MPS,
  else the CPU; ``T1PREP_DEVICE`` overrides).  The networks run there
  (:mod:`t1prep.nn`): measured on torch 2.13 the brain segmentation took
  19.58 s on the CPU and 3.59 s on MPS.
- :func:`stage_device` -- where the registrations taken over from deepmriprep
  run (affine and atlas registration).  The same device, except that they
  stay on the CPU on Apple Silicon unless ``T1PREP_DEVICE=mps`` asks for MPS
  (see :func:`mps_routing_requested`).  That is the policy deepmriprep's own
  device choice amounted to.

One op T1Prep exercises still has no MPS kernel as of torch 2.13,
``_adaptive_avg_pool3d`` (``interpolate(mode="area")``); it falls back to the
CPU automatically because ``segment.py`` sets ``PYTORCH_ENABLE_MPS_FALLBACK=1``
on macOS.

The deterministic algorithms ``segment.py`` enables slow down one op on an
accelerator for nothing T1Prep uses, linear interpolation;
:func:`interpolate` gives it its own kernel back.
"""

import os
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "interpolate",
    "mps_routing_requested",
    "release_cache",
    "resolve_device",
    "stage_device",
]

#: Escape hatch: ``T1PREP_DEVICE=cpu`` forces the whole pipeline back onto the
#: CPU, which is the reference for comparing MPS output.
_DEVICE_ENV = "T1PREP_DEVICE"


def resolve_device() -> Tuple[torch.device, bool]:
    """Return the compute device and the matching ``no_gpu`` flag.

    Honours ``T1PREP_DEVICE`` (``cpu``, ``mps``, ``cuda`` or ``auto``); with
    ``auto`` or unset, prefers CUDA, then MPS, then CPU.

    Returns:
        ``(device, no_gpu)`` where ``no_gpu`` is ``True`` only for the CPU.
    """
    requested = os.environ.get(_DEVICE_ENV, "auto").strip().lower()
    if requested in ("cpu", "mps", "cuda"):
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        elif requested == "mps" and not torch.backends.mps.is_available():
            requested = "cpu"
        return torch.device(requested), requested == "cpu"

    if torch.cuda.is_available():
        return torch.device("cuda"), False
    if torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), True


def mps_routing_requested() -> bool:
    """Whether the caller explicitly asked for the MPS-everywhere path.

    Running the registrations on MPS is **opt-in** (``T1PREP_DEVICE=mps``)
    rather than automatic.  It makes them faster, but on a 24 GB machine a
    full run exhausts unified memory: the pure-PyTorch spline resampler in
    ``spline_resize`` materialises ~27 index/gather temporaries per call over
    volumes of tens of millions of voxels, and MPS executes asynchronously, so
    Python queues far more allocations than the GPU has retired.  Capping
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` does not help, which means the memory
    is live rather than reclaimable cache.
    """
    return os.environ.get(_DEVICE_ENV, "auto").strip().lower() == "mps"


def stage_device() -> torch.device:
    """Device of the registrations taken over from deepmriprep.

    The device :func:`resolve_device` selects, except MPS, which they use only
    when :func:`mps_routing_requested`.
    """
    device, _ = resolve_device()
    if device.type == "mps" and not mps_routing_requested():
        return torch.device("cpu")
    return device


def release_cache(device: torch.device) -> None:
    """Return the accelerator's cached blocks to the system.

    PyTorch's caching allocator keeps every block it has touched.  Since
    ``segment.py`` sets ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``, which removes
    the allocator's ceiling on MPS, that accumulation has no backstop and a
    multi-stage run can exhaust unified memory and push the machine into swap.

    Args:
        device: The device whose cache should be released.  CPU is a no-op.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def interpolate(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """``torch.nn.functional.interpolate``, with its kernel on an accelerator.

    Under deterministic algorithms torch computes (bi/tri)linear and bicubic
    interpolation on CUDA and MPS with a Python decomposition, so that its
    *backward* is deterministic.  The forward kernels are deterministic
    already; the decomposition only costs.  Measured on an M2 (torch 2.15) it
    took 0.7 s instead of 14 ms to upsample onto the working grid, and an
    IXI199 run 6 s longer and 3 GB higher in peak memory.  A call that records
    no gradient therefore dispatches directly to the native interpolation
    operator, without changing the process-wide deterministic setting.  On
    the CPU, or when a gradient is recorded, this is ``F.interpolate``
    unchanged.

    T1Prep calls this instead of ``F.interpolate`` (``tests/test_device.py``
    checks that).
    """
    if (
        input.is_cpu
        or not torch.are_deterministic_algorithms_enabled()
        or (input.requires_grad and torch.is_grad_enabled())
    ):
        return F.interpolate(input, *args, **kwargs)

    names = (
        "size",
        "scale_factor",
        "mode",
        "align_corners",
        "recompute_scale_factor",
        "antialias",
    )
    if len(args) > len(names) or any(name in kwargs for name in names[: len(args)]):
        return F.interpolate(input, *args, **kwargs)
    options = dict(zip(names, args))
    options.update(kwargs)
    if set(options) - set(names):
        return F.interpolate(input, *args, **kwargs)
    options = {
        "size": None,
        "scale_factor": None,
        "mode": "nearest",
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False,
        **options,
    }

    dimensions = {
        ("linear", 3): "upsample_linear1d",
        ("bilinear", 4): "upsample_bilinear2d",
        ("bicubic", 4): "upsample_bicubic2d",
        ("trilinear", 5): "upsample_trilinear3d",
    }
    mode = options["mode"]
    if not isinstance(mode, str):
        return F.interpolate(input, *args, **kwargs)
    operator = dimensions.get((mode, input.dim()))
    if operator is None or options["antialias"]:
        return F.interpolate(input, *args, **kwargs)

    size = options["size"]
    scale_factor = options["scale_factor"]
    recompute = options["recompute_scale_factor"]
    spatial_dims = input.dim() - 2
    if size is not None and recompute is not None:
        return F.interpolate(input, *args, **kwargs)
    if size is not None and scale_factor is not None:
        return F.interpolate(input, *args, **kwargs)
    if size is not None:
        output_size = (size,) * spatial_dims if isinstance(size, int) else tuple(size)
        if len(output_size) != spatial_dims:
            return F.interpolate(input, *args, **kwargs)
        scales = (None,) * spatial_dims
    elif scale_factor is not None:
        scales = (
            (scale_factor,) * spatial_dims
            if isinstance(scale_factor, (int, float))
            else tuple(scale_factor)
        )
        if len(scales) != spatial_dims:
            return F.interpolate(input, *args, **kwargs)
        if recompute:
            output_size = tuple(
                int(input.shape[i + 2] * scales[i]) for i in range(spatial_dims)
            )
            scales = (None,) * spatial_dims
        else:
            output_size = None
    else:
        return F.interpolate(input, *args, **kwargs)

    align_corners = options["align_corners"]
    if align_corners is None:
        align_corners = False
    return getattr(torch.ops.aten, operator).default(
        input, output_size, align_corners, *scales
    )
