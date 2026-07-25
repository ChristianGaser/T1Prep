"""Compute-device selection for the deepmriprep stages.

``deepmriprep`` decides its device once, at import, with

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

so on Apple Silicon every stage it owns — affine registration, nogm
segmentation, warping, atlas registration — pins itself to the CPU no matter
what T1Prep selects.  There is no argument or environment variable upstream to
change that: each class does ``torch.device('cpu' if no_gpu else DEVICE)`` in
its ``__init__``.

The models themselves are TorchScript archives loaded through
``deepbet.utils.load_model``, and their graphs follow the *input* tensor's
device rather than the module's registered parameters.  Routing the inputs is
therefore all that is needed, and it is worth doing: measured on torch 2.13 the
same forward pass takes 15.65 s on CPU and 2.62 s on MPS (nogm), 19.58 s vs
3.59 s (brain), 15.32 s vs 5.33 s (warp).

``DEVICE`` is imported by value into several deepmriprep submodules, so each
binding has to be rebound individually — patching ``deepmriprep.utils`` alone
would miss them.  This must run *before* ``Preprocess`` and its sub-models are
constructed, because that is when the device is read.

One op T1Prep exercises still has no MPS kernel as of torch 2.13,
``_adaptive_avg_pool3d`` (deepmriprep's ``interpolate(mode="area")``); it falls
back to the CPU automatically because ``segment.py`` sets
``PYTORCH_ENABLE_MPS_FALLBACK=1`` on macOS.
"""

import importlib
import os
from contextlib import contextmanager
from typing import Iterator, List, Set, Tuple

import torch

__all__ = [
    "resolve_device",
    "route_deepmriprep",
    "release_cache",
    "mps_routing_requested",
    "stage_target_device",
    "stage_device",
]

#: deepmriprep submodules that bind ``DEVICE`` at import time.
_DEEPMRIPREP_MODULES = (
    "utils",
    "preprocess",
    "segment",
    "register",
    "smooth",
    "atlas",
)

#: Escape hatch: ``T1PREP_DEVICE=cpu`` forces the whole pipeline back onto the
#: CPU, which is the reference for comparing MPS output.
_DEVICE_ENV = "T1PREP_DEVICE"


def resolve_device() -> Tuple[torch.device, bool]:
    """Return the compute device and the matching ``no_gpu`` flag.

    Honours ``T1PREP_DEVICE`` (``cpu``, ``mps``, ``cuda`` or ``auto``); with
    ``auto`` or unset, prefers CUDA, then MPS, then CPU.

    Returns:
        ``(device, no_gpu)`` where ``no_gpu`` is ``True`` only for the CPU, so
        it can be passed straight to deepmriprep's constructors.
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

    Routing deepmriprep to MPS is **opt-in** (``T1PREP_DEVICE=mps``) rather than
    automatic.  It makes each model forward 3-6x faster, but on a 24 GB machine
    a full run exhausts unified memory: the pure-PyTorch spline resampler in
    ``spline_resize`` materialises ~27 index/gather temporaries per call over
    volumes of tens of millions of voxels, and MPS executes asynchronously, so
    Python queues far more allocations than the GPU has retired.  Capping
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` does not help, which means the memory
    is live rather than reclaimable cache.  Until the resampling is kept on the
    CPU, leaving this off preserves today's behaviour: brain segmentation on
    MPS, the deepmriprep stages on CPU.
    """
    return os.environ.get(_DEVICE_ENV, "auto").strip().lower() == "mps"


#: Stages moved to the accelerator individually, as a comma-separated list.
#: Only ``warp`` is safe today: the affine, brain and nogm stages push
#: ``spline_resize``'s 27-tap gather over 47-million-voxel volumes, which is
#: what exhausts unified memory (see :func:`mps_routing_requested`), while
#: ``run_warp_register`` works entirely at the 113x137x113 warp grid.
_STAGES_ENV = "T1PREP_MPS_STAGES"
_DEFAULT_MPS_STAGES = "warp"


def _requested_stages() -> Set[str]:
    raw = os.environ.get(_STAGES_ENV, _DEFAULT_MPS_STAGES)
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def stage_target_device(stage: str, fallback: torch.device) -> torch.device:
    """Device that *stage* should run on.

    Args:
        stage: Stage name, matched against ``T1PREP_MPS_STAGES``.
        fallback: Device to use when the stage is not selected for the
            accelerator (normally the ``Preprocess`` device).

    Returns:
        The accelerator when this stage opted in and one is available,
        otherwise *fallback*.
    """
    if stage.lower() not in _requested_stages():
        return fallback
    device, _ = resolve_device()
    return device if device.type != "cpu" else fallback


@contextmanager
def stage_device(prep, device: torch.device) -> Iterator[None]:
    """Run one ``Preprocess`` stage on *device*, restoring the old one after.

    deepmriprep moves every stage's inputs to a single ``Preprocess.device``,
    so a stage can only be relocated by swapping that attribute around the
    call.  All ``run_*`` methods return CPU-side NIfTIs, so nothing device
    specific escapes the context.

    Args:
        prep: The ``Preprocess`` instance to retarget.
        device: Device to run the enclosed stage on.
    """
    previous = prep.device
    if torch.device(device) == torch.device(previous):
        yield
        return
    prep.device = torch.device(device)
    # SyNBase builds its identity grid lazily and caches it on the instance,
    # on whichever device the first call used; a stale one would be added to
    # flows living on the new device.
    syn = getattr(getattr(prep, "warp_register", None), "syn", None)
    stale_grid = syn is not None and getattr(syn, "_grid", None) is not None
    if stale_grid and syn._grid.device != prep.device:
        syn._grid = None
    try:
        yield
    finally:
        prep.device = previous


def release_cache(device: torch.device) -> None:
    """Return the accelerator's cached blocks to the system.

    deepmriprep frees between stages with ``torch.cuda.empty_cache()``, guarded
    by ``if self.device.type == 'cuda'``, so on MPS nothing is ever released:
    the caching allocator keeps every block it has touched.  Since
    ``segment.py`` also sets ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``, which
    removes the allocator's ceiling, that accumulation has no backstop and a
    multi-stage run can exhaust unified memory and push the machine into swap.

    Args:
        device: The device whose cache should be released.  CPU is a no-op.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def route_deepmriprep(device: torch.device) -> List[str]:
    """Point deepmriprep's stages at *device*.

    Args:
        device: Device the deepmriprep stages should run on.

    Returns:
        Names of the deepmriprep submodules whose ``DEVICE`` was rebound.
        Empty when deepmriprep is not importable.
    """
    patched: List[str] = []
    for name in _DEEPMRIPREP_MODULES:
        try:
            module = importlib.import_module(f"deepmriprep.{name}")
        except ImportError:
            continue
        if hasattr(module, "DEVICE"):
            module.DEVICE = device
            patched.append(name)
    return patched
