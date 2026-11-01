"""T1Prep's neural networks: defined once, stored as safetensors, run on any backend.

T1Prep runs deepbet's LinkNet3d (skull-stripping) and deepmriprep's Unet3d
(segmentation).  Upstream ships them as TorchScript archives, which only torch
can run and which torch is retiring.  This package holds the architectures
itself, keeps the weights in a framework-neutral format, and lets any backend
run them.  The layers, from the bottom up:

``weights``
    safetensors files <-> numpy arrays, no framework involved.
``backends``
    an operation table (convolution, instance norm, ...) per framework:
    ``torch`` (default, bit-identical to the archives on CPU and MPS) and ``tinygrad``
    (optional; Metal, CUDA, AMD).  ``T1PREP_NN_BACKEND`` selects one.
``architectures``
    each network family written once, against a backend's operations.
``network``
    :class:`Network` = an architecture + its weights + a backend.
``models``
    the networks T1Prep ships: names, where their files live, and how they
    are downloaded and converted (``t1prep-download-models``).
``convert``
    TorchScript archives and torch ``state_dict``s to network files, also
    for retrained models (``python -m t1prep.nn.convert``).

The pipeline stages that run the networks -- skull-stripping and
segmentation -- are in :mod:`t1prep.stages`.

Extending it:

- **a backend** -- subclass :class:`~t1prep.nn.backends.Backend` in a new
  module of ``backends/`` and list it in ``backends._BACKENDS``;
- **an architecture** -- write ``forward(ops, weights, x)`` in a new module
  of ``architectures/``, decorate it with ``@register`` and import it there;
- **a (re)trained model** -- export its ``state_dict`` with
  :func:`~t1prep.nn.convert.save_state_dict` and put the file in
  ``models.MODEL_DIR`` under its name in ``models.NETWORKS``.
"""

from .backends import BACKEND_ENV, Backend, backend_name, backend_names, get_backend
from .network import Network

__all__ = [
    "BACKEND_ENV",
    "Backend",
    "Network",
    "backend_name",
    "backend_names",
    "get_backend",
]
