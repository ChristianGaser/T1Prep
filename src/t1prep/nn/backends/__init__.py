"""Backends: what runs the architectures.

A backend is an operation table -- the handful of tensor operations the
architectures call -- plus the plumbing to put weights on a device and to run
a network on an input.  ``torch`` is the default and reproduces deepmriprep's
TorchScript archives bit for bit on CPU and MPS; ``tinygrad`` (optional) runs on Metal,
CUDA and AMD with its own kernels.  ``T1PREP_NN_BACKEND`` selects one.

To add a backend, subclass :class:`Backend` in a new module of this package
and list it in ``_BACKENDS``; nothing else needs to know about it.
"""

from __future__ import annotations

import abc
import importlib
import os

__all__ = ["BACKEND_ENV", "Backend", "backend_name", "backend_names", "get_backend"]

#: Environment variable that selects the backend.
BACKEND_ENV = "T1PREP_NN_BACKEND"

#: Backend name -> ``"module:class"``, imported on first use, so that the
#: framework of an unused backend is never imported.  The first is the default.
_BACKENDS = {
    "torch": "t1prep.nn.backends.torch_backend:TorchBackend",
    "tinygrad": "t1prep.nn.backends.tinygrad_backend:TinygradBackend",
}

#: What to do when a backend's framework is missing.
_INSTALL_HINTS = {"tinygrad": "pip install 'T1Prep[tinygrad]' (needs Python >= 3.11)"}

_INSTANCES: dict = {}


class Backend(abc.ABC):
    """Runs architectures: an operation table plus tensor plumbing.

    An architecture (:mod:`t1prep.nn.architectures`) sees only the operations
    below and Python arithmetic (``+``, ``-``, ``*`` between tensors and with
    numbers), on 5-D ``(N, C, D, H, W)`` tensors of the backend's own type.
    Callers of :meth:`run` hand in numpy arrays or torch tensors and get the
    same kind back, so they need not know which backend runs.
    """

    #: Name under which the backend is registered.
    name: str = ""

    # -- plumbing ------------------------------------------------------------

    @abc.abstractmethod
    def load(self, arrays: dict, device) -> dict:
        """Put numpy weight *arrays* on *device*, a torch-style device or its name."""

    @abc.abstractmethod
    def run(self, forward, weights: dict, x):
        """Evaluate ``forward(self, weights, x)`` without gradients.

        A numpy *x* gives a numpy result.  A torch *x* gives a torch result:
        with the torch backend on the weights' device, otherwise on *x*'s.
        """

    def release(self) -> None:
        """Free what is kept between calls (caches, captured graphs); weights stay."""

    # -- operations ----------------------------------------------------------

    @abc.abstractmethod
    def conv(self, x, w, b=None, stride=1, padding=0):
        """3-D convolution, weights ``(C_out, C_in, k, k, k)``."""

    @abc.abstractmethod
    def conv_transpose(self, x, w, b=None, stride=1, padding=0, output_padding=0):
        """3-D transposed convolution, weights ``(C_in, C_out, k, k, k)`` as torch."""

    @abc.abstractmethod
    def instance_norm(self, x):
        """Normalise each channel over D, H, W: biased variance, eps 1e-5, no affine."""

    @abc.abstractmethod
    def relu(self, x):
        """max(x, 0)."""

    @abc.abstractmethod
    def max_pool(self, x, kernel, stride, padding):
        """3-D max-pooling; padding never wins."""

    @abc.abstractmethod
    def cat(self, a, b):
        """Concatenate along the channels."""

    @abc.abstractmethod
    def sigmoid(self, x):
        """Logistic function."""

    @abc.abstractmethod
    def softmax(self, x):
        """Softmax over the channels."""


def backend_names() -> list:
    """Registered backend names, the default first."""
    return list(_BACKENDS)


def backend_name() -> str:
    """The backend ``T1PREP_NN_BACKEND`` selects; the default when unset."""
    name = os.environ.get(BACKEND_ENV, "").strip().lower() or next(iter(_BACKENDS))
    if name not in _BACKENDS:
        raise ValueError(
            f"{BACKEND_ENV}={name!r}: expected one of {', '.join(_BACKENDS)}"
        )
    return name


def get_backend(name: str | None = None) -> Backend:
    """The shared instance of backend *name* (default: :func:`backend_name`)."""
    name = name or backend_name()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}: expected one of {', '.join(_BACKENDS)}"
        )
    if name not in _INSTANCES:
        module, cls = _BACKENDS[name].split(":")
        try:
            _INSTANCES[name] = getattr(importlib.import_module(module), cls)()
        except ImportError as exc:
            hint = _INSTALL_HINTS.get(name, "install it")
            raise RuntimeError(
                f"backend {name!r} is not available ({exc}): {hint}"
            ) from exc
    return _INSTANCES[name]
