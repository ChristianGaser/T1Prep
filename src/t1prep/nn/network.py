"""A network: an architecture, its trained weights and the backend that runs it."""

from __future__ import annotations

from pathlib import Path

from .architectures import get_architecture
from .backends import get_backend
from .weights import load_safetensors

__all__ = ["Network"]


class Network:
    """A trained network, ready to run.

    The architecture is the one the file's ``arch`` metadata names, or the
    one its weight names identify.

    Args:
        path: The network's safetensors file (:mod:`t1prep.nn.weights`).
        backend: Backend name; default from ``T1PREP_NN_BACKEND``.
        device: Where to run, as a torch-style device or its name
            (``"cpu"``, ``"mps"``, ``"cuda"``); each backend maps it to its own.

    Calling the network runs it without gradients: a numpy array in gives a
    numpy array out, a torch tensor in gives a torch tensor out -- the
    contract of the TorchScript modules it replaces.
    """

    def __init__(self, path, backend: str | None = None, device="cpu"):
        arrays, metadata = load_safetensors(path)
        self.path = Path(path)
        self.architecture = get_architecture(metadata.get("arch"), arrays)
        self.backend = get_backend(backend)
        self.weights = self.backend.load(self.architecture.canonical(arrays), device)

    def __call__(self, x):
        return self.backend.run(self.architecture.forward, self.weights, x)

    def __repr__(self) -> str:
        return f"Network({self.path.name!r}, {self.architecture.name}, {self.backend.name})"
