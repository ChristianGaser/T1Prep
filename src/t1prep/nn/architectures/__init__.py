"""Architectures: each network family, written once for every backend.

An architecture is a function ``forward(ops, weights, x)``.  *ops* is a
backend (:class:`t1prep.nn.backends.Backend`), *weights* maps the names of
the trained module's ``state_dict`` to the backend's tensors, and *x* is the
input.  The function uses only the backend's operations and arithmetic, so
any backend runs it.

``@register`` adds one, with the weight name that identifies its files and,
where trained modules named their weights differently, a function that
renames them to the names ``forward`` reads.  To add an architecture, write a
module next to these and import it at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["Architecture", "architecture_names", "get_architecture", "register"]


@dataclass(frozen=True)
class Architecture:
    """A registered architecture."""

    #: Name stored in a network file's ``arch`` metadata.
    name: str
    #: ``forward(ops, weights, x) -> y``.
    forward: Callable
    #: A weight name only this architecture's files contain.
    signature: str
    #: Maps a file's weights to the names ``forward`` reads.
    canonical: Callable


_REGISTRY: dict = {}


def register(name: str, signature: str, canonical: Callable | None = None):
    """Decorator registering ``forward`` as architecture *name*."""

    def decorate(forward):
        _REGISTRY[name] = Architecture(name, forward, signature, canonical or dict)
        return forward

    return decorate


def architecture_names() -> list:
    """Names of the registered architectures."""
    return list(_REGISTRY)


def get_architecture(
    name: str | None = None, weights: dict | None = None
) -> Architecture:
    """The architecture called *name*, or else the one *weights* belong to."""
    if name:
        if name not in _REGISTRY:
            raise ValueError(
                f"unknown architecture {name!r}: expected one of {', '.join(_REGISTRY)}"
            )
        return _REGISTRY[name]
    for architecture in _REGISTRY.values():
        if architecture.signature in architecture.canonical(weights or {}):
            return architecture
    raise ValueError(
        f"the weights match none of the architectures {', '.join(_REGISTRY)}"
    )


# Each module registers its architecture on import.
from . import linknet3d, unet3d  # noqa: E402,F401
