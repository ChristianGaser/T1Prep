"""The tinygrad backend: Metal on Apple Silicon, CUDA and AMD on Linux.

tinygrad's ``conv2d``, ``conv_transpose2d`` and ``max_pool2d`` are
N-dimensional, so each operation is a one-liner.  Networks of one
architecture and input shape share one captured graph (``TinyJit``), with the
weights among its inputs: the 18 patch models replay a single graph instead
of being scheduled call by call, which cut the patch stage threefold.

Measured on an M2 in a whole IXI199 run against torch on MPS: segmentation
14.3 instead of 19.2 s and 7.9 instead of 15.3 GB peak, but skull-stripping
7.8 instead of 1.1 s -- a network called once pays tinygrad's Python-side
scheduling of its graph -- so the run took 110 instead of 106 s, and its peak
(15 GB, set by later stages) did not move.  Results differ from torch's in
the last digits (TIV and tissue volumes identical as reported).  tinygrad's
CPU path took 426 s for the segmentation against torch's 74 s.  tinygrad
changes its API between minor releases; the version is pinned in the
``tinygrad`` extra.
"""

from __future__ import annotations

import numpy as np
from tinygrad import Device, Tensor, TinyJit

from . import Backend


class TinygradBackend(Backend):
    """Runs the architectures with tinygrad on its default accelerator or CPU."""

    name = "tinygrad"

    def __init__(self):
        self._graphs: dict = {}

    @staticmethod
    def device(device) -> str:
        """tinygrad's device for a torch-style one: its CPU, or its default accelerator."""
        kind = str(getattr(device, "type", device)).split(":")[0].lower()
        return "CPU" if kind == "cpu" else Device.DEFAULT

    def load(self, arrays, device):
        device = self.device(device)
        return {k: Tensor(v, device=device).realize() for k, v in arrays.items()}

    def run(self, forward, weights, x):
        as_numpy = isinstance(x, np.ndarray)
        host = x if as_numpy else x.detach().cpu().numpy()
        names = tuple(sorted(weights))
        device = weights[names[0]].device
        # one graph per architecture, weight shapes and input shape
        shapes = tuple(tuple(weights[n].shape) for n in names)
        key = (forward, names, shapes, host.shape, device)
        if key not in self._graphs:

            def graph(inp, *tensors):
                return forward(self, dict(zip(names, tensors)), inp).realize()

            self._graphs[key] = TinyJit(graph)
        inp = Tensor(
            np.ascontiguousarray(host, dtype=np.float32), device=device
        ).realize()
        out = self._graphs[key](inp, *(weights[n] for n in names)).numpy()
        if as_numpy:
            return out
        import torch

        return torch.from_numpy(out).to(x.device)

    def release(self):
        for graph in self._graphs.values():
            graph.reset()
        self._graphs.clear()
        # Freed buffers stay in each device's allocator cache until let go
        for name in list(getattr(Device, "_opened_devices", ())):
            allocator = getattr(Device[name], "allocator", None)
            if hasattr(allocator, "free_cache"):
                allocator.free_cache()

    def conv(self, x, w, b=None, stride=1, padding=0):
        return x.conv2d(w, b, stride=stride, padding=padding)

    def conv_transpose(self, x, w, b=None, stride=1, padding=0, output_padding=0):
        return x.conv_transpose2d(
            w, b, stride=stride, padding=padding, output_padding=output_padding
        )

    def instance_norm(self, x):
        return x.layernorm(axis=tuple(range(2, x.ndim)), eps=1e-5)

    def relu(self, x):
        return x.relu()

    def max_pool(self, x, kernel, stride, padding):
        return x.max_pool2d(
            kernel_size=(kernel,) * (x.ndim - 2), stride=stride, padding=padding
        )

    def cat(self, a, b):
        return a.cat(b, dim=1)

    def sigmoid(self, x):
        return x.sigmoid()

    def softmax(self, x):
        return x.softmax(1)
