"""The torch backend: the aten operations deepmriprep's TorchScript archives recorded.

Each operation is the eager spelling of a call the archives make, with the
same arguments, so the networks give the archives' results bit for bit
(checked on CPU and MPS for all 21 networks).  On CUDA the last bits may
differ: the archives fixed cuDNN's flags at trace time (T1Prep requests
deterministic algorithms) and TorchScript's GPU fuser merged their
activation chains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import Backend


class TorchBackend(Backend):
    """Runs the architectures with PyTorch on CPU, MPS or CUDA."""

    name = "torch"

    def __init__(self):
        self._device_types: set = set()

    def load(self, arrays, device):
        device = torch.device(device)
        self._device_types.add(device.type)
        return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}

    def run(self, forward, weights, x):
        device = next(iter(weights.values())).device
        as_numpy = isinstance(x, np.ndarray)
        tensor = torch.from_numpy(np.ascontiguousarray(x)) if as_numpy else x
        with torch.no_grad():
            y = forward(self, weights, tensor.to(device))
        return y.cpu().numpy() if as_numpy else y

    def release(self):
        if "cuda" in self._device_types:
            torch.cuda.empty_cache()
        if "mps" in self._device_types:
            torch.mps.empty_cache()

    def conv(self, x, w, b=None, stride=1, padding=0):
        return F.conv3d(x, w, b, stride, padding)

    def conv_transpose(self, x, w, b=None, stride=1, padding=0, output_padding=0):
        return F.conv_transpose3d(x, w, b, stride, padding, output_padding)

    def instance_norm(self, x):
        return F.instance_norm(x, eps=1e-5)

    def relu(self, x):
        return F.relu(x)

    def max_pool(self, x, kernel, stride, padding):
        return F.max_pool3d(x, kernel, stride, padding)

    def cat(self, a, b):
        return torch.cat([a, b], 1)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def softmax(self, x):
        return torch.softmax(x, 1)
