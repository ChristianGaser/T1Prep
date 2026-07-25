"""Memory-bounded 3-D convolution for the CPU inference path.

PyTorch's CPU ``conv3d`` lowers to an im2col matrix multiplication and
materialises the column buffer for the *whole* volume in a single allocation of
``C_in * prod(kernel) * prod(output_shape)`` elements.  At the 1.5 mm warp
template grid (padded to 128x144x128) the 32-channel layer of the warp model
therefore asks for ``32 * 27 * 2.36e6 * 4 B`` = 8.2 GB, which dominates the peak
resident memory of the whole warping stage.

Splitting the output into slabs along the slowest spatial axis splits only the
N dimension of the underlying gemm: every output element is still accumulated
over exactly the same K values, so the computation is unchanged in exact
arithmetic while the column buffer stays under ``budget`` bytes.  It is also
faster in practice, because the smaller buffers stay closer to cache.

In floating point the equivalence is not guaranteed to the last bit, because
the CPU kernel chooses its blocking from the tensor size and the accumulation
order can therefore differ; on small test volumes the two paths disagree by
about two ulp (~2e-7 relative).  On the shapes this is deployed for — the warp
model at the 1.5 mm template grid — the split result was verified *bitwise*
identical to the one-shot result at every budget from 64 MB to 1 GB.

The split is installed as a ``TorchDispatchMode`` rather than a module patch so
it also applies inside TorchScript-traced models such as ``warp_model.pt``,
whose convolutions never pass through Python.

Usage::

    from t1prep._conv_chunk import chunked_conv3d

    with chunked_conv3d():
        out = prep.run_warp_register(p0_large, p1_affine, p2_affine, wj_affine)
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["ChunkedConv3d", "chunked_conv3d"]

#: Column-buffer ceiling per convolution.  Measured on the warp model at the
#: 1.5 mm grid, the peak keeps falling until ~256 MB (14.3 GB -> 3.5 GB) and
#: flattens out below it, while smaller budgets start costing wall-clock.
DEFAULT_BUDGET = 256 << 20

#: Both spellings of the forward convolution reach the dispatcher: eager code
#: emits ``aten::convolution``, TorchScript graphs emit ``aten::_convolution``.
_CONV_OPS = (
    torch.ops.aten.convolution.default,
    torch.ops.aten._convolution.default,
)


class ChunkedConv3d(TorchDispatchMode):
    """Evaluate oversized CPU 3-D convolutions slab by slab.

    Only plain forward convolutions are redirected: 5-D float32 CPU input,
    ``groups=1``, unit stride and unit dilation.  Anything else — transposed
    convolutions, strided/dilated layers, GPU tensors, half precision — falls
    through to the stock kernel untouched.

    Args:
        budget: Maximum im2col column-buffer size in bytes.  Convolutions whose
            buffer would stay below this are left alone.
    """

    def __init__(self, budget: int = DEFAULT_BUDGET):
        super().__init__()
        self.budget = int(budget)
        self.chunked_calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _CONV_OPS and not kwargs and len(args) >= 9:
            out = self._chunk(args)
            if out is not None:
                self.chunked_calls += 1
                return out
        return func(*args, **kwargs)

    def _chunk(self, args):
        """Return the slab-wise result, or ``None`` to use the stock kernel."""
        # ``_convolution`` trails four backend flags after the shared prefix;
        # TorchScript graphs emit that variant, eager code emits ``convolution``.
        (inp, weight, bias, stride, padding, dilation, transposed,
         _output_padding, groups) = args[:9]

        if (
            transposed
            or groups != 1
            or inp.dim() != 5
            or inp.device.type != "cpu"
            or inp.dtype != torch.float32
            or any(int(s) != 1 for s in stride)
            or any(int(d) != 1 for d in dilation)
        ):
            return None

        pad = [int(p) for p in padding]
        kernel = [int(k) for k in weight.shape[2:]]
        out_shape = [
            int(inp.shape[2 + i]) + 2 * pad[i] - kernel[i] + 1 for i in range(3)
        ]
        if min(out_shape) < 1:
            return None

        # What the stock kernel would allocate for the column buffer: one row
        # per gemm K value (C_in * kernel volume), one column per output voxel.
        k_elems = int(weight.shape[1]) * kernel[0] * kernel[1] * kernel[2]
        columns = k_elems * out_shape[0] * out_shape[1] * out_shape[2]
        if columns * inp.element_size() <= self.budget:
            return None

        n_slabs = -(-columns * inp.element_size() // self.budget)
        depth = out_shape[0]
        step = max(1, -(-depth // n_slabs))
        halo = kernel[0] - 1
        depth_in = int(inp.shape[2])

        # Write slabs straight into the result and pad each slab individually:
        # a shared padded copy plus a final ``cat`` would each add a full-volume
        # temporary, which at these tensor sizes is most of what we just saved.
        out = inp.new_empty(
            (int(inp.shape[0]), int(weight.shape[0])) + tuple(out_shape)
        )
        for d0 in range(0, depth, step):
            d1 = min(d0 + step, depth)
            # Slab [d0, d1) of the padded volume needs padded planes
            # [d0, d1 + halo), i.e. input planes shifted by the D padding.
            lo, hi = d0 - pad[0], d1 + halo - pad[0]
            slab = inp[:, :, max(lo, 0):min(hi, depth_in)]
            front, back = max(-lo, 0), max(hi - depth_in, 0)
            if front or back or pad[1] or pad[2]:
                slab = F.pad(
                    slab, (pad[2], pad[2], pad[1], pad[1], front, back)
                )
            out[:, :, d0:d1] = F.conv3d(slab, weight, bias, padding=0)
        return out


@contextmanager
def chunked_conv3d(budget: int = DEFAULT_BUDGET, enabled: bool = True):
    """Context manager applying :class:`ChunkedConv3d` to enclosed code.

    Args:
        budget: Column-buffer ceiling in bytes, see :class:`ChunkedConv3d`.
        enabled: When ``False`` the context is a no-op, so callers can gate the
            optimisation without branching at every call site.

    Yields:
        The active :class:`ChunkedConv3d` instance, or ``None`` when disabled.
        Its ``chunked_calls`` counter reports how many convolutions were split.
    """
    if not enabled:
        yield None
        return
    mode = ChunkedConv3d(budget)
    with mode:
        yield mode
