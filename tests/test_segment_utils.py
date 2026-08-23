"""Tests for :mod:`t1prep._segment_utils`."""

import numpy as np
import pytest
import torch

from t1prep._segment_utils import scale_intensity


def _reference(x, low=0.5, high=99.5):
    """Percentile rescaling with log-compressed top tail, computed on the host.

    Deliberately written with plain NumPy rather than the tensor ops under
    test, so an error in the implementation cannot cancel itself out.
    """
    arr = x.detach().cpu().numpy().astype(np.float64)
    lo = np.percentile(arr[arr > 0], low)
    hi = np.percentile(arr[arr > 0], high)
    out = (arr - lo) / (hi - lo)
    return np.where(out > 1, 1 + np.log10(np.maximum(out, 1)), out)


def _volumes():
    torch.manual_seed(0)
    return {
        "typical": torch.rand(1, 1, 24, 26, 22) ** 3 * 4.0,
        "with_background": (torch.rand(1, 1, 20, 20, 20) ** 3 * 4.0)
        * (torch.rand(1, 1, 20, 20, 20) > 0.3),
        # Nothing exceeds the upper percentile, so the log branch is never
        # taken and the result must pass through untouched.
        "no_tail": torch.rand(1, 1, 16, 16, 16) * 0.4 + 0.01,
    }


@pytest.mark.parametrize("name", sorted(_volumes()))
def test_matches_reference(name):
    x = _volumes()[name]
    got = scale_intensity(x.clone()).cpu().numpy()
    np.testing.assert_allclose(got, _reference(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(_volumes()))
def test_output_is_finite(name):
    """The unused log branch must not leak -inf/NaN for values <= 1."""
    got = scale_intensity(_volumes()[name].clone())
    assert torch.isfinite(got).all()


def test_input_not_mutated():
    x = _volumes()["typical"]
    before = x.clone()
    scale_intensity(x)
    assert torch.equal(x, before)


def test_non_contiguous_input():
    """``brain_segment`` passes a cropped view, not a contiguous tensor."""
    full = torch.rand(1, 1, 30, 32, 28) ** 3 * 4.0
    view = full[:, :, 1:-2, 15:-12, :-3]
    assert not view.is_contiguous()
    got = scale_intensity(view).cpu().numpy()
    np.testing.assert_allclose(got, _reference(view), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_mps_matches_cpu():
    """Regression guard for the mask-indexing shape mismatch seen on MPS.

    The old implementation resolved ``x > 1`` once for the read and once for
    the write; when those disagreed the call raised
    "shape mismatch: value tensor of shape [N] ... indexing result of shape [M]".
    """
    x = _volumes()["typical"]
    on_cpu = scale_intensity(x.clone().to("cpu"))
    on_mps = scale_intensity(x.clone().to("mps")).cpu()
    assert torch.equal(on_cpu, on_mps)
