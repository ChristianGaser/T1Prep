"""Tests for :mod:`t1prep._intensity`."""

import numpy as np
import pytest
import torch

# Allow running tests without installing the package (repo checkout / editable dev)
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from t1prep._intensity import scale_intensity


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


# ---------------------------------------------------------------------------
# Bias-field fits: the dense SplineSmooth3D evaluation and the linear-binning
# histogram must give what the library code they replace gives.
# ---------------------------------------------------------------------------

from nxbc.filter import distrib_kde, kernelfntri  # noqa: E402
from SplineSmooth3D.SplineSmooth3D import (  # noqa: E402
    SplineSmooth3D,
    SplineSmooth3DUnregularized,
)

from t1prep import _intensity  # noqa: E402


def _fit_grid():
    """A small fit grid as _fit_log_field builds it: domain [0, 1], a mask."""
    rng = np.random.default_rng(7)
    data = rng.random((9, 11, 10)).astype(np.float32)
    mask = rng.random(data.shape) < 0.8
    return data, mask, 1 / (np.array(data.shape) - 1)


def test_dense_fit_and_predict_match_the_library():
    data, mask, vox = _fit_grid()
    lib = SplineSmooth3DUnregularized(data, vox, 1.0, domainMethod="minc", mask=mask)
    new = _intensity._DenseSplineFit(data, vox, 1.0, domainMethod="minc", mask=mask)
    rng = np.random.default_rng(8)
    for level in range(3):
        for _ in range(2):  # the first fit builds the weights, the second reuses them
            update = (rng.random(data.shape) - 0.5).astype(np.float32)
            lib.fit(update)
            new.fit(update)
            np.testing.assert_allclose(new.omegaWeights, lib.omegaWeights, rtol=1e-12)
            np.testing.assert_allclose(new.AtA, lib.AtA, rtol=1e-12)
            np.testing.assert_allclose(new.Atx, lib.Atx, rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose(new.predict(), lib.predict(), rtol=1e-10, atol=1e-13)
        lib, new = lib.promote(), new.promote()
        assert type(new) is _intensity._DenseSplineFit, f"level {level + 1}"


def test_dense_predict_matches_the_library_on_the_full_grid():
    # the predictor evaluates the fit's control points on the full-resolution grid
    data, mask, vox = _fit_grid()
    fit = SplineSmooth3DUnregularized(data, vox, 1.0, domainMethod="minc", mask=mask)
    fit.fit((np.random.default_rng(9).random(data.shape) - 0.5).astype(np.float32))
    fit.solve()
    full = np.zeros((41, 51, 46))
    full_vox = vox / _intensity._FIT_SUBSAMPLE
    lib = SplineSmooth3D(full, full_vox, 1.0, knts=fit.kntsArr, dofit=False)
    new = _intensity._DenseSplinePredict(full, full_vox, 1.0, knts=fit.kntsArr, dofit=False)
    for level in range(3):
        lib.P = fit.P.copy()
        new.P = fit.P.copy()
        np.testing.assert_allclose(new.predict(), lib.predict(), rtol=1e-10, atol=1e-13)
        fit = fit.promote()
        lib, new = lib.promote(), new.promote()
        assert type(new) is _intensity._DenseSplinePredict, f"level {level + 1}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_triangular_kde_matches_nxbc(dtype):
    data = (np.random.default_rng(3).normal(size=20000) * 0.3 + 1.0).astype(dtype)
    hist, centres, width = _intensity._triangular_kde(data, 256)
    want, _, want_centres, want_width = distrib_kde(
        data, 256, kernfn=kernelfntri, binCentreLimits=True
    )
    np.testing.assert_array_equal(centres, want_centres)
    assert width == want_width
    # nxbc sums float32 kernel values for float32 data, so allow its rounding
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(hist, want, rtol=rtol, atol=rtol * want.max())


def test_triangular_kde_leaves_constant_data_to_nxbc():
    data = np.full(100, 2.5)
    hist, centres, width = _intensity._triangular_kde(data, 16)
    want, _, want_centres, want_width = distrib_kde(
        data, 16, kernfn=kernelfntri, binCentreLimits=True
    )
    np.testing.assert_array_equal(hist, want)
    np.testing.assert_array_equal(centres, want_centres)
    np.testing.assert_array_equal(width, want_width)


def test_fit_log_field_matches_the_library(monkeypatch):
    # a smooth multiplicative field on a tissue-like image; the library's
    # classes and histogram swapped back in give the reference run
    rng = np.random.default_rng(11)
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, n) for n in (41, 51, 46)), indexing="ij")
    field = np.exp(0.15 * x - 0.1 * y * z + 0.05 * z)
    full = (field * (1.0 + 0.02 * rng.standard_normal(field.shape))).astype(np.float64)
    data_sub = full[::5, ::5, ::5]
    mask_sub = rng.random(data_sub.shape) < 0.7
    args = (full, data_sub, mask_sub, 1.0, 30, 5e-4, True)

    new = _intensity._fit_log_field(*args)
    monkeypatch.setattr(_intensity, "_DenseSplineFit", SplineSmooth3DUnregularized)
    monkeypatch.setattr(_intensity, "_DenseSplinePredict", SplineSmooth3D)
    monkeypatch.setattr(
        _intensity,
        "_triangular_kde",
        lambda d, n: (lambda h, e, c, w: (h, c, w))(
            *distrib_kde(d, n, kernfn=kernelfntri, binCentreLimits=True)
        ),
    )
    lib = _intensity._fit_log_field(*args)
    assert new.shape == lib.shape == full.shape
    np.testing.assert_allclose(new, lib, rtol=1e-6, atol=1e-7)
