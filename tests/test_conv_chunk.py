import sys
import unittest
from pathlib import Path

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is optional for the test env
    torch = None

if torch is not None:
    from t1prep._conv_chunk import ChunkedConv3d, chunked_conv3d


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestChunkedConv3d(unittest.TestCase):
    """Slab-wise convolution must agree with the stock kernel to rounding.

    Slabbing splits only the N dimension of the gemm, so the maths is
    unchanged; the CPU kernel may still pick a different accumulation order for
    a differently shaped call, which shows up as a last-ulp disagreement.  A
    halo or padding mistake would instead show up as a large error localised at
    the slab seams, which ``_assert_close`` is tight enough to catch.
    """

    #: A few ulp of float32, well below anything a real indexing bug produces.
    TOL = 1e-5

    def setUp(self):
        torch.manual_seed(0)
        self._grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self._grad)

    def _assert_close(self, expected, got):
        self.assertEqual(expected.shape, got.shape)
        scale = expected.abs().max().clamp(min=1.0)
        rel = ((expected - got).abs().max() / scale).item()
        self.assertLess(rel, self.TOL, f"slab result differs by {rel:.3e} relative")

    def _check(self, x, w, b, padding):
        expected = F.conv3d(x, w, b, padding=padding)
        with chunked_conv3d(budget=1) as mode:
            got = F.conv3d(x, w, b, padding=padding)
        self.assertGreater(mode.chunked_calls, 0, "convolution was not split")
        self._assert_close(expected, got)

    def test_identical_kernel3_pad1(self):
        x = torch.rand(1, 3, 9, 8, 7)
        w = torch.rand(4, 3, 3, 3, 3)
        self._check(x, w, torch.rand(4), 1)

    def test_identical_without_bias(self):
        x = torch.rand(1, 2, 7, 6, 5)
        w = torch.rand(3, 2, 3, 3, 3)
        self._check(x, w, None, 1)

    def test_identical_kernel5_pad2(self):
        x = torch.rand(1, 2, 11, 9, 8)
        w = torch.rand(3, 2, 5, 5, 5)
        self._check(x, w, torch.rand(3), 2)

    def test_identical_unpadded(self):
        x = torch.rand(1, 2, 8, 7, 6)
        w = torch.rand(3, 2, 3, 3, 3)
        self._check(x, w, torch.rand(3), 0)

    def test_identical_anisotropic_kernel(self):
        x = torch.rand(1, 2, 9, 8, 7)
        w = torch.rand(3, 2, 1, 3, 3)
        expected = F.conv3d(x, w, None, padding=(0, 1, 1))
        with chunked_conv3d(budget=1):
            got = F.conv3d(x, w, None, padding=(0, 1, 1))
        self._assert_close(expected, got)

    def test_budget_left_alone(self):
        """Convolutions under the budget must use the stock kernel untouched."""
        x = torch.rand(1, 2, 6, 6, 6)
        w = torch.rand(2, 2, 3, 3, 3)
        expected = F.conv3d(x, w, None, padding=1)
        with chunked_conv3d(budget=1 << 30) as mode:
            got = F.conv3d(x, w, None, padding=1)
        self.assertEqual(mode.chunked_calls, 0)
        self.assertTrue(torch.equal(expected, got))

    def test_unsupported_cases_fall_through(self):
        """Strided, grouped and transposed convolutions stay on the stock path."""
        x = torch.rand(1, 4, 8, 8, 8)
        w = torch.rand(4, 2, 3, 3, 3)
        wt = torch.rand(4, 2, 3, 3, 3)
        cases = [
            lambda t: F.conv3d(t, w, None, stride=2, padding=1, groups=2),
            lambda t: F.conv3d(t, w, None, padding=1, groups=2),
            lambda t: F.conv_transpose3d(t, wt, None, padding=1),
        ]
        for i, fn in enumerate(cases):
            expected = fn(x)
            with chunked_conv3d(budget=1) as mode:
                got = fn(x)
            with self.subTest(case=i):
                self.assertEqual(mode.chunked_calls, 0)
                self.assertTrue(torch.equal(expected, got))

    def test_non_float32_falls_through(self):
        x = torch.rand(1, 2, 6, 6, 6, dtype=torch.float64)
        w = torch.rand(2, 2, 3, 3, 3, dtype=torch.float64)
        expected = F.conv3d(x, w, None, padding=1)
        with chunked_conv3d(budget=1) as mode:
            got = F.conv3d(x, w, None, padding=1)
        self.assertEqual(mode.chunked_calls, 0)
        self.assertTrue(torch.equal(expected, got))

    def test_intercepts_torchscript_convolution(self):
        """Traced models emit ``aten::_convolution``, which must also be split.

        deepmriprep ships the warp model as a TorchScript archive, so missing
        this op spelling silently disables the optimisation.
        """
        conv = torch.nn.Conv3d(2, 3, 3, padding=1)
        traced = torch.jit.trace(conv, torch.rand(1, 2, 6, 6, 6))
        x = torch.rand(1, 2, 9, 8, 7)
        expected = traced(x)
        with chunked_conv3d(budget=1) as mode:
            got = traced(x)
        self.assertGreater(mode.chunked_calls, 0, "TorchScript conv was not split")
        self._assert_close(expected, got)

    def test_disabled_is_noop(self):
        with chunked_conv3d(enabled=False) as mode:
            self.assertIsNone(mode)

    def test_slab_count_tracks_budget(self):
        x = torch.rand(1, 4, 16, 8, 8)
        w = torch.rand(4, 4, 3, 3, 3)
        mode = ChunkedConv3d(budget=1 << 30)
        self.assertIsNone(mode._chunk((x, w, None, [1] * 3, [1] * 3, [1] * 3, False, [0] * 3, 1)))


if __name__ == "__main__":
    unittest.main()
