"""Deterministic mode must not abort CUDA runs.

Up to 0.8.0 ``segment.py`` called ``torch.use_deterministic_algorithms(True)``.
On CUDA that raises twice over: cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG`` set
before its first call, and the backward of ``grid_sample``/``interpolate``
(torchreg's affine registration) has no deterministic CUDA kernel.  There is
no CUDA device in the test environment, so this pins the settings that avoid
both: the workspace variable, and ``warn_only`` so that every deterministic
kernel is still used while the few without one warn instead of raising.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from t1prep.segment import setup_determinism  # noqa: E402


class TestSetupDeterminism(unittest.TestCase):
    def setUp(self):
        before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark,
        )

        def restore():
            torch.use_deterministic_algorithms(before[0], warn_only=before[1])
            torch.backends.cudnn.benchmark = before[2]

        self.addCleanup(restore)

    def test_deterministic_but_never_raising(self):
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        with mock.patch.dict(os.environ, env, clear=True):
            setup_determinism(0)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_a_workspace_set_by_the_user_is_kept(self):
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            setup_determinism(0)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_the_seed_reaches_torch_and_numpy(self):
        import numpy as np

        setup_determinism(7)
        a = (torch.rand(3), np.random.rand(3))
        setup_determinism(7)
        b = (torch.rand(3), np.random.rand(3))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))


if __name__ == "__main__":
    unittest.main()
