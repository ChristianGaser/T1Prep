import os
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
except ImportError:  # pragma: no cover - torch is optional for the test env
    torch = None

if torch is not None:
    from t1prep import _device


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDeviceSelection(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("T1PREP_DEVICE",)}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_explicit_cpu_request(self):
        os.environ["T1PREP_DEVICE"] = "cpu"
        device, no_gpu = _device.resolve_device()
        self.assertEqual(device.type, "cpu")
        self.assertTrue(no_gpu)

    def test_no_gpu_flag_matches_device(self):
        device, no_gpu = _device.resolve_device()
        self.assertEqual(no_gpu, device.type == "cpu")

    def test_unavailable_accelerator_falls_back_to_cpu(self):
        os.environ["T1PREP_DEVICE"] = "cuda"
        device, no_gpu = _device.resolve_device()
        if not torch.cuda.is_available():
            self.assertEqual(device.type, "cpu")
            self.assertTrue(no_gpu)

    def test_mps_routing_is_opt_in(self):
        self.assertFalse(_device.mps_routing_requested())
        os.environ["T1PREP_DEVICE"] = "mps"
        self.assertTrue(_device.mps_routing_requested())

    def test_release_cache_on_cpu_is_noop(self):
        _device.release_cache(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
