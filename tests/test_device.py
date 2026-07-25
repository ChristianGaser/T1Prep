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


class _FakeSyn:
    def __init__(self, grid=None):
        self._grid = grid


class _FakeWarp:
    def __init__(self, syn):
        self.syn = syn


class _FakePreprocess:
    """Stands in for deepmriprep's ``Preprocess``: one shared device attribute."""

    def __init__(self, device, syn=None):
        self.device = device
        self.warp_register = _FakeWarp(syn) if syn is not None else None


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDeviceSelection(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("T1PREP_DEVICE", "T1PREP_MPS_STAGES")}
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

    def test_warp_opted_in_by_default(self):
        cpu = torch.device("cpu")
        target = _device.stage_target_device("warp", cpu)
        accelerated, _ = _device.resolve_device()
        expected = accelerated if accelerated.type != "cpu" else cpu
        self.assertEqual(target, expected)

    def test_other_stages_stay_on_the_fallback(self):
        cpu = torch.device("cpu")
        for stage in ("nogm", "affine", "brain"):
            with self.subTest(stage=stage):
                self.assertEqual(_device.stage_target_device(stage, cpu), cpu)

    def test_stage_list_is_configurable(self):
        cpu = torch.device("cpu")
        os.environ["T1PREP_MPS_STAGES"] = ""
        self.assertEqual(_device.stage_target_device("warp", cpu), cpu)
        os.environ["T1PREP_MPS_STAGES"] = "warp, nogm"
        accelerated, _ = _device.resolve_device()
        expected = accelerated if accelerated.type != "cpu" else cpu
        self.assertEqual(_device.stage_target_device("nogm", cpu), expected)

    def test_stage_device_restores_previous(self):
        prep = _FakePreprocess(torch.device("cpu"))
        with _device.stage_device(prep, torch.device("cpu")):
            self.assertEqual(prep.device, torch.device("cpu"))
        self.assertEqual(prep.device, torch.device("cpu"))

    def test_stage_device_restores_after_exception(self):
        prep = _FakePreprocess(torch.device("cpu"))
        meta = torch.device("meta")  # always constructible, never the real device
        with self.assertRaises(RuntimeError):
            with _device.stage_device(prep, meta):
                self.assertEqual(prep.device, meta)
                raise RuntimeError("boom")
        self.assertEqual(prep.device, torch.device("cpu"))

    def test_stale_syn_grid_is_dropped_on_device_change(self):
        """SyNBase caches its identity grid on first use; a stale one would be
        added to flows living on the new device."""
        syn = _FakeSyn(grid=torch.zeros(1, 2, 2, 2, 3))  # cpu grid
        prep = _FakePreprocess(torch.device("cpu"), syn=syn)
        with _device.stage_device(prep, torch.device("meta")):
            self.assertIsNone(syn._grid)

    def test_release_cache_on_cpu_is_noop(self):
        _device.release_cache(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
