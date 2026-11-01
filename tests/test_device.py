import os
import re
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

    def test_registrations_stay_off_mps_unless_asked(self):
        from unittest import mock

        def resolved(kind):
            return mock.patch.object(
                _device, "resolve_device", return_value=(torch.device(kind), kind == "cpu")
            )

        with resolved("mps"):
            self.assertEqual(_device.stage_device().type, "cpu")
            os.environ["T1PREP_DEVICE"] = "mps"
            self.assertEqual(_device.stage_device().type, "mps")
        for kind in ("cuda", "cpu"):
            with resolved(kind):
                self.assertEqual(_device.stage_device().type, kind)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestInterpolate(unittest.TestCase):
    """``_device.interpolate``: torch's own kernel under deterministic algorithms.

    A tensor on the meta device stands in for an accelerator: like CUDA and
    MPS it is not ``is_cpu``, which is all torch's choice of the
    decomposition looks at.
    """

    SIZE = {"size": (4, 4, 4), "mode": "trilinear", "align_corners": False}

    def setUp(self):
        before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.addCleanup(
            lambda: torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        )
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _deterministic_inside(self, x, **kwargs):
        """Whether deterministic algorithms were on when F.interpolate ran."""
        from unittest import mock

        seen, real = [], torch.nn.functional.interpolate

        def spy(*args, **kw):
            seen.append(torch.are_deterministic_algorithms_enabled())
            return real(*args, **kw)

        with mock.patch.object(torch.nn.functional, "interpolate", spy):
            _device.interpolate(x, **{**self.SIZE, **kwargs})
        return seen[0]

    def _state(self):
        return (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    def test_an_accelerator_call_lifts_the_setting_and_restores_it(self):
        x = torch.empty(1, 1, 2, 2, 2, device="meta")
        for warn_only in (True, False):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            self.assertFalse(self._deterministic_inside(x))
            self.assertEqual(self._state(), (True, warn_only))

    def test_the_setting_is_restored_when_interpolate_raises(self):
        x = torch.empty(1, 1, 2, 2, 2, device="meta")
        with self.assertRaises(Exception):
            _device.interpolate(x, size=(4, 4, 4), mode="no-such-mode")
        self.assertEqual(self._state(), (True, True))

    def test_cpu_and_gradient_calls_are_left_alone(self):
        self.assertTrue(self._deterministic_inside(torch.rand(1, 1, 2, 2, 2)))
        x = torch.empty(1, 1, 2, 2, 2, device="meta", requires_grad=True)
        self.assertTrue(self._deterministic_inside(x))
        with torch.no_grad():
            self.assertFalse(self._deterministic_inside(x))

    def test_without_deterministic_algorithms_nothing_changes(self):
        torch.use_deterministic_algorithms(False)
        self.assertFalse(
            self._deterministic_inside(torch.empty(1, 1, 2, 2, 2, device="meta"))
        )
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_cpu_results_are_f_interpolate_s(self):
        x = torch.rand(1, 1, 5, 6, 7)
        for kwargs in (
            self.SIZE,
            {"scale_factor": 1 / 1.5, "mode": "trilinear", "align_corners": True},
            {"size": (9, 9, 9), "mode": "nearest-exact"},
        ):
            self.assertTrue(
                torch.equal(
                    _device.interpolate(x, **kwargs),
                    torch.nn.functional.interpolate(x, **kwargs),
                )
            )

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(), "no MPS device"
    )
    def test_mps_gets_the_native_kernel(self):
        x = torch.rand(1, 1, 12, 14, 12, device="mps")
        ours = _device.interpolate(x, size=(37, 41, 37), mode="trilinear")
        torch.use_deterministic_algorithms(False)
        native = torch.nn.functional.interpolate(x, size=(37, 41, 37), mode="trilinear")
        self.assertTrue(torch.equal(ours.cpu(), native.cpu()))

    def test_t1prep_never_calls_f_interpolate_directly(self):
        pattern = re.compile(r"\b(F|functional)\.interpolate\(")
        offenders = [
            f"{path.relative_to(_SRC)}:{n}"
            for path in sorted((_SRC / "t1prep").rglob("*.py"))
            if path.name != "_device.py"
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
