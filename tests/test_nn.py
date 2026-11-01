"""Tests for :mod:`t1prep.nn`, T1Prep's network framework.

The weights format, the registries and the backend selection are tested on
synthetic data.  The central promise -- the torch backend reproduces
deepmriprep's TorchScript archives bit for bit -- is tested against the
archives themselves, on small inputs, wherever the archives are on disk (a
source checkout has them in ``src/t1prep/data/models``).  The tinygrad
backend is compared with torch where tinygrad is installed.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from t1prep import nn  # noqa: E402
from t1prep.nn import architectures, backends, convert, models  # noqa: E402
from t1prep.nn.weights import load_safetensors, save_safetensors  # noqa: E402

_ARCHIVES = [
    d
    for d in (_SRC / "t1prep" / "data" / "models", models._DEEPMRIPREP_MODELS)
    if d is not None and (d / "segmentation_patch_0_model.pt").is_file()
]
ARCHIVE_DIR = _ARCHIVES[0] if _ARCHIVES else None

try:
    import tinygrad  # noqa: F401

    HAVE_TINYGRAD = True
except ImportError:
    HAVE_TINYGRAD = False


def _tmpdir(case):
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class TestWeights(unittest.TestCase):
    def test_round_trip_keeps_values_dtypes_and_metadata(self):
        rng = np.random.default_rng(0)
        tensors = {
            "a.weight": rng.standard_normal((4, 2, 3, 3, 3)).astype(np.float32),
            "b": np.arange(5, dtype=np.int64),
            "c": np.array(True),
            "d": rng.standard_normal(3).astype(np.float16),
        }
        path = _tmpdir(self) / "w.safetensors"
        save_safetensors(path, tensors, {"arch": "unet3d", "n": 3})
        back, meta = load_safetensors(path)
        self.assertEqual(meta, {"arch": "unet3d", "n": "3"})
        self.assertEqual(set(back), set(tensors))
        for name, array in tensors.items():
            self.assertEqual(back[name].dtype, array.dtype)
            np.testing.assert_array_equal(back[name], array)

    def test_file_follows_the_format(self):
        path = _tmpdir(self) / "w.safetensors"
        save_safetensors(path, {"x": np.ones((2, 3), np.float32)})
        raw = path.read_bytes()
        size = int.from_bytes(raw[:8], "little")
        self.assertEqual(size % 8, 0)  # data 8-byte aligned
        self.assertEqual(raw[8:9], b"{")
        self.assertEqual(len(raw), 8 + size + 2 * 3 * 4)

    def test_file_is_readable_by_others(self):
        # mkstemp creates 0600; a Docker image converts as root, runs as a user
        old = os.umask(0o022)
        try:
            path = _tmpdir(self) / "w.safetensors"
            save_safetensors(path, {"x": np.zeros(1, np.float32)})
        finally:
            os.umask(old)
        self.assertTrue(path.stat().st_mode & stat.S_IROTH)

    def test_a_truncated_file_is_refused(self):
        path = _tmpdir(self) / "w.safetensors"
        save_safetensors(path, {"x": np.zeros(16, np.float32)})
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises(ValueError):
            load_safetensors(path)


class TestArchitectures(unittest.TestCase):
    def test_both_families_are_registered(self):
        self.assertEqual(
            set(architectures.architecture_names()), {"unet3d", "linknet3d"}
        )

    def test_the_weights_identify_their_architecture(self):
        coarse = {"0.inp.double_conv.0.weight": 0, "1.factor": 0}
        unet = architectures.get_architecture(None, coarse)
        self.assertEqual(unet.name, "unet3d")
        self.assertEqual(
            set(unet.canonical(coarse)), {"unet.inp.double_conv.0.weight", "act.factor"}
        )
        self.assertEqual(
            architectures.get_architecture(None, {"init.0.weight": 0}).name, "linknet3d"
        )

    def test_unknown_weights_and_names_are_refused(self):
        with self.assertRaises(ValueError):
            architectures.get_architecture(None, {"fc.weight": 0})
        with self.assertRaises(ValueError):
            architectures.get_architecture("resnet")


class TestBackendSelection(unittest.TestCase):
    def test_torch_is_the_default(self):
        with mock.patch.dict(os.environ, {nn.BACKEND_ENV: ""}):
            self.assertEqual(nn.backend_name(), "torch")
        self.assertEqual(nn.backend_names()[0], "torch")

    def test_the_environment_selects(self):
        with mock.patch.dict(os.environ, {nn.BACKEND_ENV: " TinyGrad "}):
            self.assertEqual(nn.backend_name(), "tinygrad")
        with (
            mock.patch.dict(os.environ, {nn.BACKEND_ENV: "jax"}),
            self.assertRaises(ValueError),
        ):
            nn.backend_name()

    def test_every_backend_implements_the_operation_table(self):
        self.assertIsInstance(nn.get_backend("torch"), nn.Backend)
        with self.assertRaises(TypeError):
            type("Partial", (nn.Backend,), {"name": "partial"})()

    @unittest.skipIf(HAVE_TINYGRAD, "tinygrad is installed")
    def test_a_missing_framework_says_how_to_install_it(self):
        with self.assertRaisesRegex(RuntimeError, "pip install"):
            backends.get_backend("tinygrad")


@unittest.skipIf(ARCHIVE_DIR is None, "deepmriprep's TorchScript archives not on disk")
class TestAgainstTheArchives(unittest.TestCase):
    """Each network as a file reproduces its TorchScript archive."""

    #: (network, input shape): the depth-4 UNet needs multiples of 16, the
    #: LinkNet multiples of 32
    CASES = (
        ("segmentation_patch_0_model", (1, 2, 32, 32, 32)),
        ("segmentation_model", (1, 1, 32, 48, 32)),
        ("brain_extraction_bbox_model", (1, 1, 64, 64, 64)),
    )

    def setUp(self):
        self.dir = _tmpdir(self)

    def _pair(self, name):
        archive = ARCHIVE_DIR / f"{name}.pt"
        path = self.dir / f"{name}.safetensors"
        convert.convert_model(archive, path)
        return torch.jit.load(str(archive), map_location="cpu").eval(), path

    def _input(self, shape):
        return torch.from_numpy(
            np.random.default_rng(1).random(shape, dtype=np.float32)
        )

    def test_torch_backend_is_bit_identical(self):
        for name, shape in self.CASES:
            archive, path = self._pair(name)
            x = self._input(shape)
            with torch.no_grad():
                expected = archive(x)
            got = nn.Network(path, "torch", "cpu")(x)
            self.assertTrue(torch.equal(got, expected), name)

    def test_numpy_in_gives_numpy_out(self):
        _, path = self._pair("segmentation_patch_0_model")
        x = self._input((1, 2, 32, 32, 32))
        net = nn.Network(path, "torch", "cpu")
        out = net(x.numpy())
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, net(x).numpy())

    def test_a_state_dict_converts_like_the_archive(self):
        archive, path = self._pair("segmentation_model")
        pth = self.dir / "trained.pth"
        torch.save(archive.state_dict(), pth)
        convert.convert_model(pth, self.dir / "trained.safetensors")
        a, meta_a = load_safetensors(path)
        b, meta_b = load_safetensors(self.dir / "trained.safetensors")
        self.assertEqual(meta_a["arch"], meta_b["arch"])
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    @unittest.skipUnless(HAVE_TINYGRAD, "tinygrad not installed")
    def test_tinygrad_backend_agrees_with_torch(self):
        for name, shape in self.CASES:
            _, path = self._pair(name)
            x = self._input(shape)
            ref = nn.Network(path, "torch", "cpu")(x)
            got = nn.Network(path, "tinygrad", "cpu")(x)
            self.assertEqual(got.shape, ref.shape)
            np.testing.assert_allclose(
                got.numpy(), ref.numpy(), atol=2e-3, err_msg=name
            )


class TestConvert(unittest.TestCase):
    def test_coarse_keys_are_stored_canonically(self):
        state = {
            "0.inp.double_conv.0.weight": torch.ones(2, 1, 3, 3, 3),
            "1.factor": torch.ones(1),
        }
        path = _tmpdir(self) / "m.safetensors"
        convert.save_state_dict(state, path, source="x.pth")
        tensors, meta = load_safetensors(path)
        self.assertEqual(meta, {"arch": "unet3d", "source": "x.pth"})
        self.assertEqual(set(tensors), {"unet.inp.double_conv.0.weight", "act.factor"})

    def test_the_command_line_shows_its_synopsis_without_arguments(self):
        with self.assertRaises(SystemExit) as stop, mock.patch("sys.stderr"):
            convert.main([])
        self.assertEqual(stop.exception.code, 1)


class TestModelFiles(unittest.TestCase):
    def setUp(self):
        self.dir = _tmpdir(self)
        for name, value in (("MODEL_DIR", self.dir), ("_DEEPMRIPREP_MODELS", None)):
            patcher = mock.patch.object(models, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_one_file_per_network(self):
        self.assertEqual(len(models.NETWORKS), 21)
        self.assertEqual(
            models.model_path("segmentation_model"),
            self.dir / "segmentation_model.safetensors",
        )

    def test_nothing_is_fetched_when_everything_is_there(self):
        for name in models.MODEL_FILES:
            (self.dir / name).write_bytes(b"w")
        with mock.patch.object(models, "download_models", side_effect=AssertionError):
            models.prepare_model_files(verbose=False)

    @unittest.skipIf(
        ARCHIVE_DIR is None, "deepmriprep's TorchScript archives not on disk"
    )
    def test_archives_on_disk_are_converted_without_a_download(self):
        source = _tmpdir(self)
        (source / "segmentation_model.pt").symlink_to(
            ARCHIVE_DIR / "segmentation_model.pt"
        )
        with mock.patch.object(models, "_DEV_MODEL_DIR", source):
            models._convert_missing(verbose=False)
        made = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(made, ["segmentation_model.safetensors"])
        self.assertEqual(
            nn.Network(models.model_path("segmentation_model")).architecture.name,
            "unet3d",
        )


if __name__ == "__main__":
    unittest.main()
