"""T1Prep's image stages (:mod:`t1prep.stages`): deepmriprep's, without deepmriprep.

A full run is checked end to end elsewhere (byte-identical to the one
through deepmriprep's classes).  These tests pin the structure -- what the
pipeline imports, what the stages are built from -- and compare each piece
taken over from deepbet or deepmriprep with the original, networks stubbed,
wherever the original is installed.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from t1prep.nn.models import all_models_present  # noqa: E402
from t1prep.stages import preprocess, segmentation, skullstrip  # noqa: E402


def _original(case, module, name):
    """*name* from deepbet's or deepmriprep's *module*; skips *case* without it."""
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except ImportError:
        case.skipTest(f"{module.split('.')[0]} not installed")


class TestStructure(unittest.TestCase):
    def test_the_pipeline_imports_neither_deepmriprep_nor_deepbet(self):
        # in a fresh interpreter: other tests import them to compare with them
        code = (
            "import sys, t1prep.segment; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('deepmriprep', 'deepbet')))"
        )
        env = {**os.environ, "PYTHONPATH": str(_SRC), "T1PREP_NO_TRACK": "1"}
        out = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")

    def test_the_stages_run_on_t1preps_networks(self):
        if not all_models_present():
            self.skipTest("model weights not available")
        prep = preprocess.Preprocess(no_gpu=True)
        self.assertIsInstance(prep.brain_extract, skullstrip.BrainExtraction)
        self.assertIsInstance(prep.brain_segment, segmentation.BrainSegmentation)
        self.assertEqual(prep.device.type, "cpu")


class TestSkullStrip(unittest.TestCase):
    """The port against deepbet's own class, both with the same stubbed networks."""

    SHAPE = (40, 48, 36)

    def setUp(self):
        rng = np.random.default_rng(0)
        self.volume = rng.random(self.SHAPE, dtype=np.float32) * 50
        self.volume[8:32, 8:40, 6:30] += (
            500 + rng.random((24, 32, 24), dtype=np.float32) * 400
        )
        self.probability = (self.volume > 100).astype(np.float32)
        self.box = tuple(slice(0, s) for s in self.SHAPE)

    def _strip(self, cls, img):
        """Run *cls* (deepbet's or T1Prep's) on *img* with the stub's probability."""
        stage = cls.__new__(cls)  # no networks
        if hasattr(cls, "probability"):
            stage.probability = lambda x, *a, **k: (self.probability.copy(), self.box)
            return stage(img)
        stage.bbox = self.box
        stage.run_model = lambda x, *a, **k: self.probability.copy()
        brain, mask, _ = stage.run(img, threshold=0.5, n_dilate=0)
        return brain, mask

    def _images(self, affine):
        """The volume as an in-memory float32 image and as int16 and float32 files."""
        ornt = nib.io_orientation(affine)
        stored = nib.apply_orientation(
            self.volume, nib.orientations.ornt_transform([[0, 1], [1, 1], [2, 1]], ornt)
        )
        stored = np.ascontiguousarray(stored)
        yield "memory", lambda: nib.Nifti1Image(stored.copy(), affine)
        folder = tempfile.mkdtemp()
        for dtype in (np.float32, np.int16):
            path = os.path.join(folder, f"t1_{np.dtype(dtype).name}.nii")
            nib.save(nib.Nifti1Image(stored.astype(dtype), affine), path)
            yield np.dtype(dtype).name, lambda p=path: nib.load(p)

    def test_the_port_is_deepbet_and_leaves_the_input_alone(self):
        Original = _original(self, "deepbet.bet", "BrainExtraction")
        orientations = {
            "RAS": np.eye(4),
            "LAS": np.diag([-1.0, 1, 1, 1]),
            "PIR": np.array(
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]]
            ),
        }
        for label, affine in orientations.items():
            for source, load in self._images(affine):
                with self.subTest(orientation=label, source=source):
                    theirs = self._strip(Original, load())
                    img = load()
                    before = img.get_fdata(dtype=np.float32).copy()
                    ours = self._strip(skullstrip.BrainExtraction, img)
                    np.testing.assert_array_equal(
                        img.get_fdata(dtype=np.float32), before
                    )
                    for a, b in zip(ours, theirs):
                        np.testing.assert_array_equal(
                            np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
                        )
                        self.assertEqual(
                            np.asanyarray(a.dataobj).strides,
                            np.asanyarray(b.dataobj).strides,
                        )
                        np.testing.assert_array_equal(a.affine, b.affine)
                    self.assertEqual(ours[1].get_data_dtype(), np.uint8)

    def test_the_helpers_are_deepbet_s(self):
        keep_largest = _original(
            self, "deepbet.utils", "keep_largest_connected_component"
        )
        normalize = _original(self, "deepbet.utils", "normalize")
        rng = np.random.default_rng(2)
        mask = rng.random((24, 24, 24)) > 0.6
        np.testing.assert_array_equal(
            skullstrip.keep_largest_connected_component(mask.copy()),
            keep_largest(mask.copy()),
        )
        x = torch.from_numpy(rng.random((8, 8, 8), dtype=np.float32))
        self.assertTrue(
            torch.equal(skullstrip.normalize(x, 0.1, 0.9), normalize(x, 0.1, 0.9))
        )


class TestSegmentation(unittest.TestCase):
    def test_patch_weights_are_deepmriprep_s(self):
        get_patch_slices = _original(self, "deepmriprep.segment", "get_patch_slices")
        get_patch_weights = _original(self, "deepmriprep.segment", "get_patch_weights")
        corners = np.array([[0, 0, 0], [16, 8, 0], [32, 32, 16]])
        ours = [
            (slice(None), slice(None))
            + tuple(slice(int(c), int(c) + 32) for c in corner)
            for corner in corners
        ]
        sigma = 10.0 * torch.ones(3)
        expected = get_patch_weights(
            get_patch_slices(corners, (32,) * 3), (64,) * 3, (32,) * 3, sigma
        )
        for a, b in zip(
            segmentation._patch_weights(ours, (64,) * 3, (32,) * 3, sigma), expected
        ):
            self.assertTrue(torch.equal(a, b))

    def test_the_result_comes_back_on_the_working_grid(self):
        stage = segmentation.BrainSegmentation.__new__(
            segmentation.BrainSegmentation
        )  # no networks
        stage.device, stage.fill_holes, stage.chunk_conv = (
            torch.device("cpu"),
            True,
            True,
        )
        stage.coarse = lambda x: torch.ones_like(x)
        stage.refine = lambda x, p0: 2 * p0
        x = torch.rand(1, 1, 339, 411, 339)
        p0 = stage(x)
        self.assertEqual(p0.shape, x.shape)
        self.assertEqual(float(p0[0, 0, 1, 15, 0]), 2.0)  # inside the crop
        self.assertEqual(float(p0[0, 0, 0, 0, 0]), 0.0)  # in the padding


class _StubRegistration:
    """Stands in for torchreg's AffineRegistration: identity, no optimisation."""

    def __init__(self, **settings):
        self.settings = settings
        self._loss = 0.25

    def __call__(self, moving, static, return_moved=True):
        self.shapes = moving.shape, static.shape

    def get_affine(self):
        return torch.eye(3, 4)[None]


class TestAffineRegistration(unittest.TestCase):
    def test_the_brain_is_resampled_through_libcat(self):
        prep = preprocess.Preprocess.__new__(preprocess.Preprocess)  # no networks
        prep.device = torch.device("cpu")
        prep.affine_template = nib.Nifti1Image(np.ones((6, 7, 8), np.uint8), np.eye(4))
        brain = nib.Nifti1Image(
            np.random.default_rng(0).random((9, 9, 9), dtype=np.float32), np.eye(4)
        )
        mask = nib.Nifti1Image(np.ones((9, 9, 9), np.float32), np.eye(4))
        calls = []

        def resample(image, grid, **kwargs):
            calls.append(kwargs)
            return torch.zeros((1, 1, *grid.shape[1:4]))

        with (
            mock.patch.object(preprocess, "AffineRegistration", _StubRegistration),
            mock.patch.object(preprocess, "bspline_grid_sample", side_effect=resample),
        ):
            out = prep.run_affine_register(brain, mask)
        self.assertEqual(
            calls, [{"align_corners": True, "prefilter": True, "mask_value": 0}]
        )
        self.assertEqual(
            set(out), {"affine", "brain_large", "mask_large", "affine_loss"}
        )
        self.assertEqual(out["brain_large"].shape, (6, 7, 8))
        self.assertEqual(out["brain_large"].get_data_dtype(), np.float32)
        np.testing.assert_array_equal(out["affine"].values, np.eye(4))
        self.assertEqual(out["affine_loss"][0], 0.25)

    def test_the_cost_is_deepmriprep_s(self):
        MSEAndDice = _original(self, "deepmriprep.register", "MSEAndDice")
        rng = np.random.default_rng(1)
        x, y = (
            torch.from_numpy(rng.random((1, 2, 6, 6, 6), dtype=np.float32))
            for _ in range(2)
        )
        self.assertTrue(
            torch.equal(preprocess.MSEAndDice(0.5)(x, y), MSEAndDice(0.5)(x, y))
        )


if __name__ == "__main__":
    unittest.main()
