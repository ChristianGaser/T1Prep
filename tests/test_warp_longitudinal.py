"""Tests for the low-dimensional groupwise longitudinal registration.

The registration is checked against a synthetic series built by applying a
*known* velocity field to one phantom, so the recovered log Jacobian has a
ground truth to be scored against rather than only an internal objective to
report.  That distinction matters here: a groupwise fit can drive its own
objective down while producing a deformation that is four times too large and
in the wrong place, which is exactly what an unregularised version of this code
did.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from t1prep.warp_longitudinal import (  # noqa: E402
    _exp_svf,
    _gradient,
    _log_jacobian,
    _warp,
    displacement_to_mm,
    groupwise_svf,
)

_ZOOM = 1.5
_SHAPE = (56, 64, 56)


def _affine(shape=_SHAPE, zoom=_ZOOM):
    aff = np.diag([zoom, zoom, zoom, 1.0])
    aff[:3, 3] = -0.5 * (np.asarray(shape) - 1) * zoom
    return aff


def _phantom(shape=_SHAPE, seed=0):
    """A textured ellipsoid with a darker core, on a zero background."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij"
    )
    brain = np.sqrt((zz / 21) ** 2 + (yy / 25) ** 2 + (xx / 19) ** 2)
    core = np.sqrt((zz / 6) ** 2 + (yy / 8) ** 2 + (xx / 5) ** 2)
    vol = 0.85 * (brain < 1.0).astype(np.float32)
    vol[core < 1.0] = 0.15
    vol = vol + 0.12 * gaussian_filter(rng.normal(0, 1, shape), 3.0) * (brain < 1.0)
    return gaussian_filter(vol, 1.0).astype(np.float32), brain


def _truth_velocity(shape=_SHAPE, amplitude=0.10):
    """A smooth radial expansion in voxels: no singularity at the origin."""
    zz, yy, xx = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij"
    )
    env = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (13.0 ** 2))
    field = np.stack(
        [amplitude * env * zz, amplitude * env * yy, amplitude * env * xx]
    ).astype(np.float32)
    return torch.from_numpy(field)[None]


def _series(noise=0.01, seed=0):
    """Two time points related by a known expansion, split symmetrically."""
    rng = np.random.default_rng(seed + 100)
    phantom, brain = _phantom(seed=seed)
    velocity = _truth_velocity()
    base = torch.from_numpy(phantom)[None, None]
    images = []
    for sign in (-0.5, +0.5):
        warped = _warp(base, _exp_svf(sign * velocity))[0, 0].numpy()
        warped = warped + rng.normal(0, noise, _SHAPE).astype(np.float32)
        images.append(nib.Nifti1Image(warped.astype(np.float32), _affine()))
    # exp(+v/2) undoes the tp1 image, so it is tp1's map onto the average.
    truth = {
        0: _log_jacobian(_exp_svf(+0.5 * velocity))[0].numpy(),
        1: _log_jacobian(_exp_svf(-0.5 * velocity))[0].numpy(),
    }
    return images, truth, brain


class TestPrimitives(unittest.TestCase):
    """The building blocks, where a sign or axis error would be silent."""

    def test_zero_displacement_is_identity(self):
        volume = torch.randn(1, 1, 12, 14, 16)
        disp = torch.zeros(1, 3, 12, 14, 16)
        self.assertTrue(torch.allclose(_warp(volume, disp), volume, atol=1e-5))

    def test_log_jacobian_of_identity_is_zero(self):
        disp = torch.zeros(1, 3, 12, 14, 16)
        self.assertLess(float(_log_jacobian(disp).abs().max()), 1e-6)

    def test_gradient_channel_matches_array_axis(self):
        for axis in range(3):
            shape = [1, 1, 12, 14, 16]
            ramp = torch.arange(shape[axis + 2], dtype=torch.float32)
            view = [1, 1, 1, 1, 1]
            view[axis + 2] = -1
            volume = ramp.view(view).expand(shape).contiguous()
            expected = torch.zeros(3)
            expected[axis] = 1.0
            got = _gradient(volume)[0, :, 6, 7, 8]
            self.assertTrue(torch.allclose(got, expected, atol=1e-5), f"axis {axis}")

    def test_exp_of_constant_velocity_is_translation(self):
        velocity = torch.zeros(1, 3, 24, 24, 24)
        velocity[:, 0] = 2.0
        disp = _exp_svf(velocity)
        centre = disp[0, :, 12, 12, 12]
        self.assertAlmostEqual(float(centre[0]), 2.0, places=3)
        self.assertAlmostEqual(float(centre[1]), 0.0, places=5)
        self.assertAlmostEqual(float(centre[2]), 0.0, places=5)

    def test_warp_shifts_along_the_named_axis(self):
        volume = torch.zeros(1, 1, 24, 24, 24)
        volume[0, 0, 10, 12, 12] = 1.0
        disp = torch.zeros(1, 3, 24, 24, 24)
        disp[:, 0] = 2.0
        moved = _warp(volume, disp)[0, 0]
        peak = (moved == moved.max()).nonzero()[0].tolist()
        # Sampling at x + disp is a pull-back, so the feature moves to 10 - 2.
        self.assertEqual(peak, [8, 12, 12])

    def test_displacement_to_mm_uses_the_affine(self):
        disp = np.zeros((3, 4, 4, 4), dtype=np.float32)
        disp[0] = 1.0
        out = displacement_to_mm(disp, _affine(shape=(4, 4, 4)))
        self.assertEqual(out.shape, (4, 4, 4, 3))
        np.testing.assert_allclose(out[..., 0], _ZOOM, rtol=1e-6)
        np.testing.assert_allclose(out[..., 1:], 0.0, atol=1e-6)


class TestGroupwiseFit(unittest.TestCase):
    """Properties the ageing model has to hold, on a known deformation."""

    @classmethod
    def setUpClass(cls):
        cls.images, cls.truth, brain = _series()
        cls.outputs = groupwise_svf(cls.images, resolution=_ZOOM)
        # Score away from the mask edge, where the force is one-sided.
        from scipy.ndimage import binary_erosion

        cls.core = binary_erosion(brain < 0.95, iterations=5)

    def test_requires_at_least_two_time_points(self):
        with self.assertRaises(ValueError):
            groupwise_svf(self.images[:1])

    def test_rejects_mismatched_schedule(self):
        with self.assertRaises(ValueError):
            groupwise_svf(self.images, scales=(2, 1), iterations=(10,))

    def test_solution_is_unbiased(self):
        # No time point is the reference: the velocities sum to zero, which is
        # the tangent-space analogue of the rigid stage's SE(3) barycentre.
        mean = np.stack(self.outputs.coefficients).mean(axis=0)
        self.assertLess(float(np.abs(mean).max()), 1e-5)

    def test_deformations_are_diffeomorphic(self):
        for idx, logjac in enumerate(self.outputs.log_jacobians):
            det = np.exp(logjac)
            self.assertGreater(float(det.min()), 0.0, f"time point {idx} folds")
            self.assertEqual(int((det <= 0).sum()), 0)

    def test_recovers_the_volume_change_map(self):
        for idx, logjac in enumerate(self.outputs.log_jacobians):
            r = np.corrcoef(logjac[self.core], self.truth[idx][self.core])[0, 1]
            self.assertGreater(r, 0.7, f"time point {idx}: r = {r:.3f}")

    def test_does_not_inflate_the_deformation(self):
        # The membrane prior shrinks the estimate, so it may come out small --
        # but it must never come out several times larger than the truth, which
        # is the failure mode of the same fit without a prior.
        for idx, logjac in enumerate(self.outputs.log_jacobians):
            truth_peak = float(np.abs(self.truth[idx][self.core]).max())
            got_peak = float(np.abs(logjac[self.core]).max())
            self.assertLess(got_peak, 2.0 * truth_peak, f"time point {idx}")

    def test_time_points_end_up_better_aligned(self):
        volumes = [
            np.asarray(img.dataobj, dtype=np.float32) for img in self.images
        ]
        before = float(np.sqrt(((volumes[0] - volumes[1])[self.core] ** 2).mean()))
        warped = [
            _warp(
                torch.from_numpy(vol)[None, None],
                torch.from_numpy(disp)[None],
            )[0, 0].numpy()
            for vol, disp in zip(volumes, self.outputs.displacements)
        ]
        after = float(np.sqrt(((warped[0] - warped[1])[self.core] ** 2).mean()))
        self.assertLess(after, before)

    def test_more_iterations_do_not_change_the_answer(self):
        # Without the membrane prior the fit never converges: it keeps turning
        # image noise into volume change, so a longer run gives a different --
        # and worse -- answer.  Pin that down.
        longer = groupwise_svf(
            self.images, resolution=_ZOOM, scales=(2, 1), iterations=(120, 120)
        )
        for short, long_ in zip(self.outputs.log_jacobians, longer.log_jacobians):
            np.testing.assert_allclose(short, long_, atol=5e-3)


class TestCli(unittest.TestCase):
    """The CLI writes the files the bash pipeline expects to find."""

    def test_writes_jacobian_displacement_and_template(self):
        images, _, _ = _series()
        tmp = Path(tempfile.mkdtemp())
        try:
            paths = []
            for idx, img in enumerate(images):
                path = tmp / f"tp{idx + 1}.nii.gz"
                nib.save(img, str(path))
                paths.append(str(path))
            out_dir = tmp / "out"
            cmd = [
                sys.executable,
                "-m",
                "t1prep.warp_longitudinal",
                "--inputs",
                *paths,
                "--out-dir",
                str(out_dir),
                "--save-displacement",
                "--save-template",
                "--apply",
                "--iterations",
                "10",
                "10",
            ]
            env = {"PYTHONPATH": str(_SRC)}
            import os

            env = {**os.environ, **env}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)

            for idx in (1, 2):
                for suffix in (
                    "longLogJacobian",
                    "longDisplacement",
                    "longWarped",
                ):
                    expected = out_dir / f"tp{idx}_desc-{suffix}.nii.gz"
                    self.assertTrue(expected.is_file(), f"missing {expected.name}")
            self.assertTrue((out_dir / "longitudinal_average.nii.gz").is_file())

            disp = nib.load(str(out_dir / "tp1_desc-longDisplacement.nii.gz"))
            self.assertEqual(disp.shape, (*_SHAPE, 3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rejects_mismatched_subfolders(self):
        images, _, _ = _series()
        tmp = Path(tempfile.mkdtemp())
        try:
            paths = []
            for idx, img in enumerate(images):
                path = tmp / f"tp{idx + 1}.nii.gz"
                nib.save(img, str(path))
                paths.append(str(path))
            import os

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "t1prep.warp_longitudinal",
                    "--inputs",
                    *paths,
                    "--out-dir",
                    str(tmp / "out"),
                    "--out-subfolders",
                    "only_one",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(_SRC)},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("out-subfolders", result.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
