"""Tests for longitudinal Jacobian modulation.

The series is synthesised by applying a *known* velocity field to one tissue
map, so the deformation and the two time points are consistent by construction.
That matters: an earlier version of this test imposed a deformation unrelated to
the tissue difference, which made the shared-tissue mode look broken when it was
the test that was wrong.

The property that has to hold is volume preservation -- modulation moves volume
change out of the shape and into the intensity, so the integral of a modulated
map must reproduce that time point's native tissue volume.
"""

import os
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

from t1prep.warp_longitudinal import _exp_svf, _log_jacobian, _warp  # noqa: E402
from t1prep.modulate_longitudinal import (  # noqa: E402
    _reorient,
    modulate_longitudinal,
    output_name,
    resolve_inputs,
)

_WORK_SHAPE = (48, 56, 48)
_WORK_ZOOM = 1.5
_MNI_SHAPE = (36, 42, 36)
_MNI_ZOOM = 2.0
#: The synthetic y_ fields are a pure scaling, so the cross-sectional Jacobian
#: is known exactly and the test can check it was applied.
_AFFINE_SCALE = 0.90


def _grid_affine(shape, zoom):
    affine = np.diag([zoom, zoom, zoom, 1.0])
    affine[:3, 3] = -0.5 * (np.asarray(shape) - 1) * zoom
    return affine


def _write(path, data, affine):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), path)
    return path


def _series(tmp: Path, amplitude=0.05):
    """Build a two-time-point series from one tissue map and a known velocity.

    Returns:
        ``(paths, native_volumes_mm3, shared_tissue)`` where ``paths`` is a dict
        of the four input lists.
    """
    work_affine = _grid_affine(_WORK_SHAPE, _WORK_ZOOM)
    mni_affine = _grid_affine(_MNI_SHAPE, _MNI_ZOOM)

    zz, yy, xx = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in _WORK_SHAPE], indexing="ij"
    )
    radius = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    tissue = np.clip(1.0 - (radius - 11.0) / 2.0, 0, 1).astype(np.float32)

    envelope = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (14.0 ** 2))
    velocity = torch.from_numpy(
        np.stack(
            [
                amplitude * envelope * zz,
                amplitude * envelope * yy,
                amplitude * envelope * xx,
            ]
        ).astype(np.float32)
    )[None]

    paths = {"tissue": [], "displacement": [], "log_jacobian": [], "deformation": []}
    volumes = []
    for idx, sign in enumerate((-0.5, +0.5)):
        scaled = sign * velocity
        forward = _exp_svf(scaled)  # phi_i: average space -> time point i
        inverse = _exp_svf(-scaled)
        # p_i(x) = pbar(exp(-v_i)(x)) so that p_i(phi_i(x)) == pbar(x) exactly.
        native = _warp(torch.from_numpy(tissue)[None, None], inverse)[0, 0].numpy()
        volumes.append(float(native.sum()) * _WORK_ZOOM ** 3)

        paths["tissue"].append(
            _write(str(tmp / f"p1_tp{idx + 1}.nii.gz"), native, work_affine)
        )
        paths["log_jacobian"].append(
            _write(
                str(tmp / f"tp{idx + 1}_desc-longLogJacobian.nii.gz"),
                _log_jacobian(forward)[0].numpy(),
                work_affine,
            )
        )
        paths["displacement"].append(
            _write(
                str(tmp / f"tp{idx + 1}_desc-longDisplacement.nii.gz"),
                np.moveaxis(forward[0].numpy(), 0, -1) * _WORK_ZOOM,
                work_affine,
            )
        )

    mz, my, mx = np.meshgrid(*[np.arange(n) for n in _MNI_SHAPE], indexing="ij")
    mni_mm = np.stack([mz, my, mx], -1) @ mni_affine[:3, :3].T + mni_affine[:3, 3]
    for idx in range(2):
        paths["deformation"].append(
            _write(
                str(tmp / f"y_tp{idx + 1}.nii"),
                (mni_mm * _AFFINE_SCALE)[:, :, :, None, :],
                mni_affine,
            )
        )
    return paths, volumes, tissue


class TestModulation(unittest.TestCase):
    """Volume preservation and the two shared quantities."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.paths, cls.volumes, cls.tissue = _series(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, **kwargs):
        return modulate_longitudinal(
            self.paths["tissue"],
            self.paths["displacement"],
            self.paths["log_jacobian"],
            self.paths["deformation"],
            **kwargs,
        )

    def test_preserves_each_time_points_volume(self):
        # The whole point of modulation: the integral of the modulated map is
        # the tissue volume that time point actually had.
        out = self._run(tissue_source="timepoint")
        for got, expected in zip(out.native_volumes_mm3, self.volumes):
            self.assertAlmostEqual(got / expected, 1.0, places=2)

    def test_shared_tissue_matches_per_time_point_when_registration_is_right(self):
        # On a synthetic series the deformation explains the whole difference,
        # so the two modes agree.  On real data they do not, and the default is
        # 'timepoint' because that is what CAT12 does -- see
        # test_default_keeps_each_time_points_own_anatomy.
        shared = self._run(tissue_source="shared").native_volumes_mm3
        per_tp = self._run(tissue_source="timepoint").native_volumes_mm3
        for a, b in zip(shared, per_tp):
            self.assertAlmostEqual(a / b, 1.0, places=3)

    def test_default_is_timepoint_not_shared(self):
        # CAT12's ageing model keeps each time point's own segmentation and
        # applies the longitudinal Jacobian on top; sharing one tissue map is a
        # different estimator that leaves the time points differing only by a
        # smooth multiplier.
        import inspect

        signature = inspect.signature(modulate_longitudinal)
        self.assertEqual(signature.parameters["tissue_source"].default, "timepoint")

    def test_default_keeps_each_time_points_own_anatomy(self):
        # With a shared map the ratio between time points is exactly the
        # Jacobian ratio -- smooth everywhere.  With per-time-point maps it also
        # carries each segmentation's own structure, which is what makes
        # individual anatomy visible.
        shared = self._run(tissue_source="shared")
        per_tp = self._run(tissue_source="timepoint")

        def roughness(maps):
            ratio = np.where(maps[0] > 0.2, maps[1] / np.maximum(maps[0], 1e-9), np.nan)
            diffs = [np.abs(np.diff(ratio, axis=axis)) for axis in range(3)]
            return float(np.nanmean([d[np.isfinite(d)].mean() for d in diffs]))

        self.assertGreater(
            roughness(per_tp.modulated), roughness(shared.modulated)
        )

    def test_recovers_the_volume_ratio(self):
        out = self._run()
        got = out.native_volumes_mm3[1] / out.native_volumes_mm3[0]
        expected = self.volumes[1] / self.volumes[0]
        self.assertAlmostEqual(got, expected, places=2)
        # The series really does contain a volume change to find.
        self.assertGreater(expected, 1.02)

    def test_nonlinear_modulation_divides_out_the_affine(self):
        full = self._run(modulation="full").native_volumes_mm3
        nonlinear = self._run(modulation="nonlinear").native_volumes_mm3
        for a, b in zip(nonlinear, full):
            self.assertAlmostEqual(a / b, 1.0 / _AFFINE_SCALE ** 3, places=2)

    def test_shared_tissue_map_absorbs_segmentation_differences(self):
        # This is why CAT12 shares the tissue map, and why 'shared' is default:
        # an independent segmentation enters both time points identically and so
        # cancels from the contrast, instead of masquerading as volume change.
        perturbed = dict(self.paths)
        clean = np.asarray(
            nib.load(self.paths["tissue"][1]).dataobj, dtype=np.float32
        )
        rng = np.random.default_rng(0)
        from scipy.ndimage import gaussian_filter

        bias = 0.12 * gaussian_filter(rng.normal(0, 1, _WORK_SHAPE), 3.0)
        edge = (clean > 0.05) & (clean < 0.95)
        bias -= bias[edge].mean()  # volume neutral, as segmenter drift tends to be
        noisy = np.clip(clean + bias * edge, 0, 1)
        perturbed["tissue"] = [
            self.paths["tissue"][0],
            _write(
                str(self.tmp / "p1_tp2_perturbed.nii.gz"),
                noisy,
                nib.load(self.paths["tissue"][1]).affine,
            ),
        ]

        def contrast(tissue_paths, source):
            out = modulate_longitudinal(
                tissue_paths,
                self.paths["displacement"],
                self.paths["log_jacobian"],
                self.paths["deformation"],
                tissue_source=source,
            )
            return out.modulated[1] - out.modulated[0]

        errors = {}
        for source in ("timepoint", "shared"):
            reference = contrast(self.paths["tissue"], source)
            got = contrast(perturbed["tissue"], source)
            mask = (np.abs(reference) > 0) | (np.abs(got) > 0)
            signal = float(np.sqrt((reference[mask] ** 2).mean()))
            errors[source] = float(np.sqrt(((got - reference)[mask] ** 2).mean())) / signal

        self.assertLess(errors["shared"], 0.2 * errors["timepoint"])

    def test_rejects_inconsistent_inputs(self):
        with self.assertRaises(ValueError):
            modulate_longitudinal(
                self.paths["tissue"][:1],
                self.paths["displacement"],
                self.paths["log_jacobian"],
                self.paths["deformation"],
            )
        with self.assertRaises(ValueError):  # fewer than two time points
            modulate_longitudinal(
                *[[v[0]] for v in (
                    self.paths["tissue"],
                    self.paths["displacement"],
                    self.paths["log_jacobian"],
                    self.paths["deformation"],
                )]
            )
        with self.assertRaises(ValueError):
            self._run(modulation="sideways")
        with self.assertRaises(ValueError):
            self._run(tissue_source="whatever")

    def test_rejects_deformations_on_different_grids(self):
        odd = _grid_affine((30, 30, 30), 3.0)
        mz, my, mx = np.meshgrid(*[np.arange(n) for n in (30, 30, 30)], indexing="ij")
        mm = np.stack([mz, my, mx], -1) @ odd[:3, :3].T + odd[:3, 3]
        path = _write(str(self.tmp / "y_odd.nii"), mm[:, :, :, None, :], odd)
        with self.assertRaises(ValueError):
            modulate_longitudinal(
                self.paths["tissue"],
                self.paths["displacement"],
                self.paths["log_jacobian"],
                [self.paths["deformation"][0], path],
            )


class TestResolveInputs(unittest.TestCase):
    """Filename discovery, so the pipeline need not know the naming scheme."""

    def test_finds_legacy_named_outputs(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            affine = _grid_affine((8, 8, 8), 1.0)
            dirs, names = [], []
            for idx in (1, 2):
                mri = tmp / f"tp{idx}" / "mri"
                mri.mkdir(parents=True)
                bname = f"scan{idx}"
                _write(str(mri / f"p1{bname}.nii.gz"), np.zeros((8, 8, 8)), affine)
                _write(str(mri / f"y_{bname}.nii"), np.zeros((8, 8, 8)), affine)
                _write(
                    str(mri / f"{bname}_desc-longDisplacement.nii.gz"),
                    np.zeros((8, 8, 8)),
                    affine,
                )
                _write(
                    str(mri / f"{bname}_desc-longLogJacobian.nii.gz"),
                    np.zeros((8, 8, 8)),
                    affine,
                )
                dirs.append(str(mri))
                names.append(bname)
            tissue, disp, logjac, deform = resolve_inputs(dirs, names)
            self.assertTrue(all(os.path.isfile(p) for p in tissue + disp + logjac + deform))
            self.assertTrue(tissue[0].endswith("p1scan1.nii.gz"))
            self.assertTrue(deform[1].endswith("y_scan2.nii"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_names_what_is_missing(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "mri").mkdir()
            with self.assertRaises(ValueError) as ctx:
                resolve_inputs([str(tmp / "mri")], ["scan1"])
            message = str(ctx.exception)
            self.assertIn("tissue for 'scan1'", message)
            self.assertIn("--p", message)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            resolve_inputs(["a", "b"], ["only-one"])


class TestOutputNaming(unittest.TestCase):
    """Output names come from T1Prep's naming table, not a suffix invented here."""

    def test_legacy_names_double_the_mw_marker_like_cat12(self):
        # These maps are modulated twice -- longitudinal Jacobian, then spatial
        # normalisation -- which is what CAT12's ageing model records as mwmw.
        self.assertEqual(output_name("mri/p1subj.nii"), "mwmwp1subj.nii")
        self.assertEqual(output_name("mri/p2subj.nii.gz"), "mwmwp2subj.nii.gz")
        self.assertEqual(output_name("mri/p3subj.nii"), "mwmwp3subj.nii")

    def test_bids_names_double_the_modulated_marker(self):
        got = output_name("mri/sub-01_ses-1_T1w_label-GM_probseg.nii.gz")
        self.assertIn("-modulated-modulated", got)
        self.assertIn("label-GM_probseg", got)
        self.assertNotIn("_long", got)

    def test_does_not_collide_with_the_cross_sectional_modulated_map(self):
        # T1Prep writes a cross-sectional mwp1 into the same directory by
        # default; overwriting it would destroy a file the user still needs.
        self.assertNotEqual(output_name("mri/p1subj.nii"), "mwp1subj.nii")
        bids = output_name("mri/sub-01_T1w_label-GM_probseg.nii.gz")
        self.assertNotEqual(
            bids, "sub-01_space-MNI152NLin2009cAsym-nonlinear-modulated_label-GM_probseg.nii.gz"
        )

    def test_falls_back_for_unrecognised_names(self):
        self.assertEqual(
            output_name("mri/whatever.nii"), "whatever_desc-longModulated.nii"
        )


class TestOrientation(unittest.TestCase):
    """Outputs must be stored the way T1Prep stores its other MNI volumes.

    The stage computes on the RAS-canonical ``y_`` grid, but ``mwp1``/``mwp2``
    live on the template grid, whose first axis runs the other way.  Writing on
    the ``y_`` grid is geometrically correct yet stores the array mirrored with
    respect to its neighbours, so a voxel-wise comparison against ``mwp1``
    silently compares mirrored brains -- which is exactly what it looked like
    when this was wrong.
    """

    def test_reorient_flips_the_array_and_adopts_the_target_affine(self):
        ras = np.diag([1.5, 1.5, 1.5, 1.0])
        ras[:3, 3] = [-84.0, -120.0, -72.0]
        las = np.diag([-1.5, 1.5, 1.5, 1.0])
        las[:3, 3] = [84.0, -120.0, -72.0]

        volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        moved, affine = _reorient([volume], ras, las)
        np.testing.assert_allclose(affine, las)
        np.testing.assert_array_equal(moved[0], volume[::-1])

    def test_reorient_is_a_no_op_when_orientations_agree(self):
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        volume = np.zeros((2, 2, 2), dtype=np.float32)
        moved, got = _reorient([volume], affine, affine)
        np.testing.assert_allclose(got, affine)
        np.testing.assert_array_equal(moved[0], volume)

    def test_reorientation_preserves_total_volume(self):
        # A permutation and flip must not change what the map integrates to.
        ras = np.diag([1.5, 1.5, 1.5, 1.0])
        las = np.diag([-1.5, 1.5, 1.5, 1.0])
        rng = np.random.default_rng(0)
        volume = rng.random((4, 5, 6)).astype(np.float32)
        moved, _ = _reorient([volume], ras, las)
        self.assertAlmostEqual(float(moved[0].sum()), float(volume.sum()), places=4)


class TestCli(unittest.TestCase):
    def test_writes_modulated_maps(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            paths, _, _ = _series(tmp)
            out_dir = tmp / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "t1prep.modulate_longitudinal",
                    "--tissue", *paths["tissue"],
                    "--displacement", *paths["displacement"],
                    "--log-jacobian", *paths["log_jacobian"],
                    "--deformation", *paths["deformation"],
                    "--out-dir", str(out_dir),
                    "--save-shared-tissue",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(_SRC)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for idx in (1, 2):
                # CAT12's doubled "modulated warped" name, from T1Prep's table.
                self.assertTrue(
                    (out_dir / f"mwmwp1_tp{idx}.nii.gz").is_file(),
                    sorted(q.name for q in out_dir.iterdir()),
                )
            self.assertTrue((out_dir / "longitudinal_shared_tissue.nii.gz").is_file())
            modulated = nib.load(str(out_dir / "mwmwp1_tp1.nii.gz"))
            self.assertEqual(modulated.shape, _MNI_SHAPE)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_requires_one_input_style_or_the_other(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "t1prep.modulate_longitudinal",
                "--tissue", "a.nii",
                "--out-dir", "unused",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_SRC)},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--mri-dirs", result.stderr)


if __name__ == "__main__":
    unittest.main()
