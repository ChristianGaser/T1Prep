"""The unified segmentation (``--unified-seg``) and how it is handed on.

``t1prep.unified_seg`` replaces deepbet, the affine registration and
deepmriprep's networks by ``cat_surf.vol_unified_seg``.  The fit itself is
CAT-Surface's and tested there; these tests hold what T1Prep adds: the
affine in deepmriprep's convention, the brain mask, the storage order, and
how the option travels from the command lines to ``segment.py``.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import unified_seg as useg  # noqa: E402
from t1prep import report  # noqa: E402
from t1prep import t1prep as orchestrator  # noqa: E402
from t1prep.launch import options as opt  # noqa: E402
from t1prep.utils import centre_extent_volume, get_filenames  # noqa: E402


def _oblique_case():
    """An oblique image grid, an axis-aligned working grid and a mapping between them."""
    rng = np.random.default_rng(0)
    image = np.cumsum(np.cumsum(rng.random((23, 31, 19)), 0), 1).astype(np.float32)
    angle = 0.2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    image_mat = np.eye(4)
    image_mat[:3, :3] = rotation @ np.diag([1.1, 0.9, 1.3])
    image_mat[:3, 3] = [-12, -14, -11]
    grid_shape = (17, 21, 15)
    grid_mat = np.diag([0.7, 0.75, 0.8, 1.0])
    grid_mat[:3, 3] = [-6, -8, -5]
    world = np.eye(4)
    world[:3, :3] = [[1.05, 0.03, 0], [-0.02, 0.97, 0.04], [0.01, 0, 1.02]]
    world[:3, 3] = [1.5, -2, 0.7]
    return image, image_mat, grid_shape, grid_mat, world


class TestGridAffine(unittest.TestCase):
    """deepmriprep's affine maps grid_sample coordinates of the working grid
    to those of the image; it has to sample what the world mapping says."""

    def test_it_samples_where_the_world_mapping_points(self):
        image, image_mat, grid_shape, grid_mat, world = _oblique_case()
        affine = useg.grid_affine(world, image_mat, image.shape, grid_mat, grid_shape)
        grid = F.affine_grid(torch.from_numpy(affine[None, :3]).float(),
                             [1, 1, *grid_shape], align_corners=True)
        sampled = F.grid_sample(torch.from_numpy(image)[None, None], grid,
                                mode="bilinear", align_corners=True)[0, 0].numpy()

        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in grid_shape], indexing="ij"),
                       -1).reshape(-1, 3)
        template_mm = np.c_[ijk, np.ones(len(ijk))] @ grid_mat.T
        voxels = template_mm @ np.linalg.inv(world).T @ np.linalg.inv(image_mat).T
        expected = map_coordinates(image, voxels[:, :3].T, order=1).reshape(grid_shape)
        np.testing.assert_allclose(sampled, expected, rtol=0, atol=1e-6 * image.max())

    def test_the_same_grid_and_no_mapping_is_the_identity(self):
        mat = np.diag([0.5, 0.5, 0.5, 1.0])
        affine = useg.grid_affine(np.eye(4), mat, (20, 30, 40), mat, (20, 30, 40))
        np.testing.assert_allclose(affine, np.eye(4), atol=1e-12)

    def test_its_volume_factor_is_that_of_the_world_mapping(self):
        # segment.py's wj_affine: det(affine) times the ratio of the boxes
        # between the corner voxel centres -- native volume per template volume
        image, image_mat, grid_shape, grid_mat, world = _oblique_case()
        affine = useg.grid_affine(world, image_mat, image.shape, grid_mat, grid_shape)
        wj = (np.linalg.det(affine)
              * centre_extent_volume(nib.Nifti1Image(image, image_mat))
              / centre_extent_volume(nib.Nifti1Image(np.zeros(grid_shape), grid_mat)))
        self.assertAlmostEqual(wj, 1.0 / np.linalg.det(world), places=10)


class TestBrainMask(unittest.TestCase):
    @staticmethod
    def prob(brain):
        """Six posteriors times 255 with *brain* split over GM, WM and CSF."""
        p = np.zeros(brain.shape + (6,), np.uint8)
        p[..., 0] = np.round(brain * 255 * 0.5)
        p[..., 1] = np.round(brain * 255 * 0.3)
        p[..., 2] = np.round(brain * 255) - p[..., 0] - p[..., 1]
        p[..., 4] = 255 - p[..., :3].sum(-1)
        return p

    def test_the_largest_component_with_its_cavities_filled(self):
        brain = np.zeros((30, 30, 30))
        brain[5:20, 5:20, 5:20] = 1.0
        brain[10:14, 10:14, 10:14] = 0.0       # a cavity
        brain[25:28, 25:28, 25:28] = 1.0       # an island
        mask = useg.brain_mask(self.prob(brain))
        self.assertTrue(mask[5:20, 5:20, 5:20].all())
        self.assertFalse(mask[25:28, 25:28, 25:28].any())
        self.assertEqual(mask.sum(), 15 ** 3)

    def test_half_the_posterior_is_brain(self):
        brain = np.zeros((10, 10, 10))
        brain[2:8, 2:8, 2:5] = 0.51
        brain[2:8, 2:8, 5:8] = 0.49
        mask = useg.brain_mask(self.prob(brain))
        self.assertTrue(mask[2:8, 2:8, 2:5].all())
        self.assertFalse(mask[..., 5:].any())

    def test_no_brain_is_an_error(self):
        with self.assertRaises(RuntimeError):
            useg.brain_mask(self.prob(np.zeros((8, 8, 8))))


def _las_image(data):
    """*data* (RAS order) stored left to right flipped, as a LAS scan is."""
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = data.shape[0] - 1
    return nib.Nifti1Image(np.ascontiguousarray(data[::-1]), affine)


class TestTwoPasses(unittest.TestCase):
    """The affine comes from a pass on the unfiltered, RAS-ordered scan, the
    segmentation from a second one on the denoised scan on the working grid,
    with that affine fixed."""

    def run_segment(self, data, world, denoised=None):
        calls = []

        def fit(src, templates, mat, template_mat, **kwargs):
            calls.append(dict(src=np.array(src), mat=np.array(mat), kwargs=kwargs))
            shape = np.shape(src)
            return {"prob": np.zeros(shape + (6,), np.uint8), "label": np.array(src),
                    "corrected": np.array(src),
                    "params": {"affine": world if len(calls) == 1 else np.eye(4)}}

        grid = nib.Nifti1Image(np.zeros((9, 11, 13), np.float32),
                               np.diag([0.5, 0.5, 0.5, 1.0]))
        denoised = data if denoised is None else denoised
        with mock.patch.object(useg, "check_available"), \
             mock.patch.object(useg, "unified_templates", return_value=([], np.eye(4))), \
             mock.patch.object(useg.cat_surf, "vol_unified_seg", fit, create=True):
            seg = useg.segment(_las_image(denoised), _las_image(data), grid)
        return seg, calls, grid

    def test_the_first_pass_sees_the_ras_scan_and_its_affine_is_kept(self):
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        world = np.eye(4)
        world[:3, 3] = [0.3, -0.2, 0.1]
        seg, calls, grid = self.run_segment(data, world)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0]["src"], data)
        self.assertGreater(calls[0]["mat"][0, 0], 0)
        self.assertFalse(calls[0]["kwargs"]["prob"])
        for key, value in useg.AFFINE_PASS.items():
            self.assertEqual(calls[0]["kwargs"][key], value)
        np.testing.assert_allclose(
            seg["affine"],
            useg.grid_affine(world, calls[0]["mat"], data.shape, grid.affine, grid.shape))

    def test_the_second_pass_segments_the_denoised_scan_on_the_working_grid(self):
        data = np.ones((4, 5, 6), np.float32)
        seg, calls, grid = self.run_segment(data, np.eye(4), denoised=3 * data)
        np.testing.assert_array_equal(calls[0]["src"], data)       # the affine: unfiltered
        second = calls[1]
        self.assertEqual(second["src"].shape, grid.shape)
        self.assertAlmostEqual(float(second["src"][2, 2, 2]), 3.0, places=4)
        np.testing.assert_array_equal(second["mat"], grid.affine)
        self.assertEqual(second["kwargs"]["affine_mode"], "fixed")
        np.testing.assert_array_equal(second["kwargs"]["affine"], np.eye(4))
        for key, value in useg.UNIFIED_SETTINGS.items():
            self.assertEqual(second["kwargs"][key], value)
        self.assertFalse(second["kwargs"]["sanlm"])     # denoised already
        self.assertEqual(seg["label"].shape, grid.shape)

    def test_a_given_registration_skips_the_first_pass(self):
        data = np.ones((4, 5, 6), np.float32)
        world = np.diag([1.1, 1.1, 1.1, 1.0])
        world[:3, 3] = [0.3, -0.2, 0.1]
        calls = []

        def fit(src, templates, mat, template_mat, **kwargs):
            calls.append({"mat": np.array(mat), "kwargs": kwargs})
            return {"prob": np.zeros(np.shape(src) + (6,), np.uint8), "label": np.array(src),
                    "corrected": np.array(src), "params": {}}

        grid = nib.Nifti1Image(np.zeros((9, 11, 13), np.float32), np.diag([0.5, 0.5, 0.5, 1.0]))
        with mock.patch.object(useg, "check_available"), \
             mock.patch.object(useg, "unified_templates", return_value=([], np.eye(4))), \
             mock.patch.object(useg.cat_surf, "vol_unified_seg", fit, create=True):
            seg = useg.segment(_las_image(data), _las_image(data), grid, world=world)
        self.assertEqual(len(calls), 1)                     # the segmentation alone
        self.assertEqual(calls[0]["kwargs"]["affine_mode"], "fixed")
        np.testing.assert_array_equal(seg["world"], world)
        self.assertIsNone(seg["affine_params"])
        canonical = nib.as_closest_canonical(_las_image(data))
        np.testing.assert_allclose(
            seg["affine"], useg.grid_affine(world, canonical.affine, data.shape, grid.affine, grid.shape))
        # the same as with the registration estimated, when it comes out as the given one
        same, _, _ = self.run_segment(data, world)
        np.testing.assert_allclose(seg["affine"], same["affine"])
        np.testing.assert_allclose(seg["label"], same["label"])

    def test_a_given_registration_is_checked(self):
        data = np.ones((4, 5, 6), np.float32)
        grid = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
        bad = (np.eye(3), np.full((4, 4), np.nan), np.diag([1.0, 1.0, 0.0, 1.0]),
               np.vstack([np.eye(4)[:3], [0, 0, 1, 1]]))
        with mock.patch.object(useg, "check_available"), \
             mock.patch.object(useg, "unified_templates", return_value=([], np.eye(4))):
            for world in bad:
                with self.subTest(world=world.tolist()), self.assertRaises(ValueError):
                    useg.segment(_las_image(data), _las_image(data), grid, world=world)

    def test_both_inputs_share_their_grid(self):
        data = np.ones((4, 5, 6), np.float32)
        with mock.patch.object(useg, "check_available"):
            with self.assertRaises(ValueError):
                useg.segment(_las_image(np.ones((4, 5, 7), np.float32)), _las_image(data),
                             nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4)))

    def test_the_mask_lands_on_the_input_s_voxels(self):
        data = np.random.default_rng(1).random((6, 7, 8)).astype(np.float32) + 1
        mask = np.zeros(data.shape, bool)
        mask[0:2, 3:5, 4:6] = True                       # RAS order
        t1 = _las_image(data)
        brain, mask_img = useg.native_brain(t1, mask)
        self.assertEqual(mask_img.get_data_dtype(), np.uint8)
        np.testing.assert_array_equal(mask_img.affine, t1.affine)
        stored = np.asanyarray(mask_img.dataobj).astype(bool)
        np.testing.assert_array_equal(stored, mask[::-1])
        np.testing.assert_array_equal(
            np.asanyarray(brain.dataobj), np.where(mask, data, 0)[::-1])


class TestMasksBetweenTheGrids(unittest.TestCase):
    def setUp(self):
        # the scan at 1 mm, the working grid at 0.5 mm over the same box
        self.native_shape = (10, 12, 8)
        native_mat = np.eye(4)
        grid_mat = np.diag([0.5, 0.5, 0.5, 1.0])
        grid_shape = (19, 23, 15)
        self.seg = {
            "affine": useg.grid_affine(np.eye(4), native_mat, self.native_shape,
                                       grid_mat, grid_shape),
            "native_shape": self.native_shape,
            "grid": nib.Nifti1Image(np.zeros(grid_shape, np.float32), grid_mat),
        }
        self.native = np.zeros(self.native_shape, bool)
        self.native[2:8, 3:9, 2:6] = True

    def test_there_and_back(self):
        large = useg.to_working_grid(self.native, self.seg)
        self.assertEqual(large.shape, self.seg["grid"].shape)
        self.assertTrue(large[4:15, 6:17, 4:11].all())
        np.testing.assert_array_equal(useg.to_native(large, self.seg), self.native)

    def test_the_maps_are_cut_to_the_mask(self):
        shape = self.seg["grid"].shape
        seg = dict(self.seg, label=np.full(shape, 2.5, np.float32),
                   corrected=np.full(shape, 250.0, np.float32))
        mask = np.zeros(shape, bool)
        mask[3:9, 4:10, 2:8] = True
        affine, brain_large, p0_large = useg.working_grid_maps(seg, mask)
        self.assertEqual(affine.shape, (4, 4))
        np.testing.assert_array_equal(np.asanyarray(p0_large.dataobj), np.where(mask, 2.5, 0))
        np.testing.assert_array_equal(np.asanyarray(brain_large.dataobj), np.where(mask, 250, 0))
        self.assertEqual(brain_large.get_data_dtype(), np.float32)


class TestAvailability(unittest.TestCase):
    def test_an_old_cat_surf_is_named(self):
        with mock.patch.object(useg, "cat_surf", mock.Mock(spec=[], __version__="1.1.8")):
            with self.assertRaisesRegex(RuntimeError, r"cat-surf >= 1\.1\.9.*1\.1\.8"):
                useg.check_available()

    def test_missing_templates_are_named(self):
        with mock.patch.object(useg, "cat_surf", mock.Mock(vol_unified_seg=None)), \
             mock.patch.object(useg, "resolve_template_file", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "six-class"):
                useg.check_available()

    def test_a_folder_without_the_templates_is_named(self):
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(useg, "cat_surf", mock.Mock(vol_unified_seg=None)):
            with self.assertRaisesRegex(RuntimeError, "six-class.*" + re.escape(folder)):
                useg.check_available(folder)


class TestAnotherTemplateFolder(unittest.TestCase):
    """``template_dir``: the templates of a caller that brings its own."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)
        self.mat = np.diag([1.5, 1.5, 1.5, 1.0])
        rng = np.random.default_rng(1)
        for k, name in enumerate(useg.UNIFIED_TEMPLATES):
            data = (rng.random((3, 4, 5, 6)) + k).astype(np.float32)
            nib.save(nib.Nifti1Image(data, self.mat), self.folder / f"{name}.nii.gz")

    def test_the_templates_are_read_from_it_and_normalised(self):
        with mock.patch.object(useg, "resolve_template_file",
                               side_effect=AssertionError("T1Prep's folder was asked")):
            templates, mat = useg.unified_templates(self.folder)
        self.assertEqual(len(templates), 5)
        np.testing.assert_allclose(mat, self.mat)
        for tpl in templates:
            self.assertEqual(tpl.shape, (3, 4, 5, 6))
            np.testing.assert_allclose(tpl.sum(-1), 1, rtol=1e-5)

    def test_a_missing_template_is_not_looked_up_elsewhere(self):
        (self.folder / f"{useg.UNIFIED_TEMPLATES[2]}.nii.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            useg.unified_templates(self.folder)

    def test_segment_hands_it_on_and_finds_the_working_grid_itself(self):
        grid = nib.Nifti1Image(np.zeros((9, 11, 13), np.float32), np.diag([0.5, 0.5, 0.5, 1.0]))
        data = np.ones((8, 9, 10), np.float32)

        def fit(src, templates, mat, template_mat, **kwargs):
            return {"prob": None, "label": None, "corrected": None, "params": {"affine": np.eye(4)}}

        with mock.patch.object(useg, "check_available") as check, \
             mock.patch.object(useg, "unified_templates", return_value=([], np.eye(4))) as load, \
             mock.patch.object(useg, "default_working_grid", return_value=grid) as default, \
             mock.patch.object(useg.cat_surf, "vol_unified_seg", fit, create=True):
            seg = useg.segment(_las_image(data), _las_image(data), template_dir=self.folder)
        check.assert_called_once_with(self.folder)
        load.assert_called_once_with(self.folder)
        default.assert_called_once_with()
        self.assertEqual(seg["grid"].shape, grid.shape)


class TestTheDeformation(unittest.TestCase):
    """``deformation=True`` hands on what the library returns of the shooting."""

    def run_segment(self, **kwargs):
        calls = []

        def fit(src, templates, mat, template_mat, **kw):
            calls.append(kw)
            out = {"prob": None, "label": None, "corrected": None,
                   "params": {"affine": np.eye(4)}}
            if kw.get("inverse"):
                out.update(iy="iy", y="y", jac="jac")
            return out

        grid = nib.Nifti1Image(np.zeros((9, 11, 13), np.float32), np.diag([0.5, 0.5, 0.5, 1.0]))
        data = np.ones((8, 9, 10), np.float32)
        template_mat = np.diag([1.5, 1.5, 1.5, 1.0])
        with mock.patch.object(useg, "check_available"), \
             mock.patch.object(useg, "unified_templates", return_value=([], template_mat)), \
             mock.patch.object(useg.cat_surf, "vol_unified_seg", fit, create=True):
            seg = useg.segment(_las_image(data), _las_image(data), grid, **kwargs)
        return seg, calls, template_mat

    def test_it_is_not_computed_unless_asked_for(self):
        seg, calls, _ = self.run_segment()
        self.assertFalse(calls[1]["deformation"])
        self.assertFalse(calls[1]["inverse"])
        self.assertNotIn("iy", seg)

    def test_the_second_pass_returns_it_with_the_matrix_of_the_templates(self):
        seg, calls, template_mat = self.run_segment(deformation=True)
        self.assertNotIn("inverse", calls[0])            # the affine pass has no use for it
        self.assertTrue(calls[1]["deformation"] and calls[1]["inverse"])
        self.assertEqual((seg["iy"], seg["y"], seg["jac"]), ("iy", "y", "jac"))
        np.testing.assert_array_equal(seg["template_mat"], template_mat)


class TestThePublicNames(unittest.TestCase):
    def test_las_and_the_module_are_exported(self):
        import t1prep
        from t1prep import _intensity, apply_LAS

        self.assertIs(apply_LAS, _intensity.apply_LAS)
        self.assertIs(t1prep.unified_seg, useg)
        self.assertIn("apply_LAS", t1prep.__all__)
        self.assertIn("unified_seg", t1prep.__all__)


@unittest.skipUnless(
    all((_SRC / "t1prep" / "data" / "templates_MNI152NLin2009cAsym" / f"{n}.nii.gz").is_file()
        for n in useg.UNIFIED_TEMPLATES),
    "six-class templates not in this tree")
class TestTemplates(unittest.TestCase):
    def test_five_six_class_templates_on_the_shooting_grid(self):
        folder = _SRC / "t1prep" / "data" / "templates_MNI152NLin2009cAsym"
        grid = nib.load(folder / "Template_4_GS.nii.gz")
        for name in useg.UNIFIED_TEMPLATES:
            with self.subTest(template=name):
                img = nib.load(folder / f"{name}.nii.gz")
                self.assertEqual(img.shape, grid.shape[:3] + (6,))
                np.testing.assert_allclose(img.affine, grid.affine)


class TestCommandLines(unittest.TestCase):
    """--unified-seg goes from t1prep-run and the Python API to segment.py."""

    def test_the_segmentation_gets_the_flag(self):
        cmd = orchestrator._build_segment_cmd("in.nii", "m", "r", "l", use_unified_seg=True)
        self.assertIn("--unified-seg", cmd)
        self.assertNotIn("--unified-seg", orchestrator._build_segment_cmd("in.nii", "m", "r", "l"))

    def test_t1prep_run_passes_it_on(self):
        with mock.patch.object(orchestrator, "_process_single", return_value=0) as single, \
             mock.patch.object(orchestrator.os.path, "isfile", return_value=True):
            orchestrator.main(["x.nii", "--unified-seg", "--amap"])
        self.assertTrue(single.call_args.kwargs["unified_seg"])
        self.assertTrue(single.call_args.kwargs["amap"])

    def test_segment_understands_it(self):
        from t1prep import segment

        argv = ["segment", "--input", "x.nii", "--mri-dir", "m", "--report-dir", "r",
                "--label-dir", "l", "--unified-seg", "--amap"]
        with mock.patch.object(sys, "argv", argv):
            args = segment.parse_arguments()
        self.assertTrue(args.unified_seg and args.amap)


class TestReport(unittest.TestCase):
    def test_the_report_names_the_segmentation(self):
        import tempfile

        params = {"classes": ["GM", "WM", "CSF", "bone", "soft tissue", "air"],
                  "volume_ml": [600, 500, 300, 800, 2500, 5000],
                  "affine_mutual_information": 1.4, "ll": -4.7,
                  "noise_estimated": 0.02, "shooting": {"iterations": 22},
                  "time_s": {"total": 38.0}}
        world = np.diag([1.25, 1.0, 1.0, 1.0])        # template = 1.25 x native
        entry = useg.report_entry(params, world=world)
        with tempfile.TemporaryDirectory() as tmp:
            names = get_filenames(False, "sub01", "", "", "", "")
            Path(tmp, names["Report_file"]).write_text(json.dumps({"vol_tiv": {"value": 1500}}))
            report.write_t1prep_report(tmp, "sub01", False, str(Path(tmp, "sub01.nii")),
                                       segmentation=entry)
            software = json.loads(Path(tmp, names["Report_file"]).read_text())["software"]
        self.assertEqual(software["segmentation"]["method"], "unified")
        self.assertEqual(software["segmentation"]["settings"], useg.UNIFIED_SETTINGS)
        self.assertAlmostEqual(software["segmentation"]["volume_native_ml"]["GM"], 480.0)
        self.assertIsNone(software["models"])        # no network ran


class TestLauncher(unittest.TestCase):
    def test_the_backend_does_not_matter_without_networks(self):
        values = dict(opt.default_values({}, environ={}), unified_seg=True,
                      nn_backend="tinygrad")
        self.assertIn("nn_backend", opt.irrelevant(values))
        arguments = opt.build_arguments(values, {}, environ={})
        self.assertIn("--unified-seg", arguments)
        self.assertNotIn("--nn-backend", arguments)


if __name__ == "__main__":
    unittest.main()
