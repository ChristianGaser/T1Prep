import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib


# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import t1prep.utils as t1prep_utils
from t1prep.utils import (
    _content_bounds,
    crop_nifti_image_with_border,
    find_largest_cluster,
    get_filenames,
    get_packaged_data_path,
    get_ras,
    load_namefile,
    resample_and_save_nifti,
    smart_round,
    substitute_pattern,
)

class TestUtils(unittest.TestCase):

    def test_find_largest_cluster_no_cluster(self):
        """Test that an empty mask is returned when no clusters are found."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == False))
        self.assertEqual(cluster_mask.shape, binary_volume.shape)

    def test_find_largest_cluster_one_cluster(self):
        """Test that the largest cluster is correctly identified."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[2:5, 2:5, 2:5] = True
        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == binary_volume))

    def test_find_largest_cluster_multiple_clusters(self):
        """Test that the largest cluster is identified among multiple clusters."""
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[2:4, 2:4, 2:4] = True  # Smaller cluster
        binary_volume[6:9, 6:9, 6:9] = True  # Larger cluster

        expected_mask = np.zeros((10, 10, 10), dtype=bool)
        expected_mask[6:9, 6:9, 6:9] = True

        cluster_mask = find_largest_cluster(binary_volume)
        self.assertTrue(np.all(cluster_mask == expected_mask))

    def test_find_largest_cluster_min_size_filters_all(self):
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[2:4, 2:4, 2:4] = True  # size 8
        # min_size larger than cluster should yield empty
        cluster_mask = find_largest_cluster(binary_volume, min_size=9)
        self.assertTrue(np.all(cluster_mask == False))

    def test_find_largest_cluster_max_n_cluster_two(self):
        binary_volume = np.zeros((10, 10, 10), dtype=bool)
        binary_volume[1:3, 1:3, 1:3] = True  # 8 voxels
        binary_volume[6:9, 6:9, 6:9] = True  # 27 voxels
        binary_volume[4:5, 4:5, 4:6] = True  # 2 voxels

        cluster_mask = find_largest_cluster(binary_volume, max_n_cluster=2)
        # should contain the two largest clusters (27 and 8) but not the tiny one
        self.assertTrue(cluster_mask[7, 7, 7])
        self.assertTrue(cluster_mask[1, 1, 1])
        self.assertFalse(cluster_mask[4, 4, 4])

    def test_smart_round(self):
        """Test the smart_round function."""
        self.assertEqual(smart_round(0.123456789), 0.12346)
        self.assertEqual(smart_round(1.123456789), 1.123)
        self.assertEqual(smart_round(10.123456789), 10.12)
        self.assertEqual(smart_round(-0.123456789), -0.12346)
        self.assertEqual(smart_round(-1.123456789), -1.123)
        self.assertEqual(smart_round(-10.123456789), -10.12)
        self.assertEqual(smart_round(0), 0.0)

    def test_get_ras_identity(self):
        aff = np.eye(4)
        ras, dirs = get_ras(aff, 3)
        np.testing.assert_array_equal(ras, np.array([0, 1, 2]))
        np.testing.assert_array_equal(dirs, np.array([1, 1, 1]))

    def test_get_ras_swapped_xy(self):
        aff = np.eye(4)
        aff[:3, :3] = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        ras, dirs = get_ras(aff, 3)
        np.testing.assert_array_equal(ras, np.array([1, 0, 2]))
        np.testing.assert_array_equal(dirs, np.array([1, 1, 1]))

    def test_substitute_pattern_basic(self):
        pat = "{bname}_hemi-{side}{desc}{space}.{nii_ext}"
        out = substitute_pattern(pat, bname="sub-01", side="L", desc="", space="", nii_ext="nii.gz")
        self.assertEqual(out, "sub-01_hemi-L.nii.gz")

    def test_load_namefile_parsing(self):
        content = """# comment\n\nA\tfoo\tbar\nB\tbaz\n"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "names.tsv"
            p.write_text(content)
            d = load_namefile(str(p))
        self.assertEqual(d["A"], ("foo", "bar"))
        self.assertEqual(d["B"], ("baz", ""))

    def test_get_filenames_bids_vs_legacy(self):
        # This asserts the naming contract encoded in src/t1prep/data/Names.tsv.
        bname = "sub-01"
        nii_ext = "nii.gz"
        legacy = get_filenames(False, bname=bname, side="left", desc="", space="", nii_ext=nii_ext)
        bids = get_filenames(True, bname=bname, side="left", desc="", space="", nii_ext=nii_ext)
        self.assertEqual(legacy["Hemi_volume"], "lh.seg.sub-01.nii.gz")
        self.assertEqual(bids["Hemi_volume"], "sub-01_hemi-L_seg.nii.gz")

    def test_get_packaged_data_path_exists(self):
        # Should resolve both in editable (repo) mode and when installed.
        p = get_packaged_data_path("cat_surf_view_defaults.txt")
        self.assertEqual(p.name, "cat_surf_view_defaults.txt")
        self.assertTrue(p.exists())

    def test_crop_nifti_image_with_border_even_dims(self):
        data = np.zeros((9, 10, 11), dtype=np.float32)
        data[2:7, 3:9, 4:10] = 2.0
        img = nib.Nifti1Image(data, affine=np.eye(4))
        cropped = crop_nifti_image_with_border(img, border=0, threshold=1.0)
        # util pads odd dims to be even
        sx, sy, sz = cropped.shape
        self.assertEqual(sx % 2, 0)
        self.assertEqual(sy % 2, 0)
        self.assertEqual(sz % 2, 0)


class TestResampleAndSaveNifti(unittest.TestCase):
    """Kernel selection and overshoot clipping in ``resample_and_save_nifti``."""

    @staticmethod
    def _label_volume(shape=(32, 34, 30)):
        """A label-like map with sharp steps, as ``get_partition`` produces.

        Background is exactly 1.0, which is what puts it below the crop
        threshold and lets the B-spline fast path find a content box.
        """
        data = np.ones(shape, dtype=np.float32)
        lo = [s // 4 for s in shape]
        hi = [3 * s // 4 for s in shape]
        data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 2.0
        mid_lo = [3 * s // 8 for s in shape]
        mid_hi = [5 * s // 8 for s in shape]
        data[mid_lo[0]:mid_hi[0], mid_lo[1]:mid_hi[1], mid_lo[2]:mid_hi[2]] = 3.0
        return nib.Nifti1Image(data, affine=np.eye(4))

    @staticmethod
    def _rotation_grid(shape, degrees=9.0):
        """Resampling grid for a pure in-plane rotation.

        Built with the same ``align_corners`` convention the pipeline uses, so
        the grid matches how ``resample_and_save_nifti`` samples it.
        """
        import torch
        import torch.nn.functional as F
        from torchreg.utils import INTERP_KWARGS

        th = np.deg2rad(degrees)
        rot = np.array([[np.cos(th), -np.sin(th), 0.0],
                        [np.sin(th), np.cos(th), 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
        mat = torch.tensor(np.hstack([rot, np.zeros((3, 1), np.float32)]))[None]
        return F.affine_grid(mat, [1, 3, *shape],
                             align_corners=INTERP_KWARGS["align_corners"])

    def _resample(self, img, tmpdir, name, **kwargs):
        out = str(Path(tmpdir) / f"{name}.nii")
        resample_and_save_nifti(
            img, self._rotation_grid(img.shape), np.eye(4), img.header, out, **kwargs
        )
        return nib.load(out).get_fdata()

    def test_default_matches_trilinear_grid_sample(self):
        """Without ``bspline`` the kernel must stay plain trilinear."""
        import torch
        import torch.nn.functional as F
        from torchreg.utils import INTERP_KWARGS

        img = self._label_volume()
        with tempfile.TemporaryDirectory() as tmp:
            got = self._resample(img, tmp, "default")
        expected = F.grid_sample(
            torch.from_numpy(img.get_fdata().astype(np.float32))[None, None],
            self._rotation_grid(img.shape),
            align_corners=INTERP_KWARGS["align_corners"],
        )[0, 0].numpy()
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_bspline_overshoots_and_clipping_removes_it(self):
        """B-spline rings past the label range; ``clip_overshoot`` cuts it."""
        img = self._label_volume()
        data = img.get_fdata()
        with tempfile.TemporaryDirectory() as tmp:
            loose = self._resample(img, tmp, "loose", bspline=True)
            tight = self._resample(img, tmp, "tight", bspline=True,
                                   clip_overshoot=True)
        self.assertGreater(loose.max(), data.max() + 1e-3)
        self.assertGreaterEqual(tight.min(), data.min() - 1e-6)
        self.assertLessEqual(tight.max(), data.max() + 1e-6)
        # Clipping must only cut the overshoot, never alter the interior.
        np.testing.assert_allclose(tight, np.clip(loose, data.min(), data.max()))

    def test_bspline_keeps_edges_sharper_than_trilinear(self):
        """The point of the option: less blurring of the thresholded boundary."""
        img = self._label_volume()
        with tempfile.TemporaryDirectory() as tmp:
            linear = self._resample(img, tmp, "linear")
            spline = self._resample(img, tmp, "spline", bspline=True,
                                    clip_overshoot=True)

        def edge_gradient(vol):
            return np.mean(np.abs(np.gradient(vol, axis=0)))

        self.assertGreater(edge_gradient(spline), edge_gradient(linear))

    def test_clip_overshoot_applies_before_round(self):
        """Rounding must see clipped values, not the raw ringing."""
        img = self._label_volume()
        data = img.get_fdata()
        with tempfile.TemporaryDirectory() as tmp:
            got = self._resample(img, tmp, "rounded", bspline=True,
                                 clip_overshoot=True, round=True)
        self.assertLessEqual(got.max(), data.max() + 1e-6)
        np.testing.assert_allclose(got, np.round(got))


    def test_crop_fast_path_matches_full_volume_spline(self):
        """Restricting the gather to the crop box must not change the output."""
        from unittest.mock import patch

        # Large enough that the content box is a strict subset of the volume.
        img = self._label_volume((96, 96, 96))
        full_box = lambda mask, border: tuple(slice(0, n) for n in mask.shape)
        kwargs = dict(bspline=True, clip_overshoot=True)

        with tempfile.TemporaryDirectory() as tmp:
            # Sanity: the box really is smaller than the volume, so the fast
            # path is exercised rather than trivially covering everything.
            grid = self._rotation_grid(img.shape)
            import torch.nn.functional as F
            from torchreg.utils import INTERP_KWARGS
            probe = F.grid_sample(
                nib_to_tensor(img), grid,
                align_corners=INTERP_KWARGS["align_corners"],
            )[0, 0]
            box = _content_bounds(
                probe > t1prep_utils._CROP_THRESHOLD,
                t1prep_utils._CROP_BORDER + t1prep_utils._SPLINE_CROP_MARGIN,
            )
            self.assertIsNotNone(box)
            self.assertLess(
                int(np.prod([sl.stop - sl.start for sl in box])), probe.numel()
            )

            fast = self._resample(img, tmp, "fast", crop=True, align=True, **kwargs)
            with patch.object(t1prep_utils, "_content_bounds", full_box):
                full = self._resample(img, tmp, "full", crop=True, align=True,
                                      **kwargs)

        self.assertEqual(fast.shape, full.shape)
        np.testing.assert_array_equal(fast, full)


class TestContentBounds(unittest.TestCase):
    """Bounding box helper backing the B-spline crop fast path."""

    def test_returns_none_for_empty_mask(self):
        import torch

        self.assertIsNone(_content_bounds(torch.zeros(8, 8, 8, dtype=torch.bool), 2))

    def test_box_covers_content_plus_border(self):
        import torch

        mask = torch.zeros(20, 20, 20, dtype=torch.bool)
        mask[8:12, 7:13, 9:11] = True
        box = _content_bounds(mask, 3)
        self.assertEqual(box[0], slice(5, 15))
        self.assertEqual(box[1], slice(4, 16))
        self.assertEqual(box[2], slice(6, 14))

    def test_box_is_clamped_to_volume(self):
        import torch

        mask = torch.zeros(10, 10, 10, dtype=torch.bool)
        mask[0:2, 8:10, 4:6] = True
        box = _content_bounds(mask, 5)
        self.assertEqual(box[0], slice(0, 7))
        self.assertEqual(box[1], slice(3, 10))
        self.assertEqual(box[2], slice(0, 10))


def nib_to_tensor(img):
    """Volume of ``img`` as a 5D tensor, matching the resampler's layout."""
    import torch

    return torch.from_numpy(img.get_fdata().astype(np.float32))[None, None]


if __name__ == '__main__':
    unittest.main()
