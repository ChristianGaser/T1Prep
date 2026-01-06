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

from t1prep.utils import (
    crop_nifti_image_with_border,
    find_largest_cluster,
    get_filenames,
    get_packaged_data_path,
    get_ras,
    load_namefile,
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
        p = get_packaged_data_path("cat_viewsurf_defaults.txt")
        self.assertEqual(p.name, "cat_viewsurf_defaults.txt")
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

if __name__ == '__main__':
    unittest.main()
