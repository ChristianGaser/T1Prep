"""ROI volumes are looked up by label id, not matched to the CSV by position.

Up to 0.8.0 the region table was ``pd.concat([csv_rows, get_volumes(...)])``.
deepmriprep's ``get_volumes`` returns one unlabelled row per label *present*
in the warped atlas, so every CSV row after an id the atlas lacks carried the
next region's volume.  Neuromorphometrics (a default atlas) lists the vessels,
ids 63/64, which the atlas does not have: 106 of its 136 rows were shifted
(AAL3 136 of 170, Mori 70 of 128; SUIT has six labels without a CSV row and
got NaN ids).  ``get_volumes`` also merges a one-voxel region into the next.
"""

import glob
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from t1prep._atlas import roi_table, roi_tissue_volumes  # noqa: E402
from t1prep.utils import TEMPLATE_PATH_T1PREP  # noqa: E402


def _maps(shape, seed=0):
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.random(shape).astype(np.float32)) for _ in range(3)]


class TestTissueVolumesById(unittest.TestCase):
    def test_rows_follow_the_ids_and_absent_ids_are_zero(self):
        atlas = np.zeros((6, 5, 4), np.float32)
        atlas[:3] = 1
        atlas[3:5] = 4
        atlas[5, 0, 0] = 2          # a one-voxel region
        p = _maps(atlas.shape)
        table = roi_tissue_volumes(torch.from_numpy(atlas), *p, [1, 2, 3, 4])
        for row, roi in zip(table, (1, 2, 3, 4)):
            mask = atlas == roi
            expected = [p[k].numpy()[mask].sum() for k in range(3)] + [mask.sum()]
            np.testing.assert_allclose(row, expected, rtol=1e-6)
        self.assertEqual(table[1, 3], 1)    # the one-voxel region kept its voxel
        self.assertTrue(np.all(table[2] == 0))

    def test_same_as_deepmriprep_when_every_label_is_present(self):
        from deepmriprep.atlas import get_volumes

        atlas = np.repeat(np.arange(1, 5, dtype=np.float32), 30).reshape(4, 5, 6)
        p = _maps(atlas.shape, seed=1)
        ours = roi_tissue_volumes(torch.from_numpy(atlas), *p, [1, 2, 3, 4])
        theirs = get_volumes(torch.from_numpy(atlas), *p)
        np.testing.assert_allclose(ours, theirs, rtol=1e-5)

    def test_deepmriprep_merges_a_one_voxel_region(self):
        # Why it is replaced: labels 1,1,2,3,3 come back as two regions
        from deepmriprep.atlas import get_volumes

        atlas = torch.tensor([1, 1, 2, 3, 3], dtype=torch.float32).reshape(1, 1, 5)
        ones = torch.ones_like(atlas)
        self.assertEqual(len(get_volumes(atlas, ones, ones, ones)), 2)
        self.assertEqual(
            roi_tissue_volumes(atlas, ones, ones, ones, [1, 2, 3])[:, 3].tolist(),
            [2, 1, 2],
        )


class TestRoiTable(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_csv_rows_of_absent_ids_go_and_unnamed_labels_come(self):
        csv = self.tmp / "a.csv"
        csv.write_text("ROIid;ROIname\n1;one\n2;two\n3;not in the atlas\n")
        rois = roi_table(str(csv), [0, 1, 2, 5])
        self.assertEqual(rois["ROIid"].tolist(), [1, 2, 5])
        self.assertEqual(rois["ROIname"].tolist(), ["one", "two", "Region_5"])

    def test_without_a_csv_every_label_gets_a_name(self):
        rois = roi_table(str(self.tmp / "none.csv"), [0, 3, 7])
        self.assertEqual(rois["ROIname"].tolist(), ["Region_3", "Region_7"])

    def test_every_shipped_atlas_reports_each_label_once(self):
        for nii in sorted(glob.glob(os.path.join(TEMPLATE_PATH_T1PREP, "*.nii.gz"))):
            csv = nii[: -len(".nii.gz")] + ".csv"
            if not os.path.exists(csv):
                continue
            with self.subTest(atlas=os.path.basename(nii)):
                ids = np.unique(np.asarray(nib.load(nii).dataobj)).astype(np.int64)
                rois = roi_table(csv, ids)
                self.assertEqual(sorted(rois["ROIid"]), sorted(ids[ids > 0]))
                self.assertFalse(rois["ROIid"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
