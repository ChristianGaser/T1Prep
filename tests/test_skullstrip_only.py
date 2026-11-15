"""``--skullstrip-only`` writes the skull-stripped image and the brain mask.

Its help always said so, but up to 0.9.0 only the image was written: the mask
the skull-strip returns was dropped.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import nibabel as nib
    from t1prep.segment import save_skullstrip_only_outputs
except Exception as exc:  # pragma: no cover - optional heavy deps
    raise unittest.SkipTest(f"t1prep.segment unavailable: {exc}")


class TestSkullStripOnlyOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        affine = np.diag([1.0, 1.2, 0.9, 1.0])
        affine[:3, 3] = (-8.0, 4.0, 2.0)
        inside = np.zeros((8, 9, 10), dtype=np.float32)
        inside[2:6, 3:7, 2:8] = 1.0
        self.brain = nib.Nifti1Image(inside * 250.0, affine)
        # As the skull-strip returns it: float, not 0/1
        self.mask = nib.Nifti1Image(inside * 0.97, affine)

    def test_cat12_names(self):
        save_skullstrip_only_outputs(
            self.brain, self.mask, False, str(self.tmp), "sub1", "nii")
        self.assertTrue((self.tmp / "msub1.nii").is_file())
        mask = nib.load(self.tmp / "masksub1.nii")
        self.assertEqual(mask.get_data_dtype(), np.uint8)
        values = np.asanyarray(mask.dataobj)
        self.assertEqual(sorted(np.unique(values)), [0, 1])
        self.assertEqual(int(values.sum()), 4 * 4 * 6)
        np.testing.assert_allclose(mask.affine, self.brain.affine)
        self.assertEqual(mask.shape, self.brain.shape)

    def test_bids_names(self):
        save_skullstrip_only_outputs(
            self.brain, self.mask, True, str(self.tmp), "sub-01_T1w", "nii.gz")
        self.assertTrue((self.tmp / "sub-01_desc-skullstripped_T1w.nii.gz").is_file())
        self.assertTrue((self.tmp / "sub-01_desc-brain_mask.nii.gz").is_file())

    def test_without_a_mask_the_image_is_still_written(self):
        save_skullstrip_only_outputs(
            self.brain, None, False, str(self.tmp), "sub1", "nii")
        self.assertTrue((self.tmp / "msub1.nii").is_file())
        self.assertFalse((self.tmp / "masksub1.nii").exists())


if __name__ == "__main__":
    unittest.main()
