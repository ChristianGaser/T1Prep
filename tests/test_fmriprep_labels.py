"""The --fmriprep label images hold exact integers.

Up to 0.8.0 the native ``_dseg`` (GM=1, WM=2, CSF=3) and the brain mask
were the 0.5 mm label resampled *trilinearly* and saved through a uint8
header with a scale factor: on IXI199 the dseg had 256 distinct values and
not one voxel equal to 1, 2 or 3, CSF (3) and GM (1) met through WM (2) at
the pial surface, and no mask or ribbon voxel was exactly 1.  fMRIPrep and
ANTs select labels by value.
"""

import argparse
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
import torch.nn.functional as F  # noqa: E402

from t1prep import segment  # noqa: E402
from t1prep.utils import resample_and_save_nifti  # noqa: E402


def _label_map(shape=(24, 24, 24)):
    """A 0.5 mm p0 label map: CSF shell, GM shell, WM core (values 1, 2, 3)."""
    zz, yy, xx = np.indices(shape) - (np.array(shape)[:, None, None, None] - 1) / 2
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    p0 = np.select([r < 5, r < 8, r < 11], [3.0, 2.0, 1.0], 0.0)
    return nib.Nifti1Image(p0.astype(np.float32), np.diag([0.5, 0.5, 0.5, 1]))


def _native(shape=(12, 12, 12)):
    """A 1 mm native image with a uint8 header, as deepbet's mask has."""
    img = nib.Nifti1Image(np.zeros(shape, np.uint8), np.diag([1.0, 1.0, 1.0, 1]))
    img.header.set_data_dtype(np.uint8)
    return img


def _grid(native):
    """Sample the native grid over the label map's field of view."""
    return F.affine_grid(torch.eye(4)[None, :3], [1, 3, *native.shape],
                         align_corners=False)


class TestResampleLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_nearest_and_integer_dtype_give_exact_labels(self):
        native = _native()
        out = self.tmp / "dseg.nii.gz"
        resample_and_save_nifti(_label_map(), _grid(native), native.affine,
                                native.header, str(out), nearest=True, dtype=np.uint8)
        img = nib.load(out)
        self.assertEqual(img.get_data_dtype(), np.uint8)
        self.assertLessEqual(set(np.unique(img.get_fdata())), {0.0, 1.0, 2.0, 3.0})

    def test_the_old_way_was_neither(self):
        native = _native()
        out = self.tmp / "old.nii.gz"
        resample_and_save_nifti(_label_map(), _grid(native), native.affine,
                                native.header, str(out), clip=[0, 4])
        values = np.unique(nib.load(out).get_fdata())
        self.assertGreater(len(values), 4)


class TestFmriprepExtras(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_dseg_and_mask_are_exact(self):
        args = argparse.Namespace(
            input=str(self.tmp / "sub-01_T1w.nii.gz"), mri_dir=str(self.tmp),
            label_dir=str(self.tmp), report_dir=str(self.tmp), atlas="",
            surf=False, csf=True, mwp=False, wp=False, p=True, rp=False,
            lesions=False, save_h5=False, save_fmriprep=True, bids=True, gz=True,
        )
        opts = segment.OutputOptions.from_args(args)
        names = segment._output_names(opts)
        native = _native()
        segment._save_fmriprep_extras(opts, names, _label_map(), native, _grid(native))

        dseg = nib.load(self.tmp / names.native["dseg_volume"])
        mask = nib.load(self.tmp / names.native["mask_volume"])
        d, m = dseg.get_fdata(), mask.get_fdata()
        self.assertLessEqual(set(np.unique(d)), {0.0, 1.0, 2.0, 3.0})
        self.assertEqual(set(np.unique(m)), {0.0, 1.0})
        # fMRIPrep's order: GM=1, WM=2, CSF=3; WM sits in the centre
        self.assertEqual(d[6, 6, 6], 2)
        np.testing.assert_array_equal(m > 0, d > 0)


if __name__ == "__main__":
    unittest.main()
