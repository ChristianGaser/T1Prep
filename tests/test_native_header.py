"""Native outputs carry one geometry: qform = sform, pixdim = voxel sizes.

Up to 0.8.0 ``align_brain`` permuted the data axes but rewrote only
``srow_*`` and ``qoffset_*``.  Every scan stored with permuted axes (any
sagittal or coronal dcm2niix output) kept the input's qform rotation and
pixdim order: IXI199 (PSR) came out with a PSR qform on an RPS array and
pixdim (0.94, 0.94, 1.2) for voxels of (1.2, 0.94, 0.94).  nibabel and SPM
use the sform and were fine; ITK (ANTs, fMRIPrep) fell back to the qform and
placed the image up to 388 mm away, and the QA got the zooms in the wrong
order.  The existing orientation test passed ``header=None``, so it never
saw this.
"""

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deepbet.utils import reoriented_nifti  # noqa: E402

from t1prep.utils import align_brain, get_resampled_header  # noqa: E402

ZOOMS = np.array([1.2, 0.9375, 0.9375])     # voxel sizes along R-L, P-A, S-I


def _input(codes, qform_code=1, sform_code=1, shape=(20, 24, 16)):
    """A volume stored with the given axis codes, voxel sizes ZOOMS by RAS axis."""
    ras = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
    aff = np.eye(4)
    aff[:3, :3] = 0
    for col, code in enumerate(codes):
        axis, sign = ras[code]
        aff[axis, col] = sign * ZOOMS[axis]
    aff[:3, 3] = [-12.0, 17.0, -9.0]
    img = nib.Nifti1Image(np.random.default_rng(0).random(shape, np.float32), aff)
    img.header.set_qform(aff, code=qform_code)
    img.header.set_sform(aff, code=sform_code)
    return img


def _consistent(header, affine):
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    return (np.allclose(header.get_qform(), affine, atol=1e-4)
            and np.allclose(header.get_sform(), affine, atol=1e-4)
            and np.allclose(header.get_zooms()[:3], zooms, atol=1e-5))


class TestTheNativeHeader(unittest.TestCase):
    CASES = ["PSR", "ILP", "LPI", "LAS", "RAS"]

    def test_align_brain_keeps_qform_sform_and_pixdim_together(self):
        for codes in self.CASES:
            for q, s in ((1, 1), (1, 0), (0, 2)):
                with self.subTest(codes=codes, qform_code=q, sform_code=s):
                    img = _input(codes, q, s)
                    vol, aff, hdr, _ = align_brain(
                        img.get_fdata(), img.affine, img.header.copy(), np.eye(4), do_flip=0
                    )
                    self.assertTrue(_consistent(hdr, aff))
                    # same voxels at the same world positions as the input
                    out = nib.Nifti1Image(vol, aff, hdr)
                    np.testing.assert_allclose(
                        nib.as_closest_canonical(out).get_fdata(),
                        nib.as_closest_canonical(img).get_fdata(),
                    )

    def test_outputs_written_through_deepbet_stay_consistent(self):
        # the path of every native output: align_brain -> reoriented_nifti
        for codes in self.CASES:
            with self.subTest(codes=codes):
                img = _input(codes)
                vol, aff, hdr, _ = align_brain(
                    img.get_fdata(), img.affine, img.header.copy(), np.eye(4), do_flip=0
                )
                t1 = nib.Nifti1Image(vol, aff, hdr)
                can = nib.as_closest_canonical(t1).get_fdata()
                out = reoriented_nifti(can.astype(np.float32), t1.affine, t1.header)
                saved = nib.Nifti1Image.from_bytes(out.to_bytes())
                self.assertTrue(_consistent(saved.header, saved.affine))

    def test_the_working_grid_uses_the_zooms_of_the_data_axes(self):
        img = _input("PSR")
        vol, aff, hdr, _ = align_brain(
            img.get_fdata(), img.affine, img.header.copy(), np.eye(4), do_flip=0
        )
        t1 = nib.Nifti1Image(vol, aff, hdr)
        hdr_05, aff_05 = get_resampled_header(t1.header, t1.affine, np.array([0.5] * 3))
        # PSR (20, 24, 16) -> RPS (16, 20, 24) with zooms (1.2, 0.9375, 0.9375)
        np.testing.assert_array_equal(hdr_05["dim"][1:4], np.round([16 * 2.4, 20 * 1.875, 24 * 1.875]))
        np.testing.assert_allclose(np.sqrt((aff_05[:3, :3] ** 2).sum(0)), 0.5)


if __name__ == "__main__":
    unittest.main()
