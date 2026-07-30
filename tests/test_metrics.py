import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib


# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.metrics import compute_dice_nifti


def _labels(shape=(40, 44, 48)):
    """Create a simple two-class label block inside a background of zeros."""
    lab = np.zeros(shape, dtype=np.int16)
    lab[8:30, 10:34, 12:38] = 1
    lab[14:24, 16:28, 18:32] = 2
    return lab


def _affine(zoom=1.0, translation=(-20.0, -22.0, -24.0)):
    """Affine with permuted/flipped axes, as produced by many scanners."""
    aff = np.zeros((4, 4))
    aff[0, 2] = -zoom
    aff[1, 0] = -zoom
    aff[2, 1] = -zoom
    aff[:3, 3] = translation
    aff[3, 3] = 1.0
    return aff


class TestComputeDiceNiftiGeometry(unittest.TestCase):
    """Geometry handling: the NIfTI affine must be honoured, not ignored."""

    def setUp(self):
        # Keep the resampling notices out of the test output (one test opts in)
        logger = logging.getLogger("t1prep.metrics")
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, previous)

    def test_identical_images_are_not_resampled(self):
        """Matching grids yield a perfect score without any interpolation."""
        img = nib.Nifti1Image(_labels(), _affine())
        _, order, dice_per, dice_weighted, generalized = compute_dice_nifti(img, img)
        self.assertEqual(list(order), [1, 2])
        np.testing.assert_allclose(dice_per, [1.0, 1.0])
        self.assertAlmostEqual(dice_weighted, 1.0)
        self.assertAlmostEqual(generalized, 1.0)

    def test_permuted_axes_via_affine(self):
        """Reoriented images (different shape, same world content) score 1.0."""
        gt = nib.Nifti1Image(_labels(), _affine())
        pred = nib.as_closest_canonical(gt)
        self.assertNotEqual(gt.shape, pred.shape)

        _, _, dice_per, _, _ = compute_dice_nifti(gt, pred)
        np.testing.assert_allclose(dice_per, [1.0, 1.0])

    def test_padded_grid_with_matching_world_geometry(self):
        """A padded prediction covering the same world space scores 1.0."""
        lab = _labels()
        gt_affine = _affine()
        padded = np.pad(lab, 5)
        pred_affine = gt_affine.copy()
        pred_affine[:3, 3] = gt_affine[:3, :3] @ np.full(3, -5.0) + gt_affine[:3, 3]

        _, _, dice_per, _, _ = compute_dice_nifti(
            nib.Nifti1Image(lab, gt_affine), nib.Nifti1Image(padded, pred_affine)
        )
        np.testing.assert_allclose(dice_per, [1.0, 1.0])

    def test_shifted_world_position_lowers_dice(self):
        """A real world-space offset must reduce Dice instead of being ignored."""
        lab = _labels()
        gt = nib.Nifti1Image(lab, _affine())
        shifted = _affine()
        shifted[:3, 3] += np.array([6.0, 0.0, 0.0])
        pred = nib.Nifti1Image(lab, shifted)

        _, _, dice_per, _, _ = compute_dice_nifti(gt, pred)
        self.assertTrue(np.all(dice_per < 0.95))

        # Opting out compares voxel-to-voxel again (affines ignored)
        _, _, dice_voxelwise, _, _ = compute_dice_nifti(gt, pred, resample=False)
        np.testing.assert_allclose(dice_voxelwise, [1.0, 1.0])

    def test_shape_mismatch_without_resampling_raises(self):
        """resample=False keeps the strict same-shape requirement."""
        gt = nib.Nifti1Image(_labels(), _affine())
        pred = nib.Nifti1Image(np.pad(_labels(), 5), _affine())
        with self.assertRaises(ValueError):
            compute_dice_nifti(gt, pred, resample=False)

    def test_voxel_size_difference_is_resampled(self):
        """Different voxel sizes are aligned through the affines."""
        lab = _labels()
        gt = nib.Nifti1Image(lab, _affine(zoom=1.0))
        # Same object sampled at 2 mm: half the extent per axis
        coarse = lab[::2, ::2, ::2]
        pred = nib.Nifti1Image(coarse, _affine(zoom=2.0))

        _, _, dice_per, _, _ = compute_dice_nifti(gt, pred)
        # Nearest-neighbour upsampling from 2 mm keeps the large class nearly
        # perfect; the small nested class loses boundary detail by construction.
        self.assertGreater(dice_per[0], 0.95)
        self.assertGreater(dice_per[1], 0.7)

    def test_resampling_is_logged(self):
        """A geometry mismatch is reported instead of silently changing scores."""
        gt = nib.Nifti1Image(_labels(), _affine())
        pred = nib.as_closest_canonical(gt)
        with self.assertLogs("t1prep.metrics", level="WARNING") as captured:
            compute_dice_nifti(gt, pred)
        self.assertIn("resampled pred", "\n".join(captured.output))

    def test_soft_dice_with_probability_channels(self):
        """4D probability maps are resampled per channel in soft mode."""
        lab = _labels()
        probs = np.stack([(lab == 1) * 0.9, (lab == 2) * 0.8], axis=-1).astype(float)
        gt = nib.Nifti1Image(probs, _affine())
        pred = nib.as_closest_canonical(gt)
        self.assertNotEqual(gt.shape, pred.shape)

        _, order, dice_per, _, _ = compute_dice_nifti(gt, pred, round_labels=False)
        self.assertEqual(list(order), [1, 2])
        np.testing.assert_allclose(dice_per, [1.0, 1.0], atol=1e-6)

    def test_accepts_paths(self):
        """File paths (str and Path) are loaded with their affines."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.nii.gz"
            pred_path = Path(tmp) / "pred.nii.gz"
            gt = nib.Nifti1Image(_labels(), _affine())
            nib.save(gt, gt_path)
            nib.save(nib.as_closest_canonical(gt), pred_path)

            _, _, dice_per, _, _ = compute_dice_nifti(str(gt_path), pred_path)
            np.testing.assert_allclose(dice_per, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
