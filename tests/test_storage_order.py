"""Results do not depend on how a scan stores its axes; volumes use the
grid the affine refers to.

Up to 0.8.0 SANLM and the first bias fit ran on the stored voxel lattice
(both anchor their blocks or their every-5th-voxel subsampling at index 0,
the bias fit also median-filters with an even size).  The same scan stored
LPI instead of PSR gave GM -0.73%, thickness -1.3% and ROI differences up to
23%; only axis permutations were handled exactly.  And the factor that turns
template volumes into native ones used the edge-to-edge box of the native
grid and of the 1.5 mm warp template, although the affine maps normalised
coordinates of the 0.5 mm affine template with ``align_corners=True``: every
absolute volume was 0.3-1.8% too large, depending on the matrix size.
"""

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cat_surf  # noqa: E402

from t1prep._intensity import correct_bias_field  # noqa: E402
from t1prep.utils import (  # noqa: E402
    centre_extent_volume,
    from_ras_storage,
    in_ras_storage,
    to_ras_storage,
)


def _phantom(shape=(40, 44, 36), seed=0):
    """A bright ellipsoid under a smooth multiplicative bias, with noise."""
    rng = np.random.default_rng(seed)
    idx = np.indices(shape).astype(np.float64)
    centre = (np.array(shape) - 1)[:, None, None, None] / 2
    r = np.sqrt((((idx - centre) / (np.array(shape)[:, None, None, None] * 0.38)) ** 2).sum(0))
    tissue = np.where(r < 1, 1.0, 0.0) + np.where(r < 0.55, 0.3, 0.0)
    bias = 1 + 0.25 * (idx[0] / shape[0]) - 0.15 * (idx[2] / shape[2])
    return (tissue * bias + 0.01 * rng.standard_normal(shape)).astype(np.float32)


def _two_storage_orders():
    """The same image stored RAS and LPI (x and z flipped), same world grid."""
    data = _phantom()
    ras = nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0]))
    flip = np.diag([-1.0, 1.0, -1.0, 1.0])
    flip[:3, 3] = [data.shape[0] - 1, 0, data.shape[2] - 1]
    lpi = nib.Nifti1Image(np.ascontiguousarray(data[::-1, :, ::-1]), ras.affine @ flip)
    return ras, lpi


def _world(img):
    return nib.as_closest_canonical(img).get_fdata()


class TestStorageOrder(unittest.TestCase):
    def test_round_trip_and_common_ras_array(self):
        ras, lpi = _two_storage_orders()
        a = to_ras_storage(np.asanyarray(lpi.dataobj), lpi.affine)
        np.testing.assert_array_equal(a, np.asanyarray(ras.dataobj))
        np.testing.assert_array_equal(from_ras_storage(a, lpi.affine), lpi.dataobj)

    def test_sanlm_depends_on_the_lattice_unless_run_in_ras_order(self):
        ras, lpi = _two_storage_orders()
        raw = [nib.Nifti1Image(cat_surf.vol_sanlm(np.asanyarray(i.dataobj)), i.affine)
               for i in (ras, lpi)]
        # on the stored lattice a flipped copy changes the values by up to ~0.5
        self.assertGreater(np.abs(_world(raw[0]) - _world(raw[1])).max(), 1e-2)

        wrapped = [
            nib.Nifti1Image(
                from_ras_storage(
                    cat_surf.vol_sanlm(to_ras_storage(np.asanyarray(i.dataobj), i.affine)),
                    i.affine,
                ),
                i.affine,
            )
            for i in (ras, lpi)
        ]
        # identical up to SANLM's own run-to-run noise (threaded float32 sums,
        # ~2e-7 on this phantom; cat-surf 1.1.4)
        np.testing.assert_allclose(_world(wrapped[0]), _world(wrapped[1]), atol=1e-6)

    def test_the_first_bias_fit_in_ras_order(self):
        ras, lpi = _two_storage_orders()
        a = in_ras_storage(ras, correct_bias_field)
        b = in_ras_storage(lpi, correct_bias_field)
        self.assertTrue(np.array_equal(b.affine, lpi.affine))
        np.testing.assert_array_equal(_world(a), _world(b))


class TestCentreExtent(unittest.TestCase):
    def test_box_between_the_corner_voxel_centres(self):
        img = nib.Nifti1Image(np.zeros((11, 21, 5), np.uint8), np.diag([0.5, 2.0, 3.0, 1.0]))
        self.assertAlmostEqual(centre_extent_volume(img), (10 * 0.5) * (20 * 2.0) * (4 * 3.0))

    def test_independent_of_storage_order_and_of_pixdim(self):
        ras, lpi = _two_storage_orders()
        lpi.header.set_zooms((9.0, 9.0, 9.0))   # a wrong pixdim must not matter
        self.assertAlmostEqual(centre_extent_volume(ras), centre_extent_volume(lpi))


if __name__ == "__main__":
    unittest.main()
