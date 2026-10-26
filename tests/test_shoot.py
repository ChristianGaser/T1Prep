"""Geodesic shooting hands its deformations on in deepmriprep's convention.

T1Prep registers by geodesic shooting (``cat_surf.vol_shoot``) instead of
deepmriprep's warp network, and everything downstream -- the warped and
modulated maps, the y_/iy_ and ITK fields, the atlases -- reads the
deformations as deepmriprep wrote them: ``grid_sample`` grids in normalised
coordinates with reversed channels, stored in the template's orientation.
The conventions are checked with an exact integer translation standing in
for the shooting; the real shooting runs when cat-surf provides it.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cat_surf  # noqa: E402

from t1prep import _shoot  # noqa: E402

#: CAT12's template grid: LAS storage, 1.5 mm, corners (-84, -120, -72) RAS
TMAT = np.array([[-1.5, 0, 0, 84.0], [0, 1.5, 0, -120.0], [0, 0, 1.5, -72.0], [0, 0, 0, 1]])
SHAPE = (113, 137, 113)
SHIFT = np.array([2.0, -1.0, 3.0])  # template voxel v pulls from input voxel v + SHIFT


def _blob(shape):
    """A smooth, asymmetric blob, so a misplaced axis cannot go unnoticed."""
    idx = np.indices(shape).astype(np.float32)
    centre = np.array([0.4, 0.55, 0.6])[:, None, None, None] * np.array(shape)[:, None, None, None]
    return np.exp(-(((idx - centre) / np.array([14.0, 20.0, 9.0])[:, None, None, None]) ** 2).sum(0))


def _translation(gm, wm, templates, mat, tmat, inverse=True, **options):
    """Stands in for ``cat_surf.vol_shoot``: an exact integer translation."""
    tshape = templates[0].shape[:3]
    v = np.moveaxis(np.indices(tshape, dtype=np.float32), 0, -1)
    u = np.moveaxis(np.indices(gm.shape, dtype=np.float32), 0, -1)
    return {"y": v + SHIFT, "iy": u - SHIFT, "jac": np.ones(tshape, np.float32),
            "info": {"level_iterations": [0] * 5}}


class TestConventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gm = _blob(SHAPE)
        cls.wm = 0.5 * cls.gm[::-1]
        p1 = nib.Nifti1Image(cls.gm, TMAT)
        p2 = nib.Nifti1Image(cls.wm, TMAT)
        templates = ([np.zeros(SHAPE + (3,), np.float32)] * 5, TMAT)
        with mock.patch.object(_shoot, "shooting_templates", return_value=templates), \
                mock.patch.object(_shoot.cat_surf, "vol_shoot", _translation, create=True):
            cls.out = _shoot.shoot_register(p1, p2, 0.75)

    def _stored(self, name):
        img = self.out[name]
        np.testing.assert_allclose(img.affine, TMAT)
        return np.asarray(img.dataobj, dtype=np.float64)

    def test_warped_maps_pull_from_the_deformed_position(self):
        mwp1 = self._stored("mwp1")
        i, j, k = (slice(5, n - 5) for n in SHAPE)
        shifted = self.gm[7:SHAPE[0] - 3, 4:SHAPE[1] - 6, 8:SHAPE[2] - 2]
        np.testing.assert_allclose(mwp1[i, j, k], 0.75 * shifted, atol=1e-6)

    def test_modulated_maps_carry_the_volume_factor(self):
        wp2 = self._stored("wp2")       # uint8 storage: within one level of 255
        mwp2 = self._stored("mwp2")
        np.testing.assert_allclose(mwp2, 0.75 * wp2, atol=0.75 / 255 + 1e-6)

    def test_grids_are_normalised_with_reversed_channels(self):
        from deepmriprep.utils import nifti_to_tensor

        xy = nifti_to_tensor(self.out["warp_xy"]).numpy()
        yx = nifti_to_tensor(self.out["warp_yx"]).numpy()
        # RAS axis 0 runs against LAS storage: the shift reads -2, -1, +3
        step = np.array([-2.0, -1.0, 3.0]) * 2.0 / (np.array(SHAPE) - 1)
        centre = tuple(n // 2 for n in SHAPE)
        identity = np.array([2.0 * c / (n - 1) - 1.0 for c, n in zip(centre, SHAPE)])
        np.testing.assert_allclose(xy[centre], (identity + step)[::-1], atol=1e-6)
        np.testing.assert_allclose(yx[centre], (identity - step)[::-1], atol=1e-6)

    def test_modulation_does_not_depend_on_storage_order(self):
        """Maps stored RAS against CAT's LAS templates: the shooting's voxel
        Jacobian is negative there, the volume change it describes is not."""
        ras = TMAT.copy()
        ras[0, 0], ras[0, 3] = 1.5, -84.0
        p1 = nib.Nifti1Image(np.ascontiguousarray(self.gm[::-1]), ras)
        p2 = nib.Nifti1Image(np.ascontiguousarray(self.wm[::-1]), ras)

        def flipped(*args, **kwargs):
            out = _translation(*args, **kwargs)
            out["jac"] = -out["jac"]
            return out

        templates = ([np.zeros(SHAPE + (3,), np.float32)] * 5, TMAT)
        with mock.patch.object(_shoot, "shooting_templates", return_value=templates), \
                mock.patch.object(_shoot.cat_surf, "vol_shoot", flipped, create=True):
            out = _shoot.shoot_register(p1, p2, 0.75)
        np.testing.assert_allclose(np.asarray(out["jac"].dataobj), 1.0)
        self.assertGreaterEqual(float(np.asarray(out["mwp1"].dataobj).min()), 0.0)

    def test_inverse_is_finite_where_the_shooting_left_it_undefined(self):
        grid = np.zeros((4, 5, 6, 3), np.float32)
        grid[0] = np.nan
        filled = _shoot._fill_undefined(grid)
        self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(filled[1:], grid[1:])
        # the displacement of the nearest defined neighbour, here layer 1
        axes = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in grid.shape[:3]]
        identity = np.stack(np.meshgrid(*axes, indexing="ij")[::-1], axis=-1)
        np.testing.assert_allclose(filled[0] - identity[0], grid[1] - identity[1], atol=1e-6)


@unittest.skipUnless(hasattr(cat_surf, "vol_shoot"), "cat-surf without vol_shoot")
class TestShootingOnItsOwnTemplate(unittest.TestCase):
    def test_the_template_registers_onto_itself(self):
        templates, tmat = _shoot.shooting_templates()
        gm, wm = templates[-1][..., 0], templates[-1][..., 1]
        p1, p2 = nib.Nifti1Image(gm, tmat), nib.Nifti1Image(wm, tmat)
        out = _shoot.shoot_register(p1, p2, 1.0, regstr=0.5, n_threads=8)
        mwp1 = np.asarray(out["mwp1"].dataobj, dtype=np.float64)
        self.assertAlmostEqual(mwp1.sum() / gm.sum(), 1.0, delta=0.01)
        wp1 = np.asarray(out["wp1"].dataobj, dtype=np.float64)
        self.assertGreater(np.corrcoef(wp1.ravel(), gm.ravel())[0, 1], 0.99)


if __name__ == "__main__":
    unittest.main()
