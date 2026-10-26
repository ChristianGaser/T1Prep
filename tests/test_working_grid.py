"""The 0.5 mm working grid carries the geometry of its content.

deepmriprep's ``Template_05mm_bet`` is the 1.5 mm warp template upsampled
three times with aligned corner voxel centres, but its header claims 0.5 mm
voxels: 169x205x169 mm between the corner centres instead of the warp
template's 168x204x168.  Registration and warp work in normalised
coordinates and never noticed; everything that read millimetres from the
working grid (atlas and prior placement, voxel sizes for filters and size
thresholds) was 0.6% off, up to 1 mm at the far edges.
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

import deepmriprep.preprocess as dmp_preprocess  # noqa: E402
from scipy.ndimage import binary_fill_holes, map_coordinates  # noqa: E402

from t1prep import segment  # noqa: E402
from t1prep.utils import centre_extent_volume  # noqa: E402


def _corners(img):
    """World coordinates of the first and the last voxel centre, RAS."""
    img = nib.as_closest_canonical(img)
    last = np.asarray(img.shape[:3]) - 1.0
    return img.affine[:3, 3], img.affine[:3, :3] @ last + img.affine[:3, 3]


class TestWorkingGridTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = dmp_preprocess.AFFINE_TEMPLATE
        cls.warp = dmp_preprocess.WARP_TEMPLATE
        cls.grid = segment._working_grid_template(cls.header, cls.warp)

    def test_corner_centres_are_the_warp_templates(self):
        for mine, theirs in zip(_corners(self.grid), _corners(self.warp)):
            np.testing.assert_allclose(mine, theirs, atol=1e-9)
        np.testing.assert_allclose(
            np.sqrt((self.grid.affine[:3, :3] ** 2).sum(axis=0)),
            [168 / 338, 204 / 410, 168 / 338],
        )

    def test_header_follows_the_affine_and_the_data_is_shared(self):
        np.testing.assert_allclose(
            self.grid.header.get_zooms()[:3], [168 / 338, 204 / 410, 168 / 338], rtol=1e-6
        )
        self.assertEqual(self.grid.shape, self.header.shape)
        self.assertIs(self.grid.dataobj, self.header.dataobj)
        # deepmriprep's module-level template is left as it was
        np.testing.assert_allclose(self.header.header.get_zooms()[:3], [0.5, 0.5, 0.5])

    def test_one_volume_factor_serves_both_grids(self):
        self.assertAlmostEqual(
            centre_extent_volume(self.grid) / centre_extent_volume(self.warp), 1.0, places=12
        )

    def test_already_consistent_geometry_is_kept(self):
        again = segment._working_grid_template(self.grid, self.warp)
        np.testing.assert_allclose(again.affine, self.grid.affine, atol=1e-12)

    def test_the_content_sits_where_the_new_geometry_says(self):
        """The template's brain meets the warp template's better than before."""
        data = np.asanyarray(self.header.dataobj) > 0
        warp = nib.as_closest_canonical(self.warp)
        prob = np.asanyarray(warp.dataobj, dtype=np.float32)
        reference = binary_fill_holes(prob[..., 0] + prob[..., 1] > 0.5)
        ijk = np.indices(reference.shape).reshape(3, -1).astype(np.float64)
        world = warp.affine[:3, :3] @ ijk + warp.affine[:3, 3:4]

        def dice(affine):
            inv = np.linalg.inv(affine)
            vox = inv[:3, :3] @ world + inv[:3, 3:4]
            mask = map_coordinates(data.astype(np.float32), vox, order=1) > 0.5
            mask = mask.reshape(reference.shape)
            return 2 * (mask & reference).sum() / (mask.sum() + reference.sum())

        # Measured: 0.9404 against 0.9352 through the 0.5 mm header
        self.assertGreater(dice(self.grid.affine), dice(self.header.affine) + 0.003)


class TestCustomPreprocessGrid(unittest.TestCase):
    def _template_passed(self, *args, **kwargs):
        with mock.patch.object(segment.Preprocess, "__init__", return_value=None) as init:
            segment.CustomPreprocess(*args, **kwargs)
        call = init.call_args
        if len(call.args) > 1:
            return call.args[1]
        return call.kwargs.get("affine_template")

    def test_the_working_grid_replaces_the_header_geometry(self):
        template = self._template_passed(no_gpu=True)
        np.testing.assert_allclose(
            template.header.get_zooms()[:3], [168 / 338, 204 / 410, 168 / 338], rtol=1e-6
        )

    def test_an_explicit_template_is_used_as_given(self):
        mine = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
        self.assertIs(self._template_passed(no_gpu=True, affine_template=mine), mine)
        self.assertIs(self._template_passed(True, mine), mine)


if __name__ == "__main__":
    unittest.main()
