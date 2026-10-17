"""deepmriprep's affine registration resamples through libCAT.

``CustomPreprocess.run_affine_register`` swaps the ``spline_resize`` module
``deepmriprep.preprocess`` calls through for one whose ``grid_sample`` is
:func:`t1prep.utils.bspline_grid_sample`, for the duration of the call only.
"""

import sys as _sys
import unittest
from pathlib import Path as _Path
from unittest import mock

# Allow running tests without installing the package (repo checkout / editable dev)
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import deepmriprep.preprocess as dmp_preprocess  # noqa: E402
import spline_resize  # noqa: E402

from t1prep import segment, utils  # noqa: E402


class TestSplineResizeViaLibcat(unittest.TestCase):
    def test_only_grid_sample_is_swapped_and_only_while_active(self):
        original = dmp_preprocess.sr
        with segment._spline_resize_via_libcat():
            self.assertIs(dmp_preprocess.sr.grid_sample, utils.bspline_grid_sample)
            # everything else still is spline_resize's
            self.assertIs(dmp_preprocess.sr.resize, spline_resize.resize)
        self.assertIs(dmp_preprocess.sr, original)
        # the module itself was never touched
        self.assertIsNot(spline_resize.grid_sample, utils.bspline_grid_sample)

    def test_restored_after_an_exception(self):
        original = dmp_preprocess.sr
        with self.assertRaises(RuntimeError):
            with segment._spline_resize_via_libcat():
                raise RuntimeError("upstream failed")
        self.assertIs(dmp_preprocess.sr, original)

    def test_run_affine_register_goes_through_libcat(self):
        seen = {}

        def upstream(self, brain_nifti, mask_nifti):
            seen["grid_sample"] = dmp_preprocess.sr.grid_sample
            return {"brain": brain_nifti, "mask": mask_nifti}

        # an instance without __init__: no models are loaded
        prep = segment.CustomPreprocess.__new__(segment.CustomPreprocess)
        with mock.patch.object(dmp_preprocess.Preprocess, "run_affine_register", upstream):
            out = prep.run_affine_register("b", "m")
        self.assertEqual(out, {"brain": "b", "mask": "m"})
        self.assertIs(seen["grid_sample"], utils.bspline_grid_sample)
        self.assertIs(dmp_preprocess.sr, spline_resize)


if __name__ == "__main__":
    unittest.main()
