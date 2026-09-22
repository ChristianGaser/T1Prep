"""Tests for the dura removal in :mod:`t1prep.dura`.

The phantom is a concentric brain on a 0.5 mm grid -- white matter, a grey
matter ribbon, a layer of CSF -- wrapped in a shell of GM-bright dura that the
label calls CSF, which is what the skull-strip leaves behind in the scans this
step exists for.

As in :mod:`tests.test_nogm`, the negative controls carry more weight than
the detection: the CSF under the dura, and a cortex that reaches the dura with
no CSF in between, must both survive.
"""

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

# Allow running tests without installing the package (repo checkout / editable dev)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep import dura  # noqa: E402

VX = 0.5
N = 120
C = N // 2

WM, GM, CSF = 1.0, 0.67, 0.33


def _phantom(with_dura=True, csf_level=CSF, no_csf_side=False):
    """Concentric brain, optionally wrapped in dura.

    Args:
        with_dura: Add a 1.5 mm shell of GM-bright, CSF-labelled dura.
        csf_level: Intensity of the CSF.
        no_csf_side: Replace the CSF by cortex on one side, so the grey
            matter reaches the dura there with no gap.

    Returns:
        ``(brain, p0, regions)`` -- intensities, label, and a dict of boolean
        masks ``wm``, ``gm``, ``csf`` and ``dura``.
    """
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
    r = np.sqrt((zz - C) ** 2 + (yy - C) ** 2 + (xx - C) ** 2) * VX

    regions = {
        "wm": r < 20.0,
        "gm": (r >= 20.0) & (r < 23.0),
        "csf": (r >= 23.0) & (r < 25.5),
        "dura": (r >= 25.5) & (r < 27.0) if with_dura else np.zeros_like(r, bool),
    }
    if no_csf_side:
        side = regions["csf"] & (xx > C + 10 / VX)
        regions["gm"] |= side
        regions["csf"] &= ~side

    brain = np.zeros((N, N, N), np.float32)
    p0 = np.zeros((N, N, N), np.float32)
    for name, value, label in (
        ("wm", WM, 3.0),
        ("gm", GM, 2.0),
        ("csf", csf_level, 1.0),
        ("dura", 0.65, 1.0),
    ):
        brain[regions[name]] = value
        p0[regions[name]] = label
    return brain, p0, regions


class TestDuraMask(unittest.TestCase):
    def test_dura_outside_the_csf_is_removed(self):
        brain, p0, regions = _phantom()
        removed = dura.dura_mask(brain, p0, (VX, VX, VX))
        self.assertIsNotNone(removed)
        found = removed[regions["dura"]].mean()
        self.assertGreater(found, 0.9, f"only {found:.1%} of the dura removed")

    def test_csf_under_the_dura_is_kept(self):
        brain, p0, regions = _phantom()
        removed = dura.dura_mask(brain, p0, (VX, VX, VX))
        lost = removed[regions["csf"]].mean()
        self.assertLess(lost, 0.1, f"{lost:.1%} of the CSF removed")

    def test_brain_tissue_is_never_touched(self):
        brain, p0, regions = _phantom()
        removed = dura.dura_mask(brain, p0, (VX, VX, VX))
        self.assertEqual(int(removed[regions["gm"] | regions["wm"]].sum()), 0)

    def test_cortex_touching_the_dura_is_kept(self):
        """No CSF gap, no dura: the rule must not take the cortex."""
        brain, p0, regions = _phantom(no_csf_side=True)
        removed = dura.dura_mask(brain, p0, (VX, VX, VX))
        self.assertEqual(int(removed[regions["gm"] | regions["wm"]].sum()), 0)

    def test_nothing_removed_without_dura(self):
        brain, p0, _ = _phantom(with_dura=False)
        self.assertIsNone(dura.dura_mask(brain, p0, (VX, VX, VX)))

    def test_contrast_guard_stops_the_rule(self):
        """CSF 0.05 below GM: the thresholds cannot be trusted, skip."""
        brain, p0, _ = _phantom(csf_level=GM - 0.05)
        self.assertIsNone(dura.dura_mask(brain, p0, (VX, VX, VX)))

    def test_one_millimetre_grid(self):
        """At 1 mm there is no block averaging; the rule works the same."""
        brain, p0, regions = _phantom()
        brain, p0 = brain[::2, ::2, ::2], p0[::2, ::2, ::2]
        dura_1mm = regions["dura"][::2, ::2, ::2]
        removed = dura.dura_mask(brain, p0, (1.0, 1.0, 1.0))
        self.assertGreater(removed[dura_1mm].mean(), 0.9)
        self.assertEqual(int(removed[p0 >= 1.5].sum()), 0)


class TestRemoveDura(unittest.TestCase):
    def test_clears_image_and_label_but_not_the_inputs(self):
        brain, p0, regions = _phantom()
        affine = np.diag([VX, VX, VX, 1.0])
        brain_img = nib.Nifti1Image(brain, affine)
        p0_img = nib.Nifti1Image(p0, affine)

        brain_out, p0_out, removed = dura.remove_dura(brain_img, p0_img)

        self.assertIsNotNone(removed)
        self.assertTrue(np.all(np.asanyarray(brain_out.dataobj)[removed] == 0))
        self.assertTrue(np.all(np.asanyarray(p0_out.dataobj)[removed] == 0))
        # The caller's images are left alone.
        np.testing.assert_array_equal(np.asanyarray(brain_img.dataobj), brain)
        np.testing.assert_array_equal(np.asanyarray(p0_img.dataobj), p0)

    def test_returns_inputs_when_nothing_to_remove(self):
        brain, p0, _ = _phantom(with_dura=False)
        affine = np.diag([VX, VX, VX, 1.0])
        brain_img = nib.Nifti1Image(brain, affine)
        p0_img = nib.Nifti1Image(p0, affine)

        brain_out, p0_out, removed = dura.remove_dura(brain_img, p0_img)

        self.assertIsNone(removed)
        self.assertIs(brain_out, brain_img)
        self.assertIs(p0_out, p0_img)


if __name__ == "__main__":
    unittest.main()
