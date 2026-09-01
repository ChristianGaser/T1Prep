"""The native-space outputs are not necessarily RAS.

``--fmriprep`` builds the whole-head ``desc-preproc_T1w`` by pairing the input
with the native label map, which only lines up once both are in the same array
order.  The result then has to be written back into the orientation the rest of
the native outputs use.  Stamping a RAS-ordered array with a non-RAS affine
instead flips it against its own affine — silently, because the shape is
unchanged — which is what put white matter in the background on an LPI dataset.
"""

import nibabel as nib
import numpy as np
import pytest
from deepbet.utils import reoriented_nifti


def _affine(axcodes, shape, zooms=(1.0, 1.2, 0.9)):
    """A diagonal affine with the given orientation, centred on the volume."""
    ras = nib.orientations.axcodes2ornt(axcodes)
    base = np.diag(list(zooms) + [1.0])
    base[:3, 3] = -np.array(shape) * np.array(zooms) / 2.0
    transform = nib.orientations.inv_ornt_aff(ras, shape)
    return base @ transform


SHAPE = (6, 7, 8)
ORIENTATIONS = ["RAS", "LPI", "LAS", "PSR", "RPI", "ASL"]
#: Orientations that only flip axes, leaving the shape untouched.  These are
#: the dangerous ones -- a permuting orientation changes the shape and so trips
#: any shape check, while a flip corrupts the image silently.  The dataset that
#: exposed this was LPI.
FLIP_ONLY = ["LPI", "LAS", "RPI"]


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.random(SHAPE).astype(np.float32)


@pytest.mark.parametrize("axcodes", ORIENTATIONS)
def test_reoriented_nifti_round_trips(volume, axcodes):
    """A canonical array written this way reads back canonical and unchanged."""
    affine = _affine(tuple(axcodes), SHAPE)
    image = reoriented_nifti(volume, affine, None)

    back = nib.as_closest_canonical(image)
    np.testing.assert_allclose(back.get_fdata(dtype=np.float32), volume, atol=0)


@pytest.mark.parametrize("axcodes", FLIP_ONLY)
def test_stamping_the_affine_directly_corrupts_the_image(volume, axcodes):
    """The bug this guards against: same shape, wrong contents."""
    affine = _affine(tuple(axcodes), SHAPE)
    naive = nib.Nifti1Image(volume, affine)

    back = nib.as_closest_canonical(naive).get_fdata(dtype=np.float32)
    assert back.shape == volume.shape, "the shape alone never reveals the flip"
    assert not np.allclose(back, volume), (
        "a non-RAS affine must permute or flip the array; if this passes the "
        "orientation fixture is not exercising anything"
    )


@pytest.mark.parametrize("axcodes", ORIENTATIONS)
def test_label_and_image_agree_after_canonicalising(volume, axcodes):
    """Both operands must be canonicalised before they can be paired.

    Mirrors the pairing in ``save_results``: the label is stored in the native
    orientation, the input may be stored in another, and the bias fit reads
    them as plain arrays.
    """
    affine = _affine(tuple(axcodes), SHAPE)
    label = np.zeros(SHAPE, dtype=np.float32)
    label[2:4, 3:5, 4:6] = 3.0                      # a "white matter" block

    stored_label = reoriented_nifti(label, affine, None)
    stored_image = reoriented_nifti(volume, _affine(("R", "A", "S"), SHAPE), None)

    canonical_label = nib.as_closest_canonical(stored_label).get_fdata(dtype=np.float32)
    canonical_image = nib.as_closest_canonical(stored_image).get_fdata(dtype=np.float32)

    assert canonical_label.shape == canonical_image.shape
    np.testing.assert_allclose(canonical_label, label, atol=0)
    np.testing.assert_allclose(canonical_image, volume, atol=0)
