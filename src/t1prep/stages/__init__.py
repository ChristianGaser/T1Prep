"""The image stages T1Prep took over from deepmriprep.

deepmriprep (Fisch et al., MIT licence) runs a subject through ``Preprocess``:
skull-strip, affine registration onto a 0.5 mm working grid, brain
segmentation, non-linear registration and atlas placement, each a step from
NIfTI images to NIfTI images.  T1Prep kept that shape and replaced most of
the contents; these are T1Prep's own versions, independent of deepmriprep:

``preprocess``
    :class:`Preprocess`, the stages as one object, as ``segment.py`` drives
    them: skull-strip, affine registration (torchreg, B-spline resampling in
    libCAT), segmentation, geodesic shooting, atlases and ROI volumes.
``skullstrip``
    deepbet's skull-stripping on T1Prep's networks.
``segmentation``
    deepmriprep's brain segmentation on T1Prep's networks.

The networks themselves -- weights, architectures, backends -- are
:mod:`t1prep.nn`; the stages hand them torch tensors.
"""

from .preprocess import Preprocess

__all__ = ["Preprocess"]
