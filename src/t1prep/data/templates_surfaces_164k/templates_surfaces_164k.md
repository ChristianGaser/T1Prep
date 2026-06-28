# templates_surfaces_164k

This folder contains surface template files used by **T1Prep/CAT** for spherical registration, resampling, and for visualizing results in template space.

Notes:
- `?h` is a placeholder for hemisphere and typically expands to `lh` (left hemisphere) and `rh` (right hemisphere).
- The 164k meshes referenced below are derived from FreeSurfer's `fsaverage` average subject.

## FreeSurfer 164k template surfaces

Source:
- FreeSurfer `fsaverage` subject, distributed with FreeSurfer by the Laboratory for Computational Neuroimaging: https://surfer.nmr.mgh.harvard.edu/

These surfaces are used internally for spherical registration and resampling, and can be used to overlay maps/results in template space (e.g., after spherical registration and resampling).

Files:
- `?h.central.freesurfer.gii`
- `?h.sphere.freesurfer.gii`

Details:
- The **central** surface is estimated by averaging the white and pial surfaces.
- The **sphere** surface is the corresponding 164k `fsaverage` spherical registration mesh.

## Mask

Mask based on the **DK40 Atlas** of FreeSurfer:

- `?h.mask`

The mask was created by masking out region `0` (*Unknown*). It is used internally to set values to `NaN` in these regions.

These `NaN` values are ignored during:
- smoothing (masked smoothing)
- statistical analysis

## Standard deviation of mean curvature

This map holds the standard deviation of mean curvature, estimated from 120 resampled mean-curvature maps of children and elderly subjects (age 5–85 years). It can be used to weight the spherical registration:

- `?h.std_curv`
