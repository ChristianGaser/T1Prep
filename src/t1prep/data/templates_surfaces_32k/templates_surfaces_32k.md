# templates_surfaces_32k

This folder contains surface template files used by **T1Prep/CAT** for spherical registration, resampling, and for visualizing results in template space.

Notes:
- `?h` is a placeholder for hemisphere and typically expands to `lh` (left hemisphere) and `rh` (right hemisphere).
- The 32k meshes referenced below are based on Human Connectome Project (HCP) standard meshes and maintain vertex correspondence between left and right hemispheres.

## HCP-based 32k template surfaces

Source:
- https://github.com/Washington-University/Pipelines/tree/master/global/templates/standard_mesh_atlases

These surfaces are internally used for spherical registration and resampling, and can be used to overlay maps/results in template space (e.g., after spherical registration and resampling).

Files:
- `?h.central.freesurfer.gii`
- `?h.sphere.freesurfer.gii`
- `?h.inflated.freesurfer.gii`
- `?h.patch.freesurfer.gii`

Details:
- The **central** surface is estimated by averaging the white and pial surface.
- The **sphere** surface contains the transformation needed to get from the 32k HCP meshes to FreeSurfer 164k `fsaverage` and is based on:
  - `fs_LR-deformed_to-fsaverage.?.sphere.32k_fs_LR.surf.gii`
- The **cortex patch** is based on `colin.cerebral.?.flat.32k_fs_LR.surf.gii` and rotated to fit hemisphere views using:

```matlab
% lh:
spm_mesh_transform(g, spm_matrix([0 0 0 -pi/2 -pi/2 0]));

% rh:
spm_mesh_transform(g, spm_matrix([0 0 0 -pi/2 -pi/2 0 -1 1 1]));
```

## Display textures

Mean curvature and (sqrt) sulcal depth for display purposes (e.g., as underlying texture):

- `?h.mc.freesurfer.gii`
- `?h.sqrtsulc.freesurfer.gii`

## Mask

Mask based on the **DK40 Atlas** of FreeSurfer:

- `?h.mask`

The mask was created by masking out region `0` (*Unknown*). It is internally used to set values to `NaN` in these regions.

These `NaN` values are ignored during:
- smoothing (masked smoothing)
- statistical analysis

## Surfaces and thickness values (Template_T1)

Surfaces and thickness values based on the average of the **MNI152NLin2009cAsym** template (processed with `collcorr=24`).

These data should only be used to map results from 3D space (e.g., VBM or fMRI statistical results) to template space after spherical registration.

Files:
- `?h.central.Template_T1.gii`
- `?h.thickness.Template_T1`

## Merged hemispheres (fsaverage)

Merged left and right hemisphere surface based on FreeSurfer `fsaverage`:

- `mesh.central.freesurfer.gii`

## 2D index mapping (for visualization)

Index file containing the transformation from surface maps to 2D maps for visualization (used in `cat_stat_check_cov.m`):

- `fsavg.index2D_256x128.txt`
