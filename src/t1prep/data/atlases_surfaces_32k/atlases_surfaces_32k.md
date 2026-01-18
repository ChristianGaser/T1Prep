# atlases_surfaces_32k

This folder contains surface atlas annotation files on the **HCP 32k** standard mesh.

Notes:
- HCP 32k meshes are generally on a standard mesh where left and right hemispheres are aligned.
- FreeSurfer data are based on the `fsaverage` mesh, which has no vertex-to-vertex correspondence between hemispheres.
- `?h` is a placeholder for hemisphere and typically expands to `lh` (left) and `rh` (right).

## Desikan–Killiany atlas (DK40)

File:
- `?h.aparc_DK40.freesurfer.annot`

Website:
- https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation

Reference:
- Desikan RS, Ségonne F, Fischl B, Quinn BT, Dickerson BC, Blacker D, Buckner RL, Dale AM,
  Maguire RP, Hyman BT, Albert MS, Killiany RJ. *An automated labeling system for subdividing the
  human cerebral cortex on MRI scans into gyral based regions of interest.* Neuroimage.
  2006 Jul 1;31(3):968-80.

## Destrieux atlas (aparc.a2009s)

File:
- `?h.aparc_a2009s.freesurfer.annot`

Websites:
- https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
- https://surfer.nmr.mgh.harvard.edu/fswiki/DestrieuxAtlasChanges

Reference:
- Destrieux C, Fischl B, Dale A, Halgren E. *A sulcal depth-based anatomical parcellation of the
  cerebral cortex.* Human Brain Mapping (HBM) Congress 2009, Poster #541.

## HCP Multi-Modal Parcellation (HCP-MMP1.0)

File:
- `?h.aparc_HCP_MMP1.freesurfer.annot`

Conversion source (to FreeSurfer fsaverage):
- https://figshare.com/articles/HCP-MMP1_0_projected_on_fsaverage/3498446/2

Website:
- https://balsa.wustl.edu/study/show/RVVG

Reference:
- Glasser MF, Coalson TS, Robinson EC, et al. *A multi-modal parcellation of human cerebral cortex.*
  Nature. 2016;536(7615):171-178.

## Schaefer 2018 parcellations

Pattern:
- `?h.Schaefer2018_*Parcels_*Networks_order.annot`

Description:
- Local-Global Intrinsic Functional Connectivity Parcellation by Schaefer et al.
- Available at different numbers of parcels (e.g., 100, 200, 400, 600)
- Based on resting-state data from 1489 subjects

Website:
- https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal

Reference:
- Schaefer A, Kong R, Gordon EM, Zuo XN, Holmes AJ, Eickhoff SB, Yeo BT.
  *Local-Global Parcellation of the Human Cerebral Cortex From Intrinsic Functional Connectivity MRI.*
  Cerebral Cortex.
