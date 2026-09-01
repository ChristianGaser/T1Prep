# atlases_surfaces_32k

This folder contains surface atlas annotation files on the **HCP 32k** standard mesh.

Notes:
- HCP 32k meshes are generally on a standard mesh where left and right hemispheres are aligned.
- FreeSurfer data are based on the `fsaverage` mesh, which has no vertex-to-vertex correspondence between hemispheres.
- `?h` is a placeholder for hemisphere and typically expands to `lh` (left) and `rh` (right).

## Which space each atlas comes from

This mesh is the fs_LR 32k mesh: `templates_surfaces_32k/?h.sphere.freesurfer.gii`
is bit-identical to HCP's `fs_LR-deformed_to-fsaverage.?.sphere.32k_fs_LR.surf.gii`,
so vertex *i* here is vertex *i* of fs_LR.  Each atlas is therefore taken from
whichever space it was defined in:

- **DK40** and **Destrieux** are defined on fsaverage and are carried here from
  fsaverage, which is authoritative for them.
- **HCP-MMP1** and **Schaefer2018** are defined on fs_LR, and are taken from
  their own fs_LR 32k releases and dropped on by vertex index with no
  resampling.

Earlier versions shipped fsaverage projections of the latter two.  Those are
gone: the two renditions of the same atlas assign a different parcel to roughly
14 % of vertices (86.2 % agreement for Schaefer-400, 85.8 % for HCP-MMP1), a
difference that comes from the atlas authors projecting into each space
independently.  Results are therefore **not comparable with T1Prep output from
before this change**, and are now directly comparable with HCP and
CIFTI-pipeline results.

Using the fs_LR rendition also respects left/right correspondence better, which
only this mesh can express: for HCP-MMP1, 81.6 % of labelled vertices carry the
homologous area in both hemispheres, against 75.8 % for the fsaverage
rendition.  (The same figure is not meaningful for Schaefer, whose parcels are
not bilaterally homologous by construction.)

The fs_LR files were converted from the CIFTI `dlabel.nii` releases listed in
the per-atlas `.txt` files by scattering each hemisphere's labels onto the full
32492-vertex mesh -- HCP's dlabels omit the medial wall -- and re-indexing them
per hemisphere with index 0 as the medial wall.  No spatial interpolation is
involved.

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
- `?h.aparc_HCP_MMP1.annot`

Source (native fs_LR 32k):
- `Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii`

Website:
- https://balsa.wustl.edu/study/show/RVVG

Reference:
- Glasser MF, Coalson TS, Robinson EC, et al. *A multi-modal parcellation of human cerebral cortex.*
  Nature. 2016;536(7615):171-178.

## Schaefer 2018 parcellations

Pattern:
- `?h.Schaefer2018_*Parcels_17Networks_order.annot` (native fs_LR 32k)

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
