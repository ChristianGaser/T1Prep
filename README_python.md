# T1Prep – Python API Reference

This document covers the pure Python interface of **T1Prep**, including the
main pipeline, utility helpers, evaluation metrics, and the new
`t1prep.cat_surf` namespace that exposes all CAT-Surface operations without
requiring a separate `cat-surf` import.

---

## Table of Contents

- [Installation](#installation)
- [Package Layout](#package-layout)
- [Pipeline: `run_t1prep()`](#pipeline-run_t1prep)
- [CAT-Surface Interface: `t1prep.cat_surf`](#cat-surface-interface-t1prepcat_surf)
  - [I/O](#io)
  - [Surface Metrics](#surface-metrics)
  - [Surface Processing](#surface-processing)
  - [Surface Topology](#surface-topology)
  - [Volume Operations](#volume-operations)
  - [Registration](#registration)
  - [CLI Sub-module](#cli-sub-module)
- [Utilities: `t1prep.utils`](#utilities-t1preputils)
- [Metrics: `t1prep.metrics`](#metrics-t1prepmetrics)
- [Nipype Integration](#nipype-integration)

---

## Installation

```bash
# From PyPI (recommended for most users)
pip install T1Prep

# Editable install from a source checkout (for development)
pip install -e .

# Bash bootstrapper — fetches the source tree and sets up a managed venv
curl -fsSL https://raw.githubusercontent.com/ChristianGaser/T1Prep/main/scripts/install.sh | bash
```

After installation the following import works from any Python session:

```python
import t1prep
```

`cat-surf` is declared as a dependency of T1Prep and is installed
automatically.

---

## Package Layout

```
t1prep/
├── __init__.py          # top-level exports
├── t1prep.py            # run_t1prep() – pipeline entry point
├── segment.py           # volume pipeline (python -m t1prep.segment)
├── surface_estimation.py  # surface pipeline per hemisphere
├── stages/              # skull-strip, affine registration, segmentation
├── nn/                  # the networks: weights, architectures, backends
├── utils.py             # shared utilities
├── metrics.py           # Dice / evaluation metrics
├── cat_surf/
│   └── __init__.py      # re-exports all cat_surf symbols + cli sub-module
├── gui/                 # viewers (CAT_SurfView, CAT_VolView) and launcher (t1prep-gui)
├── launch/              # what the launcher knows that is not Qt
└── ...
```

---

## Pipeline: `run_t1prep()`

`run_t1prep()` is the main Python entry point.  It drives the pipeline
entirely from Python: segmentation runs in a child process via
`python -m t1prep.segment`, surface estimation runs in parallel child
processes via `python -m t1prep.surface_estimation`, and the two
hemispheres are joined in-process.  No bash interpreter is required.
Several files are processed one after another; for a parallel batch use the
bash orchestrator (`T1Prep --multi N`) or call `run_t1prep()` from several
processes.  The function returns the number of files that failed (`0` when
all succeeded).

```python
from t1prep import run_t1prep

# Single file, default options
run_t1prep("/data/sub-01_T1w.nii.gz")

# BIDS naming, custom output directory
run_t1prep(
    "/data/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz",
    out_dir="/results",
    bids=True,
)

# Several files (one after another), extra outputs
failed = run_t1prep(
    ["/data/sub-01_T1w.nii.gz", "/data/sub-02_T1w.nii.gz"],
    out_dir="/results",
    atlas=["neuromorphometrics", "suit"],
    atlas_surf=["aparc_DK40.freesurfer"],
    wp=True,             # write warped gray matter
    p=True,              # write native-space tissue maps
    csf=True,            # include CSF tissue map
    lesions=True,        # lesion detection
    gz=True,             # compress outputs
)
```

### Signature

```python
run_t1prep(
    inputs,                  # str | Path | list[str | Path]
    *,
    # output
    out_dir=None,            # str | Path; default: next to the input
    bids=False,              # BIDS derivatives names (implies gz)
    gz=False,                # .nii.gz instead of .nii
    # processing control
    no_surf=False,
    no_seg=False,
    no_sphere_reg=False,
    skullstrip_only=False,
    skip_skullstrip=False,
    pial_white=True,         # estimate the pial and white surfaces as well
    lesions=False,
    amap=False,              # refine the network segmentation with AMAP
    fmriprep=False,          # fMRIPrep-compatible results
    save_h5=False,           # deformations also as ANTs/ITK .h5
    fast=False,
    # tissue maps
    no_mwp=False,            # suppress modulated warped tissue maps
    wp=False,                # warped tissue maps (unmodulated)
    rp=False,                # affine-registered tissue maps
    p=False,                 # native-space tissue maps
    csf=False,               # include CSF in the p and rp maps
    hemisphere=False,        # keep the hemispheric label maps
    # atlas labeling
    atlas=None,              # str | list[str]; default 'neuromorphometrics', 'cobra'
    atlas_surf=None,         # str | list[str]; default DK40 and Destrieux
    no_atlas=False,          # skip the volume atlases (ROI volumes)
    no_atlas_surf=False,     # skip the surface atlases
    # tuning
    seed=0,                  # int (reproducibility)
    pre_fwhm=2.0,            # float (mm), smoothing before Marching Cubes
    median_filter=1,         # int, number of median filter passes
    vessel=1.0,              # float, vessel correction (0 switches it off)
    thickness_method=3,      # 1 = Tfs, 2 = pial/white, 3 = PBT
    no_correct_folding=False,
    # longitudinal
    initial_surf=None,       # str | Path
    long_data=None,          # str | Path
    # robustness
    no_overwrite=None,       # str: skip a subject whose results match this pattern
    retry=False,             # retry a failed step once
    no_retry=False,          # kept for compatibility; retry is off by default
    # misc
    debug=False,
    verbose=True,
    track=True,              # report a crash (no paths); False switches it off
    nn_backend=None,         # "torch" | "tinygrad"; default T1PREP_NN_BACKEND, else torch
) -> int                     # number of files that failed (0 = all succeeded)
```

---

## CAT-Surface Interface: `t1prep.cat_surf`

All functions from the [`cat-surf`](https://pypi.org/project/cat-surf/) package
are re-exported under the `t1prep.cat_surf` namespace. Nipype interfaces (or
any downstream code) therefore only need `t1prep` as a dependency:

```python
import t1prep.cat_surf as cs

# or selective imports
from t1prep.cat_surf import read_surface, write_surface, vol_thickness_pbt
```

### I/O

| Function | Description |
|----------|-------------|
| `read_surface(path)` | Load a surface mesh; returns `(vertices, faces)` |
| `write_surface(path, vertices, faces)` | Save a surface mesh |
| `read_values(path)` | Load per-vertex scalar values |
| `write_values(path, values)` | Save per-vertex scalar values |

```python
import t1prep.cat_surf as cs

v, fcs = cs.read_surface("lh.central.sub-01.gii")
thickness = cs.read_values("lh.thickness.sub-01")

cs.write_surface("lh.pial.sub-01.gii", v, fcs)
cs.write_values("lh.corrected_thickness.sub-01", thickness)
```

### Surface Metrics

| Function | Description |
|----------|-------------|
| `get_area(v, fcs)` | Per-vertex area and the total surface area |
| `get_area_normalized(v, fcs, sphere_v, sphere_fcs)` | Per-vertex area, normalized against a reference sphere |
| `euler_characteristic(v, fcs)` | Euler number |
| `point_distance(v1, fcs1, v2, fcs2)` | Per-vertex distance between two meshes, and its maximum |
| `point_distance_mean(v1, fcs1, v2, fcs2)` | Mean (FreeSurfer Tfs) per-vertex distance between two meshes |
| `hausdorff_distance(v1, fcs1, v2, fcs2)` | Hausdorff distance between two meshes |
| `sphere_radius(v, fcs)` | Average radius of a spherical mesh |
| `sulcus_depth(v, fcs)` | Sulcus depth map |
| `smoothed_curvatures(v, fcs)` | Smoothed mean curvature |
| `surf_curvature(v, fcs)` | Per-vertex curvature (`curvtype` selects the measure) |
| `count_intersections(v, fcs)` | Number of self-intersecting triangle pairs |

### Surface Processing

| Function | Description |
|----------|-------------|
| `smooth_heatkernel(v, fcs, values, fwhm)` | Heat-kernel surface smoothing |
| `smooth_mesh(v, fcs, iterations)` | Laplacian mesh smoothing |
| `reduce_mesh(v, fcs, target_faces)` | Mesh decimation to a number of faces |
| `fix_self_intersect(v, fcs)` | Remove self-intersections |
| `surf_average(surfaces)` | Average across a list of meshes |
| `surf_deform(v, fcs, volume, ...)` | Deform a surface toward an isovalue of a volume |
| `resample_to_sphere(v, fcs, sphere_v, sphere_fcs, target_sphere_v, target_sphere_fcs)` | Resample a mesh (and values) from its sphere to a target sphere |
| `resample_annot(source_surface_file, source_sphere_file, target_sphere_file, annot_in_file, annot_out_file)` | Resample an atlas annotation (file based) |

```python
import t1prep.cat_surf as cs

v, fcs = cs.read_surface("lh.central.sub-01.gii")
thickness = cs.read_values("lh.thickness.sub-01")

# Smooth thickness values with 20 mm FWHM
smoothed = cs.smooth_heatkernel(v, fcs, thickness, fwhm=20.0)
cs.write_values("lh.thickness_s20.sub-01", smoothed)
```

### Surface Topology

| Function | Description |
|----------|-------------|
| `surf_to_sphere(v, fcs)` | Map surface to unit sphere |
| `surf_to_pial_white(central_v, fcs, thickness, label)` | Derive pial/white from central surface, thickness and the label volume |
| `central_to_pial(v, fcs, thickness)` | Expand central surface to pial |
| `correct_thickness_folding(v, fcs, thickness)` | Correct thickness for folding artefacts |

### Volume Operations

| Function | Description |
|----------|-------------|
| `vol_sanlm(volume)` | Non-local-means denoising |
| `vol_blood_vessel_correction(label, voxelsize)` | Remove blood-vessel label artefacts |
| `vol_thickness_pbt(label, voxelsize)` | PBT cortical thickness: thickness map, percentage position map and the two distance maps |
| `vol_amap(volume, labels)` | AMAP tissue segmentation, started from a label volume |
| `vol_marching_cubes(volume, threshold)` | Marching-cubes surface extraction at a threshold of the volume |
| `vol2surf(volume, v, fcs, ...)` | Sample volume values onto surface |

```python
import t1prep.cat_surf as cs
import nibabel as nib

# A hemisphere label map, as T1Prep keeps it with --hemi
label_img = nib.load("lh.seg.sub-01.nii")
label = label_img.get_fdata()
vx = label_img.header.get_zooms()[:3]

# Cortical thickness (GMT) and percentage position map (PPM) via PBT
gmt, ppm, dcsf, dwm = cs.vol_thickness_pbt(label, voxelsize=vx)
nib.save(nib.Nifti1Image(gmt, label_img.affine), "lh.thickness.sub-01.nii")

# Central surface: the 0.5 level of the percentage position map
ppm_img = nib.Nifti1Image(ppm, label_img.affine)
v, fcs = cs.vol_marching_cubes(ppm_img, threshold=0.5)
cs.write_surface("lh.central.sub-01.gii", v, fcs)
```

### Registration

| Function | Description |
|----------|-------------|
| `bbreg(volume_file, lh_surface=..., rh_surface=..., ref_file=...)` | Boundary-based registration of a volume to the cortical surfaces |
| `bbreg_detect_contrast(volume_file, lh_surface=..., rh_surface=...)` | Auto-detect image contrast for BBR |
| `volume_register_nmi(fixed_file, moving_file)` | Rigid registration by normalised mutual information |
| `volume_register_robust(fixed_file, moving_file)` | Rigid same-modality registration, robust to outliers |

These four take file names and return the 4x4 transform from the moving to
the fixed image.

### CLI Sub-module

`t1prep.cat_surf.cli` provides higher-level Python wrappers (file-path based)
that mirror the original CAT-Surface command-line tools:

```python
from t1prep.cat_surf import cli

# Resample surface to sphere
cli.surf2sphere("lh.central.sub-01.gii", "lh.sphere.sub-01.gii")

# Reduce mesh
cli.surf_reduce("lh.central.sub-01.gii", "lh.reduced.gii", ratio=0.25)

# Resample thickness values to the template sphere
cli.surf_resample(
    "lh.central.sub-01.gii",            # the surface
    "lh.sphere.reg.sub-01.gii",         # its registered sphere
    "lh.sphere.freesurfer.gii",         # target sphere (data/templates_surfaces_32k)
    None,                               # no resampled surface wanted
    input_values_file="lh.thickness.sub-01",
    output_values_file="lh.thickness.resampled.sub-01",
)

# Compute surface curvature
cli.surf_curvature("lh.central.sub-01.gii", "lh.mc.sub-01")

# Map volume values onto surface
cli.vol2surf(
    "lh.central.sub-01.gii",            # surface
    "sub-01_T1w.nii.gz",                # volume
    "lh.T1.sub-01",                     # output values
)
```

Full list of `cli` functions:

| Function | Equivalent CLI tool |
|----------|---------------------|
| `surf2pial_white` | `CAT_Surf2PialWhite` |
| `surf2roi_multi` | `CAT_Surf2ROIMulti` |
| `surf2sphere` | `CAT_Surf2Sphere` |
| `surf_area` | `CAT_SurfArea` |
| `surf_average` | `CAT_SurfAverage` |
| `surf_correct_thickness_folding` | `CAT_SurfCorrectThicknessFolding` |
| `surf_curvature` | `CAT_SurfCurvature` |
| `surf_deform` | `CAT_SurfDeform` |
| `surf_distance` | `CAT_SurfDistance` |
| `surf_fix_self_intersect` | `CAT_SurfFixSelfIntersect` |
| `surf_fractal_dimension` | `CAT_SurfFractalDimension` |
| `surf_info` | `CAT_SurfInfo` |
| `surf_ratio` | `CAT_SurfRatio` |
| `surf_reduce` | `CAT_SurfReduce` |
| `surf_resample` | `CAT_SurfResample` |
| `surf_resample_annot` | `CAT_SurfResample -label` |
| `surf_resample_multi` | `CAT_SurfResampleMulti` |
| `surf_spherical_demon` | `CAT_SurfSphericalDemon` |
| `surf_sulcus_depth` | `CAT_SurfSulcusDepth` |
| `vol2surf` | `CAT_Vol2Surf` |
| `vol_amap` | `CAT_VolAmap` |
| `vol_calc` | `CAT_VolCalc` |
| `vol_local_stat` | `CAT_VolLocalStat` |
| `vol_marching_cubes` | `CAT_VolMarchingCubes` |
| `vol_realign` | `CAT_VolRealign` |
| `vol_sanlm` | `CAT_VolSanlm` |
| `vol_sheetness` | `CAT_VolSheetness` |
| `vol_shoot` | `CAT_VolShoot` |
| `vol_smooth` | `CAT_VolSmooth` |
| `vol_thickness_pbt` | `CAT_VolThicknessPbt` |

The names in the right column are those of the CAT-Surface command-line
tools; T1Prep installs none of them, the functions are their equivalents.

---

## Utilities: `t1prep.utils`

```python
from t1prep import (
    progress_bar,
    remove_file,
    resample_and_save_nifti,
    get_resampled_header,
    align_brain,
    get_filenames,
    get_volume_native_space,
)
```

| Function | Description |
|----------|-------------|
| `get_filenames(use_bids_naming, bname, side, desc, space, nii_ext)` | The output file names of an image by their code in `Names.tsv`, in the CAT12 or the BIDS convention |
| `resample_and_save_nifti(img, grid, affine, header, out_path, ...)` | Resample a NIfTI onto a sampling grid and save. `bspline=True` swaps the default trilinear kernel for a quadratic B-spline (sharper edges, ~10x slower); `clip_overshoot=True` clamps the result to the input range to remove spline ringing |
| `get_resampled_header(header, aff, new_vox_size)` | Header and affine for a new voxel size |
| `align_brain(data, aff, header, aff_ref)` | Reorient data to the orientation of a reference affine |
| `get_volume_native_space(vol_nifti, wj_affine)` | Native-space volume of a registered probability map |
| `remove_file(name)` | Delete a file if it exists |
| `progress_bar(elapsed, total, name)` | Print a progress bar with percentage and ETA |

```python
from t1prep import get_filenames

# BIDS names of the files of sub-01_T1w, compressed
names = get_filenames(True, "sub-01_T1w", "left", "", "", "nii.gz")
print(names["Label_volume"])   # sub-01_label-seg.nii.gz
print(names["mask_volume"])    # sub-01_desc-brain_mask.nii.gz
```

---

## Metrics: `t1prep.metrics`

```python
from t1prep import compute_dice_nifti

confusion, labels, dice_per_label, dice_weighted, generalized_dice = \
    compute_dice_nifti("gt_labels.nii.gz", "pred_labels.nii.gz")

print("Per-label Dice:", dice_per_label)
print("Weighted Dice: ", dice_weighted)
print("Generalized Dice:", generalized_dice)
```

### `compute_dice_nifti(gt, pred, *, round_labels=True, resample=True, affine_atol=1e-3)`

| Parameter | Description |
|-----------|-------------|
| `gt` | Ground-truth label NIfTI (path or `nib.Nifti1Image`); its voxel grid is the reference space |
| `pred` | Predicted label NIfTI (path or `nib.Nifti1Image`) |
| `round_labels` | When `True` (default), round to integer labels before scoring. Set `False` for soft/continuous Dice. |
| `resample` | When `True` (default), resample `pred` onto the grid of `gt` whenever the world-space geometries differ (shape, voxel size, orientation, rotation or translation in the NIfTI affine). Nearest neighbour for label maps, trilinear for soft maps. Set `False` to compare voxel-to-voxel, ignoring the affines. |
| `affine_atol` | Absolute tolerance (mm) below which two affines count as the same grid. |

The NIfTI affine (`sform`/`qform`, whichever nibabel selects) is honoured, so
images that share a shape but carry different rotations — or that differ in shape
while covering the same world space — are aligned before scoring instead of being
compared voxel-by-voxel. Resampling is reported through the
`t1prep.metrics` logger at `WARNING` level.

Returns a 5-tuple:

| Return value | Description |
|--------------|-------------|
| `confusion` | K×K confusion matrix |
| `labels_order` | Label values for rows/cols |
| `dice_per_label` | Per-class Dice scores |
| `dice_weighted` | Volume-weighted Dice |
| `generalized_dice` | Generalized Dice coefficient |

---

## Nipype Integration

Because all CAT-Surface functions are available under `t1prep.cat_surf`,
Nipype interfaces only need to declare `t1prep` as a dependency — no separate
`cat-surf` import is required.

A minimal Nipype `Interface` example:

```python
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits,
)
import t1prep.cat_surf as cs


class SurfReduceInputSpec(BaseInterfaceInputSpec):
    in_file  = File(mandatory=True, exists=True, desc="Input surface (.gii)")
    out_file = File(desc="Output surface (.gii)")
    ratio    = traits.Float(0.25, usedefault=True, desc="Reduction ratio")
    aggr     = traits.Int(7,    usedefault=True, desc="Aggressiveness")


class SurfReduceOutputSpec(TraitedSpec):
    out_file = File(desc="Reduced surface (.gii)")


class SurfReduce(BaseInterface):
    """Reduce surface mesh resolution using CAT-Surface via t1prep.cat_surf."""

    input_spec  = SurfReduceInputSpec
    output_spec = SurfReduceOutputSpec

    def _run_interface(self, runtime):
        import os
        out = self.inputs.out_file or self.inputs.in_file.replace(".gii", "_reduced.gii")

        v, fcs = cs.read_surface(self.inputs.in_file)
        v, fcs = cs.reduce_mesh(v, fcs,
                                 target_faces=int(len(fcs) * self.inputs.ratio),
                                 aggressiveness=self.inputs.aggr)
        cs.write_surface(out, v, fcs)

        self._out_file = out
        return runtime

    def _list_outputs(self):
        return {"out_file": self._out_file}
```

For pipeline-level processing you can combine `run_t1prep()` with
`t1prep.cat_surf` post-processing in the same workflow:

```python
from t1prep import run_t1prep
import t1prep.cat_surf as cs

# 1. Run the full T1Prep pipeline
run_t1prep(
    "/data/sub-01_T1w.nii.gz",
    out_dir="/results",
)

# 2. Post-process surfaces directly with t1prep.cat_surf
v, fcs   = cs.read_surface("/results/surf/lh.central.sub-01.gii")
thickness = cs.read_values("/results/surf/lh.thickness.sub-01")
smoothed  = cs.smooth_heatkernel(v, fcs, thickness, fwhm=20.0)
cs.write_values("/results/surf/lh.thickness_s20.sub-01", smoothed)
```
