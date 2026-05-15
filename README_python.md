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

Install T1Prep in editable (development) mode or as a regular package:

```bash
# editable install from the repository root
pip install -e .

# or via the quick-install script
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
├── segment.py           # segmentation logic
├── surface_estimation.py
├── utils.py             # shared utilities
├── metrics.py           # Dice / evaluation metrics
├── cat_surf/
│   └── __init__.py      # re-exports all cat_surf symbols + cli sub-module
└── ...
```

---

## Pipeline: `run_t1prep()`

`run_t1prep()` is the main Python entry point. It calls the `scripts/T1Prep`
bash pipeline as a subprocess with full CLI parity and returns the process
exit code.

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

# Batch: multiple files, parallel workers, extra outputs
run_t1prep(
    ["/data/sub-01_T1w.nii.gz", "/data/sub-02_T1w.nii.gz"],
    out_dir="/results",
    multi=-1,            # auto-detect CPU count
    atlas=["neuromorphometrics", "suit"],
    atlas_surf=["aparc_DK40.freesurfer"],
    wp=True,             # write warped gray matter
    p=True,              # write native-space tissue maps
    csf=True,            # include CSF tissue map
    lesions=True,        # lesion detection
    gz=True,             # compress outputs
    log_file="/results/run.log",
    stream_output=True,
)
```

### Signature

```python
run_t1prep(
    inputs,              # str | Path | list[str | Path]
    *,
    # output
    out_dir=None,        # str | Path
    bids=False,
    gz=False,
    # processing control
    no_surf=False,
    no_seg=False,
    no_sphere_reg=False,
    skullstrip_only=False,
    skip_skullstrip=False,
    pial_white=False,
    lesions=False,
    amap=False,
    fast=False,
    # tissue maps
    no_mwp=False,        # suppress modulated warped tissue maps
    wp=False,            # warped tissue maps (unmodulated)
    rp=False,            # rigid-body warped tissue maps
    p=False,             # native-space tissue maps
    csf=False,           # include CSF output
    # atlas labeling
    atlas=None,          # str | list[str]
    atlas_surf=None,     # str | list[str]
    no_atlas=False,
    # parallelization
    multi=None,          # int; -1 = auto
    min_memory=None,     # float (GB per process)
    seed=None,           # int (reproducibility)
    # pre-processing
    pre_fwhm=None,       # float (mm)
    downsample=None,     # float (mm)
    median_filter=None,  # int (kernel size)
    vessel=None,         # float (vessel-correction threshold)
    thickness_method=None,
    no_correct_folding=False,
    # longitudinal
    initial_surf=None,   # str | Path
    long_data=None,      # str | Path
    # robustness
    no_overwrite=None,   # str (filename pattern to skip)
    no_retry=False,
    # misc
    hemisphere=False,
    defaults=None,       # str | Path (alternative defaults file)
    python=None,         # str | Path (interpreter)
    install=False,
    re_install=False,
    debug=False,
    cwd=None,
    env=None,
    check=True,          # raise on non-zero exit
    log_file=None,       # str | Path
    stream_output=True,
) -> int
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
| `get_area(v, fcs)` | Total surface area |
| `get_area_normalized(v, fcs)` | Per-vertex area (normalized) |
| `euler_characteristic(v, fcs)` | Euler number |
| `point_distance(v, fcs, pts)` | Distance from points to surface |
| `point_distance_mean(v, fcs, pts)` | Mean distance from points to surface |
| `hausdorff_distance(v1, fcs1, v2, fcs2)` | Hausdorff distance between two meshes |
| `sphere_radius(v)` | Sphere radius from vertex array |
| `sulcus_depth(v, fcs)` | Sulcus depth map |
| `smoothed_curvatures(v, fcs)` | Smoothed mean curvature |
| `surf_curvature(v, fcs)` | Raw mean curvature |
| `count_intersections(v, fcs)` | Count self-intersecting triangles |

### Surface Processing

| Function | Description |
|----------|-------------|
| `smooth_heatkernel(v, fcs, values, fwhm)` | Heat-kernel surface smoothing |
| `smooth_mesh(v, fcs, iterations)` | Laplacian mesh smoothing |
| `reduce_mesh(v, fcs, ratio, aggr)` | Mesh decimation |
| `remove_intersections(v, fcs)` | Remove self-intersections |
| `surf_average(meshes)` | Average across a list of meshes |
| `surf_deform(v, fcs, ...)` | Surface deformation |
| `surf_warp(v, fcs, deform_field)` | Warp surface by deformation field |
| `resample_to_sphere(v, fcs, sphere_v, sphere_fcs)` | Resample mesh to sphere |
| `resample_annot(annot, ...)` | Resample atlas annotation |

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
| `surf_to_pial_white(central_v, fcs, thickness)` | Derive pial/white from central + thickness |
| `central_to_pial(v, fcs, thickness)` | Expand central surface to pial |
| `correct_thickness_folding(v, fcs, thickness)` | Correct thickness for folding artefacts |

### Volume Operations

| Function | Description |
|----------|-------------|
| `vol_sanlm(volume)` | Non-local-means denoising |
| `vol_blood_vessel_correction(label, voxelsize)` | Remove blood-vessel label artefacts |
| `vol_thickness_pbt(label, voxelsize)` | PBT cortical thickness estimation |
| `vol_amap(data, mask)` | AMAP tissue segmentation |
| `vol_marching_cubes(label, voxelsize, iso)` | Marching-cubes surface extraction |
| `vol2surf(volume, v, fcs, ...)` | Sample volume values onto surface |

```python
import t1prep.cat_surf as cs
import nibabel as nib

label_img = nib.load("p0sub-01_T1w.nii.gz")
label = label_img.get_fdata()
vx = label_img.header.get_zooms()[:3]

# Extract initial surface via marching cubes
v, fcs = cs.vol_marching_cubes(label, voxelsize=vx, iso=1.5)
cs.write_surface("lh.initial.gii", v, fcs)

# Cortical thickness via PBT
gmt, ppm, dcsf, dwm = cs.vol_thickness_pbt(label, voxelsize=vx)
cs.write_values("lh.thickness.sub-01", gmt)
```

### Registration

| Function | Description |
|----------|-------------|
| `bbreg(moving, fixed, ...)` | Boundary-based registration |
| `bbreg_detect_contrast(volume)` | Auto-detect image contrast for BBR |
| `volume_register_nmi(moving, fixed)` | NMI-based volume registration |
| `volume_register_robust(moving, fixed)` | Robust volume registration |

### CLI Sub-module

`t1prep.cat_surf.cli` provides higher-level Python wrappers (file-path based)
that mirror the original CAT-Surface command-line tools:

```python
from t1prep.cat_surf import cli

# Resample surface to sphere
cli.surf2sphere("lh.central.sub-01.gii", "lh.sphere.sub-01.gii")

# Reduce mesh
cli.surf_reduce("lh.central.sub-01.gii", "lh.reduced.gii", ratio=0.25)

# Resample to template sphere and apply smoothing
cli.surf_resample(
    "lh.thickness.sub-01",
    "lh.sphere.sub-01.gii",
    target_sphere="lh.sphere.freesurfer_fsaverage.gii",
    fwhm=15.0,
    output="lh.thickness.resampled.sub-01",
)

# Compute surface curvature
cli.surf_curvature("lh.central.sub-01.gii", "lh.mc.sub-01")

# Map volume values onto surface
cli.vol2surf(
    volume="sub-01_T1w.nii.gz",
    central="lh.central.sub-01.gii",
    output="lh.T1.sub-01",
)
```

Full list of `cli` functions:

| Function | Equivalent CLI tool |
|----------|---------------------|
| `surf2pial_white` | `CAT_SurfPialWhite` |
| `surf2sphere` | `CAT_SurfToSphere` |
| `surf_area` | `CAT_SurfArea` |
| `surf_average` | `CAT_SurfAverage` |
| `surf_correct_thickness_folding` | `CAT_SurfCorrectThicknessfolding` |
| `surf_curvature` | `CAT_SurfCurvature` |
| `surf_deform` | `CAT_SurfDeform` |
| `surf_distance` | `CAT_SurfDistance` |
| `surf_reduce` | `CAT_SurfReduce` |
| `surf_remove_intersections` | `CAT_SurfRemoveIntersections` |
| `surf_resample` | `CAT_SurfResample` |
| `surf_resample_annot` | `CAT_SurfResampleAnnot` |
| `surf_warp` | `CAT_SurfWarp` |
| `vol2surf` | `CAT_Vol2Surf` |
| `vol_amap` | `CAT_VolAmap` |
| `vol_blood_vessel_correction` | `CAT_VolBloodVesselCorrection` |
| `vol_marching_cubes` | `CAT_VolMarchingCubes` |
| `vol_sanlm` | `CAT_VolSanlm` |
| `vol_thickness_pbt` | `CAT_VolThicknessPbt` |

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
| `get_filenames(input_file, out_dir, bids)` | Resolve all output filenames for a given input |
| `resample_and_save_nifti(img, voxel_size, out_path)` | Resample a NIfTI to a new voxel size and save |
| `get_resampled_header(header, voxel_size)` | Build a resampled NIfTI header |
| `align_brain(img, template)` | Reorient to match template orientation |
| `get_volume_native_space(warped, deform, reference)` | Warp a volume back to native space |
| `remove_file(path)` | Delete a file, silently ignoring missing-file errors |
| `progress_bar(...)` | Drive the `progress_bar_multi.sh` progress display |

```python
from t1prep import get_filenames

fnames = get_filenames("/data/sub-01_T1w.nii.gz", out_dir="/results", bids=True)
print(fnames["brainmask"])   # /results/.../sub-01_..._desc-brain_mask.nii.gz
print(fnames["lh_thickness"])
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

### `compute_dice_nifti(gt, pred, *, round_labels=True)`

| Parameter | Description |
|-----------|-------------|
| `gt` | Ground-truth label NIfTI (path or `nib.Nifti1Image`) |
| `pred` | Predicted label NIfTI (path or `nib.Nifti1Image`) |
| `round_labels` | When `True` (default), round to integer labels before scoring. Set `False` for soft/continuous Dice. |

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
                                 ratio=self.inputs.ratio,
                                 aggr=self.inputs.aggr)
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
    bids=True,
    gz=True,
)

# 2. Post-process surfaces directly with t1prep.cat_surf
v, fcs   = cs.read_surface("/results/.../lh.central.sub-01.gii")
thickness = cs.read_values("/results/.../lh.thickness.sub-01")
smoothed  = cs.smooth_heatkernel(v, fcs, thickness, fwhm=20.0)
cs.write_values("/results/.../lh.thickness_s20.sub-01", smoothed)
```
