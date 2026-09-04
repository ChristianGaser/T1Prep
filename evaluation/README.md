# Registration evaluation

Reproducible comparison of T1Prep's spatial registration against the tools it
would replace, on public data with manual ground truth.

Everything is scored by `tools/eval_mindboggle.py`, which implements one
protocol for every method: each subject's *manual* labels are carried into a
common space by the transform under test and compared there, so nothing but
the registration differs between arms.

## Layout

| path | contents |
|------|----------|
| `data/subset20.txt` | the 20 subjects used for the volume comparison |
| `tools/` | the exact scripts used to run each competing method |
| `results/RESULTS.md` | the numbers, the protocol, and the caveats |
| `results/*.csv` | per-region, per-comparison Dice for every arm |

## Data

[Mindboggle-101](https://mindboggle.info/data.html) (Klein & Tourville 2012),
101 subjects with manually labelled cortex under the DKT protocol, CC-BY.
Download the per-cohort `*_volumes.tar.gz` (T1w plus label volumes) and
`SurfaceLabels_*.tar.gz` (labelled `.vtk` surfaces) archives.  `--mindboggle`
takes any number of roots and searches each for a directory named after the
subject, so the archives can stay split by kind:

```
mindboggle-101/
├── data/<subject>/        t1weighted.nii.gz, labels.DKT31.manual*.nii.gz
│                          plus the T1Prep outputs (mri/, surf/)
└── surfaces/<subject>/    ?h.labels.DKT31.manual.vtk
```

```bash
--mindboggle <mindboggle>/surfaces <mindboggle>/data
```

Both roots are needed for the surface route: the labels come from `surfaces/`
and `--reference-glob` finds `t1weighted.nii.gz` under `data/` to convert the
labelled surfaces out of FreeSurfer tkrRAS into scanner space.

The surface evaluation uses all 101 subjects; the volume comparison uses the
stratified 20 in `data/subset20.txt` (4 per cohort, seed 20240903), because
ANTs and CAT12 cost 9-15 minutes per subject.

## Reproducing

### Surface arms

A plain T1Prep run does not write everything the surface comparison needs, and
`--fmriprep` is the wrong way to get it -- that flag also switches the run to
BIDS naming *and* to AMAP segmentation, so the surfaces would no longer match.
Two helpers derive the missing pieces from a plain run instead:

```bash
tools/gen_sulc.py --t1prep <mb>/data                    # ?h.sulc (MSM input)
tools/gen_fslr_msm_spheres.py --t1prep <mb>/data        # fsLR + msmsulc spheres
    --subject-file data/subset20.txt                    # --fslr-only is ~40x faster
tools/run_newmsm.py --t1prep <mb>/data --work <work>/msm \
    --subject-file data/subset20.txt --jobs 4           # FSL newMSM
```

An external registration is scored by pointing at its sphere and its target:

```bash
tools/eval_mindboggle.py project --space fsLR --out-space newmsm \
    --sphere-file '<work>/msm/{subject}/{hemi}.sphere.reg.surf.gii' \
    --template-sphere 'tools/msm_data/{fshemi}.sphere.164k_fs_LR.gii' ...
```

```bash
# 1. process the cohort with T1Prep
for s in <mindboggle>/data/*/; do
    T1Prep --out-dir "$s" "$s/t1weighted.nii.gz"
done

# 2. the competing methods (see tools/ for each)
tools/run_cat12.sh   <mindboggle>/data <work>/cat12
tools/run_ants_batch.sh                     # antsRegistration, fMRIPrep config
tools/make_affine_baseline.py --mindboggle <mindboggle>/data \
    --reference <mindboggle>/data/<subj>/mri/y_*.nii --out <work>/affine

# 3. score every arm through the same protocol
tools/eval_mindboggle.py project-volume --mindboggle <mindboggle>/data \
    --t1prep <mindboggle>/data --work <work>/eval --out-space t1prep
tools/eval_mindboggle.py dice --work <work>/eval --space t1prep \
    --protocol both --subjects $(tr '\n' ' ' < evaluation/data/subset20.txt)
```

Arms produced outside T1Prep are scored with `--preresampled`, which takes
label volumes already in the target space -- that is how any other tool's
normalisation can be dropped into the same comparison.

## Tools

| script | purpose |
|---|---|
| `eval_mindboggle.py` | the protocol: project labels, score LOO and pairwise Dice |
| `make_affine_baseline.py` | Mindboggle's affine labels on the evaluation grid |
| `run_ants_batch.sh` + `ants_fmriprep.sh` | `antsRegistration` with fMRIPrep's exact JSON |
| `run_ants_antspy.py` | ANTsPy arms (note: its default `SyN` uses `reg_iterations=(40,20,0)` — no iterations at the finest level) |
| `run_cat12.sh` | CAT12 geodesic shooting |
| `gen_sulc.py` | `?h.sulc` from an existing run, without `--fmriprep` |
| `gen_fslr_msm_spheres.py` | fsLR and msmsulc spheres from an existing run |
| `build_newmsm.sh` | build FSL newMSM on macOS |
| `run_newmsm.py` | newMSM as sMRIPrep runs MSMSulc |
| `msm_data/` | sMRIPrep's MSM config and reference surfaces, verbatim |
