# Registration evaluation

Reproducible comparison of T1Prep's spatial registration against the tools it
would replace, on public data with manual ground truth.

Everything is scored by `scripts/eval_mindboggle.py`, which implements one
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
`SurfaceLabels_*.tar.gz` (labelled `.vtk` surfaces) archives and extract them
so every subject is a directory under one root.

The surface evaluation uses all 101 subjects; the volume comparison uses the
stratified 20 in `data/subset20.txt` (4 per cohort, seed 20240903), because
ANTs and CAT12 cost 9-15 minutes per subject.

## Reproducing

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
scripts/eval_mindboggle.py project-volume --mindboggle <mindboggle>/data \
    --t1prep <mindboggle>/data --work <work>/eval --out-space t1prep
scripts/eval_mindboggle.py dice --work <work>/eval --space t1prep \
    --protocol both --subjects $(tr '\n' ' ' < evaluation/data/subset20.txt)
```

Arms produced outside T1Prep are scored with `--preresampled`, which takes
label volumes already in the target space -- that is how any other tool's
normalisation can be dropped into the same comparison.
