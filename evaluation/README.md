# Registration evaluation

Reproducible comparison of T1Prep's spatial registration against the tools it
would replace, on public data with manual ground truth.

Everything is scored by `tools/eval_mindboggle.py`, which implements one
protocol for every method: each subject's *manual* labels are carried into a
common space by the transform under test and compared there, so nothing but
the registration differs between arms.

## Protocols

Once every subject's manual labels sit in one common space, there are three
ways to turn them into a number.  `--protocol` picks one.

### `pairs` — no atlas at all

Dice between every two subjects' labels directly, over all 190 pairs of the
20-subject subset.  Nothing is estimated from the group, so this measures the
registration and only the registration.  It is what surface-registration
papers report, and it is the lowest of the three because there is no averaging
to smooth boundary error.

### `loo` — leave-one-out majority-vote atlas

The atlas is built and used per subject:

1. **Tally.**  At every vertex (or voxel) of the common space, count how many
   of the *N* subjects carry each of the 31 DKT labels there.
2. **Hold one out.**  For subject *i*, subtract that subject's own vote from
   the tally, so the atlas is built from the other *N−1* only.
3. **Vote.**  The label with the highest remaining count wins the vertex.
   "Unlabelled" is barred from winning, so medial-wall vertices cannot absorb
   a parcel.
4. **Score.**  Dice that predicted parcellation against subject *i*'s own
   manual labels, per region, per hemisphere.
5. Repeat for every subject.

In code that is one line — `left[labels[i], cols] -= 1` — and it is the whole
difference from the next protocol.

This mirrors how FreeSurfer-style atlas parcellation is validated, which is
why it is the number to compare against published atlas-based figures (and
against a Buckner40/DK40 result).  Note the labeller here is a plain
per-vertex majority vote: no intensity features, no spatial prior, no
smoothing.  That isolates the registration, but it also scores lower than a
trained classifier such as `mris_ca_label` would on the same alignment.

### `atlas` — the same, but including the subject

Identical except that step 2 is skipped: one atlas is built from all *N*
subjects and every subject is scored against it.  This is the more obvious
construction, and it is what `--protocol atlas` does — but each subject then
votes for the answer it is scored against, which flips exactly the split
vertices that discriminate between registrations.

**Measured on this data (N = 20):**

| arm | `loo` | `atlas` | inflation |
|---|---|---|---|
| T1Prep volume | 0.7289 | 0.7404 | **+0.0115** [+0.0113, +0.0117] |
| T1Prep surface | 0.8154 | 0.8272 | **+0.0118** [+0.0114, +0.0123] |

That bias is larger than most of the differences this benchmark resolves —
CAT12 beats T1Prep by +0.0120 and T1Prep beats ANTs by +0.0156 under `loo`.
It inflates every arm by roughly the same amount, so it would not reorder
them, but it would make the absolute numbers incomparable with published
leave-one-out figures.  The effect is ~1/N of the vote, so it shrinks with
larger cohorts and grows sharply with smaller ones; it is not a constant that
can be subtracted.  `loo` is therefore the default.

See [`results/RESULTS.md`](results/RESULTS.md) for the numbers each protocol
produces, and `results/dice_boxplots.png` for the distributions behind them.

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
| `plot_dice.py` | boxplots of the Dice distributions behind the means |
| `msm_data/` | sMRIPrep's MSM config and reference surfaces, verbatim |
