# Registration accuracy on Mindboggle-101

All numbers are Dice between manual DKT31 cortical labels carried into a
common space by the transform under test. One protocol for every arm: nothing
but the registration differs.

* **`pairs`** — every two subjects compared directly. No atlas is involved, so
  this isolates registration alone. Comparable to the numbers surface
  registration papers report.
* **`loo`** — a leave-one-out majority-vote atlas built from the other N−1
  subjects, applied to the held-out subject. The FreeSurfer-style protocol;
  measures registration *and* the atlas the group forms.

Ground truth never passes through a second registration: labels are attached
to each subject's own anatomy geometrically, and only then moved.

---

## Volume registration — 20 subjects

Stratified subset (`../data/subset20.txt`), MNI152NLin2009cAsym, identical
1.5 mm grid, identical brain-extracted inputs, identical scoring.

| method | LOO | pairs | time / subject |
|---|---|---|---|
| Affine only (Mindboggle's own) | 0.5173 | 0.3317 | — |
| ANTs SyN, Mattes MI 100×70×50×20 | 0.6972 | 0.5672 | ~1 min |
| **ANTs — fMRIPrep's exact config** | 0.7133 | 0.5854 | ~9 min |
| **T1Prep** | 0.7289 | 0.6085 | seconds |
| **CAT12 — geodesic shooting** | **0.7410** | **0.6249** | ~15 min |

Paired differences (mean, 95 % CI over all region × comparison pairs):

| comparison | LOO | pairs |
|---|---|---|
| CAT12 − T1Prep | +0.0120 [+0.0099, +0.0142] | +0.0163 [+0.0155, +0.0171] |
| T1Prep − ANTs (fMRIPrep) | +0.0156 [+0.0136, +0.0177] | +0.0231 [+0.0223, +0.0239] |
| CAT12 − ANTs (fMRIPrep) | +0.0277 [+0.0255, +0.0299] | +0.0395 [+0.0386, +0.0403] |
| ANTs (fMRIPrep) − affine | +0.1960 | +0.2539 |
| T1Prep − affine | +0.2116 | +0.2771 |
| CAT12 − affine | +0.2236 | +0.2934 |

**Ordering: CAT12 > T1Prep > ANTs (fMRIPrep) > ANTs (MI) ≫ affine.** Every
interval is clear of zero, but the spread between the three good methods is
only ~0.03 Dice — an order of magnitude smaller than the gap to affine.

### For replacing ANTs in fMRIPrep

fMRIPrep normalises with `antsRegistration` via
`niworkflows.interfaces.norm.SpatialNormalization`, parameterised by
`niworkflows/data/t1w-mni_registration_precise_000.json`: Rigid + Affine
(Mattes, 56 bins, 25 % regular sampling, 100×100) then **SyN with CC radius 4,
100×70×50×20**, transform parameters `[0.1, 3.0, 0.0]`. `tools/ants_fmriprep.sh`
reproduces that command line field for field, run with the real ANTs binaries
(2.6.5).

T1Prep is **more accurate than that configuration** (+0.016 LOO, +0.023 pairs)
at a fraction of the runtime — a forward pass versus ~9 minutes of CC-metric
optimisation. Anatomical normalisation is a dominant cost in fMRIPrep, so the
runtime argument is at least as strong as the accuracy one.

Two caveats:

* T1Prep's warp comes from a network trained to map into
  MNI152NLin2009cAsym, and the fixed image here was T1Prep's own template.
  ANTs solves the problem from scratch with no learned prior. That is a real
  property of learned registration, not a rigged comparison, but the result
  will not automatically transfer to a template T1Prep was not trained on.
* DKT31 is cortical parcels only. Nothing here measures subcortical alignment,
  which fMRIPrep users also depend on.

---

## Surface registration — 100 subjects

| space | registration | LOO | pairs |
|---|---|---|---|
| `fsaverage` | Spherical Demons → fsaverage | 0.8193 | 0.7438 |
| `fsLR` | *same registration*, project-unprojected | 0.8200 | 0.7424 |
| `msm` | independent Spherical Demons → fsLR | 0.8249 | 0.7481 |

`fsLR` performs no registration of its own — it is a fixed barycentric remap of
the fsaverage result — so it serves as a control: it isolates what changing the
template mesh costs (nothing, ±0.001), which is what makes `msm` vs `fsLR` the
only clean registration comparison here (+0.005 [+0.004, +0.006]).

`msm` is **not** FSL's MSM; it is Spherical Demons onto the fsLR average
standing in for MSMSulc. All three agree within 0.006 Dice.

Surface Dice exceeds volume Dice substantially (0.819 vs 0.729 LOO), which
reproduces Klein & Ghosh 2010. Read the direction, not the exact difference:
vertex Dice on a surface and voxel Dice on a filled ribbon are different
measurements on different supports.

`NKI-RS-22-16` is excluded from the surface arms: its right-hemisphere labelled
surface sits ~40 mm from its own anatomy. Mindboggle's own
`label-issues_201903.txt` independently lists that subject as defective.

---

## Comparison with published numbers

Published volume-registration results are not directly comparable, for three
reasons:

1. **Metric.** Klein 2009 and Ashburner & Friston 2011 report *target overlap*
   = |deformed ∩ target| / |target|, one-sided. Dice is 2|A∩B|/(|A|+|B|).
2. **Structures.** The same methods score 0.75 on LPBA40 and 0.59 on IBSR18 —
   that gap is the label set, not the algorithm. DKT31 is cortex only, the
   hardest case, which is why its affine baseline is 0.33 rather than 0.40–0.60.
3. **Geodesic shooting is not in Klein 2009** — that is Ashburner & Friston
   2011. Klein's SPM entry is DARTEL.

Improvement over affine is the most defensible cross-study quantity:

| study | dataset | affine | best | gain |
|---|---|---|---|---|
| Ashburner & Friston 2011 | IBSR18 | 0.40 | GS2 0.590 | +0.19 |
| Ashburner & Friston 2011 | LPBA40 | 0.60 | GS2 0.751 | +0.15 |
| Klein 2009 (SyN/ART) | IBSR18 | 0.40 | ~0.55 | +0.15 |
| this work (pairs) | Mindboggle-101 | 0.33 | CAT12 0.625 | +0.29 |

That CAT12's shooting leads here is consistent with Ashburner & Friston
reporting shooting ahead of DARTEL and of Klein 2009's best on both datasets.

---

## Provenance

* Data: Mindboggle-101, CC-BY (Klein & Tourville 2012).
* ANTs 2.6.5 (macOS ARM64 binaries), `antsRegistration` + `antsApplyTransforms`.
* CAT12 via `cat_batch_cat.sh -ns -p 4` (geodesic shooting, surfaces skipped).
* T1Prep deformations are the SPM-format `y_*.nii` (5-D, native mm, affine and
  non-linear composed).
* Per-region, per-comparison Dice for every arm is in `d20_*.csv`.
