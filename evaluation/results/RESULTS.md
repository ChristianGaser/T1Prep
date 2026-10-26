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
* **`atlas`** — the same, but with the subject left *in* the atlas. Inflates
  every arm by 0.009-0.012, about the CAT12-vs-T1Prep gap, because each subject
  votes for the answer it is scored against. Not used for any number below;
  see [How the protocols are built](../README.md#protocols) for the
  construction and the measured bias.

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
| **T1Prep** | 0.7289 | 0.6085 | ~1 min |
| **CAT12 — geodesic shooting** | **0.7410** | **0.6249** | 5-15 min |

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

## Surface registration

### T1Prep vs FSL newMSM vs FreeSurfer 8.1 — 19 subjects

The 19 subjects of `subset20` that FreeSurfer 8.1 processed (all but
`MMRR-3T7T-2-1`, excluded due to preprocessing issues), the same manual labels
and the same protocol.  Every arm is scored on a 164k mesh, with the labels
attached to the midthickness (FreeSurfer) or the central surface (T1Prep),
which is the same geometry; see [the attachment
surface](#the-attachment-surface-is-a-confound-and-it-is-a-large-one).  T1Prep
and newMSM register T1Prep's surfaces; FreeSurfer registers its own, so its
row measures surface reconstruction and spherical registration together, and
the two cannot be separated because each pipeline's sphere is tied to its own
mesh.  T1Prep is 0.8.1 with the relaxed input sphere (`CAT_Surf2Sphere
-relax`, cat-surf 1.1.5), as the next release runs it.

| arm | registration | target | LOO | pairs | time / hemisphere |
|---|---|---|---|---|---|
| **T1Prep** | **relaxed sphere, Spherical Demons → fsaverage** | fsaverage 164k | **0.8463** | **0.7824** | ~50 s |
| FSL newMSM | fMRIPrep's MSMSulc config, from T1Prep's sphere and `sulc` | fs_LR 164k | 0.8354 | 0.7692 | +4 min |
| **FreeSurfer 8.1** | **recon-all: `mris_sphere`, `mris_register` → fsaverage** | fsaverage 164k | **0.8516** | **0.7906** | ~12 min |

Paired differences, with 95 % intervals from a subject-level bootstrap (4000
resamples, a pair weighted by both its subjects):

| | Δ LOO | Δ pairs | subjects better (LOO) |
|---|---|---|---|
| T1Prep − FreeSurfer | −0.005 [−0.016, +0.006] | −0.008 [−0.025, +0.011] | 7/19 |
| T1Prep − newMSM | +0.011 [+0.008, +0.013] | +0.013 [+0.009, +0.017] | 18/19 |
| newMSM − FreeSurfer | −0.016 [−0.027, −0.004] | −0.022 [−0.038, −0.002] | 1/19 |

T1Prep and FreeSurfer are level within the noise of 19 subjects: the right
hemisphere is level (−0.001 / −0.001), the left trails (−0.009 / −0.016).  The
pial attachment gives the same picture (FreeSurfer 0.8197 / 0.7425, T1Prep
0.8126 / 0.7331).  newMSM is below both.  On all 20 subjects T1Prep scores
0.8473 / 0.7834 (0.8445 / 0.7845 on fsaverage 32k with its cortex mask) and
newMSM 0.8366 / 0.7705; T1Prep's fs_LR sphere for fMRIPrep (`desc-msmsulc`,
Spherical Demons onto fs_LR) reaches 0.8546 / 0.7936.

Times are per hemisphere.  T1Prep's is the sphere (19 s) plus Spherical
Demons (30 s) on an M2 Ultra, one process; T1Prep runs the two hemispheres in
parallel.  newMSM starts from T1Prep's registration and adds ~4 min (two
OpenMP threads).  FreeSurfer's is `mris_register` aligning `?h.sphere` to the
fsaverage folding atlas, 8.4 min (median; IQR 7.2-12.6, range 4.7-55.7) read
from `FSRUNTIME@` in the `rca-surfreg` logs (38 of 38 hemispheres), plus
~3.5 min for the `mris_sphere` that makes `?h.sphere`.  Two caveats on those:
every run reports **`1 threads`** despite the command line requesting
`-threads 24`, so they are single-core timings; and the spread is
cohort-structured rather than random -- the four MMRR subjects take 59-96 min
each while every NKI, OASIS and HLN subject finishes in 12-21 min.

### The attachment surface is a confound, and it is a large one

Which surface the labels attach to moves Dice by more than most method
differences here, so it has to be fixed on a principled basis and held
identical across arms:

| pipeline | mid − pial (LOO) | mid − pial (pairs) |
|---|---|---|
| T1Prep | +0.0337 | +0.0493 |
| FreeSurfer | +0.0318 | +0.0481 |

Mindboggle's labelled surfaces sit on the **pial**, so the pial gives the
smallest vertex-to-label distance and the *worst* transfer: at the pial the
two banks of a sulcus are in contact, so a nearest-neighbour lookup jumps
across the fundus.  Counting vertices whose label matches none of their 1-ring
neighbours (OASIS-TRT-20-2 lh, FreeSurfer mesh):

| mesh | distance | unlabelled | speckle |
|---|---|---|---|
| pial | 0.44 mm | 0.4 % | **0.502 %** |
| mid | 1.12 mm | 0.5 % | **0.175 %** |
| white | 2.14 mm | 13.6 % | 0.117 % |

White has the least speckle but loses 13.6 % of vertices past the 3 mm cutoff.
Midthickness is the only geometry good on both counts - and it is what
T1Prep's central surface already is, so the two pipelines are compared on
equivalent geometry rather than on a choice tuned to the outcome.

That the mid > pial > white ordering appears in **both** pipelines is the
check that this is a property of the labelled data, not of either pipeline.

### Setup notes

The two pipelines use opposite coordinate conventions, verified per subject:

| | frame | without the correction |
|---|---|---|
| FreeSurfer surfaces | tkrRAS (`--reference-glob none`) | correct as-is |
| T1Prep surfaces | scanner RAS (`+c_ras`) | 134--166 mm off |

Applying the wrong one is not subtle - it misses by the size of the head -
but it only shows up on subjects whose `c_ras` is non-zero, and OASIS-TRT-20's
is exactly zero, so a single-cohort check would pass either way.

Scoring against FreeSurfer 8.1's own `fsaverage/surf/?h.sphere` and against
T1Prep's shipped `templates_surfaces_164k/?h.sphere.freesurfer.gii` gives
identical results to four decimals: they are the same mesh, vertex for vertex.

### Reading the newMSM result

newMSM is run exactly as sMRIPrep runs MSMSulc: the config
(`MSMSulcStrainFinalconf`) and both reference files are copied verbatim from
sMRIPrep, and the four preprocessing steps (affine regression onto the
fsLR-registered sphere, apply, re-sphere to radius 100, **invert sulc**) are
reproduced.  Its starting mesh is T1Prep's native sphere, affinely aligned
(where sMRIPrep would use FreeSurfer's `?h.sphere`), and it is driven by
T1Prep's `?h.sulc`.

The newMSM row shown here until October 2026 (0.7317 / 0.6405 on 20 subjects)
was wrong.  Its `sulc` came from cat-surf < 1.1.1, whose curvature type 11 did
not measure depth: on these surfaces it correlates with the correct map at
Spearman 0.00-0.02.  MSMSulc was registering a map without folding
information, and that, not MSM, made most of the 0.08-0.10 lead T1Prep held
here before.

With correct inputs T1Prep stays ahead, by much less: +0.011 / +0.013 on the
19 subjects above, and +0.018 [+0.014, +0.023] / +0.023 [+0.017, +0.029] for
T1Prep's own fs_LR sphere, the one fMRIPrep uses instead of running MSMSulc
(20 subjects; 19 of 20 subjects and 188 of 190 pairs better).  Both the
ordering and its size match the literature: 0.786 vs 0.766 for Spherical
Demons and MSM in Zhao's pediatric comparison, 0.881 vs 0.872 in the
SphereMorph table.  MSMSulc is a deliberately conservative,
strain-regularised *refinement*: it is meant to improve an already-aligned
sphere without introducing areal distortion, and `MSMSulcStrainFinalconf`
limits it to 15 iterations at the finest level with a strain penalty.

Cortical-parcel Dice is not the criterion MSMSulc is optimised for, so this
number should not be read as "MSM is a worse algorithm" - it says that for
carrying fold-defined labels between subjects, T1Prep's registration is
somewhat better than fMRIPrep's MSMSulc on the same surfaces, in a fraction
of the time (~40 s against ~4 min per hemisphere for the step onto fs_LR).

<!---
### Not shown: T1Prep without the relaxed sphere

Kept for later reading.  T1Prep 0.8.1 and earlier registered the sphere as
`CAT_Surf2Sphere` inflates it, without `-relax`; below, that sphere against
the relaxed one on the same central surfaces (20 subjects, fsaverage 32k
unless stated).  Per-region rows: `d20_surf_*_norelax.csv`, from 0.8.1's own
spheres (bit-identical to its output).

#### The input sphere

Spherical Demons does not start from scratch: it deforms the sphere that
`CAT_Surf2Sphere` inflates the central surface to, and inherits that sphere's
distortion.  The inflation compresses the poles and expands the medial wall;
on these 40 hemispheres it maps 19 % of the surface with more than twofold
area distortion.  cat-surf 1.1.5 adds two steps: **`-relax`** pulls the
distances between neighbours (two rings) towards their distances on the
central surface, the distance term of FreeSurfer's `mris_sphere`, and **fold
removal** (on by default) re-embeds the triangles the inflation folds.
T1Prep 0.8.1 gets the fold removal through that default; `-relax` is on from
the next release.

Same T1Prep 0.8.1 central surfaces, the same Spherical Demons call and the
same scoring; only the sphere differs:

| sphere | LOO | pairs | Δ LOO [95 % CI] | Δ pairs [95 % CI] | better |
|---|---|---|---|---|---|
| inflation only (T1Prep ≤ 0.8.0) | 0.8120 | 0.7454 | | | |
| + fold removal (T1Prep 0.8.1) | 0.8119 | 0.7454 | −0.0000 | +0.0000 | |
| **+ `-relax`** (next release) | **0.8445** | **0.7845** | **+0.0326** [+0.0273, +0.0375] | **+0.0391** [+0.0315, +0.0460] | 20/20 subjects, 187/190 pairs |

Δ against inflation only, scored at fsaverage 32k; the intervals come from a
subject-level bootstrap (4000 resamples, a pair weighted by both its
subjects).

| sphere | area-weighted mean \|log₂ area ratio\| | area distorted > 2× | folded triangles: sphere / after registration | time |
|---|---|---|---|---|
| inflation only | 0.61 | 18.9 % | 1586 in 23 hemispheres / 1554 in 14 | 9 s |
| + fold removal | 0.61 | 18.9 % | 0 / 452 in 2 | 9 s |
| + `-relax` | 0.31 | 1.3 % | 0 / 0 | 19 s |

Area ratio: the area of each triangle on the sphere over its area on the
central surface, both normalised to their totals; medians over the 40
hemispheres.  Time: one hemisphere, one process, M2 Ultra; Spherical Demons
adds ~30 s either way.

* **The gain is systematic.**  Every subject improves under LOO (+0.005 to
  +0.051), the right hemisphere twice as much as the left (+0.042 vs
  +0.023).  It is concentrated in the right lateral temporal lobe: middle
  temporal +0.229 (median 0.64 → 0.88, every subject better), inferior
  temporal +0.176, transverse temporal +0.089, then postcentral +0.080.  That
  matches the whole-fold slip (~14°) of the right inferior temporal gyrus
  which comparing T1Prep's registration with recon-all's had traced to the
  inflated sphere.
* **It is not free of local failures.**  The two registrations differ by 2°
  (median per hemisphere; 95th percentile 8°), and in a few hemispheres by a
  whole fold.  Mostly that is the correction above, but in two of the 40 it
  goes the wrong way: OASIS-TRT-20-16 rh loses its parahippocampal and
  entorhinal labels (LOO 0.72 → 0.00 and 0.68 → 0.15), MMRR-3T7T-2-2 lh its
  pre- and postcentral (0.84 → 0.32 and 0.61 → 0.23).  Both subjects still
  improve on average, and these are the only regions whose mean drops by
  more than 0.003 (rh parahippocampal −0.019, lh precentral −0.010).
* **Fold removal alone changes no Dice** (no region's mean by more than
  0.002).  Its value is a valid map, and even that does not survive
  registration without `-relax`: Spherical Demons folds two of the 40 spheres
  again (188 and 264 triangles, about half of them the ones fold removal had
  straightened).  Fold removal leaves those regions crushed, with triangles
  500-2000× smaller than the median one, and the warp flips them.  With
  `-relax` none is folded.
* **The fs_LR spheres for fMRIPrep gain as much.**  `fsLR` (`desc-reg`, the
  fsaverage registration carried into fs_LR) goes from 0.8167 / 0.7452 to
  0.8480 / 0.7839, `msm` (`desc-msmsulc`, the sphere fMRIPrep resamples to
  fs_LR with) from 0.8211 / 0.7519 to 0.8546 / 0.7936, every pair better.
* **Nothing else moved.**  T1Prep 0.8.1 without `-relax` reproduces the 0.7.x
  run shown here before (fsaverage LOO −0.003, pairs ±0.000; 164k ±0.001),
  so the gain is the sphere's alone.

#### newMSM from the unrelaxed sphere

| newMSM inputs | LOO | pairs |
|---|---|---|
| `sulc` from cat-surf < 1.1.1, sphere without `-relax` (shown until October 2026) | 0.7317 | 0.6405 |
| correct `sulc`, sphere without `-relax` | 0.7998 | 0.7256 |
| correct `sulc`, relaxed sphere | 0.8366 | 0.7705 |

Correcting the map gains +0.068 (LOO) and +0.085 (pairs), with every subject
and every pair better; T1Prep's own registrations moved by no more than 0.004
between the two T1Prep versions involved, so the gain is the map's.  The
relaxed sphere then adds +0.037 [+0.032, +0.042] and +0.045 [+0.038, +0.052],
as much as it gives Spherical Demons.

#### FreeSurfer against T1Prep without `-relax` (19 subjects, 164k)

| attachment surface | FreeSurfer 8.1 | T1Prep 0.7.6 |
|---|---|---|
| midthickness / central | 0.8516 / 0.7906 | 0.8143 / 0.7427 |
| pial | 0.8197 / 0.7425 | 0.7900 / 0.7066 |

With the unrelaxed sphere FreeSurfer led by ~0.037 (LOO) and ~0.048 (pairs),
and the margin held whichever attachment surface was chosen (+0.030 to
+0.047).  T1Prep's mid − pial difference was +0.0243 / +0.0362.
-->

---

## Distributions

![Dice distributions per method](dice_boxplots.png)

The means in the tables above summarise wide distributions, and the boxplots
change how several of them should be read.

**The spread within a method dwarfs the differences between methods.** Every
non-linear volume arm has an interquartile range about 0.10 wide (LOO) or
0.13 (pairs), the surface arms about 0.08 and 0.10-0.12, while the means
separate by 0.01-0.03.  Which region is being measured
matters far more than which of these methods produced the registration - the
ranking is a statement about averages over many regions, not a prediction for
any single parcel.

**Affine is not merely lower, it is a different kind of distribution.** Its
pairwise IQR spans 0.000-0.526 - more than a quarter of region-pairs get
essentially *no* overlap from an affine alignment - against 0.52-0.66 for
ANTs.  69 % of its pairwise values fall below 0.5, versus 10-19 % for the
non-linear methods.  That is the gap the non-linear step actually closes.

**The three surface arms differ little.** T1Prep and FreeSurfer have nearly
the same box (pairwise IQR 0.103 and 0.104, lower whiskers 0.57 and 0.59,
3.0 % and 2.6 % of values below 0.5); newMSM's is slightly wider (0.115) and
reaches a little lower (0.56, 3.3 %).  The wide newMSM box shown here before
(IQR 0.199, whisker 0.27, 17.3 % below 0.5) was the `sulc` bug.

**The volume arms overlap heavily.** ANTs, T1Prep and CAT12 have visually
similar boxes, consistent with paired differences of 0.012-0.028.  Their
ordering is reliable - the paired intervals exclude zero - but no visible
difference should be expected on any one subject.

Regenerate with `tools/plot_dice.py`.

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
* FSL newMSM built against FSL 6.0.7.23; config and reference surfaces copied
  verbatim from sMRIPrep (`tools/msm_data/`).  On macOS the build needs
  `$FSLDIR/bin/make` (Apple ships GNU Make 3.81, which cannot parse the
  `define VAR =` syntax in FSL's `rules.mk` and silently generates no compile
  rules) and libomp via `USRCXXFLAGS`/`USRLDFLAGS` - see
  `tools/build_newmsm.sh`.
* T1Prep deformations are the SPM-format `y_*.nii` (5-D, native mm, affine and
  non-linear composed).
* T1Prep surface arms (`d20_surf_*`, `d19_t1p_*`): T1Prep 0.8.1 with
  cat-surf 1.1.5, the sphere computed with `relax=True` as the next release
  does it; an end-to-end run with that change gives bit-identical spheres.
  The `fsLR` and `msm` spheres are written as `--fmriprep` writes them.
* newMSM (`d20_surf_newmsm.csv`, `d19_newmsm.csv`) starts from those spheres,
  with `?h.sulc` from cat-surf 1.1.5 (`tools/gen_sulc.py`), 2 OpenMP threads
  per hemisphere.
* FreeSurfer 8.1 (`d19_fs81*.csv`): recon-all, scored through its own
  `?h.sphere.reg`, with the labels attached to the midthickness
  (`tools/prepare_fs_midthickness.py`) and `--reference-glob none`.
* Per-region, per-comparison Dice for every arm is in `d20_*.csv` (20
  subjects) and `d19_*.csv` (the 19 with FreeSurfer output).
