# Registration accuracy on Mindboggle-101 and Neuromorphometrics

Each method's transform carries every subject's manual labels into a common
space, where they are scored by Dice; nothing but the registration differs
between methods.  Two label sets: **Mindboggle-101** (DKT31 cortical parcels;
volume and surface registration) and **Neuromorphometrics** (whole-brain
labels; volume registration).  Two protocols:

* **`pairs`**: every two subjects compared directly, the registration alone.
* **`loo`**: each subject against the majority vote of all the others, as
  atlas-based labelling is validated.

Intervals are 95 % subject-level bootstrap intervals (4000 resamples).  How the
protocols are built, and why the `atlas` protocol is not used, is described in
[the evaluation README](../README.md#protocols).  The derivations and side
results are in the long version, [RESULTS_long.md](RESULTS_long.md).

![Dice per region and comparison](dice_boxplots.png)

Each box holds every region × comparison value of a method.  The spread within
a method is far larger than the differences between methods: the rankings
below hold for averages, not for any single region.

---

## Volume registration

T1Prep registers by CAT12's geodesic shooting, ported to C
(`cat_surf.vol_shoot`), from the release after 0.8.1; up to 0.8.1 it ran
deepmriprep's warp network.  It has two settings: by default CAT12's own, the
one for VBM, and with `--fmriprep` a more flexible one
([below](#t1preps-two-settings)).

### Mindboggle-101, 20 subjects

| method | LOO | pairs | time / subject |
|---|---|---|---|
| ANTs SyN, Mattes MI | 0.6972 | 0.5672 | ~1 min |
| ANTs as sMRIPrep runs it (1 mm) | 0.7294 | 0.6073 | ~10 min |
| T1Prep ≤ 0.8.1 (deepmriprep's warp) | 0.7289 | 0.6085 | ~40 s |
| CAT12 (geodesic shooting) | 0.7410 | 0.6249 | 5-15 min |
| **T1Prep, default (for VBM)** | **0.7421** | **0.6274** | ~56 s |
| **T1Prep, `--fmriprep`** | **0.7696** | **0.6663** | ~77 s |

| T1Prep − | default, Δ LOO | default, Δ pairs | `--fmriprep`, Δ LOO | `--fmriprep`, Δ pairs |
|---|---|---|---|---|
| ANTs as sMRIPrep runs it | +0.013 [+0.008, +0.018] | +0.020 [+0.014, +0.026] | +0.040 [+0.036, +0.045] | +0.059 [+0.053, +0.065] |
| CAT12 | +0.001 [−0.005, +0.006] | +0.003 [−0.004, +0.009] | +0.029 [+0.025, +0.032] | +0.041 [+0.036, +0.047] |

Times per subject on an M2 Ultra, one subject at a time on an otherwise idle
machine.  ANTs: its registration alone at 1 mm (`antsAffineInitializer`,
`antsRegistration`) on fMRIPrep's eight threads.  T1Prep: its whole volume
pipeline with MPS, segmentation, `mwp` maps and two atlases included, of which
the shooting takes 15 s (default) or 37 s (`--fmriprep`'s setting).  With
eight subjects processed at a time, each takes about 75 s or 98 s.

### Neuromorphometrics, 35 subjects

The 35 OASIS brains of the MICCAI 2012 labelling challenge, 132 tissue labels
(98 cortical, 18 subcortical, 16 other: ventricles, white matter, cerebellum,
brainstem), both hemispheres scored together.  LOO / pairs:

| regions | ANTs as sMRIPrep runs it | T1Prep, default | T1Prep, `--fmriprep` |
|---|---|---|---|
| all | 0.7105 / 0.6106 | 0.7203 / 0.6233 | **0.7355 / 0.6422** |
| cortical | 0.6773 / 0.5639 | 0.6898 / 0.5810 | **0.7085 / 0.6042** |
| subcortical | 0.8063 / 0.7426 | **0.8128 / 0.7473** | 0.8116 / 0.7455 |
| other | 0.8064 / 0.7484 | 0.8030 / 0.7429 | **0.8154 / 0.7583** |

T1Prep − ANTs over all labels: +0.010 [+0.006, +0.014] LOO with the default
(31 of 35 subjects), +0.025 [+0.022, +0.029] with `--fmriprep` (35 of 35).  The
default trails ANTs in the other structures (−0.003 [−0.006, −0.001]);
`--fmriprep` leads everywhere, least in the subcortical nuclei.

### T1Prep's two settings

`--fmriprep` runs every scheduled Gauss-Newton iteration with half of CAT12's
regularisation.  That aligns labels better than the default (+0.028 LOO on
Mindboggle, +0.015 on Neuromorphometrics, every subject), but it is **not
meant for VBM**: its Jacobians vary more between subjects, and the modulated
maps inherit that variance.  VBM of `s6mwp1` on age with TIV as a covariate,
`--fmriprep`'s setting against the default:

| | 60 children (ds000228) | 60 adults (ds005270) |
|---|---|---|
| residual variance, unmodulated GM | −41 % | −35 % |
| residual variance, modulated GM | +29 % | +20 % |
| mean T for age | −12 % | −9 % |

Tissue alignment improves and the age effect itself is unchanged; the loss
comes from the Jacobian.  CAT26 with half the regularisation loses as much
(residual variance +17 % / +14 %).

### For replacing ANTs in fMRIPrep

ANTs is run as sMRIPrep's `init_register_template_wf` runs it: the 1 mm
MNI152NLin2009cAsym template under its brain mask, the T1w truncated
(`TruncateImageIntensity 0.01 0.999 256`) and masked, `antsAffineInitializer`,
then `t1w-mni_registration_precise_000.json` in single precision on fMRIPrep's
eight threads (`tools/ants_fmriprep.sh`).  Both pipelines start from T1Prep's
bias-corrected, brain-extracted image.

* With `--fmriprep`, T1Prep is more accurate on both label sets for every
  subject, at a fraction of ANTs' time.
* Most of the lead is the overlap of the grey-matter sheet, not where parcels
  lie along it: with the labels smoothed by 6 mm FWHM, the lead on
  Neuromorphometrics' cortex is +0.016 instead of +0.040 (pairs).
* T1Prep's comparison includes its segmentation: the shooting matches tissue
  maps, ANTs matches T1w intensities.
* An earlier version of this page ran ANTs on a 1.5 mm grid and reported a
  lead of +0.056 / +0.081; registering at 1 mm, as sMRIPrep does, accounts
  for nearly all of the difference.

---

## Surface registration — Mindboggle-101, 19 subjects

The 19 subjects FreeSurfer 8.1 processed, scored on a 164k mesh with the labels
attached to the midthickness (FreeSurfer) or the central surface (T1Prep), the
same geometry.  T1Prep and newMSM register T1Prep's surfaces, FreeSurfer its
own.

| method | LOO | pairs | time / hemisphere |
|---|---|---|---|
| FSL newMSM (fMRIPrep's MSMSulc config) | 0.8391 | 0.7751 | +4 min |
| **T1Prep** (relaxed sphere, Spherical Demons → fsaverage) | **0.8507** | **0.7892** | ~50 s |
| FreeSurfer 8.1 (`mris_sphere`, `mris_register`) | 0.8516 | 0.7906 | ~12 min |

T1Prep and FreeSurfer are level (T1Prep − FreeSurfer −0.001 [−0.009, +0.011]
LOO); T1Prep leads newMSM by +0.012 [+0.009, +0.014] (18 of 19 subjects).
newMSM runs as sMRIPrep runs MSMSulc, from T1Prep's sphere and `sulc`, and its
time comes on top of T1Prep's.  The attachment surface matters more than most
method differences (midthickness − pial: +0.034 LOO for T1Prep, +0.032 for
FreeSurfer), so it is held identical across methods.

---

## Provenance

* **Data.** Mindboggle-101 (CC-BY; Klein & Tourville 2012), the 20 subjects of
  `../data/subset20.txt`.  Neuromorphometrics: the labelled OASIS images of the
  MICCAI 2012 Multi-Atlas Labeling challenge (Landman & Warfield 2012); their
  qform lies 128 mm from their sform, so images and labels are rewritten with
  qform = sform before ANTs reads them.  VBM: OpenNeuro ds000228 (3.5-12
  years; Richardson et al. 2018) and ds005270 (23-85 years).
* **ANTs as sMRIPrep runs it** (`d20_ants_fmriprep.csv`,
  `n35_ants_fmriprep.csv.gz`): ANTsPy 0.6.3, which builds ANTs 2.6.5 with ITK
  5.4.5 (its `--version` says 2.5.1; fMRIPrep pins ANTs 2.6); the truncation
  re-implemented in NumPy.  Time: three subjects at eight threads, 589-599 s,
  28 s of it the initialiser.
* **ANTs SyN (MI)**: ANTsPy (`tools/run_ants_antspy.py`).  **CAT12**:
  `cat_batch_cat.sh -ns -p 4`.  **FreeSurfer 8.1**: recon-all, scored through
  its own `?h.sphere.reg`.  **newMSM**: FSL 6.0.7.23 (`tools/build_newmsm.sh`)
  with sMRIPrep's config and reference surfaces (`tools/msm_data/`).
* **T1Prep**: the `--fmriprep` arms are T1Prep's own runs (after 0.8.1 on
  Mindboggle, 0.8.2 on Neuromorphometrics, CPU), whose default that setting
  was at the time; the default arms re-run the shooting on the same maps
  (`tools/run_volshoot.py`), which reproduces T1Prep's own output to 2e-4 per
  region.  Surfaces: 0.8.1 with the relaxed sphere and CAT-Surface #34, as the
  next release runs them.  VBM: T1Prep 0.8.2 per setting, SPM12 with CAT12's
  factorial design; that analysis is not part of this repository.
* **Files**: per-region, per-comparison Dice of every method in `d20_*.csv`,
  `d19_*.csv` and `n35_*.csv.gz`; the figure from `tools/plot_dice.py`.

<!---
Superseded results, kept for later reading.

Superseded on 2026-10-30, kept for later reading
(`d20_ants_fmriprep_grid15.csv`).  ANTs ran fMRIPrep's command line on the
1.5 mm evaluation grid: fixed image Template_05mm_bet, nearest-neighbour
sampled onto it; no truncation, centre-of-mass start, double precision;
ANTs 2.6.5 binaries.  That fixed image matters at 1.5 mm: TemplateFlow's
template resampled onto the same grid by cubic interpolation is less sharp
and scores 0.011 lower (0.7024 LOO).

| **ANTs — fMRIPrep's exact config** | 0.7133 | 0.5854 | ~9 min |
| T1Prep's shooting − ANTs (fMRIPrep config) | +0.056 [+0.050, +0.063] | +0.081 [+0.073, +0.089] | 20/20 subjects, 190/190 pairs |

T1Prep is **more accurate than that configuration** (+0.056 LOO, +0.081 pairs)
at a fraction of the runtime: ~35 s of shooting versus ~9 minutes
of CC-metric optimisation.  deepmriprep's network, which T1Prep ran up to
0.8.1, was ahead of it by +0.016 / +0.023.

* DKT31 is cortical parcels only. Nothing here measures subcortical alignment,
  which fMRIPrep users also depend on.

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

Same T1Prep 0.8.1 central surfaces and the same scoring; only the sphere
differs, and in the last row Spherical Demons as well (CAT-Surface #34: the
global rotation applied to the sphere, not to the warp):

| sphere | LOO | pairs | Δ LOO [95 % CI] | Δ pairs [95 % CI] | better |
|---|---|---|---|---|---|
| inflation only (T1Prep ≤ 0.8.0) | 0.8120 | 0.7454 | | | |
| + fold removal (T1Prep 0.8.1) | 0.8119 | 0.7454 | −0.0000 | +0.0000 | |
| + `-relax` (cat-surf 1.1.5) | 0.8445 | 0.7845 | +0.0326 [+0.0273, +0.0375] | +0.0391 [+0.0315, +0.0460] | 20/20 subjects, 187/190 pairs |
| **+ `-relax`, #34** (next release) | **0.8488** | **0.7914** | **+0.0368** [+0.0323, +0.0412] | **+0.0460** [+0.0396, +0.0517] | 20/20 subjects, 190/190 pairs |

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

* **The gain is systematic.**  Every subject improves under LOO (+0.021 to
  +0.056 with #34, +0.005 to +0.051 without), the right hemisphere about
  twice as much as the left (+0.047 vs +0.026).  It is concentrated in the
  right lateral temporal lobe: middle temporal +0.229 (median 0.64 → 0.87,
  every subject better), inferior temporal +0.179, transverse temporal
  +0.091, then postcentral +0.080.  That matches the whole-fold slip (~14°)
  of the right inferior temporal gyrus which comparing T1Prep's registration
  with recon-all's had traced to the inflated sphere.
* **One local failure is left.**  The two registrations differ by 2° (median
  per hemisphere; 95th percentile 8°), and in a few hemispheres by a whole
  fold.  Mostly that is the correction above, but with cat-surf 1.1.5 two of
  the 40 went the wrong way: OASIS-TRT-20-16 rh lost its parahippocampal and
  entorhinal labels (LOO 0.72 → 0.00 and 0.68 → 0.15), MMRR-3T7T-2-2 lh its
  pre- and postcentral (0.84 → 0.32 and 0.61 → 0.23).  The first was a defect
  in Spherical Demons, not in the sphere: it folded its global rotation into
  the warp, whose elastic prior undid the rotation wherever the data term is
  weak, and the relaxed sphere happened to start that gyrus 13° into the
  template's medial wall.  With #34 it is 0.77 and 0.64.  MMRR-3T7T-2-2 lh
  stays (0.54 and 0.31): its central and postcentral sulci are equally deep,
  on FreeSurfer's pial surface as well, so it is registered correctly only
  from starts at most 2-6° anterior of the right position (other hemispheres
  recover from 19°), and the relaxed sphere starts it 11° anterior.  No
  region's mean drops any more: the lowest is rh pericalcarine at −0.001.
* **Fold removal alone changes no Dice** (no region's mean by more than
  0.002).  Its value is a valid map, and even that does not survive
  registration without `-relax`: Spherical Demons folds two of the 40 spheres
  again (188 and 264 triangles, about half of them the ones fold removal had
  straightened).  Fold removal leaves those regions crushed, with triangles
  500-2000× smaller than the median one, and the warp flips them.  With
  `-relax` none is folded.
* **The fs_LR spheres for fMRIPrep gain as much.**  `fsLR` (`desc-reg`, the
  fsaverage registration carried into fs_LR) goes from 0.8167 / 0.7452 to
  0.8515 / 0.7906, `msm` (`desc-msmsulc`, the sphere fMRIPrep resamples to
  fs_LR with) from 0.8211 / 0.7519 to 0.8536 / 0.7926, every pair better
  (0.8480 / 0.7839 and 0.8546 / 0.7936 with cat-surf 1.1.5).
* **Nothing else moved.**  T1Prep 0.8.1 without `-relax` reproduces the 0.7.x
  run shown here before (fsaverage LOO −0.003, pairs ±0.000; 164k ±0.001),
  so the gain is the sphere's alone.

#### newMSM from the unrelaxed sphere

| newMSM inputs | LOO | pairs |
|---|---|---|
| `sulc` from cat-surf < 1.1.1, sphere without `-relax` (shown until October 2026) | 0.7317 | 0.6405 |
| correct `sulc`, sphere without `-relax` | 0.7998 | 0.7256 |
| correct `sulc`, relaxed sphere | 0.8366 | 0.7705 |
| correct `sulc`, relaxed sphere, affine start from the #34 fs_LR sphere | 0.8400 | 0.7763 |

Correcting the map gains +0.068 (LOO) and +0.085 (pairs), with every subject
and every pair better; T1Prep's own registrations moved by no more than 0.004
between the two T1Prep versions involved, so the gain is the map's.  The
relaxed sphere then adds +0.037 [+0.032, +0.042] and +0.045 [+0.038, +0.052],
as much as it gives Spherical Demons, and #34 another +0.003 / +0.006 through
the fs_LR sphere newMSM's affine start is regressed onto.

#### FreeSurfer against T1Prep without `-relax` (19 subjects, 164k)

| attachment surface | FreeSurfer 8.1 | T1Prep 0.7.6 |
|---|---|---|
| midthickness / central | 0.8516 / 0.7906 | 0.8143 / 0.7427 |
| pial | 0.8197 / 0.7425 | 0.7900 / 0.7066 |

With the unrelaxed sphere FreeSurfer led by ~0.037 (LOO) and ~0.048 (pairs),
and the margin held whichever attachment surface was chosen (+0.030 to
+0.047).  T1Prep's mid − pial difference was +0.0243 / +0.0362.
-->
