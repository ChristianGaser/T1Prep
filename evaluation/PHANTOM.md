# Phantom evaluation

`tools/eval_phantom.py` scores T1Prep against a simulated brain whose
tissue label is known exactly. It is a **manual test**: a run takes about
15 minutes, so it is not part of `pytest`. The helpers it relies on are
covered by `tests/test_eval_phantom.py`, which does run in the normal suite.

```bash
make phantom                       # run + check, in /tmp/T1Prep_phantom (~15 min)
make phantom-check                 # check an existing run again
make phantom-pin                   # re-pin: default device and CPU (~30 min)
make phantom PHANTOM_DIR=DIR PHANTOM_ARGS=--history

python evaluation/tools/eval_phantom.py run --work DIR     # the same, directly
python evaluation/tools/eval_phantom.py check --work DIR   # exit 1 on regression
```

## The phantom

`data/phantom/` holds one anatomy, HR075 MPRAGE, re-rendered by
[mri_simulate](https://github.com/ChristianGaser/T1-MRI-Phantom) 0.10.2:

| file | contents |
|---|---|
| `…_desc-snr25Rf45T4Wmh2_T1w.nii.gz` | 0.75 mm, PIL orientation, whole head. Rician noise at WM SNR 25, RF bias field strength 45 (type 4), 34 WMHs (grade 2). |
| `…_desc-Wmh2Clean_dseg.nii.gz` | Continuous partial-volume label in T1Prep's `p0` convention: 1 CSF, 2 GM, 3 WM, 4 WMH; 2.5 is half GM, half WM. |

The label is *cleaned*: vessels and dura (GM-labelled tissue more than 4 mm
from WM, measured through the tissue) are relabelled as CSF. The image is
rendered from the uncleaned fractions, so it still shows them. That is
deliberate: a segmentation should not call them GM.

The tissue fractions come from SPM12's unified segmentation of the real scan,
plus CAT's cleanup. They do not come from T1Prep. **HR075 must stay out of
any training set** (`deepmriprep-train` can consume `mri_simulate` output),
or this test measures memorisation.

## What is measured

| group | metrics | what the ground truth allows |
|---|---|---|
| tissue | Dice (hard, soft), volume error per class, label MAE | `p0` against the label, with WMH counted as WM |
| report | TIV and CGW volume error | the volumes the JSON report states |
| boundaries | `shift_white_mm`, `shift_pial_mm` | mean displacement of the 2.5 and 1.5 iso-surfaces (volume difference / true area; positive = outward) |
| rejection | `fp_gm_far_ml`, `fp_brain_far_ml`, `missed_brain_ml` | GM more than 2 mm from true GM (dura, vessels, sinus); brain outside the true brain; missed tissue |
| WMH | lesion F1, false clusters, voxel precision, flagged volume, Dice at `p4 > 0.1` and `> 0.5`, soft Dice, volume | `p4` against the WMH fraction (label − 3); see *WMH across a set of simulations* for the grade/noise grid |
| bias | WM CV, low-frequency WM CV, CJV of `m` | uniformity of the corrected image; low-frequency CV ignores noise and denoising |
| QA | NCR, ICR, IQR (with the true noise next to them) | NCR is measured after 3×3×3 block averaging (CAT12's design), so compare it with `true_ncr_qa_res`, not with the per-voxel noise |
| thickness | paired median difference, MAE, r, central-surface distance, DK40 regional bias and Spearman | against the ground-truth surface arm, see below |
| topology | `defects_lh/rh` = \|χ − 2\| / 2 of the hemisphere label | lower is better |

The run also has to pass a set of **invariants**, whatever is pinned:

- `p0` is finite and within [0, 4].
- The output lines up with the ground truth (GM soft Dice ≥ 0.6).
- Both hemisphere labels stay within [1, 3].
- Thickness is finite and ≥ 0 at every vertex.
- The ground-truth arm is registered to the run and uses the run's partition.

### The thickness reference

Thickness is not simulated, so it has no ground truth. The surface pipeline
therefore also runs on **hemisphere labels built from the ground truth**, and
the two arms are compared vertex by vertex (nearest vertex between the central
surfaces, where they are less than 1 mm apart).

The hemisphere labels carry more than tissue. `get_partition` fills the deep
grey nuclei and ventricles with WM and clears the other hemisphere, the
cerebellum and the brainstem. For the arms to differ in the tissue label alone,
the ground-truth arm has to reuse the run's own partition:

1. **A hook records the partition.** During the T1Prep run a
   `sitecustomize` hook re-runs `get_partition` on the label lowered by 1e-4.
   After that shift, a fill is the only way a voxel can reach exactly 3.0, and
   a value of exactly 1.0 over tissue can only be a clearance. Both masks are
   written through the same reslice as `?h.seg`. The hook never changes the
   pipeline's own result, and a `fill_agreement` of 1.0 confirms the shift
   changed no fill.
2. **The ground-truth hemispheres are built from the masks.** They are made
   on the grid of the pipeline's `?h.seg`: WM inside the fills, CSF where the
   partition cleared tissue, the ground-truth label elsewhere.
3. **The surface pipeline runs on them** with the settings in
   `T1Prep_defaults.txt`.

The ground truth cannot simply be re-partitioned on the working grid. That
grid lives in the affinely registered space, and its affine does not place
native data: a ground truth resampled onto it correlated with `p0` at 0.21.

What this measures is the thickness error **the segmentation causes**. PBT's
own bias against a known thickness is a separate question. The shell
phantoms in the CAT-Surface tests and `thickness_phantom.m` in
T1-MRI-Phantom are the place for that.

## Pinning

`phantom_pinned.json` records what the code did when it was pinned. The
values are characterization, not targets.

- **Tolerance.** Each metric gets the largest of: 3× the spread between the
  pinned runs, an absolute floor, and a relative floor (`METRICS` in the
  script).
- **Direction.** Each metric also has one. A `higher` or `lower` metric fails
  only when it gets worse; an improvement is reported and fails only under
  `--strict`. A `both` metric fails on any change: a signed bias that moves is
  a behaviour change either way.
- **Model hash.** The pins store a hash of the model weights. `check` refuses
  to compare across different weights (exit 3) unless `--ignore-models` is
  given.

To re-pin after an intended change, run on two devices so the tolerance has
a measured spread:

```bash
python evaluation/tools/eval_phantom.py run --work /tmp/ph_mps
python evaluation/tools/eval_phantom.py run --work /tmp/ph_cpu --device cpu
python evaluation/tools/eval_phantom.py pin --work /tmp/ph_mps /tmp/ph_cpu
```

`run --history` also appends each run's scalars to
`results/phantom_history.csv`, for the long view.

## Reading the numbers

Absolute differences are partly a matter of **convention**. The ground truth
is SPM's idea of where GM ends; T1Prep's network learned CAT12's. A constant
offset between the two, for example the pial surface sitting 0.13 mm outward,
is not necessarily an error. What the pins catch is **change**.

For an absolute accuracy statement, compare against a clean simulation of the
same anatomy (no noise, no bias field, no WMH). That clean run measures the
convention offset, and each degradation is then read on top of it.

## What the numbers have shown so far

- **The CSF deficit is a sulcal one, not the brain mask.** The details split
  the CSF error into zones: `csf_err_rim_ml` +2.8, `csf_err_ventricles_ml`
  −6.6, `csf_err_sulci_ml` −36.8. In the sulci it goes to GM (35 ml). About
  4–8 ml of that is vessels, which the ground truth calls CSF (mri_simulate
  relabels vessels and dura as CSF).
  - Switching the dura removal off raised the CSF volume from −11.2% to −5.6%,
    but only by leaving 23 ml of extra CSF in the rim, outside the true
    intracranial CSF. That surplus masked half of the sulcal deficit.
- **`p4` was not a probability (fixed, see below).** Before 0.7.6 it was the
  WM-intensity deficit times 3 (`p3 + 2·p1 + 3·p2 − 3·m`): 0.15 inside the
  lesions, so `p4 > 0.5` found nothing, and its sum, the reported volume,
  measured contrast rather than extent.

## WMH across a set of simulations

`eval_phantom.py wmh` runs T1Prep (volumes and lesions only, ~4 min an image)
on every simulated T1w in an mri_simulate folder, pairs each image with the
label it was rendered from (`…desc-snr50Rf90T4Wmh4_T1w` ↔ `…desc-Wmh4Clean_dseg`,
no `Wmh` ↔ `…desc-Clean_dseg`), and scores the WMH map per image. The set is
not in git; point `--sims` or `T1PREP_PHANTOM_SIMS` at it.

```bash
make phantom-wmh PHANTOM_SIMS=~/Dropbox/derivatives/mri_simulate-0.10.2   # ~1 h for 14 images
make phantom-wmh-pin PHANTOM_SIMS=...                                     # re-pin
```

Scores (pinned in `results/phantom_wmh_pinned.json`):

- **Detection:**
  - lesions found: a true lesion counts when a reported lesion overlaps its
    core;
  - false clusters: reported lesions that touch no true WMH;
  - F1 over the two.
- **Extent:**
  - voxel precision of the flagged region;
  - flagged volume on the WMH-free images.

  Counts alone reward flagging a lot of white matter, because a large enough
  mask covers every lesion.
- **Calibration:**
  - Dice at `p4 > 0.5`;
  - soft Dice against the WMH fraction;
  - the error of the reported WMH volume.

The set used: one anatomy (HR075), SNR 25/50 × bias 45/90 × WMH none/grade
2/grade 4, plus SNR 25 without bias and a noise-free, bias-free, WMH-free
rendering. That is 14 images and 372 lesions.

### The calibrated lesion map (0.7.6)

`p4` is now a probability, computed in `t1prep._lesions` in four steps:

1. **Signal.** Take the deficit above and subtract its local level in deep
   WM: a 5 mm Gaussian over voxels whose deficit is below 0.2. Lesions are
   kept out of the level, so a confluent lesion cannot subtract itself.
2. **Search region.** Leave out the deep grey nuclei (affinely placed
   Neuromorphometrics atlas).
3. **Probability.** A logistic model of the signal and the log WMH prior:
   `logit p = -3.905 + 23.749·signal + 0.959·log(prior + 0.01)`.
4. **Lesions.** Keep connected components of `p > 0.1` of at least 30 mm³.

The report's WMH volume is the volume of the kept lesions; the sum of `p`
underestimates it by 4.7 ml. That volume is part of WM, not added on top:
the WM map already contains the lesions (WM probability 0.995–0.999 inside
them), and 0.7.5 counted them twice.

| 14 simulations | 0.7.5 | 0.7.6 |
|---|---|---|
| lesions found | 286/372 (77%) | 232/372 (62%) |
| false clusters, lesion / WMH-free images | 55 / 125 | 51 / 83 |
| voxel precision of the flagged region | 0.43 | **0.72** |
| flagged volume, WMH images (true 6.6–9.1 ml) | 33.8 ml | 10.4 ml |
| flagged volume, WMH-free images | 22.4 ml | **2.8 ml** |
| Dice at `p4 > 0.5` / soft Dice | 0.00 / 0.29 | **0.54 / 0.46** |
| reported WMH volume, mean error | 4.4 ml (always low) | **1.0 ml** (bias +0.2) |

0.7.5 found more lesions because its lesion mask covered three to five times
the true lesion volume. The operating point is a choice: `fit_wmh_calibration.py
fit` prints the trade-off, for example `p > 0.1` and 10 mm³ gives 263/372 found
for 2.1 ml of false volume.

How the constants came about:

- **What was fitted.** The coefficients and operating point were fitted on the
  same simulations: soft-target logistic regression, reproducible with
  `evaluation/tools/fit_wmh_calibration.py dump|fit`.
- **Held-out folds.** Leaving out one WMH grade or noise level moved the
  signal slope by < 10% and the prior weight between 0.7 and 1.45.
- **Alternatives that lost.** A per-subject z-score lost to the plain
  deficit. A high-pass whose level includes the lesions scored 0.03 F1 higher
  here but erases the centre of any lesion wider than ~10 mm, and this set has
  none that large. Growing lesions from a low threshold (hysteresis) merged up
  to 22 lesions into one.

What this set cannot tell:

- **Periventricular WMH are not tested.** mri_simulate places lesions only in
  eroded WM (≥ 3 mm from the ventricles), and T1Prep's search region excludes
  ~2 mm next to non-WM tissue. Neither is exercised here.
- **Some false positives are anatomy.** Two symmetric false positives, at the
  posterior limb of the internal capsule, appear on every image including the
  noise-free one. The tissue there is darker on T1, and the ground truth
  labels it a GM/WM mixture. They are anatomy, not noise.
- **One anatomy.** Refit with simulations of more brains.
- **`--amap` is untouched.** Its lesion map is still AMAP's excess GM
  probability with the 0.7.5 selection.

## Baseline

T1Prep 0.7.5 at `0bce7ec` with the fixes below and the calibrated WMH map,
cat-surf 1.0.29, pinned from
two runs on an Apple M-series machine: the default device (MPS) and
`--device cpu`. A full run takes 15 min: about 10.5 min for T1Prep, the rest
for the ground-truth arm and scoring.

The two devices produced bit-identical segmentations. By default only brain
extraction and the atlas warp run on MPS; the deepmriprep segmentation
stages stay on the CPU. The thickness metrics still differed by about 0.003,
most likely through the warp's effect on the hemisphere partition. That
difference is the spread their tolerances rest on.

| | | | |
|---|---|---|---|
| GM Dice / soft | 0.930 / 0.887 | GM volume | +2.3% |
| WM Dice / soft | 0.955 / 0.931 | WM volume | +3.4% |
| CSF Dice / soft | 0.832 / 0.796 | CSF volume | −11.2% |
| white-surface shift | +0.07 mm | pial shift | +0.13 mm |
| GM > 2 mm from true GM | 0.7 ml | report TIV | −1.1% |
| WMH soft Dice / Dice at 0.5 | 0.34 / 0.40 | lesions found, false clusters | 18 of 34, 7 |
| WMH volume (report) | 4.8 ml (true 6.6) | voxel precision of the lesions | 0.69 |
| WM low-freq. CV raw → `m` | 0.081 → 0.014 | QA NCR / true at QA resolution | 0.029 / 0.033 |
| thickness paired median | +0.002 mm | thickness MAE / r | 0.13 mm / 0.93 |
| DK40 regional bias | +0.01 mm (Spearman 0.94) | most thinned / thickened | pericalcarine −0.26 / insula +0.20 |

## Problems this test found

The first run turned up the following. Each is fixed, and each has a test:

| problem | fix |
|---|---|
| DK40/Destrieux region names in the native annots carried heap bytes | CAT-Surface annot reader and writer (cat-surf 1.0.29), the four templates rewritten; see [`results/cat_surface_annot_bug.md`](results/cat_surface_annot_bug.md) |
| with `--lesions`, `?h.seg` reached 3.8 (WMH values leaked into the surface label) | `get_partition` clips its input to [0, 3] |
| the environment ran cat-surf 1.0.23 against `>=1.0.27`, so the shared sulcal-barrier gate was off | reinstalled from source (1.0.29) |
| `t1prep.cat_surf` lacked 14 public cat-surf functions, `vol_pbt_barrier_reference` among them | it now re-exports everything cat-surf exports |
| `get_volume_native_space` gave 0 for permuted and a negative volume for flipped affines | uses \|det\| of the affine |
| the help text and `docs/usage.md` named the lesion map `p7…`; it is `p4…` | corrected |
| the report said "Use MPS in Python" under `T1PREP_DEVICE=cpu` | `scripts/T1Prep` reports the device `resolve_device()` picks, plus `T1PREP_DEVICE` |
| `p4` was 3× the intensity deficit, not a probability; its lesion mask flagged ~22 ml per lesion-free brain | calibrated probability with deep-GM exclusion and a size-limited threshold (`_lesions.py`) |
| the report added the WMH volume to a WM map that already contained it | WMH reported as part of WM |

Between the first run and this baseline, the median thickness of this subject
moved by about +0.1 mm (lh 2.22 → 2.32 mm). Two things changed in between:
cat-surf 1.0.23 → 1.0.29, which brought back the shared barrier reference,
and the hemisphere clip. Neither has been measured alone. Steps of that size
are why the pins record the cat-surf version.
