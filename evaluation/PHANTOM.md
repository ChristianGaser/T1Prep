# Phantom evaluation

`tools/eval_phantom.py` scores T1Prep against a simulated brain whose
tissue label is known exactly. It is a **manual test**: a run takes about
15 minutes, so it is not part of `pytest`. The helpers it relies on are
covered by `tests/test_eval_phantom.py`, which does run in the normal suite.

```bash
python evaluation/tools/eval_phantom.py run --work /tmp/phantom      # ~15 min
python evaluation/tools/eval_phantom.py check --work /tmp/phantom    # exit 1 on regression
```

## The phantom

`data/phantom/` holds one anatomy, HR075 MPRAGE, re-rendered by
[mri_simulate](https://github.com/ChristianGaser/T1-MRI-Phantom) 0.10.2:

| file | contents |
|---|---|
| `…_desc-snr25Rf45T4Wmh2_T1w.nii.gz` | 0.75 mm, PIL orientation, whole head. Rician noise at WM SNR 25, RF bias field strength 45 (type 4), 34 WMHs (grade 2). |
| `…_desc-Wmh2Clean_dseg.nii.gz` | Continuous partial-volume label in T1Prep's `p0` convention: 1 CSF, 2 GM, 3 WM, 4 WMH; 2.5 is half GM, half WM. |

The label is *cleaned*: vessels and dura (GM-labelled tissue more than 4 mm
from WM) are removed from it. The image is rendered from the uncleaned
fractions, so it still shows them. That is deliberate: a segmentation should
reject them.

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
| WMH | soft Dice, Dice at `p4 > 0.1`, lesions detected, volume, mass inside lesions | `p4` against the WMH fraction (label − 3) |
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

## Baseline

T1Prep 0.7.5 at `b9e6f25` with the fixes below, cat-surf 1.0.29, pinned from
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
| GM > 2 mm from true GM | 0.7 ml | report TIV | −1.0% |
| WMH soft Dice | 0.23 | lesions found (`p4 > 0.1`) | 18 of 34 |
| WMH volume | 2.4 ml (true 6.6) | `p4` inside lesions | mean 0.15 |
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

Between the first run and this baseline, the median thickness of this subject
moved by about +0.1 mm (lh 2.22 → 2.32 mm). Two things changed in between:
cat-surf 1.0.23 → 1.0.29, which brought back the shared barrier reference,
and the hemisphere clip. Neither has been measured alone. Steps of that size
are why the pins record the cat-surf version.
