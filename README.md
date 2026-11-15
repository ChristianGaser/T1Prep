[![Python 3.9 - 3.14](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11|%203.12|%203.13|%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache&logoColor=white)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ChristianGaser/T1Prep?display_name=tag&include_prereleases)](https://github.com/ChristianGaser/T1Prep/releases)
<!--
[![Tag](https://img.shields.io/github/v/tag/ChristianGaser/T1Prep?sort=semver)](https://github.com/ChristianGaser/T1Prep/tags)
-->
> [!WARNING]
> This project is **still in development** and might contain bugs. **If you experience any issues, please [let me know](https://github.com/ChristianGaser/T1Prep/issues)!**

<img src="T1Prep_logo.svg" alt="T1Prep logo" width="340"> 

# T1Prep: T1 PREProcessing Pipeline (aka PyCAT) 
## Table of Contents

- [What T1Prep does](#what-t1prep-does)
- [Requirements](#requirements)
- [Main Differences to CAT12](#main-differences-to-cat12)
- [Installation](#installation)
- [Running T1Prep](#running-t1prep)
- [Tools](#tools)
- [Documentation](#documentation)
- [Telemetry](#telemetry)
- [Support](#support)
- [License](#license)

---

## What T1Prep does

T1Prep is a pipeline that preprocesses T1-weighted MRI data and supports segmentation and cortical surface reconstruction. It provides a complete set of tools for efficiently processing structural MRI scans.

T1Prep partially integrates [DeepMriPrep](https://github.com/wwu-mmll/deepmriprep), which uses deep learning (DL) techniques to mimic CAT12's functionality for processing structural MRIs. For details, see:
Lukas Fisch et al., "deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks," available on arXiv at https://doi.org/10.48550/arXiv.2408.10656.

T1Prep runs DeepMriPrep's segmentation networks and the skull-stripping networks of [deepbet](https://github.com/wwu-mmll/deepbet), DeepMriPrep's brain extraction, and follows DeepMriPrep's sequence of processing steps. Since version 0.9.0 it does so with its own code (the networks are kept as safetensors files, converted from DeepMriPrep's), so neither package has to be installed.

An alternative approach (`--amap`) uses the DeepMriPrep segmentation to guide the bias field correction and the lesion detection, and as the initial estimate for the subsequent AMAP segmentation from CAT12.

The non-linear registration to MNI152NLin2009cAsym is not deepmriprep's warp network but CAT12's optimised geodesic shooting, ported to C in [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) (`cat_surf.vol_shoot`). It carries manual cortical labels between subjects more accurately than the network: with CAT12's own setting, the default, as accurately as CAT12 itself, and with the more flexible setting of `--fmriprep` more accurately than CAT12 (see [the evaluation](evaluation/results/README.md)).

Cortical surface reconstruction and thickness estimation use the algorithms from [CAT-Surface](https://github.com/ChristianGaser/CAT-Surface) (a core component of the [CAT12 toolbox](https://github.com/ChristianGaser/cat12)) via the [`cat-surf`](https://pypi.org/project/cat-surf/) Python package, which provides Python bindings to the CAT-Surface C library as pre-built wheels — no separate CAT-Surface binaries have to be installed.

It is designed for both single-subject and batch processing, with optional parallelization and flexible output naming conventions. The naming patterns are compatible with both 
CAT12 folder structures and the BIDS derivatives standard.

## Requirements
 [Python 3.9-3.14](https://www.python.org/downloads/) is required (3.10+ recommended). `pip install T1Prep` installs all necessary libraries; in a source checkout `scripts/T1Prep` creates the environment `env/` and installs them the first time it is run, or when called with the flag "--install".

> **Why prefer 3.10+?** Python 3.9 works, but PyTorch publishes no wheels for it after 2.8, so a 3.9 install is pinned to PyTorch 2.8. The GPU (MPS) kernels T1Prep relies on — `max_pool3d_with_indices`, `avg_pool3d` and `grid_sampler_3d` — only arrived in PyTorch 2.9; on 2.8 they silently fall back to the CPU. On Apple Silicon that is worth a measurable amount: one subject takes **3:02 min on Python 3.9 / PyTorch 2.8 versus 2:39 min on Python 3.12 / PyTorch 2.13**. On Linux/CUDA/CPU the two stacks perform the same, so 3.9 is a fine choice there.

## Main Differences to CAT12
- Implemented in Python, with the CAT-Surface C library coming as the pip-installed `cat-surf` package: no Matlab license and no separately installed binaries are needed.
- Newly developed pipeline to estimate cortical surface and thickness.
- Skull-stripping and segmentation use the networks of DeepMriPrep (and of deepbet, its brain extraction)
- Non-linear spatial registration is CAT12's geodesic shooting (ported to C), run with CAT12's own setting; `--fmriprep` runs every scheduled iteration with half of CAT12's regularisation, which aligns labels better but is not meant for VBM
- Longitudinal processing implements CAT12's `plasticity`, `ageing` and `both`
  models and writes CAT12's own `mwp1r*` / `mwmwp1r*` outputs, but it is
  experimental: the subject average is not segmented and there is no joint bias
  correction (see [docs/usage.md](docs/usage.md) and
  [scripts/README.md](scripts/README.md)).
- Only T1 MRI data supported.

## Installation

T1Prep is on PyPI and needs nothing but Python:

```bash
python3 -m pip install T1Prep
```

That places every command — `T1Prep` (aka `PyCAT`), `t1prep-gui`, `t1prep-run`,
`CAT_SurfView`, `CAT_VolView` and the helpers — into the active environment's
`bin/`. Model
weights are fetched on the first run (or ahead of time with
`t1prep-download-models`). `pip install "T1Prep[tinygrad]"` (Python 3.11+)
adds tinygrad as a second backend for the networks; see
[docs/usage.md](docs/usage.md#neural-networks).

A source checkout, Windows via WSL, a manual install and the Docker image are
described in **[docs/installation.md](docs/installation.md)**.

## Running T1Prep

```bash
T1Prep file.nii.gz                              # segmentation + surfaces
T1Prep --out-dir out/ --multi 4 sub-*/anat/*.nii.gz   # a batch, in parallel
t1prep-run --out-dir out/ file.nii.gz           # the single-subject Python entry
```

```python
from t1prep import run_t1prep
run_t1prep("/path/to/sub-01_T1w.nii.gz")
```

The options, the output folder structure, the naming conventions, worked
examples and the experimental longitudinal pipeline are in
**[docs/usage.md](docs/usage.md)**.

> `T1Prep` is the bash orchestrator (full features including `--multi` batch
> parallelism); `PyCAT` is the same script under a second name and differs
> only in the startup banner; `t1prep-run` is the equivalent single-subject Python entry. Add
> the environment's `bin/` to `PATH` and you never need to call anything from
> the source `scripts/` folder.

## Tools

Installed alongside the pipeline, and usable on their own:

| Command | What it is |
|---------|------------|
| `CAT_SurfView` | Surface viewer: overlays, atlases, cluster tables, figures |
| `CAT_VolView` | Volume viewer: three orthogonal slices, overlays, montages |
| `t1prep-gui` | Desktop launcher: choose images, run the pipeline, browse the results |
| `CAT_SurfResampleMulti_ui`, `CAT_SurfParameters_ui`, `CAT_Surf2ROIMulti_ui` | Command-line helpers for surface post-processing: resampling and smoothing, surface parameters, ROI values |
| `CAT_VolDiff`, `CAT_VolSmooth_ui` | Voxel-wise volume differences (`diff_*` / `diffrel_*`); volume smoothing |
| `CAT_PlotHistogram` | Histograms of volumes, surfaces or text data |
| `CAT_PlotBoxplot` | Box and violin plots of volumes, surfaces or text data |
| `CAT_GrepJson` | Print a field of the JSON reports, e.g. the TIV of every subject |
| `t1prep-bbreg` | BOLD reference and boundary-based BOLD-to-T1w registration for fMRIPrep |
| `t1prep-download-models`, `t1prep-make-apps` | Fetch the model weights ahead of time; build the macOS apps of the viewers |

```bash
CAT_SurfView lh.thickness.sub-01     # surface with an overlay
CAT_VolView T1.nii.gz p1T1.nii.gz    # two linked volume windows
```

Both viewers also render figures without opening a window, so they can be used
from a script. See **[docs/viewers.md](docs/viewers.md)** for them and
**[docs/tools.md](docs/tools.md)** for the rest.

## Documentation

| Document | Covers |
|----------|--------|
| [docs/installation.md](docs/installation.md) | pip, source checkout, WSL, manual install, Docker |
| [docs/usage.md](docs/usage.md) | Running the pipeline: options, outputs, naming, examples |
| [docs/viewers.md](docs/viewers.md) | `CAT_SurfView` and `CAT_VolView`, interactive and batch |
| [docs/tools.md](docs/tools.md) | Launcher, plot tools and the post-processing helpers |
| [scripts/README.md](scripts/README.md) | The scripts in a source checkout |
| [ENVIRONMENT_USAGE.md](ENVIRONMENT_USAGE.md) | The bundled virtual environment |
| [Agents.md](Agents.md) | Contributing: layout, style, tests |

## Telemetry
Unless you opt out, T1Prep tells its developers how many files a run
processed and how many failed, and where in the code a segmentation crashed.
File names, paths, images and host names are never sent. `--no-track`,
`track=0` in the defaults file, or `T1PREP_NO_TRACK=1` / `DO_NOT_TRACK=1` in
the environment switch it off; details in
[docs/usage.md](docs/usage.md#telemetry).

## Support
For issues and inquiries, contact [me](mailto:christian.gaser@uni-jena.de).

## License
T1Prep is distributed under the terms of the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) 
as published by the Apache Software Foundation.

The networks of [DeepMriPrep](https://github.com/wwu-mmll/deepmriprep) and
[deepbet](https://github.com/wwu-mmll/deepbet) that T1Prep runs, and the code
ported from the two packages, are under the MIT License of the Medical Machine
Learning Lab, University of Münster; see [NOTICE](NOTICE).
