# AGENTS

## Overview
This repository contains **T1Prep**, a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data. The project supports tasks such as bias-field correction, segmentation, lesion detection, cortical surface reconstruction, and integration with CAT12. The code lives in the `src` directory, source-tree/dev helper scripts are in `scripts/`, and the Qt launcher lives in `src/t1prep/gui/` on top of `src/t1prep/launch/`. Installed entry points (in the environment's `bin/`): `T1Prep` (bash orchestrator), `PyCAT` (symlink to `T1Prep`, PyCAT banner), `t1prep-gui` (also `T1Prep_gui`), `t1prep-run` (Python single-subject), `CAT_SurfView`, `CAT_VolView`, `CAT_PlotHistogram`, `CAT_PlotBoxplot`, `t1prep-make-apps`, `t1prep-download-models`, `t1prep-bbreg`.

## Project Structure

```
T1Prep/
├── src/
│   └── t1prep/
│       ├── __init__.py             # Package init; exports run_t1prep etc. on first use (no torch at import)
│       ├── t1prep.py               # Python API: run_t1prep() function
│       ├── cli_help.py             # Shared CLI behaviour: synopsis, --help,
│       │                           # --version, "--" options (see below)
│       ├── segment.py              # Volume pipeline (python -m t1prep.segment)
│       ├── surface_estimation.py   # Surface pipeline per hemisphere
│       │                           # (python -m t1prep.surface_estimation)
│       ├── nogm.py                 # Conventional non-cortical GM removal (replaces
│       │                           # the DeepMRIPrep nogm model)
│       ├── dura.py                 # Removes dura left outside the CSF by the skull-strip
│       ├── _atlas.py               # Template lookup, atlas resampling, region masks
│       ├── _intensity.py           # Bias-field fits, global intensity norm, LAS
│       ├── _lesions.py             # WMH detection and tissue-map correction
│       ├── _partition.py           # Hemisphere partition + Euler number
│       ├── _segment_utils.py       # Small shared segmentation helpers
│       ├── _device.py              # Device selection, interpolate()
│       ├── _conv_chunk.py          # Memory-bounded 3-D convolution (CPU inference)
│       ├── vessels.py              # Blood-vessel detection and correction
│       ├── qa.py                   # Image quality measures and ratings
│       ├── report.py               # JSON report writer
│       ├── glued_sulci.py          # Glued (buried) sulci as a surface quality measure
│       ├── thickness_qa.py         # Shape of the thickness distribution (quality measure)
│       ├── bids_derivatives.py     # Sidecars and dataset description for --fmriprep
│       ├── fslr.py                 # Registration of the surfaces to fsLR (--fmriprep)
│       ├── bbreg.py                # BOLD-to-T1w boundary-based registration (t1prep-bbreg)
│       ├── cat_surf/               # Re-export of the cat-surf package as t1prep.cat_surf
│       ├── _shoot.py               # Non-linear registration: geodesic shooting
│       │                           # (cat_surf.vol_shoot, replaces deepmriprep's warp)
│       ├── nn/                     # Neural networks (see "Neural Networks" below)
│       │   ├── weights.py          # safetensors <-> numpy, no framework
│       │   ├── backends/           # Backend base class + registry; torch, tinygrad
│       │   ├── architectures/      # Registry; unet3d (deepmriprep), linknet3d (deepbet)
│       │   ├── network.py          # Network = architecture + weights + backend
│       │   ├── models.py           # The shipped networks: files, download, conversion
│       │   │                       # (t1prep-download-models)
│       │   └── convert.py          # TorchScript/state_dict -> safetensors (+ CLI)
│       ├── stages/                 # T1Prep's port of deepmriprep's stage sequence
│       │   ├── preprocess.py       # Preprocess: skull-strip, affine, segmentation,
│       │   │                       # shooting, atlases -- as segment.py drives them
│       │   ├── skullstrip.py       # deepbet's skull-strip on the networks
│       │   └── segmentation.py     # deepmriprep's segmentation on the networks
│       ├── _telemetry.py           # Telemetry opt-out, crash reports without paths
│       ├── transforms.py           # ITK .txt/.h5 transforms, SPM y_ deformations
│       ├── utils.py                # Shared utilities
│       ├── _names.py               # Data folder and Names.tsv table, without torch (launcher, viewers)
│       ├── metrics.py              # Dice/evaluation metrics + CLI
│       ├── dice.py                 # CLI wrapper for metrics (python -m t1prep.dice)
│       ├── plot_boxplot.py         # Box/violin plots of volumes/surfaces (CAT_PlotBoxplot)
│       ├── plot_histogram.py       # Histograms of volumes/surfaces (CAT_PlotHistogram)
│       ├── realign_longitudinal.py
│       ├── warp_longitudinal.py
│       ├── modulate_longitudinal.py
│       ├── data/                   # Templates, atlases, default files
│       │   ├── Names.tsv                       # Output file names, CAT12 and BIDS
│       │   ├── templates_MNI152NLin2009cAsym/  # Templates, volume atlases (.nii.gz, .csv, .txt)
│       │   ├── atlases_surfaces_32k/           # Surface atlases + .txt descriptions
│       │   └── templates_surfaces_{32k,4k,164k,fsLR}/  # Surface templates
│       ├── launch/                 # What the launcher knows that is not Qt (no VTK either)
│       │   ├── options.py          # Option table -> form and T1Prep arguments
│       │   ├── progress.py         # Phases, bars and per-image states of a run
│       │   ├── results.py          # Files of a subject, report measures, viewer calls
│       │   └── inputs.py           # Finding the images of a folder
│       ├── gui/                    # PySide6/VTK visualization tools
│       │   ├── launcher.py         # Desktop launcher (t1prep-gui): window, T1Prep and
│       │   │                       # viewer processes; imports no VTK
│       │   ├── launcher_widgets.py # Its table of images, options form, results panel
│       │   ├── theme.py            # Its colours and style sheet
│       │   ├── icons.py            # Its icons, drawn in code (no image files)
│       │   ├── cat_surf_view.py    # Surface viewer (CAT_SurfView)
│       │   ├── make_apps.py        # macOS .app bundles (t1prep-make-apps)
│       │   ├── cat_vol_view.py     # Orthogonal volume viewer (CAT_VolView),
│       │   │                       # also embedded in CAT_SurfView's --volume window
│       │   ├── controls.py         # Overlay control panel shared by both viewers
│       │   ├── colormaps.py        # Colormaps and p-value colorbar labels
│       │   └── viewer_common.py    # Event claiming, drop targets, help, screenshots
├── scripts/
│   ├── README.md                   # Comprehensive script documentation
│   ├── T1Prep                      # Main CLI entry point (bash; shipped to <venv>/bin)
│   ├── PyCAT -> T1Prep             # Symlink; same CLI, PyCAT banner (logo() switches on $0)
│   ├── T1Prep_gui                  # Launcher dev wrapper (installed: t1prep-gui, T1Prep_gui)
│   ├── activate_env.sh             # Source to activate the venv
│   ├── run_with_env.sh             # Run a Python script file with env/ (needs env/)
│   ├── make_macos_apps.sh          # macOS .app bundles (installed: t1prep-make-apps)
│   ├── install.sh                  # Bash bootstrapper: download release + set up venv
│   #                                 (alternative to `pip install T1Prep` from PyPI)
│   ├── dice.sh                     # Dice similarity metric wrapper
│   ├── qa_calibrate.py             # Derive QA rating bounds from BrainWeb Phantom runs
│   ├── mem_probe.py                # Per-stage time and memory probe of the volume pipeline (dev)
│   ├── process_longitudinal.sh     # Batch longitudinal processing
│   ├── realign_longitudinal.sh     # Longitudinal rigid realignment
│   ├── warp_longitudinal.sh       # Longitudinal low-dimensional non-linear registration
│   ├── modulate_longitudinal.sh   # Longitudinal Jacobian modulation (VBM maps)
│   ├── CAT_GrepJson                # Extract fields from JSON reports
│   ├── CAT_Surf2ROIMulti_ui        # Surface ROI extraction
│   ├── CAT_SurfParameters_ui       # Curvature-based surface parameters
│   ├── CAT_SurfResampleMulti_ui    # Surface resampling & smoothing
│   ├── CAT_VolSmooth_ui            # Volumetric Gaussian smoothing
│   ├── CAT_VolDiff                 # Volume differences (cat_stat_diff via CAT_VolCalc)
│   ├── parallelize                 # Generic job parallelization engine
│   ├── progress_bar_multi.sh       # Multi-job progress bars with ETA
│   └── T1Prep_utils.sh             # Shared bash utility functions (sourced; shipped to <venv>/bin)
├── evaluation/                     # Registration benchmark + simulated-phantom accuracy test
│   ├── tools/                      # eval_mindboggle.py, eval_phantom.py, fit_wmh_calibration.py + competing tools
│   ├── data/subset20.txt           # subjects used for the volume comparison
│   ├── data/phantom/               # 14 simulated T1w + ground-truth labels (mri_simulate)
│   ├── PHANTOM.md                  # phantom metrics, the thickness reference, pinning
│   └── results/                    # README.md (results), RESULTS_long.md, per-arm CSVs, phantom_pinned.json
├── tests/
│   ├── __init__.py
│   └── test_*.py                   # One module per topic, e.g. test_nn.py, test_stages.py,
│                                   # test_orchestrator_args.py, test_launch_options.py,
│                                   # test_launch_progress.py, test_launch_results.py,
│                                   # test_launcher_gui.py, test_source_install.py
├── Makefile                        # Build/dev tasks
├── requirements.txt                # pip dependencies
├── pyproject.toml                  # Package metadata and dependencies
├── setup.py                        # Legacy setup script
├── README.md                       # Landing page: what it is, install, links
├── README_pypi.md                  # The page shown on PyPI
├── README_python.md                # Python API reference (run_t1prep, t1prep.cat_surf)
├── ENVIRONMENT_USAGE.md            # The project venv and the wrapper scripts
├── docs/
│   ├── installation.md             # pip, source checkout, WSL, manual, Docker
│   ├── usage.md                    # Running the pipeline: options, outputs, examples
│   ├── viewers.md                  # CAT_SurfView and CAT_VolView
│   └── tools.md                    # Launcher, plot tools and the post-processing helpers
├── Dockerfile                      # Container build
├── T1Prep_defaults.txt             # Default CLI options
└── Agents.md                       # This file
```

## File Dependencies and Synchronization

### CRITICAL: Keep These Files in Sync

| When you change... | Also update... |
|--------------------|----------------|
| `requirements.txt` | `pyproject.toml` -> `[project.dependencies]` section |
| `pyproject.toml` dependencies | `requirements.txt` (keep versions aligned) |
| CLI options in `scripts/T1Prep` | `docs/usage.md` -> Options section |
| CLI options in `scripts/T1Prep` | `src/t1prep/t1prep.py` -> `run_t1prep()` parameters |
| CLI options in `scripts/T1Prep` | `src/t1prep/launch/options.py` -> one `Option` in the table (the launcher's form and arguments; `tests/test_launch_options.py` holds it against `parse_args`) |
| `[project.scripts]` / `script-files` in `pyproject.toml` | `README.md`, `README_pypi.md`, `docs/installation.md`, `docs/tools.md`, `scripts/README.md`, `scripts/install.sh`, `Agents.md`, `CLAUDE.md` |
| `src/t1prep/t1prep.py` API | `docs/usage.md` -> Python API section, `README_python.md` -> Signature, `README_pypi.md` -> Key parameters |
| Default values | `T1Prep_defaults.txt` |
| Scripts in `scripts/` (add/remove/rename) | `scripts/README.md`, `Agents.md` -> Project Structure, `CLAUDE.md` |
| Atlas files in `src/t1prep/data/` | Add the `.csv` ROI table (volume atlases) and the `.txt` description file |
| Installation process | `docs/installation.md` (README keeps the one-line quick start) |
| Installation process (bash bootstrapper) | `scripts/install.sh` |
| Installation process (PyPI distribution) | `pyproject.toml`, `README_pypi.md` |
| Docker configuration | `docs/installation.md` -> Docker section |
| Docker configuration | `Dockerfile` |
| Version number | `src/t1prep/__init__.py` (canonical — pyproject.toml derives via setuptools attr, scripts/T1Prep_utils.sh reads it via awk); `make release VERSION=x.y.z` bumps, commits and tags it *before* the GitHub release ("Publish to PyPI" refuses a tag ≠ `__version__`; the Dockerfile is pinned after a successful publish); README badges |

### Core File Relationships

```
scripts/T1Prep (CLI)
       |
       v calls
src/t1prep/segment.py, surface_estimation.py, utils.py, etc.
       |
       v loads
src/t1prep/data/* (templates, atlases)
       |
       v uses
cat_surf (Python package — C-extension bindings to CAT-Surface algorithms)

scripts/CAT_Surf*_ui, scripts/CAT_VolSmooth_ui, scripts/CAT_VolDiff (post-processing helpers)
       |
       v call
cat_surf.cli (Python equivalents of the CAT-Surface command-line tools, in the cat-surf package)

src/t1prep/gui/launcher.py (Qt launcher, `t1prep-gui`)
       |
       v calls                  starts as processes of their own
T1Prep (bash orchestrator)      CAT_VolView (--headless for the pictures), CAT_SurfView

src/t1prep/launch/ (option table, run states, result files; no Qt, used by the launcher)
```

## Development Guidelines
- Use Python 3.9 or newer (supports 3.9-3.14).
- Keep functions small and well documented. Include docstrings for public functions.
- Prefer using the utilities provided in `src/t1prep/utils.py` (and related helpers in `src/t1prep/`) when possible.
- Check documentation of functions and add missing documentation.

### Environment Usage
- See [ENVIRONMENT_USAGE.md](ENVIRONMENT_USAGE.md) for activating the virtual environment and running scripts safely.
- Wrapper scripts of a source checkout:
       - [scripts/activate_env.sh](scripts/activate_env.sh) and [scripts/run_with_env.sh](scripts/run_with_env.sh)
         activate the project venv `env/` and stop when there is none
         (`scripts/T1Prep --install` creates it)
       - [scripts/T1Prep_gui](scripts/T1Prep_gui) (launcher), `scripts/dice.sh` and the longitudinal
         wrappers also work without `env/`: they take the first interpreter that has
         T1Prep's requirements (`$T1PREP_PYTHON` picks one) and put `src/` first on `PYTHONPATH`

## Common Tasks
- Show CLI options: `T1Prep --help` (installed). In a checkout without `env/`, read `usage()` and
  `help()` in `scripts/T1Prep` instead: there `--help` ends with a Y/N prompt offering to build
  `env/`, and a run builds it (several GB) without asking
- Run pipeline (example): `T1Prep --out-dir /tmp/out sub-01_T1w.nii.gz`
- Python API: `from t1prep import run_t1prep`
- Quick Python sanity check: `python -m compileall src`
- Run the launcher from the checkout: `./scripts/T1Prep_gui`

## When to Update the Documentation

The README is a landing page: overview, requirements, a one-line install and
links.  Everything else lives in `docs/`, so a change usually belongs there.

| Change | Document |
|--------|----------|
| CLI options, output structure, naming, examples | `docs/usage.md` |
| Installation, dependencies, Docker | `docs/installation.md` |
| Viewer features or options | `docs/viewers.md` |
| Launcher, surface post-processing GUIs | `docs/tools.md` |
| A new entry point, or what T1Prep *is* | `README.md` as well |

**Always update the documentation when:**
1. Adding/removing/changing CLI options in `scripts/T1Prep`
2. Changing the Python API signature in `src/t1prep/t1prep.py`
3. Adding new features or capabilities
4. Changing installation requirements or process
5. Modifying output folder structure or naming conventions
6. Adding/removing dependencies
7. Changing Docker build or run instructions
8. Fixing important bugs that affect user workflow

**Sections to check:**
- **Options** (`docs/usage.md`): Must match `scripts/T1Prep --help` output
- **Python API** (`docs/usage.md`): Must match `run_t1prep()` signature
- **Installation** (`docs/installation.md`): PyPI path (`pip install T1Prep`)
  is primary; the bash bootstrapper (`scripts/install.sh`) is the secondary
  path for users who want the full source tree.  Keep both in sync.
- **Requirements** (`README.md`): Must match `requirements.txt` / `pyproject.toml`

## Command-Line Conventions

Every tool T1Prep ships — the `T1Prep`/`PyCAT` orchestrator, the viewers, the
`CAT_*_ui` wrappers, `CAT_VolDiff`, `parallelize`, the longitudinal scripts —
answers the same way, with `CAT_VolView` as the template:

| You type | You get |
|----------|---------|
| nothing at all | the synopsis — the command line plus an overview of the options — on stderr, exit 1 |
| `--help` | the full description, exit 0 |
| `--version` | `<tool> <T1PREP_VERSION>`, exit 0 |

Options are spelled `--like-this`. The single-dash spellings that predate this
(`CAT_SurfView -overlay`, `CAT_VolDiff -s`, `parallelize -p`, and `-h`/`-v`)
remain valid so existing command lines and scripts keep working; they are
simply no longer listed in the help. `t1prep-gui`, `t1prep-download-models` and
`t1prep-make-apps` (with its wrapper `scripts/make_macos_apps.sh`) are the deliberate exception to
the no-argument rule — called bare they do their job, because that job needs no input file: open
the window, fetch the weights, write the app bundles to /Applications or ~/Applications. So do not
call `t1prep-make-apps` bare just to see its options.

The behaviour lives in one place per language, so a new tool inherits it
rather than re-implementing it:

- **Python** — `src/t1prep/cli_help.py`. Build the parser with its
  `ArgumentParser` instead of `argparse.ArgumentParser`, and call
  `parser.exit_without_arguments(argv)` before `parse_args`. The class hides
  every single-dash spelling from the synopsis and the option list, adds
  `--version`, and makes a failed parse point at `--help`. Pass
  `RawTextHelpFormatter` from the same module for help strings that are
  hand-wrapped and already name their defaults.
- **Bash** — `print_usage` and `print_version` in `scripts/T1Prep_utils.sh`.
  Define a `usage()` that calls
  `print_usage "<synopsis>" "<option line>" …` — an empty argument separates
  groups — and keep the long text in `help()`. `parallelize` and
  `progress_bar_multi.sh` carry local copies of both on purpose: they are
  meant to run without sourcing `T1Prep_utils.sh`.

## Adding New CLI Options

When adding a new CLI option, update these files in order:

1. `scripts/T1Prep` - Add a `case` label in `parse_args()` (and the option to `takes_value()` if it takes a value), then to `usage()` and `help()`
2. `src/t1prep/t1prep.py` - Add parameter to `run_t1prep()` function
3. `src/t1prep/launch/options.py` - Add one `Option` to the table if the launcher should offer it
4. `T1Prep_defaults.txt` - Add default value if applicable
5. `docs/usage.md` - Document the new option

Give the option a `--` spelling, list it in `usage()` as well as `help()`, and
keep any old single-dash spelling only as a hidden alias.

## Adding New Atlases

When adding atlas files to `src/t1prep/data/`:

1. **Volume atlases** go in `templates_MNI152NLin2009cAsym/`
   - Add `<atlas_name>.nii.gz` (the atlas file)
   - Add `<atlas_name>.csv` (the ROI table, `;`-separated; the ROI volumes are read with it, and
     the launcher lists only atlases that have it)
   - Add `<atlas_name>.txt` (description shown as the atlas's tool tip in the launcher)

2. **Surface atlases** go in `atlases_surfaces_32k/`
   - Add `lh.<atlas_name>.annot` and `rh.<atlas_name>.annot`
   - Add `lh.<atlas_name>.txt` (description shown as the atlas's tool tip in the launcher)

## Neural Networks

`src/t1prep/nn/` holds the networks T1Prep runs: deepbet's LinkNet3d
(skull-strip) and deepmriprep's Unet3d (one coarse and 18 patch segmentation
models). Each architecture is written once, against the operation table of a
backend; the weights are safetensors files, made once from deepmriprep's
TorchScript archives (`models.prepare_model_files`) and kept in the user
cache, in a folder named after `models.MODEL_RELEASE`. `T1PREP_NN_BACKEND`
(`torch`, default, or `tinygrad`) selects the backend: `T1Prep --nn-backend`
and `nn_backend` in the defaults file set it, `t1prep-run`/`run_t1prep` pass
it to `segment.py`; the device follows `T1PREP_DEVICE`. The package
docstring (`nn/__init__.py`) is the map.

`src/t1prep/stages/` runs them: `Preprocess` keeps the sequence of
deepmriprep's pipeline (skull-strip, affine registration, segmentation,
non-linear registration, atlases) in T1Prep's own code, which imports neither
deepmriprep nor deepbet. Skull-strip and segmentation run deepbet's and
deepmriprep's trained networks, and the affine registration is deepmriprep's
method on torchreg. The non-linear registration is not deepmriprep's warp
network but geodesic shooting (`_shoot.py`), and non-cortical grey matter is
removed by rule (`nogm.py`). Keep the framework (`nn/`) free of pipeline knowledge, and the
stages free of framework details.

- **New backend**: subclass `nn.backends.Backend` in a new module of
  `nn/backends/` (implement its abstract operations, `load` and `run`) and add
  it to `_BACKENDS` there. Add a comparison against torch to `tests/test_nn.py`.
- **New architecture**: write `forward(ops, weights, x)` in a new module of
  `nn/architectures/`, decorate it with `@register(name, signature=...)`, and
  import it at the bottom of `architectures/__init__.py`. Use only the
  backend's operations and arithmetic.
- **Retrained model**: export the `state_dict` with
  `nn.convert.save_state_dict` (or `python -m t1prep.nn.convert model.pth`),
  not with `torch.jit.trace`, and give the file its name from `nn.models.NETWORKS`.
  Ship new weights as the asset of a new GitHub release, set
  `nn.models.MODEL_RELEASE` to its tag (installs then fetch them into a
  folder of their own instead of keeping the old weights, and stop
  converting deepmriprep's archives), and replace the archives in
  `src/t1prep/data/models`, which a source checkout converts before any
  download.
  New weights change every output: re-pin the phantom (`make phantom-pin`).
- **Invariant**: the torch backend must stay bit-identical to the archives
  on CPU and MPS (`tests/test_nn.py::TestAgainstTheArchives`), and the stages
  to deepbet's and deepmriprep's code (`tests/test_stages.py`, networks
  stubbed); a change that breaks either changes T1Prep's results.

## Ignore Rules
- Coding agents should treat any files or folders matched by `.gitignore` as out of scope (do not search, edit, or base decisions on them) unless the user explicitly asks to work with those ignored paths.

## Coding Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) style.
- If available in your environment, format with `black` and verify with `black --check src scripts`.
- If available, lint with `flake8 --select=F src` (or `ruff check --select F src`): unused and undefined names. The style classes are not enforced; `black` decides the layout of new code.
- Keep indentation at four spaces.
- Provide docstrings for all public functions and classes.
- Keep code clean and readable.
- Resize tensors with `t1prep._device.interpolate`, not `F.interpolate`: under the deterministic algorithms `segment.py` enables, torch computes linear interpolation on CUDA and MPS through a slow decomposition (0.7 s instead of 14 ms onto the working grid on an M2). `tests/test_device.py` fails on a direct call.

## Performance Optimization
- For compute-heavy voxel-wise operations, consider PyTorch or libCAT (`cat_surf`; numba was evaluated and rejected).
- Prefer clear, correct implementations first; optimize only when needed and measured.

## Commit Guidance
- Break up work into small, logically separate commits.
- Commit messages use the form `type: short summary` where `type` can be `feat`, `fix`, `docs`, `chore`, etc.
- Reference issues when relevant, e.g. `fix: handle missing atlas path (#12)`.
- Use short (<50 char) summaries and include a blank line before the body.
- Wrap body lines at 80 characters.

**Commit type examples:**
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `chore:` maintenance, dependencies
- `refactor:` code restructuring without behavior change
- `test:` adding or fixing tests

## Shell Scripts
- Validate scripts in `scripts/` with `shellcheck` (`-S error` must report nothing).
- Quick syntax check: `bash -n scripts/T1Prep`.

## Programmatic Checks
- Install dependencies via `pip install -r requirements.txt`.
- Run `python -m compileall src` to ensure files compile.
- Run `pytest` from the repository root before submitting (CI runs it on Python 3.9-3.14; the tests put `src/` on `sys.path` themselves).

## Pull Request Guidelines

When creating a PR:
1. Summarize the changes and link to relevant issues.
2. Mention any new dependencies or setup steps.
3. Ensure `python -m compileall src` completed without errors and describe the result in the PR body.
4. Run the linters before a PR is merged: `shellcheck` for the bash scripts and `flake8` (or `ruff`) for Python. The repository is clean at the levels the checklist names (shellcheck errors, the pyflakes class `F` of flake8); the style classes of flake8 (line length and the like) and shellcheck's warnings are not, so there the rule is to add none.

### PR Checklist

- [ ] `python -m compileall src` passes
- [ ] `shellcheck -S error scripts/*.sh scripts/T1Prep scripts/parallelize scripts/CAT_*_ui scripts/CAT_GrepJson` reports nothing, and no new warnings (for shell changes)
- [ ] `flake8 --select=F src` reports nothing (or `ruff check --select F src`)
- [ ] Dependencies synced between `requirements.txt` and `pyproject.toml`
- [ ] README.md updated if user-facing changes
- [ ] Docstrings added for new public functions

## Launcher Development

The Qt launcher (`t1prep-gui`) is the graphical front end of the pipeline.

**Key files:**
- `src/t1prep/launch/options.py` - the option table: labels, defaults, flags
- `src/t1prep/launch/progress.py` - phases, bars and per-image states of a run
- `src/t1prep/launch/results.py` - files of a subject, report measures, viewer calls
- `src/t1prep/launch/inputs.py` - finding the images of a folder
- `src/t1prep/gui/launcher.py` - the window, the T1Prep process, the viewer processes
- `src/t1prep/gui/launcher_widgets.py`, `theme.py`, `icons.py` - table, form, results panel, look (dark only), icons

**When modifying the launcher:**
- An option is one `Option` in the table; the form and the arguments follow from it
- Keep VTK and OpenGL out of it: the viewers draw, each in a process of its own
  (`tests/test_launcher_gui.py` fails when a VTK module gets imported)
- What is not Qt belongs in `src/t1prep/launch/`, where it is tested without a window
- For atlas selections, ensure `.txt` description files exist for the tool tips

## Testing

- Unit tests are in `tests/`
- Run with `pytest` from the repository root
- Test file naming: `test_<module>.py`

## Notes

These instructions are a starting point for contributors. Update this file as the project evolves.
