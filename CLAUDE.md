# CLAUDE.md – T1Prep Project

> For full contributor documentation see [Agents.md](Agents.md).

## Sub-Agent Routing Rules
- **Always sequential:** All tasks (security, performance, style, refactoring) must be processed sequentially.
- **No parallelization:** Only one sub-agent or one check may be active at a time.
- **Workflow:** First execute `security`, then `performance`, then `style`. Wait for each step to complete.
- **Dependencies:** B tasks must wait for the output of A tasks.

## Background Execution Rules
 
Run in background automatically:
 
- Web research and documentation lookups
- Codebase exploration and analysis
- Security audits and performance profiling
- Any task where results aren't immediately needed
- Research or analysis tasks (not file modifications)
- Results aren't blocking your current work

## Overview

**T1Prep** is a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data (bias-field correction, segmentation, lesion detection, cortical surface reconstruction, CAT12 integration). Code lives in `src/`, helper/dev scripts in `scripts/`, Flask web UI in `src/t1prep/webui/`.

Entry points are installed into the environment's `bin/` (the canonical way to
run T1Prep): `T1Prep` (bash orchestrator), `PyCAT` (symlink to `T1Prep`; same
CLI, PyCAT startup banner), `t1prep-ui`, `t1prep-run` (Python
single-subject), `CAT_SurfView`, `CAT_VolView`, `CAT_PlotHistogram`, `t1prep-make-apps`, `t1prep-download-models`, `t1prep-bbreg`. The
`scripts/` folder
is a source-tree/dev fallback and should not be put on `PATH`.

## Key Commands

```bash
# CLI (from <venv>/bin on PATH; or ./scripts/T1Prep in a source checkout)
T1Prep --help
T1Prep --out-dir /tmp/out file.nii.gz

# Python API
from t1prep import run_t1prep

# Web UI
t1prep-ui --port 5050

# Sanity check
python -m compileall src

# Tests
pytest

# Voxel-wise volume differences to the first image (cat_stat_diff via CAT_VolCalc)
CAT_VolDiff [--rel] [--glob] tp1.nii.gz tp2.nii.gz

# Recalibrate the QA rating bounds from a processed BrainWeb Phantom set
python scripts/qa_calibrate.py /path/to/BWP/report

# Histogram of volumes, surfaces or text data (CAT12's cat_plot_histogram.m)
CAT_PlotHistogram [--dist kernel|none|normal|...] [--mean] [--save PREFIX] file...

# Longitudinal: rigid realignment, then the low-dimensional ageing deformation
./scripts/process_longitudinal.sh --long-model ageing tp1.nii.gz tp2.nii.gz
./scripts/warp_longitudinal.sh --inputs tp1.nii.gz tp2.nii.gz --out-dir DIR
./scripts/modulate_longitudinal.sh --mri-dirs D1 D2 --names tp1 tp2 --out-dir DIR

# Score the spherical registration against the Mindboggle-101 manual labels
python evaluation/tools/eval_mindboggle.py project --mindboggle DIR... --t1prep DIR --work DIR
python evaluation/tools/eval_mindboggle.py dice --work DIR --protocol both

# Accuracy against the simulated phantom (manual, ~15 min): run, then check the pins
make phantom                     # = eval_phantom.py run + check in /tmp/T1Prep_phantom
make phantom-check PHANTOM_DIR=DIR
make phantom-pin                 # re-pin after an intended change (two devices, ~30 min)
make phantom-wmh                 # WMH over the 14 simulations in evaluation/data/phantom (~1 h)
python evaluation/tools/fit_wmh_calibration.py fit --work DIR  # refit WMH_CALIBRATION (after dump)

# Linting / formatting
black src scripts
flake8 src scripts       # or: ruff check src scripts
shellcheck scripts/*.sh
```

## Environment

Always use the wrapper scripts – they auto-activate the virtual environment:

| Script | Purpose |
|--------|---------|
| `scripts/activate_env.sh` | Activate venv manually |
| `scripts/run_with_env.sh <script>` | Run any Python script with correct env |
| `scripts/run_with_env.sh src/t1prep/gui/cat_surf_view.py` | Surface viewer from a checkout (installed: `CAT_SurfView`) |
| `scripts/run_with_env.sh src/t1prep/gui/cat_vol_view.py` | Volume viewer from a checkout (installed: `CAT_VolView`) |
| `scripts/make_macos_apps.sh` | Build macOS `.app` bundles (installed: `t1prep-make-apps`) |
| `scripts/T1Prep_ui` | Launch Web UI (installed: `t1prep-ui`) |

See [ENVIRONMENT_USAGE.md](ENVIRONMENT_USAGE.md) for details.

## Critical: Files to Keep in Sync

| When you change… | Also update… |
|------------------|--------------|
| `requirements.txt` | `pyproject.toml` → `[project.dependencies]` |
| `pyproject.toml` dependencies | `requirements.txt` |
| CLI options in `scripts/T1Prep` | `src/t1prep/t1prep.py`, `src/t1prep/webui/app.py`, `src/t1prep/webui/templates/index.html`, `T1Prep_defaults.txt`, `docs/usage.md` |
| `[project.scripts]` / `script-files` entry points in `pyproject.toml` | `README.md`, `README_pypi.md`, `docs/installation.md`, `scripts/install.sh`, `Agents.md`, `CLAUDE.md` |
| `src/t1prep/t1prep.py` API | `docs/usage.md` → Python API section |
| Scripts in `scripts/` (add/remove/rename) | `scripts/README.md`, `Agents.md` → Project Structure, `CLAUDE.md` |
| Viewer features or options | `docs/viewers.md` |
| The other GUI tools | `docs/tools.md` |
| Installation process | `docs/installation.md`, `README.md`, `README_pypi.md`, `scripts/install.sh` (bash bootstrapper is secondary to `pip install T1Prep`) |
| Docker configuration | `docs/installation.md`, `Dockerfile` |
| Version number | `src/t1prep/__init__.py` is the single source of truth — `pyproject.toml` derives via `setuptools.dynamic`, `scripts/T1Prep_utils.sh` awks it, `Makefile` bumps it via `make release`. Also update README badges + git tag. |

## Command-Line Conventions

Every tool behaves the same way, with `CAT_VolView` as the template:

| You type | You get |
|----------|---------|
| nothing at all | the synopsis — the command line plus an overview of the options — on stderr, exit 1 |
| `--help` | the full description, exit 0 |
| `--version` | `<tool> <T1PREP_VERSION>`, exit 0 |

Options are spelled `--like-this`. Single-dash spellings that predate this
(`CAT_SurfView -overlay`, `CAT_VolDiff -s`, `parallelize -p`, `-h`, `-v`) stay
valid so existing command lines keep working, but they are hidden from the
help. Two launchers are the deliberate exception to the no-argument rule:
`t1prep-ui` and `t1prep-download-models` do their job instead of printing the
synopsis.

The behaviour lives in one place per language — reuse it, do not re-implement:

- **Python** — `src/t1prep/cli_help.py`: use `ArgumentParser` from it instead
  of `argparse.ArgumentParser`, and call `parser.exit_without_arguments(argv)`
  before `parse_args`. It hides the single-dash spellings, adds `--version`,
  and points a failed parse at `--help`.
- **Bash** — `print_usage` and `print_version` in `scripts/T1Prep_utils.sh`:
  define a `usage()` that calls `print_usage "<synopsis>" "<option line>" …`,
  and leave the long text in `help()`. `parallelize` and
  `progress_bar_multi.sh` keep local copies on purpose — they stay standalone.

## Adding New CLI Options (order matters)

1. `scripts/T1Prep`
2. `src/t1prep/t1prep.py` → `run_t1prep()` parameters
3. `src/t1prep/webui/app.py`
4. `src/t1prep/webui/templates/index.html`
5. `T1Prep_defaults.txt`
6. `docs/usage.md`

The option gets a `--` spelling, is listed in `usage()` as well as `help()`,
and keeps any old single-dash spelling only as a hidden alias.

## Adding New Atlases

- **Volume atlases** → `src/t1prep/data/templates_MNI152NLin2009cAsym/`: add `<name>.nii.gz` + `<name>.txt`
- **Surface atlases** → `src/t1prep/data/atlases_surfaces_32k/`: add `lh.<name>.annot`, `rh.<name>.annot` + `lh.<name>.txt`

## Coding Style

- Python 3.9–3.12, PEP 8, 4-space indentation
- Docstrings for all public functions and classes
- Format with `black`, lint with `flake8`/`ruff`, check shell scripts with `shellcheck`
- For compute-heavy voxel-wise operations consider PyTorch or Numba; optimize only after measuring

## Commit Conventions

```
type: short summary (<50 chars)

Body wrapped at 80 chars. Reference issues when relevant.
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`

## PR Checklist

- [ ] `python -m compileall src` passes
- [ ] `shellcheck scripts/*.sh` passes (shell changes)
- [ ] `flake8 src` / `ruff check src` passes
- [ ] `requirements.txt` ↔ `pyproject.toml` in sync
- [ ] Documentation updated for user-facing changes (`docs/usage.md`, `docs/viewers.md`, `docs/tools.md`)
- [ ] Docstrings added for new public functions

## Ignore Rules

Treat `.gitignore`-matched files as out of scope — do not search, edit, or base decisions on them unless explicitly asked.
