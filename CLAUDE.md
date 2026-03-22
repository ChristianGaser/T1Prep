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

**T1Prep** is a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data (bias-field correction, segmentation, lesion detection, cortical surface reconstruction, CAT12 integration). Code lives in `src/`, helper scripts in `scripts/`, Flask web UI in `webui/`.

## Key Commands

```bash
# CLI
./scripts/T1Prep --help
./scripts/T1Prep --out-dir /tmp/out file.nii.gz

# Python API
from t1prep import run_t1prep

# Web UI
./scripts/T1Prep_ui --port 5050

# Sanity check
python -m compileall src

# Tests
pytest

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
| `scripts/cat_viewsurf.sh` | Launch CAT surface viewer |
| `scripts/T1Prep_ui` | Launch Web UI |

See [ENVIRONMENT_USAGE.md](ENVIRONMENT_USAGE.md) for details.

## Critical: Files to Keep in Sync

| When you change… | Also update… |
|------------------|--------------|
| `requirements.txt` | `pyproject.toml` → `[project.dependencies]` |
| `pyproject.toml` dependencies | `requirements.txt` |
| CLI options in `scripts/T1Prep` | `src/t1prep/t1prep.py`, `webui/app.py`, `webui/templates/index.html`, `T1Prep_defaults.txt`, `README.md` |
| `src/t1prep/t1prep.py` API | `README.md` → Python API section |
| Scripts in `scripts/` (add/remove/rename) | `scripts/README.md`, `Agents.md` → Project Structure, `CLAUDE.md` |
| Installation process | `README.md`, `scripts/install.sh` |
| Docker configuration | `README.md`, `Dockerfile` |
| Version number | `pyproject.toml`, `Makefile`, README badges |

## Adding New CLI Options (order matters)

1. `scripts/T1Prep`
2. `src/t1prep/t1prep.py` → `run_t1prep()` parameters
3. `webui/app.py`
4. `webui/templates/index.html`
5. `T1Prep_defaults.txt`
6. `README.md`

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
- [ ] `README.md` updated for user-facing changes
- [ ] Docstrings added for new public functions

## Ignore Rules

Treat `.gitignore`-matched files as out of scope — do not search, edit, or base decisions on them unless explicitly asked.
