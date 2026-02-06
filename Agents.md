# AGENTS

## Overview
This repository contains **T1Prep**, a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data. The project supports tasks such as bias-field correction, segmentation, lesion detection, cortical surface reconstruction, and integration with CAT12. The code lives in the `src` directory, helper scripts are in `scripts/`, and a Flask-based web UI is in `webui/`.

## Project Structure

```
T1Prep/
├── src/
│   └── t1prep/
│       ├── __init__.py             # Package init, exports run_t1prep
│       ├── t1prep.py               # Python API: run_t1prep() function
│       ├── segment.py              # Segmentation logic
│       ├── segmentation_utils.py
│       ├── utils.py                # Shared utilities
│       ├── metrics.py              # Dice/evaluation metrics
│       ├── dice.py                 # Dice coefficient calculation
│       ├── realign_longitudinal.py
│       ├── bin/                    # Compiled CAT-Surface binaries
│       ├── data/                   # Templates, atlases, default files
│       │   ├── templates_MNI152NLin2009cAsym/  # Volume atlases + .txt descriptions
│       │   └── atlases_surfaces_32k/           # Surface atlases + .txt descriptions
│       └── gui/                    # PySide6/VTK visualization tools
├── scripts/
│   ├── T1Prep                      # Main CLI entry point (bash)
│   ├── T1Prep_ui                   # GUI launcher
│   ├── install.sh                  # Installation script
│   ├── parallelize                 # Parallel processing helper
│   ├── utils.sh                    # Bash utility functions
│   └── ...                         # Other utility scripts
├── webui/
│   ├── app.py                      # Flask application
│   ├── templates/index.html        # Main web interface
│   └── static/styles.css           # CSS styling
├── tests/
│   ├── __init__.py
│   └── test_utils.py               # Unit tests
├── Makefile                        # Build/dev tasks
├── requirements.txt                # pip dependencies
├── pyproject.toml                  # Package metadata and dependencies
├── setup.py                        # Legacy setup script
├── README.md                       # User documentation
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
| CLI options in `scripts/T1Prep` | `README.md` -> Options section |
| CLI options in `scripts/T1Prep` | `src/t1prep/t1prep.py` -> `run_t1prep()` parameters |
| CLI options in `scripts/T1Prep` | `webui/app.py` -> form handling and defaults |
| `src/t1prep/t1prep.py` API | `README.md` -> Python API section |
| Default values | `T1Prep_defaults.txt` |
| Atlas files in `src/t1prep/data/` | Add corresponding `.txt` description file |
| Installation process | `README.md` -> Installation section |
| Installation process | `scripts/install.sh` |
| Docker configuration | `README.md` -> Docker section |
| Docker configuration | `Dockerfile` |
| Version number | `pyproject.toml`, release tags, README badges |

### Core File Relationships

```
scripts/T1Prep (CLI)
       |
       v calls
src/t1prep/segment.py, utils.py, etc.
       |
       v loads
src/t1prep/data/* (templates, atlases)
       |
       v calls
src/t1prep/bin/* (CAT-Surface binaries)

webui/app.py (Web UI)
       |
       v calls
scripts/T1Prep (via subprocess)
       |
       v renders
webui/templates/index.html
       |
       v styled by
webui/static/styles.css
```

## Development Guidelines
- Use Python 3.9 or newer (supports 3.9-3.12).
- Keep functions small and well documented. Include docstrings for public functions.
- Prefer using the utilities provided in `src/t1prep/utils.py` (and related helpers in `src/t1prep/`) when possible.
- Check documentation of functions and add missing documentation.

## Common Tasks
- Show CLI options: `./scripts/T1Prep --help`
- Run pipeline (example): `./scripts/T1Prep --out-dir /tmp/out sub-01_T1w.nii.gz`
- Python API: `from t1prep import run_t1prep`
- Quick Python sanity check: `python -m compileall src`
- Run Web UI: `cd webui && python app.py`

## When to Update README.md

**Always update README.md when:**
1. Adding/removing/changing CLI options in `scripts/T1Prep`
2. Changing the Python API signature in `src/t1prep/t1prep.py`
3. Adding new features or capabilities
4. Changing installation requirements or process
5. Modifying output folder structure or naming conventions
6. Adding/removing dependencies
7. Changing Docker build or run instructions
8. Fixing important bugs that affect user workflow

**README.md sections to check:**
- **Options**: Must match `scripts/T1Prep --help` output
- **Python API**: Must match `run_t1prep()` function signature
- **Installation**: Must match `scripts/install.sh` behavior
- **Requirements**: Must match `requirements.txt` / `pyproject.toml`

## Adding New CLI Options

When adding a new CLI option, update these files in order:

1. `scripts/T1Prep` - Add the option to argparse/getopts
2. `src/t1prep/t1prep.py` - Add parameter to `run_t1prep()` function
3. `webui/app.py` - Add form field handling if applicable
4. `webui/templates/index.html` - Add UI element if applicable
5. `T1Prep_defaults.txt` - Add default value if applicable
6. `README.md` - Document the new option

## Adding New Atlases

When adding atlas files to `src/t1prep/data/`:

1. **Volume atlases** go in `templates_MNI152NLin2009cAsym/`
   - Add `<atlas_name>.nii.gz` (the atlas file)
   - Add `<atlas_name>.txt` (description shown in Web UI tooltip)

2. **Surface atlases** go in `atlases_surfaces_32k/`
   - Add `lh.<atlas_name>.annot` and `rh.<atlas_name>.annot`
   - Add `lh.<atlas_name>.txt` (description shown in Web UI tooltip)

## Coding Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) style.
- If available in your environment, format with `black` and verify with `black --check src scripts`.
- If available, lint with `flake8 src scripts` (or `ruff check src scripts` if the project migrates to Ruff).
- Keep indentation at four spaces.
- Provide docstrings for all public functions and classes.
- Keep code clean and readable.

## Performance Optimization
- For compute-heavy voxel-wise operations, consider PyTorch or Numba.
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
- Validate scripts in `scripts/` with `shellcheck`.
- Quick syntax check: `bash -n scripts/T1Prep`.

## Programmatic Checks
- Install dependencies via `pip install -r requirements.txt`.
- Run `python -m compileall src` to ensure files compile.
- Run `pytest` before submitting (when tests are present and configured).

## Pull Request Guidelines

When creating a PR:
1. Summarize the changes and link to relevant issues.
2. Mention any new dependencies or setup steps.
3. Ensure `python -m compileall src` completed without errors and describe the result in the PR body.
4. Use the following linters to pass before a PR is merged. They're installed automatically by the Codex environment's Setup Script: shellcheck for *.sh and *.bash and flake8 (or ruff) for Python.

### PR Checklist

- [ ] `python -m compileall src` passes
- [ ] `shellcheck scripts/*.sh` passes (for shell changes)
- [ ] `flake8 src` passes (or `ruff check src`)
- [ ] Dependencies synced between `requirements.txt` and `pyproject.toml`
- [ ] README.md updated if user-facing changes
- [ ] Docstrings added for new public functions

## Web UI Development

The Flask-based Web UI (`webui/`) provides a browser interface for T1Prep.

**Key files:**
- `webui/app.py` - Flask routes, form handling, subprocess calls to T1Prep
- `webui/templates/index.html` - Jinja2 template with form elements and JavaScript
- `webui/static/styles.css` - CSS styling

**When modifying the Web UI:**
- Form fields should match CLI options in `scripts/T1Prep`
- Add tooltip help text matching CLI option descriptions
- For atlas selections, ensure `.txt` description files exist for hover help
- Test both light and dark mode (CSS variables in `:root`)

## Testing

- Unit tests are in `tests/`
- Run with `pytest` (when configured)
- Test file naming: `test_<module>.py`

## Notes

These instructions are a starting point for contributors. Update this file as the project evolves.
