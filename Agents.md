# AGENTS

## Overview
This repository contains **T1Prep**, a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data. The project supports tasks such as bias-field correction, segmentation, lesion detection and integration with CAT12. The code lives in the `src` directory and helper scripts are in `scripts/`.

## Project Structure

```
T1Prep/
├── src/
├── scripts/
├── ...
├── Makefile
├── requirements.txt
├── README.md
├── pyproject.toml
└── setup.py
```

## Development Guidelines
- Use Python 3.9 or newer.
- Keep functions small and well documented. Include docstrings for public functions.
- Prefer using the utilities provided in `src/t1prep/utils.py` (and related helpers in `src/t1prep/`) when possible.
- Check documentation of functions and add missing documentation.

## Common tasks
- Show CLI options: `./scripts/T1Prep --help`
- Run pipeline (example): `./scripts/T1Prep --out-dir /tmp/out sub-01_T1w.nii.gz`
- Python API: `from t1prep import run_t1prep`
- Quick Python sanity check: `python -m compileall src`

## Coding style
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

## Shell scripts
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
4. Use the following linters to pass before a PR is merged. They’re installed automatically by the Codex environment’s Setup Script: shellcheck for *.sh and *.bash and flake8 (or ruff) for Python.

## Notes
These instructions are a starting point for contributors. Update this file as the project evolves.