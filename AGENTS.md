# AGENTS

## Overview
This repository contains **T1Prep**, a Python-based pipeline for preprocessing and segmenting T1-weighted MRI data. The project supports tasks such as bias-field correction, lesion detection and integration with CAT12. The code lives in the `src` directory and helper scripts are in `scripts/`.

## Development Guidelines
- Use Python 3.8 or newer.
- Follow [PEP 8](https://peps.python.org/pep-0008/) style. Format new Python code with `black`.
- Keep functions small and well documented. Include docstrings for public functions.
- Prefer using the utilities provided in `src/utils.py` when possible.

## Commit Guidance
- Break up work into small, logically separate commits.
- Commit messages use the form `type: short summary` where `type` can be `feat`, `fix`, `docs`, `chore`, etc.
- Reference issues when relevant, e.g. `fix: handle missing atlas path (#12)`.

## Programmatic Checks
- Install dependencies via `pip install -r requirements.txt`.
- Run `python -m compileall src` to ensure files compile.
- If tests are added in the future, run `pytest` before submitting.

## Pull Request Guidelines
When creating a PR:
1. Summarize the changes and link to relevant issues.
2. Mention any new dependencies or setup steps.
3. Ensure `python -m compileall src` completed without errors and describe the result in the PR body.

## Notes
These instructions are a starting point for contributors. Update this file as the project evolves.
