.PHONY: help version release clean zip cp_binaries phantom phantom-check phantom-score phantom-pin phantom-wmh phantom-wmh-pin
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# src/t1prep/__init__.py is the single source of truth for the version;
# pyproject.toml (setuptools.dynamic) and the bash scripts read it from there.
#
# Release in this order, so that the tag carries the new version (PyPI builds
# the tag, and "Publish to PyPI" refuses a tag that differs from __version__;
# v0.8.0 was tagged before its bump and never reached PyPI):
#   make release VERSION=0.8.1     # on a clean main: bump, commit, tag v0.8.1
#   git push origin main v0.8.1
#   then create the GitHub release from tag v0.8.1.  Once PyPI has it, the
#   "Update Dockerfile" workflow pins the Docker image to that version.
# ---------------------------------------------------------------------------
CURRENT_VERSION := $(shell sed -n -E 's/^__version__ = "([^"]+)".*/\1/p' src/t1prep/__init__.py)
VERSION ?= $(CURRENT_VERSION)

ZIPFILE = T1Prep_$(VERSION).zip

BIN ?= CAT*

# Phantom accuracy test (evaluation/PHANTOM.md).  PHANTOM_DIR holds the run;
# the re-pin also uses $(PHANTOM_DIR)_cpu.  PHANTOM_ARGS is passed through,
# e.g. PHANTOM_ARGS=--history, or "-- --no-overwrite" for T1Prep options.
PHANTOM_DIR  ?= /tmp/T1Prep_phantom
PHANTOM_ARGS ?=
# mri_simulate renderings with several noise/bias/WMH settings
PHANTOM_SIMS ?= $(or $(T1PREP_PHANTOM_SIMS),evaluation/data/phantom)
EVAL_PHANTOM  = ./scripts/run_with_env.sh evaluation/tools/eval_phantom.py

# print available commands
help:
	-@echo Available commands:
	-@echo "  release       Bump __version__, commit and tag: make release VERSION=x.y.z"
	-@echo "  zip           Zip the project (current version)"
	-@echo "  clean         Clean up artifacts and permissions"
	-@echo "  cp_binaries   Copy CAT-Surface binaries [BIN=CAT_MyBinary]"
	-@echo "  version       Show the current version"
	-@echo "  phantom       Run T1Prep on the simulated phantom and check the pins (~15 min)"
	-@echo "  phantom-check Check an existing phantom run against the pins [PHANTOM_DIR=...]"
	-@echo "  phantom-score Re-score an existing phantom run"
	-@echo "  phantom-pin   Run on the default device and on the CPU, then re-pin (~30 min)"
	-@echo "  phantom-wmh   WMH detection on every simulation in PHANTOM_SIMS, checked against its pins"
	-@echo "  phantom-wmh-pin  The same, then re-pin the WMH scores"

# show the current version (the one `make release` bumps from)
version:
	-@echo "$(CURRENT_VERSION)"

# remove .DS_Store files and correct file permissions
clean:
	-@find . -type f -name .DS_Store -exec rm {} \;
	-@chmod -R a+r,g+w,o-w .
	-@find . -type f \( -name "*.sh" \) -exec chmod a+x {} \;
	-@find . -type f \( -name "*_ui" \) -exec chmod a+x {} \;

# zip release
zip: clean
	-@echo zip
	-@test ! -d T1Prep || rm -r T1Prep
	-@mkdir T1Prep
	-@rsync -av . T1Prep --exclude env --exclude '.*' --exclude Makefile --exclude Windows-Installation.txt --exclude test --exclude evaluation/data/phantom
	-@zip ${ZIPFILE} -rm T1Prep

# bump __version__, commit and tag v$(VERSION); see the header for the order
update: release
release:
	@if [ "$(VERSION)" = "$(CURRENT_VERSION)" ]; then \
	  echo "Usage: make release VERSION=x.y.z (current: $(CURRENT_VERSION))" >&2; exit 1; fi
	@echo "$(VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-]?[A-Za-z0-9]+)*$$' || \
	  { echo "'$(VERSION)' is not a version like 0.8.1" >&2; exit 1; }
	@test "$$(git branch --show-current)" = main || \
	  { echo "Release from main" >&2; exit 1; }
	@test -z "$$(git status --porcelain --untracked-files=no)" || \
	  { echo "Commit or stash your changes first" >&2; exit 1; }
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
	  echo "Tag v$(VERSION) exists already" >&2; exit 1; fi
	@perl -pi -e 's/^__version__ = "[^"]*"/__version__ = "$(VERSION)"/' src/t1prep/__init__.py
	@grep -q '^__version__ = "$(VERSION)"' src/t1prep/__init__.py || \
	  { echo "Failed to set __version__" >&2; exit 1; }
	@git commit -q -m "chore: bump version to $(VERSION)" src/t1prep/__init__.py
	@git tag -a "v$(VERSION)" -m "T1Prep $(VERSION)"
	@echo "Bumped $(CURRENT_VERSION) -> $(VERSION) and tagged v$(VERSION). Next:"
	@echo "  git push origin main v$(VERSION)"
	@echo "  then create the GitHub release from tag v$(VERSION)"

# copy binaries after cross-compiling
cp_binaries: 
	-@echo copy binaries matching $(BIN)
	-@for i in src/t1prep/bin/Linux/$(BIN); do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-pc-linux/Progs/`basename $${i}` src/t1prep/bin/Linux/ ; done
	-@for i in src/t1prep/bin/Windows/$(BIN); do cp ~/Dropbox/GitHub/CAT-Surface/build-x86_64-w64-mingw32/Progs/`basename $${i}` src/t1prep/bin/Windows/ ; done
	-@for i in src/t1prep/bin/MacOS/$(BIN); do cp ~/Dropbox/GitHub/CAT-Surface/build-native-arm64/Progs/`basename $${i}` src/t1prep/bin/MacOS/ ; done
	-@for i in src/t1prep/bin/LinuxARM64/$(BIN); do cp ~/Dropbox/GitHub/CAT-Surface/build-aarch64-none-elf/Progs/`basename $${i}` src/t1prep/bin/LinuxARM64/ ; done

# phantom accuracy test: T1Prep + ground-truth surface arm, then the pinned check
# (no "-" prefix: a regression has to fail the make run)
phantom:
	$(EVAL_PHANTOM) run --work $(PHANTOM_DIR) --check $(PHANTOM_ARGS)

phantom-check:
	$(EVAL_PHANTOM) check --work $(PHANTOM_DIR)

phantom-score:
	$(EVAL_PHANTOM) score --work $(PHANTOM_DIR) $(PHANTOM_ARGS)

# re-pin after an intended change: two devices give the tolerances a spread
phantom-pin:
	$(EVAL_PHANTOM) run --work $(PHANTOM_DIR) $(PHANTOM_ARGS)
	$(EVAL_PHANTOM) run --work $(PHANTOM_DIR)_cpu --device cpu $(PHANTOM_ARGS)
	$(EVAL_PHANTOM) pin --work $(PHANTOM_DIR) $(PHANTOM_DIR)_cpu

# WMH detection over a set of simulations (T1Prep without surfaces, ~4 min/image)
phantom-wmh:
	$(EVAL_PHANTOM) wmh --work $(PHANTOM_DIR)_wmh --sims "$(PHANTOM_SIMS)" --check $(PHANTOM_ARGS)

phantom-wmh-pin:
	$(EVAL_PHANTOM) wmh --work $(PHANTOM_DIR)_wmh --sims "$(PHANTOM_SIMS)" $(PHANTOM_ARGS)
	$(EVAL_PHANTOM) pin --work $(PHANTOM_DIR)_wmh
