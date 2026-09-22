.PHONY: help version release clean zip cp_binaries phantom phantom-check phantom-score phantom-pin phantom-wmh phantom-wmh-pin
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Single source of truth for the project version.
#
# Bump VERSION to the new release.  Set PREV_VERSION to whatever VERSION
# was on the last release — the `release` target uses an *exact*-match sed
# to rewrite occurrences of PREV_VERSION to VERSION in the one file that
# doesn't auto-derive (src/t1prep/__init__.py).
#
# Auto-derived (no manual update needed):
#   - pyproject.toml      → version = {attr = "t1prep.__version__"}
#   - scripts/utils.sh    → reads __version__ from src/t1prep/__init__.py
#   - scripts/T1Prep      → sources scripts/utils.sh
#   - Dockerfile          → installs T1Prep from PyPI (pin via build-arg
#                            T1PREP_VERSION=$(VERSION) at `docker build` time)
# ---------------------------------------------------------------------------
PREV_VERSION := 0.7.4
VERSION      := 0.7.5

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
	-@echo "  release       Bump version files from PREV_VERSION to VERSION"
	-@echo "  zip           Run release and zip the project"
	-@echo "  clean         Clean up artifacts and permissions"
	-@echo "  cp_binaries   Copy CAT-Surface binaries [BIN=CAT_MyBinary]"
	-@echo "  version       Show current PREV_VERSION / VERSION"
	-@echo "  phantom       Run T1Prep on the simulated phantom and check the pins (~15 min)"
	-@echo "  phantom-check Check an existing phantom run against the pins [PHANTOM_DIR=...]"
	-@echo "  phantom-score Re-score an existing phantom run"
	-@echo "  phantom-pin   Run on the default device and on the CPU, then re-pin (~30 min)"
	-@echo "  phantom-wmh   WMH detection on every simulation in PHANTOM_SIMS, checked against its pins"
	-@echo "  phantom-wmh-pin  The same, then re-pin the WMH scores"

# show resolved versions (sanity check before `make release`)
version:
	-@echo "PREV_VERSION = $(PREV_VERSION)"
	-@echo "VERSION      = $(VERSION)"
	-@echo
	-@echo "Tracked locations:"
	-@grep -H '^__version__'  src/t1prep/__init__.py

# remove .DS_Store files and correct file permissions
clean:
	-@find . -type f -name .DS_Store -exec rm {} \;
	-@chmod -R a+r,g+w,o-w .
	-@find . -type f \( -name "*.sh" \) -exec chmod a+x {} \;
	-@find . -type f \( -name "*_ui" \) -exec chmod a+x {} \;

# zip release
zip: release
	-@echo zip
	-@test ! -d T1Prep || rm -r T1Prep
	-@mkdir T1Prep
	-@rsync -av . T1Prep --exclude env --exclude '.*' --exclude Makefile --exclude Windows-Installation.txt --exclude test --exclude evaluation/data/phantom
	-@zip ${ZIPFILE} -rm T1Prep

# prepare a release: rewrite PREV_VERSION → VERSION in the few files that
# don't auto-derive.  Uses exact-match seds so nothing unrelated can be hit.
# When PREV_VERSION == VERSION (e.g. fresh checkout, no bump yet) this is a
# safe no-op.
update: release
release: clean
	-@if [ "$(PREV_VERSION)" = "$(VERSION)" ]; then \
	    echo "PREV_VERSION == VERSION ($(VERSION)) — nothing to bump."; \
	  else \
	    echo "Bumping $(PREV_VERSION) -> $(VERSION)"; \
	    sed -i "" 's/^__version__ = "$(PREV_VERSION)"/__version__ = "$(VERSION)"/' src/t1prep/__init__.py; \
	    echo "Done. Update PREV_VERSION := $(VERSION) in the Makefile before the next release."; \
	  fi

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
