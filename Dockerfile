# syntax=docker/dockerfile:1
FROM python:3.12-slim

# T1Prep version to install from PyPI.  This line is bumped automatically by
# the "Update Dockerfile" GitHub Actions workflow once a release has reached
# PyPI, so the committed Dockerfile always builds a reproducible, pinned
# version (0.8.0 never reached PyPI, so this stays at 0.7.9 until the next
# release does).  Override
# at build time to build a different version, or pass an empty value to install
# the latest release instead:
#
#     docker build --build-arg T1PREP_VERSION=0.4.4 -t t1prep .   # specific
#     docker build --build-arg T1PREP_VERSION=      -t t1prep .   # latest
ARG T1PREP_VERSION=0.9.0

LABEL org.opencontainers.image.authors="Christian Gaser <christian.gaser@uni-jena.de>"
LABEL org.opencontainers.image.source="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="T1Prep"
LABEL org.opencontainers.image.description="T1 PREProcessing Pipeline (aka PyCAT) — pure-Python distribution"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime deps for the scientific-Python wheel stack (numpy, scipy,
# torch, nibabel, cat-surf).  No build toolchain or Java is needed since
# T1Prep is now distributed as a pure-Python package and pulls in
# pre-built C-extension wheels (cat-surf, torch) from PyPI.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install T1Prep from PyPI.  The weights are fetched and converted further
# down, at build time, into T1PREP_MODEL_DIR, so the image needs no network on
# first run and any uid can read them — see t1prep.nn.models._resolve_model_dir().
RUN set -eux; \
    if [ -n "${T1PREP_VERSION}" ]; then \
        pip install --no-cache-dir "T1Prep==${T1PREP_VERSION}"; \
    else \
        pip install --no-cache-dir T1Prep; \
    fi; \
    python -c "import t1prep, sys; sys.stdout.write(f'Installed T1Prep {t1prep.__version__}\n')"

# Drop privileges.  /data is the expected mount point for user volumes.
# The home directory is made traversable because the image is routinely run
# with --user to match the host uid (see docs/installation.md).  The runtime
# enters WORKDIR while still privileged, so a 0700 home would leave the
# process started in a directory it is not allowed to return to.
RUN useradd -m -u 1000 -s /bin/bash t1prep \
 && mkdir -p /data \
 && chown -R t1prep:t1prep /data \
 && chmod 755 /home/t1prep
ENV T1PREP_MODEL_DIR=/opt/t1prep/models
RUN t1prep-download-models \
 && chmod -R a+rX /opt/t1prep
USER t1prep
WORKDIR /data
VOLUME ["/data"]

# Default: run the T1Prep CLI.  Examples:
#
#   # show help
#   docker run --rm t1prep
#
#   # process a NIfTI file mounted from the host
#   docker run --rm -v $PWD:/data t1prep \
#       --input /data/sub-01_T1w.nii.gz --out-dir /data/out
#
#   # drop into an interactive Python REPL with t1prep already importable
#   docker run --rm -it --entrypoint python t1prep
#
#   # an image built without the t1prep-download-models step: keep the
#   # weights in a volume so they are fetched only once
#   docker run --rm -v t1prep-models:/opt/t1prep/models \
#       --entrypoint t1prep-download-models t1prep
ENTRYPOINT ["python", "-m", "t1prep.t1prep"]
CMD ["--help"]
