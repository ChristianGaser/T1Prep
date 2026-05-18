# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Optional version pin.  Leave empty (the default) to install the latest
# T1Prep release from PyPI, or set to a specific PEP 440 version for a
# reproducible build:
#
#     docker build --build-arg T1PREP_VERSION=0.4.4 -t t1prep .
#
# Versioning is delegated entirely to pip / PyPI metadata — there is no
# longer a separate T1PREP_VERSION baked into the image filesystem.
ARG T1PREP_VERSION=

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

# Install T1Prep from PyPI.  Model weights are NOT downloaded at build
# time; they are fetched lazily on first use into the running container's
# user cache (~/.cache/t1prep/models) — see t1prep._models.prepare_model_files().
RUN set -eux; \
    if [ -n "${T1PREP_VERSION}" ]; then \
        pip install --no-cache-dir "T1Prep==${T1PREP_VERSION}"; \
    else \
        pip install --no-cache-dir T1Prep; \
    fi; \
    python -c "import t1prep, sys; sys.stdout.write(f'Installed T1Prep {t1prep.__version__}\n')"

# Drop privileges.  /data is the expected mount point for user volumes.
RUN useradd -m -u 1000 -s /bin/bash t1prep \
 && mkdir -p /data \
 && chown -R t1prep:t1prep /data
USER t1prep
WORKDIR /home/t1prep
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
#   # pre-download model weights (otherwise fetched on first run)
#   docker run --rm -v t1prep-models:/home/t1prep/.cache/t1prep t1prep \
#       --entrypoint t1prep-download-models
ENTRYPOINT ["python", "-m", "t1prep.t1prep"]
CMD ["--help"]
