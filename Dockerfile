# syntax=docker/dockerfile:1
FROM python:3.12-slim

# --- selection knobs
# SOURCE: 'release' (default) downloads a ZIP archive for the given tag;
#         'git' clones the repo at the ref you specify (branch/tag/commit).
ARG T1PREP_SOURCE=release
ARG T1PREP_VERSION=v0.2.5
ARG T1PREP_REF=main              # used only when T1PREP_SOURCE=git

LABEL org.opencontainers.image.authors="Christian Gaser <christian.gaser@uni-jena.de>"
LABEL org.opencontainers.image.source="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="T1Prep"
LABEL org.opencontainers.image.version="${T1PREP_VERSION}"
LABEL org.opencontainers.image.description="T1 PREProcessing Pipeline (aka PyCAT)"
LABEL org.opencontainers.image.t1prep.source="${T1PREP_SOURCE}"
LABEL org.opencontainers.image.t1prep.ref="${T1PREP_REF}"

# --- base deps
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    bash \
      ca-certificates \
      wget \
      unzip \
      zip \
      default-jre-headless \
      build-essential \
      git \
 && rm -rf /var/lib/apt/lists/*

# --- fetch & install T1Prep
WORKDIR /opt
RUN set -eux; \
    if [ "${T1PREP_SOURCE}" = "git" ]; then \
        echo "Cloning T1Prep @ ${T1PREP_REF} from GitHub..."; \
        git clone https://github.com/ChristianGaser/T1Prep.git /opt/T1Prep; \
        cd /opt/T1Prep; \
        # Try to fetch the requested ref (branch/tag/commit) efficiently.
        git fetch --depth=1 origin "${T1PREP_REF}" || true; \
        git fetch --depth=1 origin "refs/tags/${T1PREP_REF}:refs/tags/${T1PREP_REF}" || true; \
        git checkout --detach "${T1PREP_REF}" || git checkout "${T1PREP_REF}"; \
        rm -rf /opt/T1Prep/.git; \
    else \
        echo "Downloading release ${T1PREP_VERSION}..."; \
        wget -q --progress=dot:giga \
          "https://github.com/ChristianGaser/T1Prep/archive/refs/tags/${T1PREP_VERSION}.zip" \
          -O /opt/source.zip; \
        unzip -q /opt/source.zip -d /opt; \
        src_dir="$(find /opt -maxdepth 1 -type d -name 'T1Prep-*' | head -n 1)"; \
        test -n "${src_dir}"; \
        mv "${src_dir}" /opt/T1Prep; \
        rm /opt/source.zip; \
    fi; \
    /opt/T1Prep/scripts/T1Prep --install

# --- runtime env
ENV PATH="/opt/T1Prep/scripts:${PATH}"
WORKDIR /opt/T1Prep

# --- drop privileges
RUN useradd -m -u 1000 -s /bin/bash t1prep \
 && chown -R t1prep:t1prep /opt/T1Prep
USER t1prep

# Default: show help
ENTRYPOINT ["/opt/T1Prep/scripts/T1Prep"]
CMD ["--help"]
