FROM ubuntu:22.04
FROM python:3.12-slim

ARG T1PREP_VERSION=0
ARG T1PREP_RELEASE=0.2.0
ARG T1PREP_REVISION=beta
ENV T1PREP_TAG=${T1PREP_RELEASE}${T1PREP_REVISION:+.${T1PREP_REVISION}}

LABEL org.opencontainers.image.authors="Christian Gaser <christian.gaser@uni-jena.de>"
LABEL org.opencontainers.image.source="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="T1Prep"
LABEL org.opencontainers.image.version="${T1PREP_VERSION}"
LABEL org.opencontainers.image.revision="${T1PREP_REVISION}"
LABEL org.opencontainers.image.description="T1 PREProcessing Pipeline (aka PyCAT)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        unzip \
        zip \
        default-jre \
    && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate --progress=bar:force -P /opt https://github.com/ChristianGaser/T1Prep/releases/download/${T1PREP_TAG}/T1Prep${T1PREP_TAG}.zip \
    && unzip -q /opt/T1Prep${T1PREP_TAG}.zip -d /opt \
    && rm -f /opt/T1Prep${T1PREP_TAG}.zip \
    && /opt/T1Prep/scripts/T1Prep --re-install

WORKDIR /opt/T1Prep

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the image
COPY . .

# Expose helper scripts on PATH so the entrypoint can locate them
ENV PATH="/opt/T1Prep/scripts:${PATH}"

RUN T1Prep

ENTRYPOINT ["/opt/T1Prep/scripts/T1Prep"]
CMD ["--help"]

