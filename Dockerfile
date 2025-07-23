FROM ubuntu:latest

ARG T1PREP_VERSION=0
ARG T1PREP_RELEASE=0.2.0
ARG T1PREP_REVISION=beta
# Calculated
ENV T1PREP_TAG=${T1PREP_RELEASE}${T1PREP_REVISION:+.${T1PREP_REVISION}}

LABEL org.opencontainers.image.authors="Christian Gaser <christian.gaser@uni-jena.de>"
LABEL org.opencontainers.image.source="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.url="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.documentation="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.version="${T1PREP_VERSION}"
LABEL org.opencontainers.image.revision="${T1PREP_REVISION}"
LABEL org.opencontainers.image.vendor="Structural Brain Mapping Group"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="T1Prep"
LABEL org.opencontainers.image.description="T1 PREProcessing Pipeline (aka PyCAT)"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
        build-essential unzip zip default-jre \
    && apt-get clean \
    && rm -rf \
        /tmp/hsperfdata* \
        /var/*/apt/*/partial \
        /var/lib/apt/lists/* \
        /var/log/apt/term*

RUN wget --no-check-certificate --progress=bar:force -P /opt https://github.com/ChristianGaser/T1Prep/releases/download/${T1PREP_TAG}/T1Prep${T1PREP_TAG}.zip \
    && unzip -q /opt/T1Prep${T1PREP_TAG}.zip -d /opt \
    && rm -f /opt/T1Prep${T1PREP_TAG}.zip \
    && /opt/T1Prep/scripts/T1Prep --re-install \
    && ln -s /opt/T1Prep/scripts/* /usr/local/bin/T1Prep

RUN T1Prep

ENTRYPOINT ["T1Prep"]

CMD ["--help"]