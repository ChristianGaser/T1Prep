FROM python:3.12-slim

LABEL org.opencontainers.image.authors="Christian Gaser <christian.gaser@uni-jena.de>"
LABEL org.opencontainers.image.source="https://github.com/ChristianGaser/T1Prep"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="T1Prep"
LABEL org.opencontainers.image.description="T1 PREProcessing Pipeline (aka PyCAT)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        unzip \
        zip \
        default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/T1Prep

# Copy repository into the image
COPY . .

# Install Python dependencies into a local virtual environment
RUN scripts/T1Prep --install \
    && ln -s /opt/T1Prep/scripts/T1Prep /usr/local/bin/T1Prep

ENTRYPOINT ["T1Prep"]
CMD ["--help"]

