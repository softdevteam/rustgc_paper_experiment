# Use BuildKit features
# syntax=docker/dockerfile:1.4

ARG PEXECS=1
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

FROM debian:latest as base

FROM base as build
WORKDIR /app

ARG FULL
ARG PEXECS
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

ENV FULL=$FULL
ENV PEXECS=$PEXECS
ENV EXPERIMENTS=$EXPERIMENTS
ENV SUITES=$SUITES
ENV MEASUREMENTS=$MEASUREMENTS

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get -y install make bc perl build-essential curl git cmake python3.11 python3.11-venv python3.11-distutils python3-pip \
    libtinfo-dev libzip-dev ninja-build gdb pipx rsync \
    libdwarf-dev libunwind-dev libboost-dev libfontconfig1-dev fontconfig libboost-iostreams-dev \
    libboost-all-dev libboost-program-options-dev libboost-regex-dev zlib1g-dev zstd libelf-dev elfutils \
    libdw-dev pkg-config libssl-dev zlib1g-dev libzstd-dev liblzma-dev \
    libffi-dev libedit-dev llvm-dev clang procps autotools-dev xz-utils \
    gperf bison flex xvfb time

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set environment variables for Rust and Cargo
ENV CARGO_HOME=/cargo
ENV RUSTUP_HOME=/rustup
ENV PATH=$CARGO_HOME/bin:$PATH

# Install rustup and Rust nightly toolchain
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly

# Show versions for debugging
RUN rustc --version && cargo --version

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip --break-system-packages && \
    pip install .[dev] --break-system-packages

COPY . .

RUN mkdir -p /app/artefacts

RUN --mount=type=cache,target=/cache \
    if [ "$FULL" = "false" ]; then \
    ./fetch_binaries.sh --out-dir /cache && \
    cp -r /cache/bin /app/artefacts/ && \
    ls -ls /app/artefacts/bin; \
    fi

RUN  invoke build-benchmarks $EXPERIMENTS $SUITES $MEASUREMENTS

FROM scratch as log_export
COPY --from=build /app/experiment.log /docker-run-full.log

FROM build as runtime

ARG BUILD_QUICK=false
ARG PEXECS
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

ENV BUILD_QUICK=$BUILD_QUICK
ENV PEXECS=$PEXECS
ENV EXPERIMENTS=$EXPERIMENTS
ENV SUITES=$SUITES
ENV MEASUREMENTS=$MEASUREMENTS

WORKDIR /app

RUN pip install --upgrade pip --break-system-packages && \
    pip install .[dev] --break-system-packages

RUN ls -la
run invoke run-benchmarks $PEXECS $EXPERIMENTS $SUITES $MEASUREMENTS

