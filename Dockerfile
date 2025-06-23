# Use BuildKit features
# syntax=docker/dockerfile:1.4

FROM debian:latest as base

FROM base as build
WORKDIR /app

# Add build argument for prebuilt binaries
ARG PREBUILT_BINS=false

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get -y install make build-essential curl git cmake python3.11 python3.11-venv python3.11-distutils python3-pip \
    libtinfo-dev libzip-dev ninja-build gdb pipx rsync \
    libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev \
    libboost-all-dev libboost-program-options-dev libboost-regex-dev zlib1g-dev zstd libelf-dev elfutils \
    libdw-dev pkg-config libssl-dev zlib1g-dev libzstd-dev liblzma-dev \
    libffi-dev libedit-dev llvm-dev clang procps autotools-dev \
    gperf bison flex xvfb

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

RUN --mount=type=cache,target=/app/artefacts \
    if [ "$USE_PREBUILT_BINARIES" = "false" ]; then \
    invoke build-benchmarks; \
    else \
    echo "Using prebuilt binaries"; \
    fi


FROM scratch as log_export
COPY --from=build /app/experiment.log /docker-run-full.log

FROM debian:latest as runtime
WORKDIR /app
COPY --from=build /app/artefacts /app/artefacts

CMD invoke run-benchmarks $PEXECS \
    --experiments=$EXPERIMENTS \
    --suites=$SUITES \
    --metrics=$MEASUREMENTS

