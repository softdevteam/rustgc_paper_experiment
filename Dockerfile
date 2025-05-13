FROM debian:latest as base

FROM base as build_alloy
WORKDIR /app
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
  rm -f /etc/apt/apt.conf.d/docker-clean && \
  apt-get update && \
  apt-get -y install make build-essential curl git cmake python3 \
    libtinfo-dev libzip-dev ninja-build gdb pipx rsync \
    libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev \
    libboost-all-dev libboost-program-options-dev libboost-regex-dev zlib1g-dev zstd libelf-dev elfutils \
    libdw-dev pkg-config libssl-dev zlib1g-dev libzstd-dev liblzma-dev \
    libffi-dev libedit-dev llvm-dev clang procps autotools-dev \
    gperf bison flex xvfb
COPY . builder
RUN mkdir -p /app/alloy/bin
RUN --mount=type=cache,target=/app/builder/alloy/ \
    --mount=type=cache,target=/app/builder/heaptrack/ \
    --mount=type=cache,target=/app/builder/bdwgc/ \
    cd builder && make build-alloy && \
    cp -r alloy/bin /app/alloy/bin && \
    cp -r heaptrack /app/ && \
    cp -r bdwgc /app/

FROM base as build_benchmarks
WORKDIR /app
COPY --from=build_alloy /app .
RUN make build-benchmarks

