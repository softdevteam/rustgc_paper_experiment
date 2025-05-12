FROM debian:latest
WORKDIR /app
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
  rm -f /etc/apt/apt.conf.d/docker-clean && \
  apt-get update && \
  apt-get -y install make build-essential curl git cmake python3 \
    libtinfo-dev libzip-dev ninja-build gdb pipx rsync \
    libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev \
    libboost-program-options-dev zlib1g-dev zstd elfutils \
    pkg-config libssl-dev zlib1g-dev liblzma-dev libffi-dev libedit-dev \
    llvm-dev clang procps autotools-dev gperf bison flex xvfb


COPY . .
RUN ls

# Cache build artifacts
RUN --mount=target=/var/cache/build,type=cache,sharing=locked \
    make build && \
    cp -r alloy /var/cache/build/alloy && \
    cp -r libgc /var/cache/build/libgc && \
    cp -r heaptrack /var/cache/build/heaptrack && \
    cp -r benchmarks /var/cache/build/benchmarks

CMD ["make", "bench"]

