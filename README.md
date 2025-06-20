# Experiment for "Garbage Collection for Rust: The Finalizer Frontier"

This is the experiment for the paper ["Garbage Collection for Rust: The
Finalizer Frontier"](https://arxiv.org/abs/2504.01841) by Jacob Hughes and
Laurence Tratt. The repository for the paper can be found at
https://github.com/softdevteam/rustgc_paper.

## Quickstart

To quickly run the experiments using prebuilt binaries, first make sure your
system meets the [requirements](#system-requirements-1) and has the necessary
[dependencies](#dependencies-1) installed. Also, ensure that Docker is running
before proceeding.

Then, execute the following commands:

```sh
git clone https://github.com/softdevteam/rustgc_paper_experiment
cd rustgc_paper_experiment
make run-quick
```

This will run each experiment 5 times using the pre-built Docker image. Raw
experiment data will be saved in the `results` directory at the project root,
and the generated plots will be available in the `plots` directory.

For more information and alternative setup methods, please refer to the
sections below.

## Reproducing our experiments

We offer two ways to run our experiments:

1. **[Using the Provided Docker Image (Recommended)](#using-the-docker-image):**
 This method uses our pre-built Docker image, which comes with all necessary
dependencies and offers the simplest setup experience. Note that results may be
slightly less accurate than running on bare metal due to potential
virtualization overhead.

2. **[Running Natively (Bare Metal)](#running-natively-bare-metal):**
Alternatively, you can run the experiments directly on your own system. This
involves manually installing all required dependencies, but may yield
more accurate benchmarking results by avoiding virtualization overhead.

> [!WARNING]
> **Platform Limitations**: Bare-metal experiments with Alloy have only been
> tested on Linux systems with x86-64 hardware. Running bare-metal experiments
> on macOS is not supported natively due to platform-dependent features in
> Alloy;  macOS users must therefore use provided Docker image, but note that
> this will be slower because it relies on QEMU-based emulation. Support for
> other operating systems, including BSD-like platforms, is currently unknown
> and untested. Other OSes may also require additional dependencies that we are
> unaware of.

## Using the Docker Image (Recommended)

This approach relies on a pre-built Docker image that includes all required
dependencies, providing a consistent environment across different host systems.
To run experiments using this method, ensure your host machine meets the
following requirements and has the necessary dependencies installed.

### System Requirements

- Any 64-bit OS capable of running Docker with Linux container support (e.g.,
Linux, macOS with Docker Desktop, or Windows with Docker Desktop)
- x86_64 or ARM architecture
- At least 8 GB RAM and 30 GB free disk space
- Internet connection

### Dependencies

- `git`
- `GNU make`
- `docker-engine`
- `docker-buildkit`

If you need to install Docker, refer to [Appendix A: Installing Docker on
Debian](#appendix-a-installing-docker-on-debian) for a quick Debian-specific
guide. For installation instructions on other platforms, please see the
[official Docker documentation](https://docs.docker.com/engine/install/).

---

### Running the experiments

Begin by cloning the repository and navigating into its directory:

```sh
git clone https://github.com/softdevteam/rustgc_paper_experiment
cd rustgc_paper_experiment
```

You can now run one of the two docker-based experiment methods.

#### 1. Using Prebuilt Binaries (Recommended for Quick Setup)

This method uses prebuilt binaries for the different Alloy and Benchmark
configurations within the Docker image. It is ideal if you want to save time or
quickly verify the experiments without building everything from source.

With the Docker service running, execute the following command:

```sh
make run-quick
```

This will execute each experiment 5 times.

#### 2. Building Configurations from Source (For Full Replication)

This method involves building all components from source within the Docker
environment. It is recommended if you wish to fully replicate the experiments
as described in our paper, or if you want to inspect or modify the source code
during the process.

With the Docker service running, execute the following command:

```sh
make run-full
```

This will run each experiment 30 times -- the same number of iterations we used
in the paper.

## Filtering which experiments are run

We provide the following additional environment variables which can be used to
run a subset of the available experiments.

`BENCHMARKS` -- The benchmarks to run, provided as a space separated list containing any combination of the following: `alacritty`, `binary-trees`, `fd`, `grmtools`, `regex-redux`, `ripgrep`, or `som`.

`EXPERIMENTS` -- The experiments to run, provided as a space separated list containing any combination of the following: `gcvs`, `elision`, `premopt`.

`METRICS` -- `perf` or `mem`. Note that the bulk of running time needed to run
these experiments comes from the `mem` experiments, so you may prefer to just run the `perf` ones.

All variables can be used in combination with eachother.

## Running the experiments on bare-metal

We recommend that you use the docker image as there are lots of required
dependencies and the Dockerfile guarantees that they are pinned to the correct
version, however, if you wish to run the experiments on your own machine you
can do so as follows.

### System Requirements
- 64-bit Linux (x86_64)
- 4GiB RAM minimum (8GiB recommended)
- At least 60GiB disk-space for build artefacts
- 1GiB disk-space for each benchmark iteration. (e.g. to reproduce our
  experiment using 30 iterations, you will need 30GiB of disk-space.) [^1]

[^1]: Around 99% of the disk usage for benchmarking results comes from our
detailed heap profiling traces recorded from the memory experiments. If you
only want to run performance benchmarks, you can avoid this while also
dramatically speeding up the time needed to run the benchmarks.

### Dependencies

#### Debian / Ubuntu

```
apt install -y make build-essential curl git cmake python3 libtinfo-dev libzip-dev ninja-build gdb pipx rsync libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev libboost-program-options-dev zlib1g-dev zstd elfutils pkg-config libssl-dev zlib1g-dev liblzma-dev libffi-dev libedit-dev llvm-dev clang procps autotools-dev gperf bison flex xvfb
```

#### Arch

```
sudo pacman -Syu --needed base-devel make curl git cmake python ncurses libzip ninja gdb python-pipx rsync libdwarf libunwind boost boost-libs zlib zstd elfutils pkgconf openssl xz libffi libedit llvm clang procps-ng autoconf automake gperf bison flex xorg-server-xvfb
```

### Building and running on bare-metal

Once you have installed the required dependencies, you can run the experiments with the following:

`make bare-metal`

The same environment variables as above will also work here too.
