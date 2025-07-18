# Experiment for "Garbage Collection for Rust: The Finalizer Frontier"

This is the experiment for the paper ["Garbage Collection for Rust: The
Finalizer Frontier"](https://arxiv.org/abs/2504.01841) by Jacob Hughes and
Laurence Tratt. The repository for the paper can be found at
https://github.com/softdevteam/rustgc_paper.


## Quickstart

To quickly run the experiments using prebuilt binaries, first make sure your
system meets the [requirements](#system-requirements-1) and has the necessary
[dependencies](#dependencies-1) installed.

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


## Table of Contents

- [Reproducing our experiments](#reproducing-our-experiments)
  - [Using the Docker Image (Recommended)](#using-the-docker-image-recommended)
    - [System Requirements](#system-requirements)
    - [Dependencies](#dependencies)
    - [Running the experiments](#running-the-experiments)
      - [1. Using Prebuilt Binaries (Recommended for Quick Setup)](#1-using-prebuilt-binaries-recommended-for-quick-setup)
      - [2. Building Configurations from Source (For Full Replication)](#2-building-configurations-from-source-for-full-replication)
  - [Running the experiments on bare-metal](#running-the-experiments-on-bare-metal)
    - [System Requirements](#system-requirements-1)
    - [Dependencies](#dependencies-1)
    - [Building and running on bare-metal](#building-and-running-on-bare-metal)
- [Customizing Experiment Runs](#customizing-experiment-runs)
- [Differences from Initial Submission](#differences-from-initial-submission)
  - [Experiment Modifications](#experiment-modifications)
  - [Alloy Modifications](#alloy-modifications)
  - [Observed Differences in Results](#observed-differences-in-results)


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

If you need to install Docker, refer to [Appendix: Installing Docker on
Debian](#appendix-installing-docker-on-debian) for a quick Debian-specific
guide. For installation instructions on other platforms, please see the
[official Docker documentation](https://docs.docker.com/engine/install/).

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

## Running the experiments on bare-metal

We recommend that you use the docker image as there are lots of required
dependencies and the Dockerfile guarantees that they are pinned to the correct
version, however, if you wish to run the experiments on your own machine you
can do so as follows.

> [!CAUTION]
> **Platform Limitations**: Bare-metal experiments with Alloy have only been
> tested on Linux systems with x86-64 hardware. Running bare-metal experiments
> on macOS is not supported natively due to platform-dependent features in
> Alloy;  macOS users must therefore use provided Docker image, but note that
> this will be slower because it relies on QEMU-based emulation. Support for
> other operating systems, including BSD-like platforms, is currently unknown
> and untested. Other OSes may also require additional dependencies that we are
> unaware of.

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

The experiments require a lot of development tools and libraries. Below, you’ll
find package installation commands for the two tested Linux distributions:
Debian and Arch Linux. Alternatively, you can install the packages manually and
use the checklist to track your progress. Most other distributions should
provide equivalent packages, though names may vary and we have not tested them.

<details>
  <summary>Debian</summary>

To install all dependencies at once, run the following command (with `sudo` or `doas`):

```sh
apt install -y make build-essential curl git cmake python3 libtinfo-dev libzip-dev ninja-build gdb pipx rsync libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev libboost-program-options-dev zlib1g-dev zstd elfutils pkg-config libssl-dev liblzma-dev libffi-dev libedit-dev llvm-dev clang procps autotools-dev gperf bison flex xvfb
```

Alternatively, you can install packages individually and check them off below:

- [ ] make
- [ ] build-essential
- [ ] curl
- [ ] git
- [ ] cmake
- [ ] python3
- [ ] libtinfo-dev
- [ ] libzip-dev
- [ ] ninja-build
- [ ] gdb
- [ ] pipx
- [ ] rsync
- [ ] libdwarf-dev
- [ ] libunwind-dev
- [ ] libboost-dev
- [ ] libboost-iostreams-dev
- [ ] libboost-program-options-dev
- [ ] zlib1g-dev
- [ ] zstd
- [ ] elfutils
- [ ] pkg-config
- [ ] libssl-dev
- [ ] liblzma-dev
- [ ] libffi-dev
- [ ] libedit-dev
- [ ] llvm-dev
- [ ] clang
- [ ] procps
- [ ] autotools-dev
- [ ] gperf
- [ ] bison
- [ ] flex
- [ ] xvfb

</details>

<details>
  <summary>Arch Linux</summary>

To install all dependencies at once, run the following command (with `sudo` or `doas`):

```sh
sudo pacman -Syu --needed base-devel make curl git cmake python ncurses libzip ninja gdb python-pipx rsync libdwarf libunwind boost boost-libs zlib zstd elfutils pkgconf openssl xz libffi libedit llvm clang procps-ng autoconf automake gperf bison flex xorg-server-xvfb
```

Alternatively, you can install packages individually and check them off below:

- [ ] base-devel
- [ ] make
- [ ] curl
- [ ] git
- [ ] cmake
- [ ] python
- [ ] ncurses
- [ ] libzip
- [ ] ninja
- [ ] gdb
- [ ] python-pipx
- [ ] rsync
- [ ] libdwarf
- [ ] libunwind
- [ ] boost
- [ ] boost-libs
- [ ] zlib
- [ ] zstd
- [ ] elfutils
- [ ] pkgconf
- [ ] openssl
- [ ] xz
- [ ] libffi
- [ ] libedit
- [ ] llvm
- [ ] clang
- [ ] procps-ng
- [ ] autoconf
- [ ] automake
- [ ] gperf
- [ ] bison
- [ ] flex
- [ ] xorg-server-xvfb

</details>

### Building and running on bare-metal

Once you have installed the required dependencies, you can run the experiments with the following:

`make bare-metal`

The same environment variables as above will also work here too.


## Customizing Experiment Runs

> [!NOTE]
> Running the full experiment suite with all configurations, as described in our
> paper, is a time- and resource-intensive process. On our benchmarking server
> (64 cores, 128 GiB RAM), a complete run takes over 48 hours and consumes
> approximately 300 GiB of disk space for build artifacts and raw results. To
> save time and storage, you may wish to run only a subset of experiments or
> reduce the number of iterations (or increase them, if desired). This can be
> easily configured using environment variables, which let you specify exactly
> what to run.

You can customise which experiments and benchmarks are run using the following
environment variables.

**These variables can be used whether you are running experiments via Docker or directly on bare metal.**

- **`EXPERIMENTS`**
  Space-separated list of experiments to run.

  Options:
  - **`gcvs`**: Compare Alloy with other memory management approaches (e.g.,
  Rc). Not all benchmark suites have the same variants, but each will include a
  variant using the baseline system allocator and the program’s original memory
  strategy.
  - **`elision`**: Evaluates the *finalizer elision optimization* (see Section
  5 of the paper) by comparing two Alloy configurations: with elision and
  without elision.
  - **`premopt`**: Evaluates the cost of *premature finalizer prevention*
  (Section 6 of the paper) by comparing three Alloy configurations: naive
  (barriers for every garbage-collected pointer), opt (unnecessary barriers
  optimized away), and none (idealized version with no barriers; this is unsound).

  **Default:** `EXPERIMENTS="gcvs elision premopt"`

- **`SUITES`**
  Space-separated list of benchmark suites to run.

  Options:
  - **`alacritty`**: Terminal emulator workload ([repo](https://github.com/alacritty/alacritty)).
  - **`binary-trees`**: Classic binary trees microbenchmark ([repo](#)).
  - **`fd`**: A rust alternative to the UNIX `find` command ([repo](https://github.com/sharkdp/fd)).
  - **`grmtools`**: Parsing benchmark of the grmtools error recovery algorithm ([repo](https://github.com/softdevteam/grmtools)).
  - **`regex-redux`**: Regular expression processing benchmark ([repo](#)).
  - **`ripgrep`**: Real-world text searching workload ([repo](https://github.com/BurntSushi/ripgrep)).
  - **`som-rs-ast`**: Smalltalk interpreter (AST variant) ([repo](https://github.com/Hirevo/som-rs)).
  - **`som-rs-bc`**: Smalltalk interpreter (bytecode variant) ([repo](https://github.com/Hirevo/som-rs)).
  - **`yksom`**: Alternative Smalltalk interpreter for Alloy configuration comparisons only ([repo](https://github.com/softdevteam/yksom)).
    *Note:* `yksom` does not run with the `gcvs` experiment.

  **Default:** `SUITES="alacritty binary-trees fd grmtools regex-redux ripgrep som-rs-ast som-rs-bc yksom"`

- **`MEASUREMENTS`**
  Specifies which types of data to record.

  Options:
  - **`perf`**: Collects performance data (wall-clock and system-time).
  - **`mem`**: Gathers detailed memory allocation data for the `gcvs` experiment using [KDE HeapTrack](https://github.com/KDE/heaptrack) (resource-intensive).
  - **`metrics`**: Records high-level experiment metrics (e.g., collection counts, pause times, finalizers run, etc ).

  **Default:** `MEASUREMENTS="perf mem metrics"`
  *Note:* The `mem` measurement is the most resource-intensive. For most purposes, using just `perf` and `metrics` will suffice and is much faster.

- **`PEXECS`**
  Number of process executions (iterations) per experiment.

  **Default:** `PEXECS=5` for the quick prebuilt binary Docker image, or `PEXECS=30` otherwise (as in our paper).
  *Note:* Fewer iterations will run faster but result in wider confidence intervals and potentially less statistically significant results.

You can combine these environment variables in any way to customize which
experiments are run.

**Example:**

To run the Docker prebuilt experiments with 10 process executions and only the `perf` and `metrics` measurements:

```sh
PEXECS=10 MEASUREMENTS="perf metrics" make run-quick
```


## Differences from Initial Submission

Since the initial submission, we have updated both the experimental evaluation
and Alloy itself. These changes may affect how the data is presented, as well
as the results themselves. Below, we outline the main modifications and any
observed impact on the data.

### Experiment Modifications

We have pre-emptively included additional metrics that were requested by
reviewers during the peer review process. However, for some of these, we
currently only have the raw data available. This will be updated before the
paper revision deadline. Metrics highlighted in **bold** below were explicitly
requested by reviewers; the others were added to improve the accuracy and
completeness of comparisons.

  These include:
  - **Baseline allocator results for each benchmark suite**
  - **GC cycle counts**
  - **A breakdown of GC pause times**
  - **Different heap sizes**
  - More detailed finalizer breakdown, including both recursive drop calls and the initial outer drop method.
  - Bug fixes to the heap metric breakdown, allowing more accurate recording of the number of different shared memory types.

### Alloy Modifications

Since the original submission, we have made several improvements to Alloy that may impact experimental results.
*Only changes with potential runtime impact are listed below.*

- A new implementation of the finalization queue that uses BDWGC’s `finalize_on_demand` API
  [[PR #179](https://github.com/softdevteam/alloy/pull/179), [PR #177](https://github.com/softdevteam/alloy/pull/177)]

- Alloy now always dynamically links against BDWGC’s `libgc` library, which may influence compiler optimizations such as inlining
  [[PR #189](https://github.com/softdevteam/alloy/pull/189), [PR #187](https://github.com/softdevteam/alloy/pull/187), [PR #183](https://github.com/softdevteam/alloy/pull/183), [PR #185](https://github.com/softdevteam/alloy/pull/185), [PR (bdwgc) #29](https://github.com/softdevteam/bdwgc/pull/29)]

- The default global allocator is now set to BDWGC’s allocator automatically, so users no longer need to specify it with `#[global_allocator]`. While this change is unlikely to affect performance, we cannot completely rule it out
  [[PR #192](https://github.com/softdevteam/alloy/pull/192)]


## A brief guide to the experiment process

The `Makefile` serves as the entry point to the experiment. Whether you choose
to run experiments inside Docker or on bare metal, it firsts downloads Alloy,
checks out a fixed version of it, and builds several configurations for the
different experiments. These binaries are stored in `artefacts/bin/alloy/`.

Next, each benchmark is downloaded, checked out to a fixed version, and
compiled with the different Alloy configurations. Some benchmarks include
harnesses which are part of this repo itself instead of downloaded, these can
be found in `src/`. Some benchmark variants uses patches found in `patch/`
which are automatically applied. The patches are named accordingly, identifying
which experiment variant they belong to. The resulting binaries are placed in
`artefacts/bin/benchmarks/`.

Once all binaries are prepared, ReBench is used to run the experiments, with
raw data saved to `results/`. The data is then processed and visualized using
Python scripts, most of which are organized as Invoke tasks. The final plots
are written to `plots/`, ready for direct inclusion in the paper.

## Appendix: Installing Docker on Debian

Install Docker Engine on Debian 11/12 by running the following as root:

```sh
sudo bash -c '
apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
'
```

Test that it's worked by running the `hello-world` container:

```sh
sudo docker run hello-world
```

See [Docker’s official docs](https://docs.docker.com/engine/install/debian/) for troubleshooting.

