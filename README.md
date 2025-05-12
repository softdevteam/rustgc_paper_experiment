# Experiment for "Garbage Collection for Rust: The Finalizer Frontier"

This is the experiment for the paper ["Garbage Collection for Rust: The
Finalizer Frontier"](https://arxiv.org/abs/2504.01841) by Jacob Hughes and
Laurence Tratt. The repository for the paper can be found at
https://github.com/softdevteam/rustgc_paper.

## Quickstart

We provide a docker file so that you can easily run the experiments by cloning
the repository and running a single command:

`make docker-bench`

This will ensure all dependencies are installed, build the different variants
of Alloy needed for the experiments, and then run them. It will output the
results in the following two top-level directories: 
    - `results`: the raw data from the experiments. Includes perf data, memory
      traces, and Alloy metrics for each iteration.
    - `plots`: the various plots and tables as seen in the paper. These are
      `.ltx` and `.pdf` files.

You can examine the data in `plots` individually, or you can rebuild our paper
with the data from your run by first downloading the paper source:

```
git clone https://github.com/softdevteam/rustgc_paper
cd rustgc_paper
```

Then copy the `plots` directory over, replacing the one in `rustgc_paper`. You
can then run `make` from inside `rustgc_paper` which will build a file
`rustgc_paper.pdf` with your data in.

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
apt install -y install make build-essential curl git cmake python3 libtinfo-dev libzip-dev ninja-build gdb pipx rsync libdwarf-dev libunwind-dev libboost-dev libboost-iostreams-dev libboost-program-options-dev zlib1g-dev zstd elfutils pkg-config libssl-dev zlib1g-dev liblzma-dev libffi-dev libedit-dev llvm-dev clang procps autotools-dev gperf bison flex xvfb
```

#### Arch

```
sudo pacman -Syu --needed base-devel make curl git cmake python ncurses libzip ninja gdb python-pipx rsync libdwarf libunwind boost boost-libs zlib zstd elfutils pkgconf openssl xz libffi libedit llvm clang procps-ng autoconf automake gperf bison flex xorg-server-xvfb
```

### Building and running the experiments

To reproduce the results in the paper, simply run `make all`, which by default
uses the same benchmarking and graph processing parameters from the paper.
