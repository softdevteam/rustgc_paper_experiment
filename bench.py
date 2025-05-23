import os
import shutil
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from invoke import task
from tqdm import tqdm

# --- Configuration ---
ALLOY_REPO = "https://github.com/softdevteam/alloy"
ALLOY_COMMIT = "master"
LIBGC_PATH = "bdwgc/lib"

METRICS = ["mem", "perf"]

EXPERIMENTS = {
    "gcvs": {
        "metrics": ["mem", "perf"],
        "config_template": "gcvs.{metric}.config.toml",  # Now uses metrics
    },
    "premopt": {
        "configs": ["none", "opt", "naive"],
        "metrics": ["mem", "perf"],
        "config_template": "premopt.{config}.{metric}.config.toml",
    },
    "elision": {
        "configs": ["naive", "opt"],
        "metrics": ["mem", "perf"],
        "config_template": "elision.{config}.{metric}.config.toml",
    },
}

BENCHMARKS = {
    "alacritty": {
        "alacritty": {
            "url": "https://github.com/alacritty/alacritty.git",
            "rev": "1063706f8e8a84139e5d2b464a4978e9d840ea17",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "fd": {
        "fd": {
            "url": "https://github.com/sharkdp/fd",
            "rev": "a4fdad6ff781b5b496c837fde24001b0e46973d6",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "grmtools": {
        "grmtools": {
            "url": "https://github.com/softdevteam/grmtools",
            "rev": "a0972be0777e599a3dbca710fb0a595c39560b69",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "som": {
        "som-rs": {
            "url": "https://github.com/Hirevo/som-rs",
            "rev": "35b780cbee765cca24201fe063d3f1055ec7f608",
            "gcvs_configs": ["baseline", "gc", "rc"],
            "bins": ["som-interpreter-ast", "som-interpreter-bc"],
        },
        "yksom": {
            "url": "https://github.com/softdevteam/yksom",
            "rev": "master",
        },
    },
    "ripgrep": {
        "ripgrep": {
            "url": "https://github.com/burntsushi/ripgrep",
            "rev": "de4baa10024f2cb62d438596274b9b710e01c59b",
            "gcvs_variants": ["baseline", "gc", "arc"],
        }
    },
    # "binary_trees": {
    #     "binary_trees": {
    #         "path": "binary_trees",
    #         "gcvs_variants": ["baseline", "gc", "arc"],
    #     }
    # },
}


# --- Path Configuration ---
ALLOY_DIR = Path("alloy").resolve()
BIN_DIR = Path("bin").resolve()
BENCHMARK_BIN_DIR = BIN_DIR / "benchmarks"
BUILD_DIR = Path("build").resolve()
BENCHMARK_BUILD_DIR = BUILD_DIR / "benchmarks"
SRC_DIR = Path("src").resolve()
CONFIG_DIR = Path("configs")
BENCHMARKS_DIR = Path("benchmarks").resolve()
RESULTS_DIR = Path("results").resolve()
REBENCH_EXEC = Path("venv/bin/rebench").resolve()


LIBGC_REPO = "https://github.com/softdevteam/bdwgc"
LIBGC_COMMIT = "e49b178f892d8e4b65785029c4fba3480850ce62"
LIBGC_DIR = Path("bdwgc").absolute()
LIBGC_BUILD_DIR = LIBGC_DIR / "build"
LIBGC_LIB_DIR = LIBGC_DIR / "lib"

BUILD_ENV = os.environ.copy()
BUILD_ENV["LD_LIBRARY_PATH"] = f"{LIBGC_LIB_DIR}"
BUILD_ENV["RUSTFLAGS"] = f"-L {LIBGC_LIB_DIR}"

import errno
import tempfile


def symlink_ignore_exists(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def rebench(pexecs, conf, outfile, experiment):
    return [
        str(REBENCH_EXEC),
        "-R",
        "-D",
        "--invocations",
        str(pexecs),
        "--iterations",
        "1",
        "-df",
        str(outfile),
        str(conf),
        experiment,
    ]


def mk_benchmark_wrapper(path, name, experiment, metric, metricsdir):
    wrapper = path / name
    alloy_log = (
        metricsdir / f"{name}.$INVOCATION.{experiment}-{path.name}.$BENCHMARK.csv"
    )
    with open(wrapper, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f"INVOCATION=$1\n")
        f.write(f"BENCHMARK=$1\n")
        f.write(f"shift 2\n")
        if experiment == "gcvs" and path.name != "gc":
            f.write(f"export GC_DONT_GC=true\n")
        f.write(f'export ALLOY_LOG="{alloy_log}"\n')
        f.write(f'$(realpath $(dirname "$0"))/{name}-inner "$@"')
    st = os.stat(wrapper)
    os.chmod(wrapper, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _bench_inner(c, pexecs, experiment, benchmark, suite, metric):
    if suite == "yksom" and experiment == "gcvs":
        return

    outdir = RESULTS_DIR / experiment / benchmark / metric
    metricsdir = outdir / "metrics" / "runtime"
    metricsdir.mkdir(parents=True, exist_ok=True)
    benchmarks = BIN_DIR / "benchmarks" / experiment / benchmark / suite
    # td = Path("testtemp")
    # td.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        if benchmark == "som":
            symlink_ignore_exists(
                SRC_DIR / benchmark / "som-rs" / "core-lib", td / "SOM"
            )

        for cfg in benchmarks.iterdir():
            if not cfg.is_dir():
                continue
            (td / cfg.name).mkdir(parents=True, exist_ok=True)

            bins = [bin for bin in (cfg / metric).iterdir()]
            for bin in bins:
                symlink_ignore_exists(bin, (td / cfg.name) / f"{bin.name}-inner")
                mk_benchmark_wrapper(
                    (td / cfg.name), bin.name, experiment, metric, metricsdir
                )

        conf = BENCHMARKS_DIR / benchmark / "rebench" / f"{suite}.conf"
        symlink_ignore_exists(conf, td / conf.name)
        resultfile = outdir / f"{conf.stem}.csv"
        cmd = rebench(pexecs, conf, resultfile, experiment)
        with c.cd(td):
            c.run(" ".join(cmd))


def parse_args(e, b, m):
    e = e.split() if e else EXPERIMENTS.keys()
    b = b.split() if b else BENCHMARKS.keys()
    m = m.split() if m else METRICS
    return (e, b, m)


@task
def bench(c, pexecs=1, experiments=None, benchmarks=None, metrics=None):
    experiments, benchmarks, metrics = parse_args(experiments, benchmarks, metrics)
    for e in experiments:
        for b in benchmarks:
            for s in BENCHMARKS[b]:
                for m in metrics:
                    _bench_inner(c, pexecs, e, b, s, m)


def _clean_inner(experiment, benchmark, metric):
    resultdir = RESULTS_DIR / experiment / benchmark / metric
    shutil.rmtree(resultdir, ignore_errors=True)


@task
def clean_results(c, pexecs=1, experiments=None, benchmarks=None, metrics=None):
    experiments, benchmarks, metrics = parse_args(experiments, benchmarks, metrics)
    for e in experiments:
        for b in benchmarks:
            for m in metrics:
                _clean_inner(e, b, m)
