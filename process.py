#! /usr/bin/env python

import csv
import gc
import glob
import math
import os
import pprint
import random
import sys
from os import listdir, stat
from pathlib import Path
from statistics import geometric_mean, stdev

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"errorbar.capsize": 2})

results = {}
pp = pprint.PrettyPrinter(indent=4)

PEXECS = int(os.environ["PEXECS"])

EXPERIMENTS = {
    "gcrc": r"GcRc",
    "elision": r"Elision",
    "premopt": r"PremOpt",
}

SUITES = {
    "som-rs-ast": r"\somrsast",
    "som-rs-bc": r"\somrsbc",
    "yksom": r"\yksom",
}

CFGS = {
    "bopt-perf": "Barriers Opt",
    "bnaive-perf": "Barriers Naive",
    "bnone-perf": "Barriers None",
    "bopt-mem": "Barriers Opt",
    "bnaive-mem": "Barriers Naive",
    "bnone-mem": "Barriers None",
}

METRICS = {
    "finalizers registered": "Finalizable Objects",
    "finalizers completed": "Total Finalized",
    "barriers visited": "Barrier Chokepoints",
    "Gc allocated": "Allocations (Gc)",
    "Box allocated": "Allocations (Box)",
    "Rc alocated": "Allocations (Rc)",
    "Arc allocated": "Allocations (Arc)",
    "STW pauses": r"Gc Cycles",
}


def bootstrap(
    values, kind, method, num_bootstraps=10000, confidence=0.99, symmetric=True
):
    res = stats.bootstrap(
        (values,),
        statistic=kind,
        n_resamples=num_bootstraps,
        confidence_level=confidence,
        method=method,
        vectorized=True,
    )

    value = kind(values)
    ci_lower, ci_upper = res.confidence_interval
    if symmetric:
        margin = max(value - ci_lower, ci_upper - value)
        data = {
            "value": value,
            "ci": margin,
        }
    else:
        data = {
            "value": value,
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high,
        }

    return pd.Series(data)


def bootstrap_geomean_ci(means, num_bootstraps=10000, confidence=0.99, symmetric=False):
    # We use the BCa (bias-corrected and accelerated) bootstrap method. This
    # can provide more accurate CIs over the more straightforward percentile
    # method but it is more computationally expensive -- though this doesn't
    # matter so much when we run this using PyPy.
    #
    # This is generally better for smaller sample sizes such as ours (where the
    # number of pexecs < 100), and where the dataset is not known to be
    # normally distributed.
    #
    # We could also consider using the studentized bootstrap method which
    # libkalibera tends to prefer when deealing with larger sample sizes.
    # Though this is more computationally expensive and the maths looks a bit
    # tricky to get right!
    print("Hello")
    print(means)
    method = "Bca"
    return bootstrap(means, stats.gmean, method, num_bootstraps, confidence, symmetric)


def bootstrap_mean_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.mean, "percentile", num_bootstraps, confidence, symmetric=True
    )


def bootstrap_max_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.max, "percentile", num_bootstraps, confidence, symmetric=True
    )


def plot_bar(filename, means, errs, width, kind):
    sns.set(style="whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(width, 4))

    # means = means.rename(columns=CFGS)
    # errs = errs.rename(columns=CFGS)
    means.plot(kind="bar", ax=ax, width=0.8, yerr=errs)

    # if len(means.columns) == 1:
    #     plt.gca().get_legend().remove()

    ax.set_xticklabels(means.index, rotation=45, ha="right")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if kind == "perf":
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel("Wall-clock time (ms)\n(lower is better)")
    else:
        ax.set_ylabel("Maximum resident set size (KiB)\n(lower is better)")
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
    ax.grid(linewidth=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_tick_params(which="minor", size=0)
    ax.yaxis.set_tick_params(which="minor", width=0)
    ax.xaxis.label.set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")


def mk_table(filename, vals, cis, columns):
    df = pd.DataFrame()
    for v, c in zip(vals.columns, cis.columns):
        df[v] = vals[v].round(2).astype(str) + " \pm " + cis[c].round(3).astype(str)

    with open(filename, "w") as f:
        f.write(df.unstack().rename(columns=columns).to_latex(index=False))


def ci(row, pexecs):
    Z = 2.576  # 99% interval
    return Z * (row / math.sqrt(pexecs))


def parse_metrics(mdir):
    csvs = glob.glob(f"{mdir}/*.log")
    m = []
    for f in csvs:
        df = pd.read_csv(f)
        base = os.path.splitext(os.path.basename(f))[0].split("-")
        df["configuration"] = base[0]
        df["benchmark"] = base[1]
        df = df.drop(
            [
                "elision enabled",
                "premature finalizer prevention enabled",
                "premopt enabled",
            ],
            axis=1,
        )
        m.append(df)
    return pd.concat(m, ignore_index=True)


def human_readable_bytes(x, pos=None):
    if x < 1024:
        return f"{x} B"
    elif x < 1024**2:
        return f"{x/1024:.1f} KiB"
    elif x < 1024**3:
        return f"{x/1024**2:.1f} MiB"
    else:
        return f"{x/1024**3:.1f} GiB"


def parse_heaptrack(dir):
    files = glob.glob(f"{dir}/*.massif")
    data = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        with open(f, "r") as f:
            for line in f:
                if line.startswith("snapshot="):
                    snapshot = int(line.split("=")[1])
                elif line.startswith("time="):
                    time = float(line.split("=")[1])
                elif line.startswith("mem_heap_B="):
                    mem_heap = int(line.split("=")[1])
                    data.append(
                        {
                            "configuration": base[0],
                            "benchmark": base[1],
                            "snapshot": snapshot,
                            "time": time,
                            "mem_heap_B": mem_heap,
                        }
                    )
    return pd.DataFrame(data)


def plot_mem_time_series(data, outdir):
    for benchmark, cfgs in data.groupby("benchmark"):
        print(benchmark)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{benchmark}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Usage")
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))

        for cfg, snapshot in cfgs.groupby("configuration"):
            ax.plot(snapshot["time"], snapshot["mem_heap_B"], label=f"{cfg}")

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            outdir / f"{benchmark.lower()}.svg", format="svg", bbox_inches="tight"
        )


print(f"==> processing results for {sys.argv[1:]}")

infile = sys.argv[1]
resultsdir = Path(infile).parent
suite = Path(sys.argv[2]).parts[-2]
experiment = Path(sys.argv[2]).parts[-3]
datapoint = Path(sys.argv[2]).stem
outdir = Path(sys.argv[2]).parent
outfile = sys.argv[2]

df = pd.read_csv(infile, sep="\t", skiprows=4, index_col="benchmark")
pexecs = int(df["invocation"].max())
assert pexecs == PEXECS

perf = (
    df.loc[df["criterion"] == "total"]
    .groupby(["benchmark", "executor"])["value"]
    .apply(bootstrap_mean_ci)
    .unstack()
)

plot_bar(
    outdir / "perf.svg",
    perf["value"],
    perf["ci"],
    width=8,
    kind="perf",
)

max_rss = (
    df.loc[df["criterion"] == "MaxRSS"]
    .groupby(["benchmark", "executor"])["value"]
    .apply(bootstrap_mean_ci)
    .unstack()
)

plot_bar(outdir / "max_rss.svg", max_rss["value"], max_rss["ci"], width=8, kind="mem")

mem_data = parse_heaptrack(resultsdir / "heaptrack")

mem = (
    mem_data.groupby(["benchmark", "configuration"])["mem_heap_B"]
    .apply(bootstrap_mean_ci)
    .unstack()
)
#
plot_bar(outdir / "mem_average.svg", mem["value"], mem["ci"], kind="mem", width=8)
#
mem_summary = (
    mem.groupby("configuration")["value"].apply(bootstrap_geomean_ci).unstack()
)

max = (
    mem_data.groupby(["benchmark", "configuration"])["mem_heap_B"]
    .apply(bootstrap_max_ci)
    .unstack()
)

plot_bar(
    outdir / "mem_max.svg",
    max["value"],
    max["ci"],
    kind="mem",
    width=8,
)

max_summary = (
    max.groupby("configuration")["value"].apply(bootstrap_geomean_ci).unstack()
)
plot_mem_time_series(mem_data, outdir / "mem")

raw = parse_metrics(outdir / "metrics").groupby(["benchmark", "configuration"])
metrics = raw.mean()
cis = raw.std().apply(ci, pexecs=pexecs)
mk_table(outdir / "metrics.tex", metrics, cis, columns=METRICS)
