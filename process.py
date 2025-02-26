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
    "gcvs-gc": "Alloy",
    "gcvs-rc": "RefCount (non-atomic)",
    "gc": "Alloy",
    "rc": "RefCount (non-atomic)",
    "premopt-opt": "Barriers Opt",
    "premopt-naive": "Barriers Naive",
    "premopt-none": "Barriers None",
    "premopt-opt": "Barriers Opt",
    "elision-naive": "Elision Naive",
    "elision-opt": "Elision Opt",
    "naive": "Naive",
    "none": "None",
    "opt": "Opt",
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


def pretty_name(col):
    l = col.split()
    if len(l) == 1:
        return CFGS[l[0]]
    else:
        return CFGS[l[1]]


def plot_bar(title, filename, data, width, unit):
    values = data[0]
    errs = data[1]
    sns.set(style="whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(width, 4))

    values = values.rename(columns=pretty_name)
    errs = errs.rename(columns=pretty_name)
    values.plot(kind="bar", ax=ax, width=0.8, yerr=errs)

    ax.legend().set_title(None)
    if len(values.columns) == 1:
        ax.legend().remove()

    ax.set_xticklabels(values.index, rotation=45, ha="right")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if unit == "ms":
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
    elif unit == "b":
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
    elif unit == "kb":
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
    else:
        raise ValueError("Unknown unit")
    ax.set_ylabel(title)
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
    print(f"==> Created plot: {filename}")


def plot_mem_time_series(data, outdir):
    for benchmark, cfgs in data.groupby("benchmark"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"{benchmark}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Usage")
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_tick_params(which="minor", size=0)
        ax.yaxis.set_tick_params(which="minor", width=0)

        for cfg, snapshot in cfgs.groupby("configuration"):
            ax.plot(snapshot["time"], snapshot["mem_heap_B"], label=f"{cfg}")

        ax.legend()
        ax.grid(linewidth=0.1)
        plt.tight_layout()
        plt.savefig(
            outdir / f"{benchmark.lower()}.svg", format="svg", bbox_inches="tight"
        )
        plt.close(fig)
        print(
            f"==> Saved time-series memory results to {outdir / f"{benchmark.lower()}.svg"}"
        )


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
        df["configuration"] = base[1]
        df["benchmark"] = base[2]
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
        if base[0] != "1":
            continue
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
                            "configuration": base[1],
                            "benchmark": base[2],
                            "snapshot": snapshot,
                            "time": time,
                            "mem_heap_B": mem_heap,
                        }
                    )
    return pd.DataFrame(data)


def parse_perfdata(csv):
    print(f"==> parsing perf data from {csv}")
    df = pd.read_csv(csv, sep="\t", skiprows=4, index_col="benchmark")
    pexecs = int(df["invocation"].max())
    assert pexecs == PEXECS
    perf = df[df["criterion"] == "total"].rename(columns={"value": "wallclock"})
    perf = perf[["executor", "wallclock"]]
    rss = df[df["criterion"] == "MaxRSS"].rename(columns={"value": "maxrss"})
    rss = rss[["executor", "maxrss"]]
    df = pd.merge(perf, rss, on=["benchmark", "executor"]).groupby(
        ["benchmark", "executor"]
    )
    return df


def aggregate(grouped, col, method):
    df = grouped[col].apply(method).unstack().unstack()
    return (df["value"], df["ci"])


def process_perf(resultsdir, outdir):
    pdata = parse_perfdata(resultsdir / "perf.csv")

    perf = aggregate(pdata, "wallclock", bootstrap_mean_ci)
    # maxrss = aggregate(pdata, "maxrss", bootstrap_mean_ci)
    # print(perf)

    plot_bar(
        "Wall-clock time (ms)\n(lower is better)",
        outdir / "perf.svg",
        perf,
        width=8,
        unit="ms",
    )


def process_mem(resultsdir, outdir):
    mdata = parse_heaptrack(resultsdir / "heaptrack")
    plot_mem_time_series(mdata, outdir / "mem")
    mdata = mdata.groupby(["benchmark", "configuration"])
    avgmem = aggregate(mdata, "mem_heap_B", bootstrap_mean_ci)
    maxmem = aggregate(mdata, "mem_heap_B", bootstrap_max_ci)

    # print(avgmem)
    plot_bar(
        "Average heap usage (KiB)\n(lower is better)",
        outdir / "avg_heap.svg",
        avgmem,
        width=8,
        unit="b",
    )
    plot_bar(
        "Max heap usage (KiB)\n(lower is better)",
        outdir / "max_heap.svg",
        avgmem,
        width=8,
        unit="b",
    )


def main():

    print(f"==> processing results for {sys.argv[1:]}")
    resultsdir = Path(sys.argv[2])
    outdir = Path(sys.argv[1])

    perf = resultsdir / "perf.csv"
    mem = resultsdir / "mem.csv"

    if not os.path.exists(perf) and not os.path.exists(mem):
        print(f"No results for {resultsdir}. Exiting...")
        sys.exit()

    if os.path.exists(perf):
        process_perf(resultsdir, outdir)

    if os.path.exists(mem):
        process_mem(resultsdir, outdir)


if __name__ == "__main__":
    main()

# raw = parse_metrics(resultsdir / "metrics").groupby(["benchmark", "configuration"])
# metrics = raw.mean()
# cis = raw.std().apply(ci, pexecs=10)
# mk_table(outdir / "metrics.tex", metrics, cis, columns=METRICS)
