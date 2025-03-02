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
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats

PLOT_DIR = None
RESULTS_DIR = None
EXPERIMENT = None
BIN = None
PEXECS = int(os.environ["PEXECS"])
Z = 2.576  # 99% interval

# ============== HELPERS =================


def print_success(message):
    print(f"\033[92m✓ {message}\033[0m")


def print_warning(message):
    """Print a warning message in yellow."""
    print(f"\033[93m⚠ {message}\033[0m")


def print_info(message):
    print(f"\033[94mℹ {message}\033[0m")


def print_error(message):
    print(f"\033[91m✗ {message}\033[0m")


def bytes_formatter(max_value):
    units = [
        ("B", 1),
        ("KiB", 1024),
        ("MiB", 1024 * 1024),
        ("GiB", 1024 * 1024 * 1024),
    ]

    for unit, factor in reversed(units):
        if max_value >= factor:
            break

    def format_func(x, pos):
        return f"{x/factor:.2f}"

    return FuncFormatter(format_func), unit


# ============== PLOT FORMATTING =================

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # LaTeX and font settings
        "text.usetex": True,
        "svg.fonttype": "none",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "sans-serif",
        # Basic graph axes styling
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        # Grid and line settings
        "lines.linewidth": 1,
        "grid.linewidth": 0.25,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        # Tick settings
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        # Legend and figure settings
        # "legend.title_fontsize": 0,
        "errorbar.capsize": 2,
    }
)

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
    "gcvs-rc": "RC",
    "premopt-opt": "Barriers Opt",
    "premopt-naive": "Barriers Naive",
    "premopt-none": "Barriers None",
    "premopt-opt": "Barriers Opt",
    "elision-naive": "Elision Naive",
    "elision-opt": "Elision Opt",
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

# ============== STATISTICS =================


def ci(row, pexecs):
    return Z * (row / math.sqrt(pexecs))


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


def normalize_time(df):
    group["normalized_time"] = (group["timestamp"] - group["timestamp"].min()) / (
        group["time"].max() - group["time"].min()
    )
    return group


def aggregate(grouped, col, method, unstack=True):
    df = grouped[col].apply(method).unstack()
    if unstack:
        df = df.unstack()
    else:
        df = df.reset_index()
    return (df["value"], df["ci"])


def normalize(df, baseline_col):
    timecol = "normalized_time"
    df[timecol] = df[timecol].astype(float)

    normcols = [
        "mem",
        "mem_ci",
        "peak_heap_usage",
        "peak_heap_usage_ci",
        "mean_heap_usage",
        "mean_heap_usage_ci",
    ]
    cmps = df["configuration"][~(df["configuration"] == baseline_col)].unique()

    baseline = (
        df[df["configuration"] == baseline_col]
        .sort_values(timecol)
        .reset_index(drop=True)
        .set_index(timecol)
        .sort_index()
    )

    def find_nearest(time_value):
        idx = np.abs(baseline.index - time_value).argmin()
        return baseline.iloc[idx]

    def normalize_value(row, value_col, timecol):
        if baseline.empty:
            return np.nan
        nearest = find_nearest(row[timecol])
        return row[value_col] / nearest[value_col]

    def normalize_ci(row, value_col, ci_col, timecol):
        if baseline.empty:
            return np.nan
        nearest = find_nearest(row[timecol])
        normalized_value = row[value_col] / nearest[value_col]
        return (
            np.sqrt(
                (row[ci_col] / row[value_col]) ** 2
                + (nearest[ci_col] / nearest[value_col]) ** 2
            )
            * normalized_value
            * Z
        )

    for value_col, ci_col in zip(normcols[::2], normcols[1::2]):
        df.loc[df["configuration"].isin(cmps), value_col] = df[
            df["configuration"].isin(cmps)
        ].apply(lambda row: normalize_value(row, value_col, timecol), axis=1)
        df.loc[df["configuration"].isin(cmps), ci_col] = df[
            df["configuration"].isin(cmps)
        ].apply(lambda row: normalize_ci(row, value_col, ci_col, timecol), axis=1)

    df = df.drop(df[df["configuration"] == baseline_col].index)
    return df


def normalize_time(df):
    for (c, b, p), group in df.groupby(["configuration", "benchmark", "pexec"]):
        min = group["time"].min()
        max = group["time"].max()
        idxs = group.index

        # Normalize time to 0-1 scale
        df.loc[idxs, "normalized_time"] = (df.loc[idxs, "time"] - min) / (max - min)
    return df


def interpolate(df, oversampling=1):
    interpolated = []

    for (c, b, p), group in df.groupby(["configuration", "benchmark", "pexec"]):
        samples = int(group["snapshot"].max() * oversampling)
        # print(f"Interpolating {c} with {samples} samples.")
        dist = np.linspace(0, 1, samples)
        # Aggregate duplicate normalized time values by calculating mean
        aggregated = (
            group.sort_values("normalized_time")
            .groupby("normalized_time")["mem"]
            .mean()
        )

        # Reindex to standard time points and interpolate
        series = (
            aggregated.reindex(index=np.union1d(aggregated.index, dist))
            .interpolate(method="linear")
            .loc[dist]
        )
        interpolated.append(
            pd.DataFrame(
                {
                    "configuration": c,
                    "benchmark": b,
                    "pexec": p,
                    "normalized_time": dist,
                    "mem": series.values,
                }
            )
        )

    # Concatenate all interpolated dataframes
    df = pd.concat(interpolated, ignore_index=True)

    df = (
        df.groupby(["configuration", "benchmark", "normalized_time"])["mem"]
        .agg(mem=("mean"), mem_ci=(lambda x: ci(x.std(), PEXECS)))
        .reset_index()
    )

    peak_memory, peak_ci = aggregate(
        df.groupby(["configuration", "benchmark"]),
        "mem",
        bootstrap_max_ci,
        unstack=True,
    )
    peak_memory = peak_memory.unstack().reset_index()
    peak_memory.rename(columns={0: "peak_heap_usage"}, inplace=True)
    peak_ci = peak_ci.unstack().reset_index()
    peak_ci.rename(columns={0: "peak_heap_usage_ci"}, inplace=True)
    df = df.merge(peak_memory, on=["configuration", "benchmark"])
    df = df.merge(peak_ci, on=["configuration", "benchmark"])

    mean_memory, mean_ci = aggregate(
        df.groupby(["configuration", "benchmark"]),
        "mem",
        bootstrap_mean_ci,
        unstack=True,
    )
    mean_memory = mean_memory.unstack().reset_index()
    mean_memory.rename(columns={0: "mean_heap_usage"}, inplace=True)
    mean_ci = mean_ci.unstack().reset_index()
    mean_ci.rename(columns={0: "mean_heap_usage_ci"}, inplace=True)
    df = df.merge(mean_memory, on=["configuration", "benchmark"])
    df = df.merge(mean_ci, on=["configuration", "benchmark"])
    return df


# ============== GRAPHS =================


def plot_perf_bar(outfile, values, errs, width):
    fig, ax = plt.subplots(figsize=(width, 4))
    values = values.rename(columns=CFGS)
    errs = errs.rename(columns=CFGS)
    values.plot(kind="bar", ax=ax, width=0.8, yerr=errs)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.legend().set_title(None)
    ax.set_xticklabels(values.index, rotation=45, ha="right")
    ax.set_ylabel("Wall-clock time (ms)\n(lower is better)")
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(f"Plotted graph: {EXPERIMENT}:{BIN}:perf")


def plot_mem_bar(outfile, values, errs, width):
    fig, ax = plt.subplots(figsize=(width, 4))
    values = values.rename(columns=CFGS)
    errs = errs.rename(columns=CFGS)
    means = values.drop(["peak_heap_usage"], axis=1)
    means_errs = errs.drop(["peak_heap_usage_ci"], axis=1)
    peaks = values.drop(["mean_heap_usage"], axis=1)
    peaks_errs = errs.drop(["mean_heap_usage_ci"], axis=1)
    means.plot(kind="bar", ax=ax, alpha=0.3, width=0.8, hatch="///", yerr=means_errs)
    peaks.plot(kind="bar", ax=ax, width=0.8, alpha=0.6, yerr=peaks_errs)
    formatter, unit = bytes_formatter(np.max(values["peak_heap_usage"].max()))

    ax.legend().set_title(None)
    ax.set_xticklabels(values.index, rotation=45, ha="right")
    ax.set_ylabel(f"Memory Usage ({unit}s)")
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(f"Plotted graph: {EXPERIMENT}:{BIN}:mem")


def plot_mem_time_series(outfile, benchmarks, rows, cols, cmp=False):
    benchmarks["configuration"] = benchmarks["configuration"].replace(CFGS)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()

    num_cfgs = benchmarks["configuration"].nunique()
    num_benchmarks = benchmarks["benchmark"].nunique()
    colours = [sns.color_palette("colorblind")[i] for i in range(num_cfgs)]

    formatter, unit = bytes_formatter(np.max(benchmarks["mem"].max()))

    for i, (bench, results) in enumerate(benchmarks.groupby("benchmark")):
        ax = axes[i]
        ax.set_title(f"{bench}")
        if not cmp:
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))
        mems = []
        for j, (cfg, samples) in enumerate(results.groupby("configuration")):
            samples = samples.sort_values("normalized_time")
            (real,) = ax.plot(
                samples["normalized_time"],
                samples["mem"],
                color=colours[j],
            )

            ax.fill_between(
                samples["normalized_time"],
                samples["mem"] - samples["mem_ci"],
                samples["mem"] + samples["mem_ci"],
                alpha=0.2,
                color=colours[j],
            )

            # Plot mean heap usage as line
            mean = samples["mean_heap_usage"].iloc[0]
            mean_ci = samples["mean_heap_usage_ci"].iloc[0]
            mean = ax.axhline(
                y=mean,
                color=colours[j],
                linestyle=":",
                alpha=0.7,
            )

            if cmp:
                ax.axhline(
                    y=1,
                    color="grey",
                    linestyle="-",
                    alpha=0.5,
                )

            # Plot peak heap usage as line
            peak = samples["peak_heap_usage"].iloc[0]
            peak_ci = samples["peak_heap_usage_ci"].iloc[0]
            peak = ax.axhline(
                y=peak,
                color=colours[j],
                linestyle="--",
                alpha=0.5,
            )

            mems.append(real)

    # Remove extra subplots
    for i in range(num_benchmarks, rows * cols):
        fig.delaxes(axes[i])

    if not cmp:
        fig.supylabel(f"Memory usage ({unit}s)", y=0.5, x=0.02, rotation=90)
    fig.supxlabel(f"Normalized Time", x=0.51)

    fig.legend(
        handles=[
            Line2D([], [], color=col, label=f"{cfg}", linestyle="-")
            for col, cfg in zip(colours, benchmarks["configuration"].unique().tolist())
        ],
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.51, 1.02),
    )
    plt.tight_layout(rect=[0.01, 0.01, 1, 1])
    plt.savefig(outfile, format="svg", bbox_inches="tight")
    print_success(f"Plotted graph: {EXPERIMENT}:{BIN}:mem:profiles")


def parse_metrics(dir):
    files = glob.glob(f"{dir}/*.csv")
    headers = [
        "elision enabled",
        "premature finalizer prevention enabled",
        "premopt enabled",
        "finalizers registered",
        "finalizers elided",
        "finalizers completed",
        "barriers visited",
        "Gc allocated",
        "Box allocated",
        "Rc allocated",
        "Arc allocated",
        "STW pauses",
    ]
    data = []
    for f in files:
        df = pd.read_csv(f, names=headers, skiprows=1)
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["pexec"] = base[0]
        df["configuration"] = base[1]
        df["benchmark"] = base[2]
        data.append(df)
    return pd.concat(data, ignore_index=True)


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
                            "configuration": base[1],
                            "benchmark": base[2],
                            "pexec": base[0],
                            "snapshot": snapshot,
                            "time": time,
                            "mem": mem_heap,
                        }
                    )
    return pd.DataFrame(data)


def parse_sampler(dir):
    csvs = glob.glob(f"{dir}/*.csv")
    data = []
    for f in csvs:
        df = pd.read_csv(f, header=0, names=["time", "mem"]).astype(float)
        df = df.assign(snapshot=range(0, len(df)))
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["pexec"] = base[0]
        df["configuration"] = base[1]
        df["benchmark"] = base[2]
        data.append(df)
    return pd.concat(data, ignore_index=True)


def parse_rebench(csv):
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


def add_gcvs_overview_entry():
    pass


def process_perf():
    df = parse_rebench(RESULTS_DIR / "perf.csv")
    perf, cis = aggregate(df, "wallclock", bootstrap_mean_ci)
    # maxrss = aggregate(pdata, "maxrss", bootstrap_mean_ci)

    plot_perf_bar(
        PLOT_DIR / "perf.svg",
        perf,
        cis,
        width=8,
    )


def process_gcvs():
    process_perf()
    # RSS data tends to be an unreliable metric since it includes memory of
    # shared libraries, heap, stack, and code segments. See [1]
    #
    # [1]: https://community.ibm.com/community/user/aiops/blogs/riley-zimmerman/2021/07/05/memory-measurements-part3
    rss = parse_sampler(resultsdir / "samples")
    rss = interpolate(normalize_time(rss))
    rss["configuration"] = rss["configuration"] + " rss"
    mem = parse_heaptrack(RESULTS_DIR / "heaptrack")
    mem = interpolate(normalize_time(mem), oversampling=0.1)
    plot_mem_time_series(PLOT_DIR / "profiles.svg", mem, 7, 4)
    add_gcvs_overview_entry()


def process_premopt():
    metrics = parse_metrics(RESULTS_DIR / "metrics")

    # Basic sanity checking
    premopt = metrics[metrics["configuration"] == "premopt-opt"]
    naive = metrics[metrics["configuration"] == "premopt-naive"]
    none = metrics[metrics["configuration"] == "premopt-none"]

    assert (
        premopt["premature finalizer prevention enabled"].astype("int")
        == 1 & premopt["premopt enabled"].astype("int")
        == 1
    ).all()

    assert (
        naive["premature finalizer prevention enabled"].astype("int")
        == 1 & naive["premopt enabled"].astype("int")
        == 0
    ).all()

    assert (
        none["premature finalizer prevention enabled"].astype("int")
        == 0 & none["premopt enabled"].astype("int")
        == 0
    ).all()

    process_perf()
    mem = parse_heaptrack(RESULTS_DIR / "heaptrack")
    mem = interpolate(normalize_time(mem), oversampling=0.05)
    norm = normalize(mem, baseline_col="premopt-none")
    plot_mem_time_series(PLOT_DIR / "profiles", norm, 7, 4, cmp=True)


def process_elision():
    process_perf()
    mem = parse_heaptrack(RESULTS_DIR / "heaptrack")
    mem = interpolate(normalize_time(mem), oversampling=0.05)
    norm = normalize(mem, baseline_col="elision-naive")
    plot_mem_time_series(PLOT_DIR / "profiles", norm, 7, 4, cmp=True)


def main():
    global RESULTS_DIR
    global PLOT_DIR
    global BIN
    global EXPERIMENT

    RESULTS_DIR = Path(sys.argv[2])
    PLOT_DIR = Path(sys.argv[1])
    BIN = Path(sys.argv[1]).parts[-1]
    EXPERIMENT = Path(sys.argv[1]).parts[-2]

    if not os.path.exists(RESULTS_DIR / "perf.csv") and not os.path.exists(
        RESULTS_DIR / "mem.csv"
    ):
        print_error(f"No data found for {EXPERIMENT}:{BIN}.")
        sys.exit()

    if EXPERIMENT == "gcvs":
        process_gcvs()
    elif EXPERIMENT == "premopt":
        process_premopt()
    elif EXPERIMENT == "elision":
        process_elision()


if __name__ == "__main__":
    main()
