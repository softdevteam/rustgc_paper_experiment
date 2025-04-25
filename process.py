#! /usr/bin/env python

import csv
import gc
import glob
import math
import os
import pprint
import random
import sys
import warnings
from os import listdir, stat
from pathlib import Path
from statistics import geometric_mean, stdev

from matplotlib.font_manager import FontProperties

warnings.simplefilter(action="ignore", category=FutureWarning)
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
DRYRUN = os.environ.get("DRYRUN", "false") in ["true", "on", "1", "yes"]
RESULTS_DIR = None
STATS_FILE = None
PEXECS = int(os.environ["PEXECS"])
EXPERIMENTS = os.environ.get("EXPERIMENTS", "").split()
BENCHMARKS = os.environ.get("BENCHMARKS", "").split()
PLOTS = os.environ.get("METRICS", "").split()
BOOTSTRAP = os.environ.get("BOOTSTRAP", "true") in ["true", "on", "1", "yes"]
Z = 2.576  # 99% interval


PLOTMAP = {
    "ripgrep": {"r": 0.12},
    "som-rs-ast": {"p": (9, 2), "r": 0.12},
    "som-rs-bc": {"p": (9, 2), "r": 0.4},
    "alacritty": {"p": (8, 3), "r": 0.4},
}

RESOLUTION = {
    "ripgrep": 0.12,
    "som-rs-ast": 0.12,
    "som-rs-bc": 0.4,
    "alacritty": 0.4,
}

# ============== HELPERS ================


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


def format_number(number):
    suffixes = ["", "K", "M", "B", "T"]
    magnitude = 0
    while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
        number /= 1000.0
        magnitude += 1
    return f"{number:.1f}{suffixes[magnitude]}".replace(".0", "")


def format_bytes(number, unit=None):
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]

    if unit is None:
        magnitude = 0
        while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
            number /= 1000.0
            magnitude += 1
    else:
        magnitude = suffixes.index(unit)
        number /= 1000.0**magnitude

    return f"{number:.2f}{suffixes[magnitude]}".replace(".00", "")


def ltxify(s):
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in s.split())


def fmt_value_ci(row):
    # if "ci" in row:
    #     ci = f"(± {row['ci']:.2f})"
    # else:
    ci = f"({row['lower']:.2f}-{row['upper']:.2f})"
    # upper = row['upper']:.2f
    return f"{row['value']:.2f} " + ci


def ltx_value_ci(row):
    # if "ci" in row:
    #     ci = f"± {row['ci']:.2f}"
    # else:
    ci = f"({row['lower']:.2f}-{row['upper']:.2f})"
    s = f"{row['value']:.2f} \\footnotesize{{{ci}}}"
    return s


def format_value_ci_asym(row):
    if math.isnan(row["ci"]):
        return "-"
    s = f"row['value']:.2f row['ci']"
    return s


def format_mem(row):
    return format_mem_ci(value, lower, upper)
    # ci = f"({format_bytes(row['lower'])}-{format_bytes(row['upper'])})"
    # return f"{format_bytes(row['value'])} " + ci


def format_mem_ci(value, lower, upper, unit=None, symmetric=False):
    if symmetric:
        ci = f"(± {format_bytes(value - lower, unit)})"
    else:
        ci = f"({format_bytes(lower, unit)}-{format_bytes(upper, unit)})"
    return f"{format_bytes(value,unit)} " + f"\\small{{{ci}}}"


def ltx_mem_ci(row):
    ci = f"({format_bytes(row['lower'])}-{format_bytes(row['upper'])})"
    s = f"{format_bytes(row['value'])} \\footnotesize{{{ci}}}"
    return s


# ============== PLOT FORMATTING =================

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # LaTeX and font settings
        "text.usetex": True,
        "svg.fonttype": "none",
        "text.latex.preamble": r"\usepackage{sfmath}",
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
        "errorbar.capsize": 1,
    }
)

SUITES = {
    "som-rs-ast": r"\somrsast",
    "som-rs-bc": r"\somrsbc",
    "alacritty": r"\alacritty",
    "yksom": r"\yksom",
    "grmtools": r"\grmtools",
    "binary-trees": r"\binarytrees",
    "binary-t": r"\binarytrees",  # it's a mem error
    "grmtool": r"\grmtools",  # it's a mem error
    "regex-redux": r"\regexredux",
    "ripgrep": r"\ripgrep",
    "fd": r"\fd",
}

CFGS = {
    "gcvs-gc": "Alloy",
    "gcvs-rc": r"\texttt{Rc<T>}",
    "gcvs-arc": r"\texttt{Arc<T>}",
    "gcvs-typed-arena": "Typed Arena",
    "gcvs-typed_arena": "Typed Arena",
    "gcvs-rust-gc": "Rust-GC",
    "premopt-opt": "Barriers Opt",
    "premopt-naive": "Barriers Naive",
    "premopt-none": "Barriers None",
    "premopt-opt": "Barriers Opt",
    "elision-naive": "No elision",
    "elision-opt": "Elision",
}

BMS = {
    # alacritty
    "cursor_motion": "Cur. Motion",
    "dense_cells": "Dense Cells",
    "light_cells": "Light Cells",
    "scrolling": "Scroll",
    "scrolling_bottom_region": "Scroll Region Bot.",
    "scrolling_bottom_small_region": "Scroll Region Bot.\n(small)",
    "scrolling_fullscreen": "Scroll Fullscreen",
    "scrolling_top_region": "Scroll Region Top",
    "scrolling_top_small_region": "Scroll Region Top\n(small)",
    "unicode": "Unicode",
    # ripgrep
    "linux_alternates": "Alternates",
    "linux_alternates_casei": "Alternates\n(case insensitive)",
    "linux_literal": "Literal",
    "linux_literal_casei": "Literal\n(case insensitive)",
    "linux_literal_default": "Literal\n(default)",
    "linux_literal_mmap": "Literal mmap",
    "linux_literal_casei_mmap": "Literal mmap\n(case insensitive)",
    "linux_literal_default": "Literal\n(default)",
    "linux_word": "Word",
    "linux_unicode_greek": "Unicode Greek",
    "linux_unicode_greek_casei": "Unicode Greek\n(case insensitive)",
    "linux_unicode_word_1": "Unicode Word",
    "linux_unicode_word_2": "Unicode Word (II)",
    "linux_re_literal_suffix": "Regex Literal Suffix",
    # fd
    "command-execution": "Command Exec.",
    "command-execution-large-output": "Command Exec.\n(large output)",
    "file-extension": "File Extension",
    "file-type": "File Type",
    "no-pattern": "No Pattern",
    "simple-pattern": "Simple Pattern",
    "simple-pattern-HI": "Simple Pattern\n(incl. hidden \& .gitignore)",
    # grmtools
    "eclipse": "Eclipse",
    "hadoop": "Hadoop",
    "jenkins": "Jenkins",
    "spring": "Spring",
}

CFG_ORDER = [
    "gcvs-gc",
    "gcvs-arc",
    "gcvs-rc",
    "elision-naive",
    "elision-opt",
    "premopt-naive",
    "premopt-none",
    "premopt-opt",
]


class Plotter:
    def __init__(
        self,
        suite,
        resolution,
        max_cols,
        perf_width,
        perf_height,
        mem_width,
        mem_height,
        barwidth,
        hzticks,
        legend_above=False,
        barh=False,
    ):
        self.suite = suite
        self.resolution = resolution
        self.max_cols = max_cols
        self.perf_width = perf_width
        self.perf_height = perf_height
        self.mem_width = mem_width
        self.mem_height = mem_height
        self.barwidth = barwidth
        self.hzticks = hzticks
        self.legend_above = legend_above
        self.barh = barh

    def plot_time_series(self, benchmarks, outdir, cmp=False):
        if DRYRUN:
            return
        # Convert configuration to categorical with custom order
        # Only include categories that exist in your data to avoid NaNs
        available_configs = benchmarks["configuration"].unique()
        ordered_configs = [cfg for cfg in CFG_ORDER if cfg in available_configs]
        benchmarks["configuration"] = pd.Categorical(
            benchmarks["configuration"], categories=ordered_configs, ordered=True
        )
        benchmarks = benchmarks.sort_values("configuration")
        num_bms = benchmarks["benchmark"].nunique()
        benchmarks = benchmarks.copy().replace({**CFGS, **BMS})

        axhcolours = ["#152238", "#d85128", "#000000"]
        axhlinestyles = ["--", ":", "-"]

        subplot_width = self.mem_width
        subplot_height = self.mem_height

        cols = min(num_bms, self.max_cols)
        rows = math.ceil(num_bms / cols)

        fig_width = subplot_width * cols
        fig_height = subplot_height * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        plt.tight_layout()

        num_cfgs = benchmarks["configuration"].nunique()
        colours = [sns.color_palette("colorblind")[i] for i in range(num_cfgs)]

        formatter, unit = bytes_formatter(np.max(benchmarks["mem"].max()))

        if not cmp:
            fig.supylabel(
                f"Memory usage ({unit}s)", fontsize=32, y=0.5, x=0.01, rotation=90
            )

        for i, (bench, results) in enumerate(benchmarks.groupby("benchmark")):
            ax = axes[i]
            ax.set_title(f"{bench}", fontsize=32, pad=20)
            if not cmp:
                ax.yaxis.set_major_formatter(FuncFormatter(formatter))
            else:
                ax.axhline(
                    y=1,
                    color="grey",
                    linestyle="-",
                    alpha=1,
                )

            ax.tick_params(
                axis="both",
                labelsize=18,
                width=0.2,
                pad=12,
            )
            mems = []
            for j, (cfg, samples) in enumerate(results.groupby("configuration")):
                samples = samples.sort_values("normalized_time")
                (real,) = ax.plot(
                    samples["normalized_time"],
                    samples["mem"],
                    color=colours[j],
                    label=cfg,
                )

                ax.fill_between(
                    samples["normalized_time"],
                    samples["lower"],
                    samples["upper"],
                    alpha=0.2,
                    color=colours[j],
                )
                # Plot mean heap usage as line
                mean = samples["mean"].iloc[0]
                # mfmt = format_mem_ci(
                #     mean,
                #     samples["mean_lower"].iloc[0],
                #     samples["mean_upper"].iloc[0],
                #     unit,
                #     symmetric=True,
                # )
                mean = ax.axhline(
                    y=mean,
                    color=axhcolours[j],
                    linestyle=axhlinestyles[j],
                    alpha=0.7,
                    label=f"Mean",
                )

                if cmp:
                    ax.axhline(
                        y=1,
                        color="grey",
                        linestyle="-",
                        alpha=0.5,
                    )

            box = ax.get_position()

            ax.legend().set_visible(False)

        for i in range(num_bms, rows * cols):
            fig.delaxes(axes[i])

        fig.supxlabel(f"Normalized Time", fontsize=32, x=0.515, y=0.005)
        plt.tight_layout(rect=[0.01, 0.01, 1, 1], h_pad=4, w_pad=4)
        plt.savefig(
            outdir / f"{self.suite}.svg", format="svg", bbox_inches="tight", dpi=300
        )
        # plt.savefig(outfile, format="svg", bbox_inches="tight", dpi=300)
        print_success(
            f"Plotted graph: {outdir.parts[-4]}:{outdir.parts[-3]}:{self.suite}:profiles"
        )
        plt.close()

    def plot_perf(self, results, outdir):
        if DRYRUN:
            return
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        available_configs = results["configuration"].unique()
        ordered_configs = [cfg for cfg in CFG_ORDER if cfg in available_configs]
        results["configuration"] = pd.Categorical(
            results["configuration"], categories=ordered_configs, ordered=True
        )
        results = results.sort_values("configuration")
        print(results)
        results = results.copy().replace({**CFGS, **BMS})
        num_cfgs = results["configuration"].nunique()
        num_benchmarks = results["benchmark"].nunique()
        colours = ["#152238", "#d85128", "#000000"]
        linestyles = ["--", ":", "-"]
        df = results.pivot(
            index="benchmark",
            columns="configuration",
            values=["value", "ci"],
        )
        df = df.sort_index(ascending=False)
        cfgs = results["configuration"].unique()
        fig, ax = plt.subplots(1, 1, figsize=(self.perf_width, self.perf_height))

        if self.barh:
            df.plot(
                kind="barh",
                y="value",
                xerr="ci",
                ax=ax,
                width=self.barwidth,
            )
            ax.set_xlabel("Wall-clock time (ms). Lower is better.", fontsize=6)
            ax.set_yticklabels(df.index, rotation=0)
            ax.yaxis.label.set_visible(False)
            ax.xaxis.set_major_formatter(formatter)
        else:
            df.plot(
                kind="bar",
                y="value",
                yerr="ci",
                ax=ax,
                width=self.barwidth,
            )
            ax.set_ylabel("Wall-clock time (ms). Lower is better.", fontsize=6)
            if self.hzticks:
                ax.set_xticklabels(df.index, rotation=0, ha="center")
            else:
                ax.set_xticklabels(df.index, rotation=45, ha="right")
            ax.xaxis.label.set_visible(False)
            ax.yaxis.set_major_formatter(formatter)

        for j, config in enumerate(cfgs):
            gmean = results[results["configuration"] == config]["geomean"].iloc[0]
            lower = results[results["configuration"] == config]["geomean_lower"].iloc[0]
            upper = results[results["configuration"] == config]["geomean_upper"].iloc[0]

            # Add the horizontal line with a label
            ax.axvline(
                x=gmean,
                color=colours[j],
                linestyle=linestyles[j],
                linewidth=0.8,
                alpha=0.7,
                label=f"G.mean: {gmean:.2f}ms ({lower:.2f}-{upper:.2f})",
            )

        handles, labels = ax.get_legend_handles_labels()
        handles = handles[2:] + handles[0:2]
        labels = labels[2:] + labels[0:2]

        legend = ax.legend(
            handles,
            labels,
            title=f"\\textsc{{{self.suite}}}",
            title_fontsize=7,
            loc="upper center",
            ncols=2,
            fontsize=6,
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.5)
        frame.set_edgecolor("grey")

        if self.legend_above and self.suite == "grmtools":
            plt.subplots_adjust(top=0.70)
            legend.set_bbox_to_anchor((0.5, 1.30))
        elif self.legend_above and self.suite == "fd":
            plt.subplots_adjust(top=0.20)
            legend.set_bbox_to_anchor((0.5, 1.80))
        elif self.legend_above and self.suite == "ripgrep":
            plt.subplots_adjust(top=0.30)
            legend.set_bbox_to_anchor((0.5, 1.70))

        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)

        plt.tight_layout()
        plt.tick_params(labelsize=6, width=0.2, pad=3)
        plt.savefig(outdir / f"{self.suite}.svg", format="svg", bbox_inches="tight")
        print_success(
            f"Plotted graph: {outdir.parts[-3]}:{outdir.parts[-2]}:{self.suite}:perf:individual"
        )
        plt.close()


grmplot = Plotter(
    "grmtools",
    resolution={"gcvs": 0.05, "elision": 0.01},
    max_cols=4,
    barwidth=0.8,
    perf_width=8,
    perf_height=3.5,
    mem_width=7.5,
    mem_height=5,
    hzticks=True,
    barh=True,
    legend_above=True,
)

PLOT_LAYOUTS = {
    "alacritty": Plotter(
        "alacritty",
        resolution={"gcvs": 0.4, "elision": 0.4},
        max_cols=5,
        barwidth=0.4,
        perf_width=9,
        perf_height=2,
        mem_width=7.5,
        mem_height=4,
        hzticks=True,
    ),
    "ripgrep": Plotter(
        "ripgrep",
        resolution={"gcvs": 0.14, "elision": 0.05, "premopt": 0.14},
        max_cols=7,
        barwidth=0.4,
        perf_width=9,
        perf_height=1.7,
        mem_width=7.5,
        mem_height=5,
        hzticks=True,
        legend_above=True,
    ),
    "fd": Plotter(
        "fd",
        resolution={"gcvs": 0.04, "elision": 0.04},
        max_cols=7,
        barwidth=0.3,
        perf_width=9,
        perf_height=1.7,
        mem_width=7.5,
        mem_height=5.5,
        hzticks=True,
        legend_above=True,
    ),
    "grmtool": grmplot,
    "grmtools": grmplot,
    "som-rs-ast": Plotter(
        "som-rs-ast",
        resolution={"gcvs": 0.12, "elision": 0.12},
        max_cols=7,
        barwidth=0.8,
        perf_width=9,
        perf_height=2,
        mem_width=7.5,
        mem_height=4,
        hzticks=False,
    ),
    "som-rs-bc": Plotter(
        "som-rs-bc",
        resolution={"gcvs": 0.4, "elision": 0.4},
        max_cols=7,
        barwidth=0.8,
        perf_width=9,
        perf_height=2,
        mem_width=7.5,
        mem_height=4,
        hzticks=False,
    ),
}


# METRICS = {
#     "finalizers registered": "Finalizable Objects",
#     "finalizers completed": "Total Finalized",
#     "barriers visited": "Barrier Chokepoints",
#     "Gc allocated": "Allocations (Gc)",
#     "Box allocated": "Allocations (Box)",
#     "Rc alocated": "Allocations (Rc)",
#     "Arc allocated": "Allocations (Arc)",
#     "STW pauses": r"Gc Cycles",
# }

BASELINE = {
    "som-rs-ast": "gcvs-rc",
    "som-rs-bc": "gcvs-rc",
    "grmtool": "gcvs-rc",
    "grmtools": "gcvs-rc",
    "binary-t": "gcvs-arc",
    "binary-trees": "gcvs-arc",
    "regex-redux": "gcvs-arc",
    "alacritty": "gcvs-arc",
    "fd": "gcvs-arc",
    "ripgrep": "gcvs-arc",
    "premopt": "premopt-none",
    "elision": "elision-naive",
}

PERF_PLOT_WIDTHS = {
    "alacritty": 8,
    "binary-trees": 8,
    "regex-redux": 8,
    "ripgrep": 8,
    "static-web-server": 8,
    "som": 8,
    "grmtools": 8,
    "fd": 8,
}

PROFILE_PLOTS = {
    "grmtools": {"r": 1, "c": 4},
    "alacritty": {"r": 1, "c": 4},
    "som": {"r": 7, "c": 4},
    "som": {"r": 7, "c": 4},
}

# ============== STATISTICS =================


def pdiff(a, b):
    return (a / (a + b)) * 100


def ci(row, pexecs):
    return Z * (row / math.sqrt(pexecs))


def ci_inl(value, std, pexecs):
    ci = Z * (value / math.sqrt(pexecs))
    lower = value - ci
    upper = value + ci
    return pd.Series({"value": value, "ci": ci, "upper": upper, "lower": lower})


def bootstrap(
    values, kind, method, num_bootstraps=10000, confidence=0.99, symmetric=True
):

    # if DRYRUN:
    #     # This should never be used for real, but it's useful to prevent things
    #     # taking forever when trying to quickly debug the script
    #     num_bootstraps = 1000

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
            "lower": res.confidence_interval.low,
            "upper": res.confidence_interval.high,
        }
    return pd.Series(data)


def bootstrap_max_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.max, "percentile", num_bootstraps, confidence, symmetric=True
    )


def bootstrap_mean_ci(raw_data, num_bootstraps=10000, confidence=0.99):
    return bootstrap(
        raw_data, np.mean, "percentile", num_bootstraps, confidence, symmetric=False
    )


def bootstrap_geomean_ci(means, num_bootstraps=10000, confidence=0.99, symmetric=True):
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


def geomean_with_ci(row, confidence=0.99):
    """Calculate geometric mean and CI for a DataFrame row"""
    # Clean data and validate
    clean_vals = row.dropna()
    n = len(clean_vals)

    # Handle edge cases
    if n == 0 or (clean_vals <= 0).any():
        return pd.Series([np.nan] * 3, index=["value", "lower", "upper"])

    # Log-transform and calculate statistics
    log_vals = np.log(clean_vals)
    mean_log = np.mean(log_vals)
    std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

    # Calculate confidence interval
    if n == 1:
        return pd.Series(
            [np.exp(mean_log), np.nan, np.nan],
            index=["value", "lower", "upper"],
        )

    sem_log = std_log / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

    # Convert back to original scale
    return pd.Series(
        [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
        index=["value", "lower", "upper"],
    )


def arith_mean_ci(series):
    n = len(series)
    mean = series.mean()
    std_err = series.std(ddof=1) / (n**0.5)  # Standard error
    margin_of_error = stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err  # t-score * SE
    return pd.Series(
        {
            "value": mean,
            "ci": margin_of_error,
            "lower": mean - margin_of_error,
            "upper": mean + margin_of_error,
        }
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
        "ci",
        "mean",
        "mean_ci",
    ]
    cmps = df["configuration"][~(df["configuration"] == baseline_col)].unique()

    baseline = (
        df[df["configuration"] == baseline_col]
        .sort_values(timecol)
        .reset_index(drop=True)
        .set_index(timecol)
        .sort_index()
    )
    print(baseline)

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
        ].apply(
            lambda row: normalize_value(row, value_col, timecol),
            axis=1,
        )
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

    def bootstrap_ci(series, n_bootstrap=10000, ci=0.99):
        bootstrap_samples = np.random.choice(
            series, size=(n_bootstrap, len(series)), replace=True
        )
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        lower = np.percentile(bootstrap_means, (1 - ci) * 100 / 2)
        upper = np.percentile(bootstrap_means, 100 - (1 - ci) * 100 / 2)
        return pd.Series(
            {
                "mem": np.mean(series),
                "lower": lower,
                "upper": upper,
                "ci": upper - lower,
            }
        )

    # Concatenate all interpolated dataframes
    interpolated = pd.concat(interpolated, ignore_index=True)

    stats = (
        interpolated.groupby(["configuration", "benchmark", "normalized_time"])["mem"]
        .apply(bootstrap_ci)
        .unstack()
        # .rename(columns={"value": "mem"})
        .reset_index()
    )

    mean = (
        df.groupby(["configuration", "benchmark"])["mem"]
        .apply(arith_mean_ci)
        .unstack()
        .rename(
            columns={
                "value": "mean",
                "lower": "mean_lower",
                "upper": "mean_upper",
                "ci": "mean_ci",
            }
        )
        .reset_index()
    )
    stats = stats.merge(mean, on=["configuration", "benchmark"])
    return stats


# ============== GRAPHS =================


def write_stat(stat):
    with open(STATS_FILE, "a") as f:
        f.write(stat + "\n")


def write_stats(df, experiment, fmt, summary=False):
    def ltxcmd(kind):
        ltxmap = {
            "best": "best",
            "worst": "worst",
            "best_all": "bestsi",
            "worst_all": "worstsi",
        }
        return ltxmap[kind]

    ltxfmt = {"perf": lambda x: f"{x:0.2f}", "mem": format_bytes}

    df = df.fillna("")

    for idx, row in df.iterrows():
        if summary:
            write_stat(f"% Summary stats: {experiment}:{idx}:{fmt}")
            latex_name = experiment + fmt + idx.split("-")[1]
        else:
            latex_name = (
                experiment + fmt + idx[0].replace("-", "") + idx[1].split("-")[1]
            )
            write_stat(
                f"% Config stats: {experiment}:{idx[0]}:{idx[1].split('-')[1]}:{fmt}"
            )
        for (kind, name), value in row.items():
            if not value:
                continue
            if name == "diff_pct":
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}pct{{{value:0.2f}\\%\\xspace}}"
                )
            if name == "diff_ratio":
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}ratio{{{value:0.2f}\\$\\times\\$\\xspace}}"
                )
            elif name == "benchmark" and not summary:
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}benchmark{{{ltxify(value)}\\xspace}}"
                )
            elif name == "suite" and summary:
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}suite{{{SUITES[value]}\\xspace}}"
                )
            elif name == "value":
                write_stat(
                    f"\\newcommand\\{latex_name}{ltxcmd(kind)}value{{{ltxfmt[fmt](value)}\\xspace}}"
                )
        write_stat("")


def write_table(outfile, df, summary=False, include_html=True, mem=False):
    df = df.copy()
    if mem:
        df["ltxval"] = df.apply(ltx_mem_ci, axis=1)
    else:
        df["ltxval"] = df.apply(ltx_value_ci, axis=1)
    if summary:
        index = "suite"
    else:
        index = ["suite", "benchmark"]
        df["benchmark"] = df["benchmark"].apply(lambda x: ltxify(x))
    ltxtable = df.pivot(
        index=index,
        columns="configuration",
        values="ltxval",
    ).fillna("-")

    latex_tabular = ltxtable.to_latex(
        index=True,
        escape=False,
        column_format="l" + "r" * len(df.columns),
        caption=None,
        label=None,
        header=True,
        position=None,
    )

    # Removes lines before \begin{tabular} and after \end{tabular}
    latex_tabular = "\n".join(
        line
        for line in latex_tabular.split("\n")
        if "begin{table}" not in line and "end{table}" not in line
    )

    with open(outfile, "w") as f:
        f.write(latex_tabular)

    print_success(f"Plotted table: {outfile.parts[-2]}:{outfile.stem.replace('_',':')}")

    if not include_html:
        return

    df["valci"] = df.apply(fmt_value_ci, axis=1)
    df = df.pivot(
        index=index,
        columns="configuration",
        values="valci",
    )
    df = df.fillna("-")
    t = outfile.parts[-2].upper() + " Summary"
    html = f"""
        <html>
        <head>
            <title>{t}</title>
        </head>
        <body>
            <h2>{t}</h2>
            {df.to_html()}
        </body>
        </html>
    """

    with open(outfile.with_suffix(".html"), "w") as f:
        f.write(html)


def plot_alacritty(results):
    pass


def plot_perf(suite, outdir, results):
    if DRYRUN:
        return
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    results = results.copy().replace({**CFGS, **BMS})
    num_cfgs = results["configuration"].nunique()
    num_benchmarks = results["benchmark"].nunique()
    colours = ["#152238", "#d85128"]
    linestyles = ["--", ":"]
    df = results.pivot(
        index="benchmark",
        columns="configuration",
        values=["value", "ci"],
    )
    cfgs = results["configuration"].unique()
    fig, ax = plt.subplots(1, 1)
    width = 9
    height = 2
    barwidth = 0.4
    hzticks = True

    df.plot(
        kind="bar",
        y="value",
        yerr="ci",
        ax=ax,
        width=barwidth,
    )

    for j, config in enumerate(cfgs):
        # Get the value where you want to place the horizontal line
        # For example, you might want the mean value for each configuration
        gmean = results[results["configuration"] == config]["geomean"].iloc[0]
        lower = results[results["configuration"] == config]["geomean_lower"].iloc[0]
        upper = results[results["configuration"] == config]["geomean_upper"].iloc[0]

        # Add the horizontal line with a label
        ax.axhline(
            y=gmean,
            color=colours[j],
            linestyle=linestyles[j],
            linewidth=0.8,
            alpha=0.7,
            label=f"G.mean: {gmean:.2f}ms ({lower:.2f}-{upper:.2f})",
        )

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[2:] + handles[0:2]
    labels = labels[2:] + labels[0:2]
    fig.set_size_inches(9, 2)

    legend = ax.legend(
        handles,
        labels,
        title=f"\\textsc{{{suite}}}",
        title_fontsize=7,
        loc="upper center",
        ncols=2,
        fontsize=6,
    )
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor("grey")

    if hzticks:
        # ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.set_xticklabels(df.index, rotation=0, ha="center")
    else:
        ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Wall-clock time (ms). Lower is better.", fontsize=6)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    plt.tight_layout()
    plt.tick_params(labelsize=6, width=0.2, pad=3)
    plt.savefig(outdir / f"{suite}.svg", format="svg", bbox_inches="tight")
    print_success(
        f"Plotted graph: {outdir.parts[-3]}:{outdir.parts[-2]}:{suite}:perf:individual"
    )
    plt.close()


def parse_rt_metrics(dir):
    files = glob.glob(f"{dir / "runtime"}/*.csv")
    data = []
    for f in files:
        flags = (
            pd.read_csv(f, usecols=[0, 1, 2])
            .tail(1)
            .replace({"true": True, "false": False})
        )
        df = pd.read_csv(f, usecols=list(range(3, 12))).tail(1).astype(float)
        base = os.path.splitext(os.path.basename(f))[0].split(".")

        exp = base[2].split("-")[0]
        cfg = base[2].split("-")[1]

        if exp == "gcvs":
            assert flags.all().all()
        elif exp == "premopt":
            assert flags["elision enabled"].all()
            if cfg == "opt":
                assert flags["pfp enabled"].all()
                assert flags["premopt enabled"].all()
            elif cfg == "naive":
                assert flags["pfp enabled"].all()
                assert (~flags)["premopt enabled"].all()
            else:
                assert (~flags)["pfp enabled"].all()
                assert (~flags)["premopt enabled"].all()
        elif exp == "elision":
            assert flags["pfp enabled"].all()
            assert flags["premopt enabled"].all()
            if cfg == "opt":
                assert flags["elision enabled"].all()
            else:
                assert (~flags)["elision enabled"].all()
        else:
            print_error(f"Unknown experiment {exp}")
            sys.exit(1)

        df["suite"] = base[0].rstrip("-harness")
        df["pexec"] = base[1]
        df["configuration"] = base[2]
        df["benchmark"] = base[3]
        data.append(df)

    return pd.concat(data, ignore_index=True)

    # return parse_ht_summary(dir).merge(
    #     df, on=["suite", "configuration", "benchmark", "pexec"]
    # )


def parse_wrk(dir):
    files = glob.glob(f"{dir}/*.csv")
    data = []
    for f in files:
        df = pd.read_csv(f).tail(1).astype(float)
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["suite"] = "static-web-server"
        df["configuration"] = base[2]
        df["benchmark"] = base[3]
        df = df.drop(columns=["latency_stdev", "latency_min", "latency_max"])
        data.append(df)

    if not data:
        return pd.DataFrame()

    df = pd.concat(data, ignore_index=True)
    return df.apply(pd.to_numeric, errors="ignore")


def parse_ht_summary(dir):
    files = glob.glob(f"{dir}/*summary.csv")
    data = []
    for f in files:
        df = pd.read_csv(f).tail(1).astype(float)
        base = os.path.splitext(os.path.basename(f))[0].split(".")
        df["suite"] = base[0].rstrip("-harness")
        df["pexec"] = base[1]
        df["configuration"] = base[2]
        df["benchmark"] = base[3]
        df = df.drop(columns=["temporary allocations"])
        data.append(df)

    if not data:
        return pd.DataFrame()
    df = pd.concat(data, ignore_index=True)
    return df


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
                            "suite": base[0].rstrip("-harness"),
                            "pexec": base[1],
                            "configuration": base[2],
                            "benchmark": base[3],
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


def parse_results(expdir):
    results = {}
    for prog in os.scandir(expdir):
        if not prog.is_dir():
            print_warning(f"Skipping unknown file {prog}...")
            continue

        if prog.name not in BENCHMARKS:
            continue

        print_info(prog)

        data = Path(prog.path) / "perf.csv"

        perf = pd.read_csv(data, sep="\t", comment="#", index_col="suite")
        pexecs = int(perf["invocation"].max())
        perf = perf[perf["criterion"] == "total"].rename(
            columns={"executor": "configuration"}
        )
        perf = perf[["benchmark", "configuration", "value"]]
        # perf_metrics = parse_rt_metrics(
        #     Path(prog.path) / "perf" / "metrics", kind="perf"
        # )

        if "mem" in PLOTS:
            mem = parse_heaptrack(Path(prog.path) / "mem" / "heaptrack")
            # mem_metrics = parse_rt_metrics(
            #     Path(prog.path) / "mem" / "metrics", kind="mem"
            # )
            # mem_metrics = mem_metrics.merge(
            #     parse_ht_summary(Path(prog.path) / "mem" / "heaptrack"),
            #     on=["suite", "configuration", "benchmark", "pexec"],
            # )
        else:
            mem = []
            mem_metrics = []

        if prog.name == "static-web-server":
            perf = parse_wrk(Path(prog.path) / "perf" / "wrk")

        results[prog.name] = (perf, mem, [], [])
    return results


def process_rt_metrics(metrics):
    def gmean_zeroes(series):
        positive_series = series[(series > 0)]
        if len(positive_series) == 0:
            return np.nan
        return np.exp(np.log(positive_series).mean())

    df = (
        metrics.groupby(["suite", "configuration", "benchmark"])
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["benchmark"])
    )

    df = df.groupby(["suite", "configuration"]).apply(gmean_zeroes).round().fillna(0)

    return df


def process_stats(df, experiment, summary=False):

    def diff(a, b):
        diff = a - b
        ratio = a / b
        pct = (ratio - 1) * 100
        return {"diff_raw": diff, "diff_ratio": ratio, "diff_pct": pct}

    def baseline(suite, data, col, lower, upper, kind):
        bl = BASELINE[suite] if experiment == "gcvs" else BASELINE[experiment]
        if col != bl:
            # We compare the cfg with a common baseline
            val = data[("value", bl)]
            si = not (upper < data[("lower"), bl] or lower > data[("upper"), bl])
        else:
            others = data["value"].drop(col)
            blidx = others.idxmin() if kind == "best" else others.idxmax()
            val = data[("value", blidx)]
            si = not (upper < data[("lower"), blidx] or lower > data[("upper"), blidx])
        return (val, si)

    def calculate_diffs(df):
        worst = []
        best = []
        worst_all = []
        best_all = []
        for idx, row in df.iterrows():
            suite = idx[0] if not summary else idx
            for config in df["value"].columns:
                value = row[("value", config)]
                lower = row[("lower", config)]
                upper = row[("upper", config)]
                d = {
                    "suite": suite,
                    "configuration": config,
                    "value": value,
                    "upper": upper,
                    "lower": lower,
                }

                if not summary:
                    d["benchmark"] = idx[1]

                (best_other, best_si) = baseline(
                    suite, row, config, lower, upper, "best"
                )
                (worst_other, worst_si) = baseline(
                    suite, row, config, lower, upper, "worst"
                )

                worst_data = d | diff(value, best_other)
                best_data = d | diff(worst_other, value)

                if not worst_si:
                    worst.append(worst_data)
                if not best_si:
                    best.append(best_data)

                worst_all.append(worst_data)
                best_all.append(best_data)

        return {
            "worst": pd.DataFrame(worst),
            "best": pd.DataFrame(best),
            "worst_all": pd.DataFrame(worst_all),
            "best_all": pd.DataFrame(best_all),
        }

    df = df.copy()
    # df["lower"] = df["value"] - df["ci"]
    # df["upper"] = df["value"] + df["ci"]
    if summary:
        index = "suite"
    else:
        index = ["suite", "benchmark"]
    pivot = df.pivot_table(
        index=index,
        columns="configuration",
        values=["value", "lower", "upper"],
    )

    diffs = calculate_diffs(pivot)
    if summary:
        statidx = "configuration"
    else:
        statidx = ["suite", "configuration"]

    stats = pd.concat(
        {
            k: (
                v.loc[v.groupby(statidx)["diff_raw"].idxmax()].set_index(statidx)
                if not v.empty
                else pd.DataFrame()
            )
            for k, v in diffs.items()
        },
        axis=1,
    )

    stats.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in stats.columns]
    )

    return stats


def process_summary(df):
    gmean = (
        df.groupby(["suite", "configuration"])["value"]
        .apply(geomean_with_ci)
        .unstack()
        .reset_index()
    )
    return gmean


def process_wrk(df, prog, experiment):
    pexecs = df.groupby(["suite", "configuration", "benchmark"]).size().max()
    perf = (
        df.copy()
        .groupby("configuration")
        .agg(
            reqs=("requests", "mean"),
            reqs_ci=("requests", lambda x: ci(x.std(), PEXECS)),
            reqs_per_sec=("requests_per_sec", "mean"),
            reqs_per_sec_ci=("requests_per_sec", lambda x: ci(x.std(), PEXECS)),
            bytes_transferred=("bytes", "mean"),
            bytes_transferred_ci=("bytes", lambda x: ci(x.std(), PEXECS)),
            bytes_transferred_per_sec=("bytes_transfer_per_sec", "mean"),
            bytes_transferred_per_sec_ci=(
                "bytes_transfer_per_sec",
                lambda x: ci(x.std(), PEXECS),
            ),
            latency=("latency_mean_ms", "mean"),
            latency_ci=("latency_mean_ms", lambda x: ci(x.std(), PEXECS)),
        )
        .reset_index()
    )

    df = perf.transpose()
    html = f"""
        <html>
        <head>
            <title>Static-Web-Server GCVS</title>
        </head>
        <body>
            <h2>Static-Web-Server GCVS</h2>
            {df.to_html()}
        </body>
        </html>
    """

    with open("plots/gcvs/static-web-server/perf.html", "w") as f:
        f.write(html)

    #

    print(perf.transpose())
    return perf


def process_perf(df, prog, experiment):
    pexecs = df.groupby(["suite", "configuration", "benchmark"]).size().max()
    perf = (
        df.copy()
        .groupby(["suite", "configuration", "benchmark"])["value"]
        .apply(arith_mean_ci)  # Get single value
        .unstack()
        .reset_index()  # Flatten multi-index
    )

    if perf["benchmark"].nunique() > 1:
        stats = process_stats(perf, experiment)
        write_stats(stats, experiment, fmt="perf")

    return perf


def process_mem(df, prog, experiment):
    pexecs = df.groupby(["suite", "configuration", "benchmark"]).size().max()
    mem = (
        df.copy()
        .groupby(["suite", "configuration", "benchmark"])["mem"]
        .apply(arith_mean_ci)
        .unstack()
        .reset_index()
    )

    # if mem["benchmark"].nunique() > 1:
    #     stats = process_stats(mem, experiment)
    #     write_stats(stats, experiment, fmt="mem")

    for suite, results in df.groupby("suite"):
        if suite not in PLOT_LAYOUTS:
            continue
        layout = PLOT_LAYOUTS[suite]
        outdir = PLOT_DIR / experiment / prog / "profiles"
        outdir.mkdir(parents=True, exist_ok=True)
        m = interpolate(
            normalize_time(results), oversampling=layout.resolution[experiment]
        )
        # if experiment in ["premopt", "elision"]:
        #     cmp = True
        #     m = normalize(m, BASELINE[experiment])
        # else:
        #     cmp = False
        layout.plot_time_series(m, outdir, cmp=False)
    return mem


def parse_stats(e, kind):
    data = []
    dir = RESULTS_DIR / e
    for prog in os.scandir(dir):
        if not prog.is_dir():
            print_warning(f"Skipping unknown file {prog}...")
            continue

        if prog.name not in BENCHMARKS:
            continue
        d = parse_rt_metrics(dir / prog / kind / "metrics")
        if kind == "mem":
            d.merge(
                parse_rt_metrics(dir / prog / kind / "heaptrack"),
                on=["suite", "configuration", "benchmark", "pexec"],
            )
        data.append(d)
    return pd.concat(data)


def process_experiment(experiment):
    print_info(f"Processing {experiment} results...")
    results = parse_results(RESULTS_DIR / experiment)
    perfs = []
    mems = []

    def sanity_check(prog, df):
        if df.empty:
            print_error(f"{experiment}:{prog} has missing data")
            return False
        runs = df.groupby(["suite", "configuration", "benchmark"]).size()
        if (runs != runs.iloc[0]).all():
            print_error(
                f"{experiment}:{prog} has an inconsistent number of pexecs: {pruns}"
            )
            return False
        return True

    for prog, (perfraw, memraw, perfmetrics, memmetrics) in results.items():
        print_info(f"Processing {prog}...")

        # if prog == "static-web-server":
        #     perf = process_wrk(perfraw, prog, experiment)
        #     print(perf)
        #     continue

        perfs_ok = "perf" in PLOTS and sanity_check(prog, perfraw)
        # perfms_ok = "perf" in PLOTS and sanity_check(prog, perfmetrics)
        mems_ok = "mem" in PLOTS and sanity_check(prog, memraw)
        # memms_ok = "mem" in PLOTS and sanity_check(prog, memmetrics)

        if perfs_ok:
            perf = process_perf(perfraw, prog, experiment)

            perfsum = process_summary(perfraw)
            perfs.append(perfsum)

            plotdf = perf.copy().merge(
                perfsum.rename(
                    columns={
                        "value": "geomean",
                        "ci": "geomean_ci",
                        "upper": "geomean_upper",
                        "lower": "geomean_lower",
                    }
                ),
                on=["suite", "configuration"],
            )

            outdir = PLOT_DIR / experiment / prog / "perf"
            outdir.mkdir(parents=True, exist_ok=True)
            for suite, results in plotdf.groupby("suite"):
                if suite in PLOT_LAYOUTS:
                    PLOT_LAYOUTS[suite].plot_perf(results, outdir)

            if perf["benchmark"].nunique() == 1:
                perfs.append(perf.drop(columns=["benchmark"]))
            else:
                write_table(PLOT_DIR / experiment / prog / "table.tex", perf)

        if mems_ok:
            mem = process_mem(memraw, prog, experiment)
            if mem["benchmark"].nunique() == 1:
                mems.append(mem.drop(columns=["benchmark"]))
            else:
                mems.append(process_summary(memraw.rename(columns={"mem": "value"})))

    # overall = (
    #     pd.concat(perfs, ignore_index=True)
    #     .groupby("configuration")
    #     .apply(geomean_with_ci)
    #     .unstack()
    #     .reset_index()
    # )

    #
    def conv_fmt(x):
        if isinstance(x, (int, float)):
            return f"{format_number(x)}"
        return x

    if "perf" in PLOTS:
        perfs = pd.concat(perfs, ignore_index=True)
        stats = process_stats(perfs, experiment, summary=True)
        write_stats(stats, experiment, fmt="perf", summary=True)
        perfs["suite"] = perfs["suite"].replace(SUITES)
        write_table(PLOT_DIR / experiment / "perf_summary.tex", perfs, summary=True)

    if "mem" in PLOTS:
        mems = pd.concat(mems, ignore_index=True)
        # stats = process_stats(mems, experiment, summary=True)
        # write_stats(stats, experiment, fmt="mem", summary=True)
        mems["suite"] = mems["suite"].replace(SUITES)
        mems["configuration"] = mems["configuration"].replace(CFGS)
        write_table(
            PLOT_DIR / experiment / "mem_summary.tex", mems, summary=True, mem=True
        )


def process_finalizers(perf):
    flzrs = (
        perf.groupby(["suite", "configuration", "benchmark"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby(["suite", "configuration"])[
            ["finalizers registered", "finalizers completed", "finalizers elidable"]
        ]
        .sum()
    )
    naive = flzrs.xs("elision-naive", level="configuration")["finalizers registered"]
    elision = flzrs.xs("elision-opt", level="configuration")["finalizers registered"]
    cmpl = flzrs.xs("elision-opt", level="configuration")["finalizers completed"]

    diff = pd.DataFrame()
    diff["naive"] = naive
    diff["opt registered"] = elision
    diff["opt elided"] = naive.sub(elision)
    diff["opt completed"] = cmpl
    diff["pct elided"] = (diff["opt elided"] / diff["naive"]) * 100

    return diff.fillna(0)


def process_allocs(perf):
    df = (
        perf.groupby(["suite", "configuration", "benchmark"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby(["suite", "configuration"])[
            [
                "Gc allocated",
                "Rc allocated",
                "Arc allocated",
                "Box allocated",
                "Gc reclaimed",
            ]
        ]
        .sum()
    )
    gc = df.xs("elision-opt", level="configuration")
    gc["total_heap"] = gc.sum(axis=1)
    gc["total_rcs"] = gc["Rc allocated"] + gc["Arc allocated"]
    gc["gcr"] = (gc["Gc allocated"] / gc["total_heap"]) * 100
    return gc


def main():
    global RESULTS_DIR
    global PLOT_DIR
    global STATS_FILE

    RESULTS_DIR = Path(sys.argv[2])
    PLOT_DIR = Path(sys.argv[1])
    STATS_FILE = PLOT_DIR / "experiment_stats.tex"

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(STATS_FILE):
        print_warning(f"File {STATS_FILE} already exists. Removing...")
        os.remove(STATS_FILE)

    print_info(f"Will process the the following experiments: {EXPERIMENTS}")
    print_info(f"Will process the the following benchmarks: {BENCHMARKS}")
    print_info(f"Will generate the the following plots: {PLOTS}")

    if DRYRUN:
        print_warning(
            f"DRYRUN enabled: no plots will be generated and CIs will be incorrect."
        )

    if not BOOTSTRAP:
        print_warning(f"BOOTSTRAP disabled: CI formula will be used for arith. mean")

    flzrs = pd.DataFrame()
    allocs = pd.DataFrame()
    for e in EXPERIMENTS:
        perf = process_experiment(e)

    #     perf = parse_stats(e, "perf")
    #     if e == "elision":
    #         flzrs = process_finalizers(perf)
    #         allocs = process_allocs(perf)
    #
    # allocs["use elided"] = (flzrs["opt elided"] + allocs["Gc allocated"]) < allocs[
    #     "total_heap"
    # ]
    # allocs["managed"] = (
    #     np.maximum(allocs["elided"], allocs["Gc reclaimed"])
    #     if allocs["use elided"].all()
    #     else allocs["Gc reclaimed"]
    # )
    # allocs["gcrt"] = (
    #     (allocs["managed"] + allocs["Gc allocated"]) / allocs["total_heap"]
    # ) * 100
    # allocs["opt registered"] = flzrs["opt registered"]
    # allocs["opt elided"] = flzrs["opt elided"]
    # output = allocs[
    #     [
    #         "Gc allocated",
    #         "Arc allocated",
    #         "Rc allocated",
    #         "Box allocated",
    #         "gcr",
    #         "gcrt",
    #         "opt registered",
    #         "opt elided",
    #     ]
    # ]
    # print(output)
    #
    # latex_rows = []
    # for index, row in allocs.iterrows():
    #     ltxsuite = SUITES[index]
    #     formatted_row = [
    #         format_number(row["Gc allocated"]),
    #         format_number(row["Arc allocated"]),
    #         format_number(row["Rc allocated"]),
    #         format_number(row["Box allocated"]),
    #         f'{row["gcr"]:.2f}\\%',
    #         f'{row["gcrt"]:.2f}\\%',
    #         format_number(row["opt registered"]),
    #         format_number(row["opt elided"]),
    #     ]
    #     latex_rows.append(f"{ltxsuite} {'& ' + ' & ' .join(formatted_row)}\\\\")
    #
    # latex_table = (
    #     r"""
    # \begin{tabular}{lrrrrrrrrr}
    # \toprule
    # & \multicolumn{4}{c}{Allocated} & \multicolumn{2}{c}{GC Ratios} & \multicolumn{2}{c}{Optimizations} \\
    # \cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
    # Suite & Gc & Arc & Rc & Box & gcr & gcrt & Registered & Elided \\
    # \midrule
    # """
    #     + "\n".join(latex_rows)
    #     + r"""
    # \bottomrule
    # \end{tabular}
    # """
    # )
    #
    # print(latex_table)


# if not flzrs.empty:
#     stats[]


if __name__ == "__main__":
    main()
