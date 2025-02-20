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


def plot(filename, means, errs, width, kind):
    sns.set(style="whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(width, 4))

    means = means.rename(columns=CFGS)
    errs = errs.rename(columns=CFGS)
    means.plot(kind="bar", ax=ax, width=0.8, yerr=errs)

    if len(means.columns) == 1:
        plt.gca().get_legend().remove()

    ax.set_xticklabels(means.index, rotation=45, ha="right")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if kind == "perf":
        ax.set_ylabel("Wall-clock time (ms)\n(lower is better)")
    else:
        ax.set_ylabel("Maximum resident set size (KiB)\n(lower is better)")
    ax.grid(linewidth=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_tick_params(which="minor", size=0)
    ax.yaxis.set_tick_params(which="minor", width=0)
    ax.xaxis.label.set_visible(False)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")


def overall_gmean(data):
    means = data.groupby(["executor", "benchmark"])["value"].mean()
    gmeans = []
    cis = []

    for config in means.index.get_level_values(0).unique():
        data = means[config]
        gmean = stats.gmean(data)

        data_array = np.array(data).reshape(-1, 1)
        boot_result = stats.bootstrap(
            (data_array,),
            stats.gmean,
            n_resamples=9999,
            confidence_level=0.99,
            method="BCa",  # Bias-corrected and accelerated bootstrap
        )

        ci_lower, ci_upper = boot_result.confidence_interval
        gmeans.append(
            {
                "configuration": config,
                "geometric_mean": gmean,
            }
        )
        cis.append(
            {
                "configuration": config,
                "ci_lower": gmean - ci_lower[0],
                "ci_upper": ci_upper[0] - gmean,
            }
        )

    gidf = pd.DataFrame(gmeans).set_index("configuration")
    cidf = pd.DataFrame(cis).set_index("configuration").transpose().to_numpy()
    return (gidf, cidf)


def mk_table(filename, vals, cis, columns):
    df = pd.DataFrame()
    for v, c in zip(vals.columns, cis.columns):
        df[v] = vals[v].round(2).astype(str) + " \pm " + cis[c].round(3).astype(str)

    with open(filename, "w") as f:
        f.write(df.unstack().rename(columns=columns).to_latex(index=False))


def ci(row, pexecs):
    Z = 2.576  # 99% interval
    return Z * (row / math.sqrt(pexecs))


def means(csv):
    global pexecs
    pexecs = int(df["invocation"].max())
    assert pexecs == PEXECS

    perf = (
        df.loc[df["criterion"] == "total"]
        .groupby(["benchmark", "executor"])["value"]
        .mean()
    )
    mem = (
        df.loc[df["criterion"] == "MaxRSS"]
        .groupby(["benchmark", "executor"])["value"]
        .mean()
    )
    return (perf, mem)


def raw_metrics(mdir):
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


print(f"==> processing results for {sys.argv[1:]}")

infile = sys.argv[1]
infile_metrics = Path(infile).parent / "metrics"
suite = Path(sys.argv[2]).parts[-2]
experiment = Path(sys.argv[2]).parts[-3]
datapoint = Path(sys.argv[2]).stem
outdir = Path(sys.argv[2]).parent
outfile = sys.argv[2]
outfile_gmean = outdir / f"{datapoint}_gmean.svg"
out_metrics_gmean = outdir / "metrics_gmean.tex"

df = pd.read_csv(infile, sep="\t", skiprows=4, index_col="benchmark")
pexecs = int(df["invocation"].max())
assert pexecs == PEXECS

perf = (
    df.loc[df["criterion"] == "total"]
    .groupby(["benchmark", "executor"])
    .agg(value=("value", "mean"), ci=("value", lambda x: ci(x.std(), pexecs)))
)

mem = (
    df.loc[df["criterion"] == "MaxRSS"]
    .groupby(["benchmark", "executor"])
    .agg(value=("value", "mean"), ci=("value", lambda x: ci(x.std(), pexecs)))
)

plot(
    outdir / "perf.svg",
    perf["value"].unstack(),
    perf["ci"].unstack(),
    width=8,
    kind="perf",
)
plot(
    outdir / "mem.svg", mem["value"].unstack(), mem["ci"].unstack(), width=8, kind="mem"
)


raw = raw_metrics(infile_metrics).groupby(["benchmark", "configuration"])
metrics = raw.mean()
cis = raw.std().apply(ci, pexecs=pexecs)
mk_table(outdir / "metrics.tex", metrics, cis, columns=METRICS)
