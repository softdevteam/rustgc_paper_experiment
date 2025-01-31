#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import numpy as np
import pandas as pd
import pprint
import csv
from scipy import stats
from pathlib import Path
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import glob

matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'errorbar.capsize': 2})

results = {}
pp = pprint.PrettyPrinter(indent=4)

PEXECS = int(os.environ['PEXECS'])

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

print(f"==> processing results for {sys.argv[1:]}")


def plot_perf(filename, means, errs, width):
    sns.set(style="whitegrid")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(width, 4))

    means = means.rename(index=CFGS)
    means.plot(kind='bar', ax=ax, width=0.8, yerr=errs)

    if len(means.columns) == 1:
        plt.gca().get_legend().remove()

    ax.set_xticklabels(means.index, rotation=45, ha='right')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Wall-clock time (ms)\n(lower is better)')
    ax.grid(linewidth=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(which='minor', size=0)
    ax.yaxis.set_tick_params(which='minor', width=0)
    ax.xaxis.label.set_visible(False)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")


def overall_gmean(data):
    means = data.groupby(['executor', 'benchmark'])['value'].mean()
    gmeans = []
    cis = []

    for config in means.index.get_level_values(0).unique():
        data = means[config]
        gmean = stats.gmean(data)

        data_array = np.array(data).reshape(-1, 1)
        boot_result = stats.bootstrap(
            (data_array, ),
            stats.gmean,
            n_resamples=9999,
            confidence_level=0.99,
            method='BCa'  # Bias-corrected and accelerated bootstrap
        )

        ci_lower, ci_upper = boot_result.confidence_interval
        gmeans.append({
            'configuration': config,
            'geometric_mean': gmean,
        })
        cis.append({
            'configuration': config,
            'ci_lower': gmean - ci_lower[0],
            'ci_upper': ci_upper[0] - gmean,
        })

    gidf = pd.DataFrame(gmeans).set_index('configuration')
    cidf = pd.DataFrame(cis).set_index('configuration').transpose().to_numpy()
    return (gidf, cidf)


METRICS = {
    'finalizers registered': 'Finalizable Objects',
    'finalizers completed': 'Total Finalized',
    'barriers visited': 'Barrier Chokepoints',
    'Gc allocated': 'Allocations (Gc)',
    'Box allocated': 'Allocations (Box)',
    'Rc alocated': 'Allocations (Rc)',
    'Arc allocated': 'Allocations (Arc)',
    'STW pauses': r'Gc Cycles',
}

infile = sys.argv[1]
infile_metrics = Path(infile).parent / 'metrics'
suite = Path(sys.argv[2]).parts[-2]
experiment = Path(sys.argv[2]).parts[-3]
outdir = Path(sys.argv[2]).parent
out_perf = sys.argv[2]
out_perf_gmean = outdir / 'perf_gmean.svg'
out_metrics = outdir / 'metrics.tex'
out_metrics_gmean = outdir / 'metrics_gmean.tex'

pexecs = 0


def mk_table(filename, vals, cis, columns):
    df = pd.DataFrame()
    for v, c in zip(vals.columns, cis.columns):
        df[v] = vals[v].round(2).astype(str) + ' \pm ' + cis[c].round(
            3).astype(str)

    with open(filename, "w") as f:
        f.write(df.unstack().rename(columns=columns).to_latex(index=False))


def ci(row, pexecs):
    Z = 2.576  # 99% interval
    return Z * (row / math.sqrt(pexecs))


def raw_perf(csv):
    df = pd.read_csv(csv, sep='\t', skiprows=4, index_col='benchmark')
    df = df[df['value'] != 0]
    global pexecs
    pexecs = int(df['invocation'].max())
    # assert pexecs == int(df['invocation'].min())
    assert pexecs == PEXECS
    return df


def raw_metrics(mdir):
    csvs = glob.glob(f"{mdir}/*.log")
    m = []
    for f in csvs:
        df = pd.read_csv(f)
        base = os.path.splitext(os.path.basename(f))[0].split('-')
        df['configuration'] = base[0]
        df['benchmark'] = base[1]
        df = df.drop([
            'elision enabled', 'premature finalizer prevention enabled',
            'premopt enabled'
        ],
                     axis=1)
        m.append(df)
        return pd.concat(m, ignore_index=True)


data = raw_perf(infile)

metrics = raw_metrics(infile_metrics).groupby(['benchmark',
                                               'configuration']).mean()
cis = metrics.std().apply(ci, pexecs=pexecs, axis=0)

mk_table(out_metrics, metrics, cis, columns=METRICS)
