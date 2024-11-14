#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import numpy as np
import pandas as pd
import pprint
import csv
from scipy.stats import t

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'cm'
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
# })

matplotlib.rcParams.update({'errorbar.capsize': 2})
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

results = {}
pp = pprint.PrettyPrinter(indent=4)

PEXECS = int(os.environ['PEXECS'])
ITERS = int(os.environ['ITERS'])

UNIT_LATEX_MAP = {
    "perf_gc": r"\ourgc",
    "bt_alloy": r"\ourgc",
    "rr_alloy": r"\ourgc",
    "rr_rc": r"\rc",
    "gc": r"\ourgc",
    "rc": r"\rc",
    "perf_rc": r"\rc",
    "bt_rc": r"\rc",
    "bt_rust_gc": r"\rustgc",
    "bt_typed_arena": r"\typedarena",
    "finalise_naive": r"\fnaive",
    "finalise_elide": r"\felide",
    "barriers_none": r"\bnone",
    "barriers_naive": r"\bnaive",
    "barriers_opt": r"\bopt",

    "requests": r"Reqs.",
    "duration_ms": r"Req. Time (ms).",
    "requests_per_sec": r"(Req/s)",
    "bytes": r"Transferred (B)",
    "bytes_transfer_per_sec": r"Transferred (B/s))",
    "latency_mean_ms": r"Latency (ms)",
}
vm_mapper = {
        "yksom_elision": r"\textsc{yksom}",
        "yksom_barriers": r"\textsc{yksom}",
        "som_rs_bc_elision": r"\textsc{som-rs$_\textrm{bc}$}",
        "som_rs_ast_elision": r"\textsc{som-rs$_\textrm{ast}$}",
        "som_rs_bc_barriers": r"\textsc{som-rs$_\textrm{bc}$}",
        "som_rs_ast_barriers": r"\textsc{som-rs$_\textrm{ast}$}",
        "som_rs_bc_perf": r"\textsc{som-rs$_\textrm{bc}$}",
        "som_rs_ast_perf": r"\textsc{som-rs$_\textrm{ast}$}",
        }
name_map = {
        'finalise_naive' : 'Naive',
        'finalise_elide' : 'Elision',
        'barriers_opt' : 'Opt',
        'barriers_none' : 'None',
        'barriers_naive' : 'Naive',
        'perf_gc' : r'\textsc{Alloy}',
        'gc' : r'\textsc{Alloy}',
        'rc' : r'\textsc{Rc}',
        'perf_rc' : r'\textsc{Rc}'
        }

unit_mapper = {
        'grmtools': 'MiB',
        'som_rs_bc_elision': 'MiB',
        'som_rs_ast_elision': 'MiB',
        'yksom_elision': 'MiB',
        'yksom_barriers': 'MiB',
        'som_rs_bc_barriers': 'MiB',
        'som_rs_ast_barriers': 'MiB',
        'som_rs_bc_perf': 'MiB',
        'som_rs_ast_perf': 'MiB',
        }
mapper = {
        "perf_gc": "gc",
        "perf_rc": "rc",
        "finalise_naive": "fnaive",
        "finalise_elide": "felide",
        "barriers_none": "bnone",
        "barriers_naive": "bnaive",
        "barriers_opt": "bopt",
        "all" : "all"
        }

def mean(l):
    return math.fsum(l) / float(len(l))

def pdiff(bl, cmp):
        return abs(((cmp - bl) / abs(bl)) * 100)

def flatten(l):
  return [y for x in l for y in x]

def human(num):
    print(num)
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

class PExec:
    def __init__(self, num, cfg, benchmark, iters, peaks):
        self.num = num
        self.cfg = cfg
        self.benchmark = benchmark
        self.iters = iters
        self.peaks = peaks

    def __repr__(self):
        return f"Pexec('{self.num}', '{self.cfg}', '{self.benchmark}', '{self.iters}', '{self.peaks}')"

class Experiment:
    def __init__(self, name, pexecs):
        self.name = name
        self.pexecs = pexecs

    def latex_name(self):
        return f"\\{self.name}".replace("_","")

    def geomean(self, cfg, benchmark='all'):
        return geometric_mean(self.iters(cfg, benchmark))

    def geomean_mem(self, cfg, benchmark='all'):
        return geometric_mean(self.peaks(cfg, benchmark))

    def ci_mean(self, cfg):
        l = self.iters(cfg, 'all')
        return confidence_interval(l)

    def ci_mean_mem(self, cfg):
        l = self.peaks(cfg, 'all')
        return confidence_interval(l)

    def ci_geomean(self, cfg):
        l = self.iters(cfg, 'all')
        log = np.log(l)
        n = len(log)
        mean_log = np.mean(log)
        std_log = np.std(log, ddof=1)
        confidence = 0.99
        degrees_of_freedom = n - 1
        t_value = np.abs(t.ppf((1 - confidence) / 2, degrees_of_freedom))

        margin_of_error = t_value * (std_log / np.sqrt(n))
        ci_lower = np.exp(mean_log - margin_of_error)
        ci_upper = np.exp(mean_log + margin_of_error)
        return (float(ci_lower), float(ci_upper))

    def mean(self, cfg, benchmark='all'):
        l = self.iters(cfg, benchmark)
        return math.fsum(l) / float(len(l))

    def speedup(self, normalised_to, benchmark):
        pass

    def num_pexecs(self):
        return max([pexec.num for pexec in self.pexecs])

    def num_iters(self):
        return len(self.pexecs[0].iters)

    def cfgs(self):
        return sorted(list({p.cfg for p in self.pexecs}))

    def stats(self, cfg, benchmark = 'all'):
        totals = {}
        for items in self.gc_stats[cfg].values():
            for k,v in items.items():
                totals.setdefault(k, []).append(v)
        for k,v in totals.items():
            val = flatten(v)
            if max(val) > 0:
                st = round(geometric_mean(flatten(v)))
                totals[k] = st
            else:
                totals[k] = 0
        return totals

    def iters(self, cfg, benchmark):
        if benchmark == 'all':
            return flatten([p.iters for p in self.pexecs if p.cfg == cfg])
        else:
            return flatten([p.iters for p in self.pexecs if (p.cfg == cfg and p.benchmark == benchmark)])

    def peaks(self, cfg, benchmark, units='mb'):
        def unit(x):
            if units == 'kb':
                return x;
            elif units == 'mb':
                return x / 1024
        if benchmark == 'all':
            return flatten([[unit(v) for v in p.peaks] for p in self.pexecs if p.cfg == cfg])
        else:
            return flatten([[unit(v) for v in p.peaks] for p in self.pexecs if (p.cfg == cfg and p.benchmark == benchmark)])

    def benchmarks(self):
        return sorted(list({p.benchmark for p in self.pexecs}))

    def diff(self, cfg, baseline, benchmark = 'all', geomean = False):
        if geomean:
            return self.geomean(baseline, benchmark) - self.geomean(cfg, benchmark)
        return self.mean(baseline, benchmark) - self.mean(cfg, benchmark)

    def speedup(self, cfg, baseline, benchmark = 'all', geomean = False):
        if geomean:
            return self.geomean(baseline, benchmark) / self.geomean(cfg, benchmark)
        return self.mean(baseline, benchmark) / self.mean(cfg, benchmark)

    def percent(self, cfg, baseline, benchmark = 'all', geomean = False):
        if geomean:
            bl = self.geomean(baseline, benchmark)
            cmp = self.geomean(cfg, benchmark)
        else:
            bl = self.mean(baseline, benchmark)
            cmp = self.mean(cfg, benchmark)
        return pdiff(bl, cmp)
        # return (abs(a - b) / b) * 100

    def dump_stats(self):
        baseline = {
            "perf": "perf_rc",
            "som_rs_bc_elision": "finalise_naive",
            "som_rs_ast_elision": "finalise_naive",
            "yksom_elision": "finalise_naive",
            "yksom_barriers": "barriers_none",
            "som_rs_bc_barriers": "barriers_none",
            "som_rs_ast_barriers": "barriers_none",
            "som_rs_bc_perf": "perf_rc",
            "som_rs_ast_perf": "perf_rc",
        }
        stats = dict((mapper[cfg], {}) for cfg in self.cfgs())
        for cfg in self.cfgs():
            for benchmark in self.benchmarks():
                speedup = self.speedup(cfg, baseline[self.name], benchmark)
                if speedup < 1:
                    percentage = (1 - speedup) * 100
                else:
                    percentage = (speedup - 1) * 100
                stats[mapper[cfg]][benchmark.lower()] = {
                    'mean' : f"{self.mean(cfg, benchmark):.2f}",
                    'diff' : f"{self.diff(cfg, baseline[self.name], benchmark):.2f}",
                    'speedup' : f"{speedup:.2f}",
                    'percent' : f"{percentage:.2f}\\%",
                    # 'direction' : f"{"slowdown" if speedup < 1 else "speedup"}",
                }
                for (k,v) in self.gc_stats[cfg][benchmark].items():
                    if cfg not in ['perf_gc', 'perf_rc']:
                        stats[mapper[cfg]][benchmark.lower()][k.replace('_','')] = human(int(mean(v)))
            speedup = self.speedup(cfg, baseline[self.name], 'all', geomean = True)
            if speedup < 1:
                percentage = (1 - speedup) * 100
            else:
                percentage = (speedup - 1) * 100
            # Use geomean for all benchmarks
            stats[mapper[cfg]]['all'] = {
                'geomean' : f"{self.geomean(cfg, benchmark):.2f}",
                'diff' : f"{self.diff(cfg, baseline[self.name], 'all', geomean = True):.2f}",
                'speedup' : f"{self.speedup(cfg, baseline[self.name], 'all', geomean = True):.2f}",
                'percent' : f"{percentage:.2f}\\%",
                # 'direction' : f"{"slowdown" if speedup < 1 else "speedup"}",
            }
            totals = {}
            if cfg not in ['perf_gc', 'perf_rc']:
                for (k,v) in self.stats(cfg).items():
                    print(k)
                    stats[mapper[cfg]]['all'][k] = human(v)
                    speedup = v / self.stats(baseline[self.name])[k]
                    if speedup < 1:
                        percentage = (1 - speedup) * 100
                    else:
                        percentage = (speedup - 1) * 100
                    direction = "fewer" if speedup < 1 else "more"
                    stats[mapper[cfg]]['all'][k + 'pdiff'] = f'{percentage:.2f}\\%'
                    stats[mapper[cfg]]['all'][k + 'direction'] = direction

            # for items in self.gc_stats[cfg].values():
            #     for k,v in items.items():
            #         totals.setdefault(k.replace('_',''),[]).append(v)
            # for k,v in totals.items():
            #     st = round(geometric_mean(flatten(v)))
            #     # bl = stats[mapper[baseline[self.name]]]['all'][k]
            #     # print(bl)
            #     stats[mapper[cfg]]['all'][k] = human(st)
        # print(totals)


                # print([[v for (k,v)] for (k,v) in items.items()])
                # for (ki, vi) in v:
                # stats[mapper[cfg]][benchmark.lower()][k] = human(int(mean(v)))
        for cfg, info in stats.items():
            if cfg in mapper[baseline[self.name]]:
                continue
            max_key = max(info, key=lambda k: float(info[k].get('speedup', float('-inf'))))
            min_key = min(info, key=lambda k: float(info[k].get('speedup', float('-inf'))))
            # print(f"greatest diff: {max_key}, {info[max_key]['speedup']}")
            # print(f"lowest diff: {min_key}, {info[min_key]['speedup']}")

            stats[cfg]['bestperf'] = stats[cfg][max_key].copy()
            stats[cfg]['bestperf']['name'] = max_key
            stats[cfg]['worstperf'] = stats[cfg][min_key].copy()
            stats[cfg]['worstperf']['name'] = min_key
        # pprint.pprint(stats, width=1)
        # pprint.pprint(self.gc_stats)

        return stats

def load_exp(exp_name):
    exp_dir = os.path.join(os.getcwd(), 'results', exp_name)
    rbdata = os.path.join(exp_dir, "results.data")
    pexecs = {}
    stats =  {}
    with open(rbdata) as f:
        for l in f.readlines():
            if l.startswith("#"):
                continue
            l = l.strip()
            if len(l) == 0:
                continue
            s = [x.strip() for x in l.split()]

            if s[4] not in ["total", "MaxRSS"]:
                continue

            # A PExec can be uniquely identified by a tuple of (invocation, benchmark, cfg)
            invocation = int(s[0])
            benchmark = s[5]
            cfg = s[6]
            iter = s[1]


            if cfg not in stats:
                stats[cfg] = {}
            if benchmark not in stats[cfg]:
                stats[cfg][benchmark] = {
                    'finalisers_registered': [],
                    'finalisers_run': [],
                    'gc_allocations': [],
                    'rust_allocations': [],
                    'gc_cycles': [],
                }

            if (invocation, benchmark, cfg) not in pexecs:
                pexecs[(invocation, benchmark, cfg)] = PExec(invocation, cfg, benchmark, [], [])

            if s[4] == "MaxRSS":
                peak_mem = float(s[2])
                pexecs[(invocation, benchmark, cfg)].peaks.append(peak_mem)
            elif s[4] == "total":
                time = float(s[2])
                pexecs[(invocation, benchmark, cfg)].iters.append(time)

    experiment = Experiment(exp_name, list(pexecs.values()))

    for cfg in stats.keys():
        p = os.path.join(exp_dir, f"{cfg}.log")
        if os.path.exists(p):
            with open(p, "r") as f:
                for l in f.readlines():
                    s = [x.strip() for x in l.split(',')]
                    bm = s[0]
                    stats[cfg][bm]['finalisers_registered'].append(int(s[1]))
                    stats[cfg][bm]['finalisers_run'].append(int(s[2]))
                    stats[cfg][bm]['gc_allocations'].append(int(s[3]))
                    stats[cfg][bm]['rust_allocations'].append(int(s[4]))
                    stats[cfg][bm]['gc_cycles'].append(int(s[5]))
    experiment.gc_stats = stats
    return experiment

# def plot_overview_mem(name, exps, variant = 'Mem'):
#     legend ={
#         'barriers' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'elision' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'barriers_mem' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'elision_mem' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'perf_mem' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#     }
#     exps = sorted(exps, key = lambda e: e.name)
#     index = [vm_mapper[e.name] for e in exps]
#     if variant == 'Mem':
#         geomeans = [[e.geomean_mem(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#         cis = [[e.ci_mean_mem(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#         name += '_mem'
#     else:
#         geomeans = [[e.geomean(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#         cis = [[e.ci_mean(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#
#     sns.set(style="whitegrid")
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     fig, ax = plt.subplots(figsize=(5, 3))
#
#     df = pd.DataFrame(geomeans, index=index)
#     errs = pd.DataFrame(cis, index=index)
#
#
#     plot = df.plot(kind='barh', width=0.8, ax=ax)
#     plot.margins(x=0.01)
#
#     leg = ax.legend(legend[name])
#     # if variant == 'Mem':
#     #     leg.remove()
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.grid(linewidth=0.25)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_tick_params(which='minor', size=0)
#     ax.yaxis.set_tick_params(which='minor', width=0)
#     plt.yticks(range(0, len(index)), index, ha="right")
#     if variant == 'Mem':
#         ax.set_xlabel('Max RSS (MiB) (lower is better)', labelpad=20)
#     else:
#         ax.set_xlabel('Wall-clock time (ms) (lower is better)', labelpad=20)
#     # ax.set_yticks(range, len(exp.benchmarks()))
#     # ax.set_ytickslabels(exp.benchmarks())
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     ax.xaxis.set_major_formatter(formatter)
#     plt.tight_layout()
#     plt.savefig(f"plots/{name}.svg", format="svg", bbox_inches="tight")
#     print("Graph saved to '%s'" % f"plots/{name}.svg")
#
# def plot_overview_bar(name, exps):
#     legend ={
#         'barriers' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'elision' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#         'perf' : [name_map[cfg] for cfg in sorted(exps[0].cfgs())],
#     }
#     exps = sorted(exps, key = lambda e: e.name)
#     index = [vm_mapper[e.name] for e in exps]
#     geomeans = [[e.geomean(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#     cis = [[e.ci_mean(cfg) for cfg in sorted(e.cfgs())] for e in exps]
#
#     sns.set(style="whitegrid")
#     # plt.rc('text', usetex=True)
#     # plt.rc('font', family='serif')
#     fig, ax = plt.subplots(figsize=(5, 3))
#
#     df = pd.DataFrame(geomeans, index=index)
#     errs = pd.DataFrame(cis, index=index)
#
#
#     plot = df.plot(kind='barh', width=0.8, ax=ax, xerr=errs)
#     plot.margins(x=0.01)
#
#     ax.legend(legend[name])
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.grid(linewidth=0.25)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_tick_params(which='minor', size=0)
#     ax.yaxis.set_tick_params(which='minor', width=0)
#     plt.yticks(range(0, len(index)), index, ha="right")
#     ax.set_xlabel('Wall-clock time (ms) (lower is better)', labelpad=20)
#     # ax.set_yticks(range, len(exp.benchmarks()))
#     # ax.set_ytickslabels(exp.benchmarks())
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     ax.xaxis.set_major_formatter(formatter)
#     plt.tight_layout()
#     plt.savefig(f"plots/{name}.svg", format="svg", bbox_inches="tight")
#     print("Graph saved to '%s'" % f"plots/{name}.svg")

def plot_perf(filename, means, errs):
    sns.set(style="whitegrid")
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 4))

    means.plot(kind='bar', ax=ax, width=0.8, yerr=errs)

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
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")

# def plot_memh(exp):
#     means = [[mean(exp.peaks(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#     cis = [[confidence_interval(exp.peaks(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#
#     sns.set(style="whitegrid")
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='sans-serif')
#     fig, ax = plt.subplots(figsize=(5, 3))
#     df = pd.DataFrame(zip(*means), index=exp.benchmarks())
#     errs = pd.DataFrame(zip(*cis), index=exp.benchmarks())
#     plot = df.plot(kind='barh', width=0.8, ax=ax, xerr=errs)
#     plot.margins(x=0.01)
#     ax.legend([name_map[c] for c in exp.cfgs()]).remove()
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_xlabel(f'Max RSS ({unit_mapper[exp.name]}) (lower is better)', labelpad=20)
#     ax.grid(linewidth=0.25)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_tick_params(which='minor', size=0)
#     ax.yaxis.set_tick_params(which='minor', width=0)
#     plt.yticks(range(0, len(exp.benchmarks())), exp.benchmarks(), ha="right")
#     # ax.set_yticks(range, len(exp.benchmarks()))
#     # ax.set_ytickslabels(exp.benchmarks())
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     ax.xaxis.set_major_formatter(formatter)
#     plt.tight_layout()
#     plt.savefig(f"plots/{exp.name}_mem.svg", format="svg", bbox_inches="tight")
#     print("Graph saved to '%s'" % f"{exp.name}_mem.svg")
#
# def ploth(exp):
#     means = [[mean(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#     cis = [[confidence_interval(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#
#     sns.set(style="whitegrid")
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='sans-serif')
#     fig, ax = plt.subplots(figsize=(5, 3))
#     df = pd.DataFrame(zip(*means), index=exp.benchmarks())
#     errs = pd.DataFrame(zip(*cis), index=exp.benchmarks())
#     plot = df.plot(kind='barh', width=0.8, ax=ax, xerr=errs)
#     plot.margins(x=0.01)
#     ax.legend([name_map[c] for c in exp.cfgs()])
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_xlabel(f'Wall-clock time (ms) (lower is better)', labelpad=20)
#     ax.grid(linewidth=0.25)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_tick_params(which='minor', size=0)
#     ax.yaxis.set_tick_params(which='minor', width=0)
#     plt.yticks(range(0, len(exp.benchmarks())), exp.benchmarks(), ha="right")
#     # ax.set_yticks(range, len(exp.benchmarks()))
#     # ax.set_ytickslabels(exp.benchmarks())
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     ax.xaxis.set_major_formatter(formatter)
#     plt.tight_layout()
#     plt.savefig(f"plots/{exp.name}.svg", format="svg", bbox_inches="tight")
#     print("Graph saved to '%s'" % f"{exp.name}_mem.svg")

# def plot_barh(exp):
#     means = [[mean(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#     cis = [[confidence_interval(exp.iters(cfg, benchmark)) for benchmark in exp.benchmarks()] for cfg in exp.cfgs()]
#
#     sns.set(style="whitegrid")
#     # plt.rc('text', usetex=False)
#     # plt.rc('font', family='sans-serif')
#     fig, ax = plt.subplots(figsize=(8, 3))
#     df = pd.DataFrame(zip(*means), index=exp.benchmarks())
#     # errs = pd.DataFrame(zip(*cis), index=exp.benchmarks())
#     plot = df.plot(kind='barh', width=0.8, ax=ax)
#     plot.margins(x=0.01)
#     ax.legend(exp.cfgs()).remove()
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_ylabel('Wall-clock time (ms)\n(lower is better)')
#     ax.grid(linewidth=0.25)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_tick_params(which='minor', size=0)
#     ax.yaxis.set_tick_params(which='minor', width=0)
#     plt.xticks(range(0, len(exp.benchmarks())), exp.benchmarks(), rotation = 45, ha="right")
#     # ax.set_yticks(range, len(exp.benchmarks()))
#     # ax.set_ytickslabels(exp.benchmarks())
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     ax.yaxis.set_major_formatter(formatter)
#     plt.tight_layout()
#     plt.savefig(f"plots/{exp.name}.svg", format="svg", bbox_inches="tight")
#     print("Barh saved to '%s'" % f"{exp.name}.svg")

# def make_table(exp):
#     bms = exp.benchmarks()
#     with open(f"plots/{exp.name}.tex", "w") as f:
#             # f.write(f" & {bm}")
#         for (cfg, values) in exp.gc_stats.items():
#             f.write(f"{cfg} &")
#             for bm in exp.benchmarks():
#                 f.write(f" {mean(values[bm]['finalisers_run'])} &\n")


# def write_stats(f, exp):
#     def depth(d):
#         depth = 0
#         while(1):
#             if not isinstance(d, dict):
#                 break
#             d = d[next(iter(d))]
#             depth += 1
#         return depth
#
#     def make_args(d, arg):
#         if not isinstance(d, dict):
#             f.write(f"{d}")
#             return
#         for k, v in d.items():
#             f.write(f"\\ifthenelse{{\equal{{#{arg}}}{{{k.replace('_','')}}}}}{{%\n")
#             make_args(v, arg + 1)
#             f.write(f"}}%\n")
#             f.write(f"{{")
#         f.write(f"\\error{{Invalid argument}}%\n")
#
#         for k, v in d.items():
#             f.write(f"}}")
#     stats = exp.dump_stats()
#     f.write(f"\\newcommand{{{exp.latex_name()}}}[{depth(stats)}]{{%\n")
#     make_args(stats, 1)
#     f.write(f"}}%\n")

def confidence_interval(row, pexecs):
    margin_of_error = t.ppf(0.995, pexecs - 1) * (row['std_dev'] / math.sqrt(pexecs))
    return margin_of_error

def load_awfy_results(f):
    raw = pd.read_csv(f, sep='\t', skiprows=4, index_col='benchmark')
    raw = raw[raw['value'] != 0]
    pexecs = raw['invocation'].max();
    iters = raw['iteration'].max();

    df = raw.groupby(['benchmark', 'executor']).agg(mean=('value', 'mean'), std_dev=('value', 'std'))
    df['ci'] = df.apply(confidence_interval, pexecs=pexecs, axis=1)
    df = df.drop(['std_dev'], axis=1)
    df = df.unstack()

    means = df.drop('ci', axis=1).rename(columns={'mean' : 'value'})
    cis = df.drop(['mean'], axis=1).rename(columns={'ci' : 'value'})

    filename = f'plots/{exp_name(f)}.svg'

    plot_perf(filename, means, cis)

def load_sws_results(f):
    raw = pd.read_csv(f, header=None)

    headers = ['requests', 'duration_ms', 'requests_per_sec', 'bytes', 'bytes_transfer_per_sec', 'connect_errors', 'read_errors', 'write_errors', 'http_errors', 'timeouts', 'latency_min', 'latency_max', 'latency_mean_ms', 'latency_stdev']

    raw.columns = headers
    raw = raw.drop(['connect_errors', 'read_errors', 'write_errors', 'http_errors', 'timeouts', 'latency_min', 'latency_max', 'latency_stdev'], axis=1)

    pd.set_option('display.float_format', '{:.4f}'.format)
    df = raw.mean().to_frame(name='mean')
    df.rename(columns={0: 'metric'}, inplace=True)
    df['std_dev'] = raw.std()
    df['ci'] = df.apply(confidence_interval, pexecs=len(raw.index), axis=1)
    df = df.drop(['std_dev'], axis=1)

    suffix = os.path.splitext(os.path.basename(f))[0]
    return (df, suffix)

def load_data(f):
    if "sws_benchmarks" in f:
        return load_sws_results(f)
    elif "awfy_benchmarks" in f:
        return load_awfy_results(f)

def exp_name(f):
    if "barriers" in f:
        exp = "barriers"
    elif "finalise" in f:
        exp = "elision"
    else:
        exp = "perf"
    if "sws_benchmarks" in f:
        return f'sws_{exp}'
    # AWFY
    if "som_rs_ast_perf" in f:
        return 'som_rs_ast_pef'
    if "som_rs_ast_finalise" in f:
        return 'som_rs_ast_finalise'
    if "som_rs_ast_barriers" in f:
        return 'som_rs_ast_barriers'
    if "som_rs_bc_perf" in f:
        return 'som_rs_bc_pef'
    if "som_rs_bc_finalise" in f:
        return 'som_rs_bc_finalise'
    if "som_rs_bc_barriers" in f:
        return 'som_rs_bc_barriers'
    if "yksom_finalise" in f:
        return 'yksom_finalise'
    if "yksom_barriers" in f:
        return 'yksom_barriers'

EXP_MAP = {
    "sws_benchmarks": r"sws",
    "barriers": r"barriere",
}

CFG_LATEX_MAP = {
    "raw_gc": r"\ourgc",
    "raw_arc": r"\rc",
    "perf_gc": r"\ourgc",
    "perf_rc": r"\rc",
    "finalise_naive": r"\fnaive",
    "finalise_elide": r"\felide",
    "barriers_none": r"\bnone",
    "barriers_naive": r"\bnaive",
    "barriers_opt": r"\bopt",
}
print(f"==> processing results for {sys.argv[1:]}")

def add_suffixes_to_columns(df, suffix):
    df.columns = [f'{col}_{suffix}' for col in df.columns]
    return df


def plot_table(name, exp):
    def merge_mean_and_ci(row, mean, ci):
        ci = r'\footnotesize{$\pm$' + f'{row[ci]:.3f}' + r'}'
        return  f'{row[mean]:.2f} ' + ci

    ltx_df = pd.DataFrame()
    cols = exp.columns
    means = cols[::2]
    cis = cols[1::2]
    for m, ci in zip(means, cis):
        cfg = [k for k in CFG_LATEX_MAP.keys() if k in m]
        assert len(cfg) == 1
        cfg = cfg[0]
        ltx_df[CFG_LATEX_MAP[cfg]] = exp.apply(merge_mean_and_ci, mean=m, ci=ci, axis = 1)

    ltx_df.rename(index=lambda x: f"{UNIT_LATEX_MAP[x]}", inplace=True)

    with open(f'plots/{name}_table.tex', 'w') as f:
        f.write(ltx_df.sort_index().to_latex())

args = sys.argv[1:]

if len(args) > 1:
    raw_data = [load_data(arg) for arg in sys.argv[1:]]
    # Combine the results from different files into a single data table
    raw_data = pd.concat([add_suffixes_to_columns(d, s) for d, s in raw_data], axis=1)
    name = args[1]
else:
    raw_data = load_data(args[0])
    name = args[0]

# plot_perf(raw_data)

# plot_table(exp_name(name), raw_data)
