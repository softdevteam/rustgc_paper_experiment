#! /usr/bin/env python

import gc, math, random, os, sys
from os import listdir, stat
from statistics import geometric_mean, stdev
import pandas as pd

def mean(l):
    return math.fsum(l) / float(len(l))

def confidence_interval(l):
    Z = 2.576  # 99% interval
    return Z * (stdev(l) / math.sqrt(len(l)))

if os.path.exists('results/raw_gc.csv') and os.path.exists('results/raw_arc.csv'):
    gc = pd.read_csv('results/raw_gc.csv', header=None).values.tolist()
    arc = pd.read_csv('results/raw_arc.csv', header=None).values.tolist()

    with open('results/perf.csv', 'w') as f:
        gc_reqs = [rs[2] for rs in gc]
        arc_reqs = [rs[2] for rs in arc]
        gc_ci = confidence_interval(gc_reqs)
        arc_ci = confidence_interval(arc_reqs)
        f.write(f"\\sws,\\ourgc,{mean(gc_reqs):2f},{gc_ci:3f},{gc_ci:3f}\n")
        f.write(f"\\sws,\\arc,{mean(arc_reqs):2f},{arc_ci:3f},{arc_ci:3f}\n")

if os.path.exists('results/raw_barriers_naive.csv') \
    and os.path.exists('results/raw_barriers_opt.csv') \
    and os.path.exists('results/raw_barriers_none.csv'):

    barriers_opt = pd.read_csv('results/raw_barriers_opt.csv', header=None).values.tolist()
    barriers_naive = pd.read_csv('results/raw_barriers_naive.csv', header=None).values.tolist()
    barriers_none = pd.read_csv('results/raw_barriers_none.csv', header=None).values.tolist()

    with open('results/barriers.csv', 'w') as f:
        opt_reqs = [rs[2] for rs in barriers_opt]
        none_reqs = [rs[2] for rs in barriers_none]
        naive_reqs = [rs[2] for rs in barriers_naive]
        opt_ci = confidence_interval(opt_reqs)
        none_ci = confidence_interval(none_reqs)
        naive_ci = confidence_interval(naive_reqs)
        f.write(f"\\sws,barriers_opt,{mean(opt_reqs):2f},{opt_ci:3f},{opt_ci:3f}\n")
        f.write(f"\\sws,barriers_none,{mean(none_reqs):2f},{none_ci:3f},{none_ci:3f}\n")
        f.write(f"\\sws,barriers_naive,{mean(naive_reqs):2f},{naive_ci:3f},{naive_ci:3f}\n")

if os.path.exists('results/raw_finalise_elide.csv') \
    and os.path.exists('results/raw_finalise_naive.csv'):

    finalise_naive = pd.read_csv('results/raw_finalise_naive.csv', header=None).values.tolist()
    finalise_elide = pd.read_csv('results/raw_finalise_elide.csv', header=None).values.tolist()

    with open('results/elision.csv', 'w') as f:
        naive_reqs = [rs[2] for rs in finalise_naive]
        elide_reqs = [rs[2] for rs in finalise_elide]
        naive_ci = confidence_interval(naive_reqs)
        elide_ci = confidence_interval(elide_reqs)
        f.write(f"\\sws,finalise_naive,{mean(naive_reqs):2f},{naive_ci:3f},{naive_ci:3f}\n")
        f.write(f"\\sws,finalise_elide,{mean(elide_reqs):2f},{elide_ci:3f},{elide_ci:3f}\n")
