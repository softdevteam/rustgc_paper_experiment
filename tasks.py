import os

from invoke import task

import results
from artefacts import HEAPTRACK
from build import BenchmarkSuite, Experiments, Metric
from util import timer


def _parse_args(pexecs, exps=None, suites=None, measurements=None):
    def _to_list(val):
        return val.split() if isinstance(val, str) else val

    return Experiments.all(pexecs).filter(
        experiments=_to_list(exps),
        suites=_to_list(suites),
        measurements=_to_list(measurements),
    )


def _build_alloy(experiments: "Experiments", warn_if_empty=False):
    cfgs = experiments.alloy_variants(only_missing=True)
    if not cfgs:
        if warn_if_empty:
            print("Nothing to do")
        return

    print(
        f"{len(cfgs)} Alloy variant(s) require installing for {len(experiments.experiments)} experiment(s):"
    )
    [print(f"  {a.name}") for a in cfgs]
    with timer("Building", sum(a.steps for a in cfgs)):
        for a in cfgs:
            a.build()


def list_exps(c):
    r = Experiments.all()
    es = r.experiments
    for e in es:
        for c in e.configurations():
            print(f"{c.name} --> {c.baseline}")


@task
def build_alloy(c, experiments=None, measurements=None):
    """Build all alloy configurations"""
    exps = _parse_args(pexecs=0, exps=experiments, measurements=measurements)
    _build_alloy(exps, warn_if_empty=True)


@task
def build_heaptrack(c):
    HEAPTRACK.build(c)


@task
def build_benchmarks(c, experiments=None, suites=None, measurements=None):
    """Build all benchmarks for all configurations"""
    build_heaptrack(c)
    exps = _parse_args(
        pexecs=0, exps=experiments, suites=suites, measurements=measurements
    )
    _build_alloy(exps)

    cfgs = exps.configurations(only_missing=True)
    if not cfgs:
        print("Nothing to do")
        return
    print(f"Found {len(cfgs)} benchmark configuration(s):")
    [print(f"  {cfg.name}") for cfg in cfgs]
    for cfg in cfgs:
        cfg.build(c=c)


@task
def run_benchmarks(c, pexecs, experiments=None, suites=None, measurements=None):
    exps = _parse_args(
        int(pexecs), exps=experiments, suites=suites, measurements=measurements
    )

    with timer(
        f"Running {len(exps.experiments)} experiments for {exps.total_iters} iterations",
        exps.total_iters,
        detailed=True,
    ):
        exps.run(c, pexecs)


@task
def clean_alloy(c, experiments=None):
    exps = Experiments.new()

    if experiments:
        exps = exps.filter_experiments(experiments)

    to_remove = set(cfg.alloy for cfg in exps.configurations() if cfg.alloy.installed)

    if not to_remove:
        print("No Alloy configurations installed")
        return

    print(f"Found {len(to_remove)} Alloy configuration(s):")
    [print(f"  {os.path.relpath(a.bin)}") for a in to_remove]

    response = input(f"Are you sure you want to remove them? (y/n): ").strip().lower()
    if response == "y":
        for alloy in to_remove:
            alloy.remove(including_src=False)
        print(f"{len(alloy)} benchmark configurations removed.")
    else:
        print("cancelled.")


@task
def process_results(c):
    results.process_results()


@task
def clean_results(c, suites=None):
    exps = Experiments()
    if suites:
        exps = exps.filter_suites(suites)

    exps.clean()


@task
def clean_benchmarks(c, suites=None, including_src=False):
    """Clean all benchmarks for all configurations"""

    exps = Experiments.new()
    if suites:
        exps = exps.filter_suites(suites)

    cfgs = exps.configurations(only_installed=True)

    if not cfgs:
        print("No benchmark programs installed")
        return
    print(f"Found {len(cfgs)} configurations:")
    [print(f"  {os.path.relpath(cfg.bin)}") for cfg in cfgs]

    response = input(f"Are you sure you want to remove them? (y/n): ").strip().lower()
    if response == "y":
        for cfg in cfgs:
            cfg.remove(including_src)
        print(f"{len(cfgs)} benchmark configurations removed.")
    else:
        print("cancelled.")


@task
def clean(c):
    response = (
        input("Are you sure you want to remove everything? (y/n): ").strip().lower()
    )
    if response == "y":
        clean_alloy(c)
        clean_benchmarks(c)
    else:
        print("cancelled.")
