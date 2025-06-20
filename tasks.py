import os

from invoke import task

from build import BenchmarkSuite, CustomExperiment, Experiments, Metric
from util import timer


@task
def build_alloy(c):
    """Build all alloy configurations"""
    exps = Experiments.new()
    cfgs = exps.configurations(only_missing=True)
    alloy_cfgs_needed = set(cfg.alloy for cfg in cfgs if not cfg.alloy.installed)
    alloy_build_steps = sum(a.steps for a in alloy_cfgs_needed)
    with timer("Building missing alloy configurations", alloy_build_steps):
        for alloy in alloy_cfgs_needed:
            alloy.build()


@task
def build_benchmarks(c, suites=None):
    """Build all benchmarks for all configurations"""

    exps = Experiments.new()
    if suites:
        exps = exps.filter_suites(suites)

    cfgs = exps.configurations(only_missing=True)

    alloy_cfgs_needed = set(cfg.alloy for cfg in cfgs if not cfg.alloy.installed)
    alloy_build_steps = sum(a.steps for a in alloy_cfgs_needed)

    print(f"Found {len(alloy_cfgs_needed)} Alloy configuration(s):")
    [print(f"  {os.path.relpath(a.path)}") for a in alloy_cfgs_needed]

    with timer("Building missing alloy configurations", alloy_build_steps):
        for alloy in alloy_cfgs_needed:
            alloy.build()

    with timer("Building benchmark configurations", exps.build_steps):
        for cfg in cfgs:
            cfg.build()


@task
def run_benchmarks(c, pexecs, experiments=None, suites=None, metric=None):
    pexecs = int(pexecs)
    exps = Experiments.new(pexecs)
    if suites:
        exps = exps.filter_suites(suites)

    if experiments:
        exps = exps.filter_experiments(experiments)

    if metric:
        exps = exps.filter_metric(metric)

    total_iters = exps.run_steps

    with timer(
        f"Running {len(exps.experiments)} experiments for {pexecs} iterations",
        total_iters,
        detailed=True,
    ):
        exps.run(c, pexecs)


@task
def process_benchmarks(c, experiments=None, suites=None, metric=None):
    exps = Experiments()
    if suites:
        exps = exps.filter_suites(suites)

    if experiments:
        exps = exps.filter_experiments(experiments)

    if metric:
        exps = exps.filter_metric(metric)

    exps.process(c)


@task(
    help={
        "suite": "The benchmark suite to run the comparison on.",
        "cfg": "Path to configuration to test.",
        "name": "Name of configuration to test.",
    },
    iterable=["cfg", "name"],
)
def compare(c, experiment_name, cfg, name, suite=None):
    if not suite:
        print(
            "A benchmark suite must be provided with the '--suite' flag. Valid suites:"
        )
        for b in BenchmarkSuite.all():
            print(b.name)
        return

    if not cfg:
        raise ValueError("Configurations must be supplied with the --cfg flag")

    if not name:
        raise ValueError(
            "Each configuration must be be supplied a name with the --name flag"
        )

    if len(cfg) != len(name):
        raise ValueError(
            "The number of provided configurations and names must be the same"
        )

    s = next((b for b in BenchmarkSuite.all() if b.name == suite))
    e = CustomExperiment(experiment_name, s, Metric.PERF, cfg, name)
    exps = Experiments(experiments=[e])
    exps.run(c, 10)
    exps.process(c)


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
