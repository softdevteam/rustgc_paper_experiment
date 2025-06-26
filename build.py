import logging
import os
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set

import yaml

import benchmarks
from artefacts import *
from util import command_runner


class DoubleQuotedStr(str):
    pass


def double_quoted_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(DoubleQuotedStr, double_quoted_str_representer)

REBENCH_EXEC = Path(".venv/bin/rebench").resolve()
REBENCH_CONF = Path("rebench.conf").resolve()
RESULTS_DIR = Path("results").resolve()
PATCH_DIR = Path("patch").resolve()
DEFAULT_PEXECS = 30
PEXECS = 30
DEFAULT_MEASUREMENTS = ["perf"]


class Metric(Enum):
    PERF = "perf"
    METRICS = "metrics"
    HEAPTRACK = "heaptrack"


class ExperimentProfile(Enum):

    def __init__(self, value, latex, alloy_flags=None):
        self._value_ = value
        self.latex = latex
        self.alloy_flags = alloy_flags

    @property
    def experiment(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def full(self) -> str:
        return f"{self.experiment}-{self.value}"

    @property
    def path(self) -> Path:
        exp = self.name.lower()
        return Path(self.experiment) / Path(exp)

    @classmethod
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiment"]:
        return [
            Experiment(tuple(set(cls) & set(suite.profiles)), m, suite, pexecs)
            for suite in BenchmarkSuite.all()
            for m in cls.measurements
        ]


class GCVS(ExperimentProfile):
    GC = ("gc", "Alloy")
    RC = ("rc", r"\texttt{Rc<T>}")
    ARC = ("arc", r"\texttt{Arc<T>}")
    BASELINE = ("baseline", r"Baseline", {"gc-default-allocator": False})
    TYPED_ARENA = ("typed-arena", "Typed Arena")
    RUST_GC = ("rust-gc", "Rust-GC")

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF]


class PremOpt(ExperimentProfile):
    NAIVE = (
        "naive",
        "Barriers Naive",
        {"premature-finalizer-prevention-optimize": False},
    )
    OPT = ("opt", "Barriers Opt")
    NONE = (
        "none",
        "Barriers None",
        {
            "premature-finalizer-prevention": False,
            "premature-finalizer-prevention-optimize": False,
        },
    )

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF, Metric.METRICS]


class Elision(ExperimentProfile):
    NAIVE = ("naive", "No elision", {"finalizer-elision": False})
    OPT = ("opt", "Elision")

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF, Metric.METRICS]


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    crate: Crate
    cmd_args: str
    benchmarks: Tuple["Benchmark"]
    gcvs: Optional[Tuple[ExperimentProfile]] = None
    deps: Optional[Tuple[Crate]] = ()
    setup: Optional[Tuple[str]] = None
    teardown: Optional[Tuple[str]] = None

    @property
    def path(self) -> Path:
        pass

    @property
    def latex(self) -> str:
        return f"\\{self.name.replace('-','')}"

    @property
    def results(self):
        from results import SuiteData

        results = SuiteData(self, [Metric.PERF, Metric.METRICS])
        return results

    def raw_data(self, measurement) -> Path:
        return RESULTS_DIR / self.name / measurement.value / "results.data"

    @property
    def args(self) -> str:
        return self.cmd_args

    @property
    def profiles(self) -> tuple[ExperimentProfile]:
        p = tuple(PremOpt) + tuple(Elision)
        if self.gcvs:
            # self.gcvs is probably a set or list; convert to tuple for concatenation
            p = p + self.gcvs + (GCVS.BASELINE, GCVS.GC)
        return p

    @classmethod
    def all(cls) -> Set["BenchmarkSuite"]:
        return {
            # BenchmarkSuite(
            #     "alacritty",
            #     ALACRITTY,
            #     benchmarks.ALACRITTY_ARGS,
            #     benchmarks.ALACRITTY,
            #     [GCVS.ARC],
            #     deps=[VTE_BENCH],
            #     setup=[
            #         "Xvfb",
            #         ":99",
            #         "-screen",
            #         "0",
            #         "1280x800x24",
            #         "-nolisten",
            #         "tcp",
            #     ],
            #     teardown=True,
            # ),
            # BenchmarkSuite(
            #     "fd", FD, benchmarks.FD_ARGS, benchmarks.FD, [GCVS.ARC], deps=[LINUX]
            # ),
            # BenchmarkSuite(RIPGREP, [GCVS.ARC], "ripgrep"),
            BenchmarkSuite(
                "som-rs-ast",
                SOMRS_AST,
                benchmarks.SOMRS_ARGS,
                benchmarks.SOM,
                (GCVS.RC,),
            ),
            BenchmarkSuite(
                "som-rs-bc", SOMRS_BC, benchmarks.SOMRS_ARGS, benchmarks.SOM, (GCVS.RC,)
            ),
            # BenchmarkSuite("yksom", YKSOM, benchmarks.YKSOM_ARGS, benchmarks.SOM),
            # BenchmarkSuite(
            #     "grmtools",
            #     PARSER_BENCH,
            #     [GCVS.RC],
            #     deps=[GRMTOOLS, CACTUS, REGEX],
            # ),
        }


@dataclass(frozen=True)
class Experiment:
    profiles: Tuple[ExperimentProfile]
    measurement: Metric
    suite: "BenchmarkSuite"
    pexecs: int = 0

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    @command_runner(description="Benchmarking", allow_failure=True)
    def _rebench(self, config, pexecs):
        return [
            str(REBENCH_EXEC),
            "-R",
            "-D",
            "--invocations",
            str(pexecs),
            "--iterations",
            "1",
            config,
            self.name,
        ]

    def run(self, c, pexecs, config):
        self.results.parent.mkdir(parents=True, exist_ok=True)
        if self.suite.setup:
            setup = " ".join([str(x) for x in self.suite.setup])
            logging.info(f"Running setup: {setup}")
            setup_proc = subprocess.Popen(
                self.suite.setup,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._rebench(config, pexecs)

        if self.suite.teardown:
            logging.info(f"Killing setup proc: {setup_proc.pid}")
            setup_proc.kill()
            setup_proc.wait()
            if not setup_proc.poll():
                logging.error(f"Setup process {setup_proc.pid} still running!")
            else:
                logging.error(
                    f"Setup process {setup_proc.pid} terminated with status ({setup_proc.poll()})"
                )

    def configurations(self, only_installed=False, only_missing=False):
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        identical = [GCVS.GC, PremOpt.OPT, Elision.OPT]
        executors = set()

        for p in self.profiles:
            name = "default" if p in identical else p.full
            is_metrics = self.measurement == Metric.METRICS
            alloy = Alloy(p, metrics=is_metrics)
            executors.add(Executor(self, self.suite, self.measurement, name, alloy))

        if only_installed:
            executors = [self for self in executors if self.installed]
        elif only_missing:
            executors = [self for self in executors if not self.installed]
        return executors

    @property
    def results(self) -> Path:
        return RESULTS_DIR / self.suite.name / self.measurement.value / "results.data"

    @property
    def experiment(self):
        expected = self.profiles[0].experiment

        assert all([x.experiment == expected for x in self.profiles])
        return self.profiles[0].experiment

    def process(self):
        from results import Results

        if not self.results.exists():
            logging.info(f"No results to process for {self.name}")
            return


        # print(plotter.mem_measurements)
        # print(plotter.wallclock)

    @property
    def name(self) -> str:
        return f"{self.suite.name}-{self.experiment}-{self.measurement.value}"

    @property
    def build_steps(self) -> int:
        return sum([cfg.steps for cfg in self.configurations(only_missing=True)])


@dataclass(frozen=True)
class Executor:
    experiment: "Experiment"
    suite: "BenchmarkSuite"
    metric: "Metric"
    id: str
    alloy: Alloy

    @property
    def name(self):
        is_metrics = self.metric == Metric.METRICS
        base = f"{self.suite.name}-{self.id}"
        return f"{base}-metrics" if is_metrics else base

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / "benchmarks" / self.suite.name

    @property
    def path(self) -> Path:
        return self.install_prefix / self.name

    @property
    def metrics_data(self) -> Path:
        return self.experiment.results.parent / f"{self.id}"

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / "benchmarks" / self.suite.name / self.name

    @property
    def installed(self) -> bool:
        return self.path.exists()

    @property
    def patch_suffix(self) -> Optional[str]:
        if self.alloy.profile == GCVS.BASELINE:
            return None
        elif self.alloy.profile in [PremOpt.NAIVE, PremOpt.NONE, Elision.NAIVE]:
            return "gc"
        else:
            return self.alloy.profile.value

    @property
    def env(self):
        return {"RUSTC": self.alloy.path}

    @command_runner(description="Building", dry_run=DRY_RUN)
    def _cargo_build(self):
        return [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            self.suite.crate.cargo_toml,
            "--target-dir",
            self.build_dir,
        ]

    @property
    def build_steps(self):
        return self.suite.crate.steps

    def build(self):
        if self.installed:
            logging.info(
                f"Skipping {self.name}: {os.path.relpath(self.path)} already exists"
            )
            return

        self.suite.crate.repo.fetch()
        logging.info(f"Starting build: {self.name}")
        self.install_prefix.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        for lib in self.suite.deps:
            if lib.repo:
                lib.repo.fetch()

        with ExitStack() as patchstack:
            crates = self.suite.deps + (self.suite.crate,)
            [patchstack.enter_context(c.repo.patch(self.patch_suffix)) for c in crates]
            self._cargo_build()
            target_bin = self.build_dir / "release" / self.suite.crate.name
            if not target_bin.exists():
                print(str(target_bin))
                raise BuildError(f"Build target does not exist")
            logging.info(f"Symlinking {target_bin} -> {self.path}")
            os.symlink(target_bin, self.path)

        logging.info(
            f"Build finished: {self.name}, installed at '{os.path.relpath(self.path)}'"
        )


class CustomExperiment(Experiment):
    _name: str
    _configurations: List["CustomExecutor"]

    def __init__(self, exp_name, suite, metric, cfgs, names):
        # Call parent class constructor to initialize suite field
        super().__init__([], suite, metric)

        self._name = exp_name

        # Initialize your custom configurations
        self._configurations = [
            CustomExecutor(self, suite, name, Path(cfg))
            for cfg, name in zip(cfgs, names)
        ]

    def configurations(self) -> List["CustomExecutor"]:
        return self._configurations

    @property
    def name(self) -> str:
        return self._name

    @property
    def experiment(self):
        return "custom"

    @property
    def results(self) -> Path:
        return (
            RESULTS_DIR / self.measurement.value / "custom" / f"{self.suite.name}.csv"
        )


@dataclass
class Experiments:
    pexecs: int
    experiments: List[Experiment]
    _config: Optional[Path] = None

    def filter(self, *, suites=None, experiments=None, measurements=None):
        if suites is not None:
            self.experiments = [e for e in self.experiments if e.suite.name in suites]
        if experiments is not None:
            self.experiments = [
                e for e in self.experiments if e.experiment in experiments
            ]
        if measurements is not None:
            self.experiments = [
                e for e in self.experiments if e.measurement.value in measurements
            ]
        return self

    def run(self, c, pexecs):
        for e in self.experiments:
            e.run(c, pexecs, self.config)

    def process(self, c):
        exp_name = "premopt"
        suite_exps = [e for e in self.experiments if e.experiment == exp_name]
        suites = set(e.suite for e in suite_exps)

        all_exps = []

        for s in suites:
            raw = s.results
            exps = [e for e in suite_exps if e.suite == s]
            for e in exps:
                if e.measurement == Metric.PERF and e.results.exists():
                    results = raw.for_experiment(e)
                    all_exps.append(results.summary().data)
                    # print(e.name)
                    # print()
                    # print(data.summary().without_errs())
                    # print()
                    # print("=====")
        from results import Overall

        overview = Overall(all_exps)
        overview.mk_perf_table()

    def remove(self):
        for e in self.experiments:
            e.results.unlink(missing_ok=True)

    def alloy_variants(self, only_installed=False, only_missing=False):
        l = [cfg.alloy for cfg in self.configurations()]
        if only_installed:
            l = [a for a in l if a.installed]
        elif only_missing:
            l = [a for a in l if not a.installed]
        return list({a.name: a for a in l}.values())

    @property
    def build_steps(self):
        return sum(cfg.build_steps for cfg in self.configurations(only_missing=True))

    @property
    def total_iters(self):
        unique = sum(len(cfg.suite.benchmarks) for cfg in self.configurations())
        print(self.pexecs)
        return unique * self.pexecs

    @property
    def config(self) -> Path:
        if self._config and self._config.exists():
            return self._config

        exp_part = {}
        exec_part = {}
        bm_part = {}

        executors = self.configurations()
        for cfg in executors:
            exec_part[cfg.name] = {
                "path": str(cfg.install_prefix),
                "executable": cfg.path.name,
            }
            if cfg.metric == Metric.METRICS:
                exec_part[cfg.name].update(
                    {
                        "env": {
                            "GC_LOG_DIR": str(cfg.metrics_data),
                            "LD_PRELOAD": str(
                                cfg.alloy.install_prefix / "lib" / "libgc.so"
                            ),
                        }
                    }
                )

        for e in self.experiments:
            exp_part[e.name] = {
                "suites": [e.suite.name],
                "executions": [cfg.name for cfg in executors if cfg.experiment == e],
                "data_file": str(e.results),
            }

            bm_part[e.suite.name] = {
                "gauge_adapter": e.gauge_adapter,
                "command": DoubleQuotedStr(e.suite.args),
                "benchmarks": [
                    (
                        bench.name
                        if bench.extra_args is None
                        else {
                            bench.name: {
                                "extra_args": DoubleQuotedStr(bench.extra_args)
                            }
                        }
                    )
                    for bench in e.suite.benchmarks
                ],
            }

        config = RebenchConfig(
            experiments=exp_part,
            executors=exec_part,
            benchmark_suites=bm_part,
        )

        config.write_to_file(REBENCH_CONF)
        self._config = REBENCH_CONF
        return self._config

    def configurations(
        self, only_installed=False, only_missing=False
    ) -> List["Executor"]:
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        executors = set()
        for e in self.experiments:
            executors.update(e.configurations(only_installed, only_missing))
        return executors

    @classmethod
    def all(cls, pexecs: int = 30) -> "Experiments":
        profiles = [GCVS, PremOpt, Elision]
        return cls(pexecs, [e for p in profiles for e in p.experiments(pexecs)])


@dataclass
class RebenchConfig:
    experiments: Dict[str, Any]
    executors: Dict[str, Any]
    benchmark_suites: Dict[str, Any]

    @property
    def runs(self):
        return {"max_invocation_time": 360}

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    # def _add_suite(self, suite):
    #     self.benchmark_suites = {
    #         f"{suite.name}": {
    #             "gauge_adapter": self.gauge_adapter,
    #             "command": DoubleQuotedStr(suite.args),
    #             "benchmarks": [
    #                 (
    #                     bench.name
    #                     if bench.extra_args is None
    #                     else {
    #                         bench.name: {
    #                             "extra_args": DoubleQuotedStr(bench.extra_args)
    #                         }
    #                     }
    #                 )
    #                 for bench in suite.benchmarks
    #             ],
    #         }
    #     }

    @classmethod
    def from_custom(cls, suite, experiment_name, cfg_names, cfg_bins):
        experiment = {
            f"{experiment_name}": {"suites": [suite_name], "executions": cfg_names}
        }
        config_data = {
            "experiments": experiment,
            "executors": {},
            "benchmark_suites": {},
        }

        rebench = cls(**config_data)
        rebench._add_suite(suite)
        return rebench

    def write_to_file(self, file_path: Path):
        """Write the configuration to a YAML file."""
        header = "# -*- mode: yaml -*-\n"
        yaml_content = yaml.dump(
            asdict(self),
            explicit_start=False,
            sort_keys=False,
            default_flow_style=False,
            width=1000,
            allow_unicode=True,
        )

        file_path.write_text(header + yaml_content)
