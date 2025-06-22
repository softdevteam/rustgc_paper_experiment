import logging
import os
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

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
DEFAULT_PEXECS = 30
DEFAULT_MEASUREMENTS = ["perf"]


class Metric(Enum):
    PERF = "perf"
    METRICS = "metrics"
    HEAPTRACK = "heaptrack"


class Stats(Enum):
    GC_ALLOCS = "Gc allocated"
    RC_ALLOCS = "Rc allocated"
    ARC_ALLOCS = "Arc allocated"
    BOX_ALLOCS = "Box allocated"
    MUTATOR_TIME = "mutator time"
    GC_TIME = "GC time"
    GC_CYCLES = "num GCs"
    BARRIERS_VISITED = "barriers visited"
    FLZR_REGISTERED = "finalizers registered"
    FLZR_COMPLETED = "finalizers completed"
    FLZR_ELIDABLE = "finalizers elidable"
    WALLCLOCK = "wallclock"
    SYS = "sys"

    def __lt__(self, other):
        return self.value < other.value


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


class GCVS(ExperimentProfile):
    GC = ("gc", "Alloy")
    RC = ("rc", r"\texttt{Rc<T>}")
    ARC = ("arc", r"\texttt{Arc<T>}")
    BASELINE = ("baseline", r"Baseline", {"gc-default-allocator": False})
    TYPED_ARENA = ("typed-arena", "Typed Arena")
    RUST_GC = ("rust-gc", "Rust-GC")

    @classmethod
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiments"]:
        measurements = [Metric.PERF]

        return [
            Experiment(list(cls), m, b, pexecs)
            for b in BenchmarkSuite.all()
            for m in measurements
        ]


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
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiment"]:
        measurements = [Metric.PERF, Metric.METRICS]

        return [
            Experiment(list(cls), m, b, pexecs)
            for b in BenchmarkSuite.all()
            for m in measurements
        ]


class Elision(ExperimentProfile):
    NAIVE = ("naive", "No elision", {"finalizer-elision": False})
    OPT = ("opt", "Elision")

    @classmethod
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiment"]:
        measurements = [Metric.PERF, Metric.METRICS]

        return [
            Experiment(list(cls), m, b, pexecs)
            for b in BenchmarkSuite.all()
            for m in measurements
        ]


class BenchmarkSuite(NamedTuple):
    name: str
    crate: Crate
    cmd_args: str
    benchmarks: List["Benchmark"]
    gcvs: Optional[List[ExperimentProfile]] = None
    deps: Optional[List[Crate]] = []
    setup: Optional[List[str]] = None
    teardown: Optional[List[str]] = None

    @property
    def path(self) -> Path:
        pass

    @property
    def latex(self) -> str:
        return f"\\{self.name.replace('-','')}"

    @property
    def args(self) -> str:
        return self.cmd_args

    @property
    def profiles(self) -> List[ExperimentProfile]:
        p = list(PremOpt) + list(Elision)
        if self.gcvs:
            p.extend(self.gcvs + [GCVS.BASELINE, GCVS.GC])
        return p

    @classmethod
    def all(cls) -> List["BenchmarkSuite"]:
        return [
            BenchmarkSuite(
                "alacritty",
                ALACRITTY,
                benchmarks.ALACRITTY_ARGS,
                benchmarks.ALACRITTY,
                [GCVS.ARC],
                deps=[VTE_BENCH],
                setup=[
                    "Xvfb",
                    ":99",
                    "-screen",
                    "0",
                    "1280x800x24",
                    "-nolisten",
                    "tcp",
                ],
                teardown=True,
            ),
            # BenchmarkSuite(
            #     "fd", FD, benchmarks.FD_ARGS, benchmarks.FD, [GCVS.ARC], deps=[LINUX]
            # ),
            # BenchmarkSuite(RIPGREP, [GCVS.ARC], "ripgrep"),
            BenchmarkSuite(
                "som-rs-ast", SOMRS_BC, benchmarks.SOMRS_ARGS, benchmarks.SOM, [GCVS.RC]
            ),
            BenchmarkSuite(
                "som-rs-bc", SOMRS_AST, benchmarks.SOMRS_ARGS, benchmarks.SOM, [GCVS.RC]
            ),
            BenchmarkSuite("yksom", YKSOM, benchmarks.YKSOM_ARGS, benchmarks.SOM),
            # BenchmarkSuite(
            #     "grmtools",
            #     PARSER_BENCH,
            #     [GCVS.RC],
            #     deps=[GRMTOOLS, CACTUS, REGEX],
            # ),
        ]


@dataclass
class Experiment:
    profiles: List[ExperimentProfile]
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

    @property
    def steps(self):
        bms = len(self.suite.benchmarks)
        return bms * len(self.configurations()) * self.pexecs

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

    @property
    def results(self) -> Path:
        return (
            RESULTS_DIR
            / self.measurement.value
            / self.experiment
            / f"{self.suite.name}.csv"
        )

    @property
    def experiment(self):
        expected = self.profiles[0].experiment

        assert all([x.experiment == expected for x in self.profiles])
        return self.profiles[0].experiment

    def process(self):
        from results import Results

        results = Results.from_raw_data(self)
        summary = results.summary()
        print(summary)

        # print(plotter.mem_measurements)
        # print(plotter.wallclock)

    @property
    def name(self) -> str:
        return f"{self.suite.name}-{self.experiment}-{self.measurement.value}"

    def configurations(
        self, only_installed=False, only_missing=False
    ) -> List["Executor"]:
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")
        intersection = set(self.profiles) & set(self.suite.profiles)
        cfgs = (Executor(self.suite, self.measurement, p, self) for p in intersection)

        if only_installed:
            cfgs = filter(lambda e: e.installed, cfgs)
        elif only_missing:
            cfgs = filter(lambda e: not e.installed, cfgs)

        return list(cfgs)

    @property
    def build_steps(self) -> int:
        return sum([cfg.steps for cfg in self.configurations(only_missing=True)])


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
    experiments: List[Experiment]
    _config: Optional[Path] = None

    # @classmethod
    # def from_custom(cls, experiments, pexecs=0):
    #     return cls(
    #         experiments=experiments,
    #     )

    @staticmethod
    def new(pexecs=0, benchmarks=None, profiles=None, metrics=None):
        profiles = profiles or [GCVS, PremOpt, Elision]
        exps = [exp for p in profiles for exp in p.experiments(pexecs)]
        return Experiments(list(filter(lambda e: e.configurations(), exps)))

    @property
    def all(self) -> List["Experiment"]:
        return self.experiments

    def filter_suites(self, suites):
        self.experiments = list(
            filter(lambda e: e.suite.name in suites, self.experiments)
        )
        return self

    def filter_experiments(self, experiments):
        self.experiments = list(
            filter(lambda e: e.experiment in experiments, self.experiments)
        )
        return self

    def filter_metric(self, metric):
        self.experiments = list(
            filter(lambda e: e.metric.name.lower() == metric, self.experiments)
        )
        return self

    def run(self, c, pexecs):
        for e in self.experiments:
            e.run(c, pexecs, self.config)

    def process(self, c):
        for e in self.experiments:
            e.process()

    def remove(self):
        for e in self.experiments:
            e.results.unlink(missing_ok=True)

    @property
    def config(self) -> Path:
        if self._config and self._config.exists():
            return self._config

        exp_part = {}
        exec_part = {}
        bm_part = {}

        for e in self.experiments:
            exp_part[e.name] = {
                "suites": [e.suite.name],
                "executions": [cfg.name for cfg in e.configurations()],
                "data_file": str(e.results),
            }

            for cfg in e.configurations():
                exec_part[cfg.name] = {
                    "path": str(cfg.install_prefix),
                    "executable": cfg.path.name,
                    "env": {
                        "GC_PRINT_STATS": "true",
                        "GC_LOG_DIR": str(cfg.gc_log_dir),
                    },
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

    @property
    def build_steps(self) -> int:
        return sum([e.build_steps for e in self.experiments])

    @property
    def run_steps(self) -> int:
        return sum([e.steps for e in self.experiments])

    def configurations(
        self, only_installed=False, only_missing=False
    ) -> List["Executor"]:
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        cfgs = []
        for e in self.experiments:
            cfgs.extend(e.configurations(only_installed, only_missing))

        return cfgs


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
