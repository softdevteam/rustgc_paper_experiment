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
DEFAULT_PEXECS = 30
PEXECS = 30
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

    @classmethod
    def experiments(cls, pexecs=DEFAULT_PEXECS) -> List["Experiment"]:
        return [
            Experiment(list(set(cls) & set(suite.profiles)), m, suite, pexecs)
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

    @property
    def build_steps(self) -> int:
        return sum([cfg.steps for cfg in self.configurations(only_missing=True)])


@dataclass(frozen=True)
class Executor:
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
        return self.install_prefix / self.suffix

    @property
    def stats_dir(self) -> Path:
        return self.experiment.results.parent / "stats" / self.suite.name

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / "benchmarks" / self.suite.name / self.suffix

    @property
    def env(self):
        return {"RUSTC": self.alloy.path}

    @prepare_build
    def build(self):
        for lib in self.suite.deps:
            if lib.repo:
                lib.repo.fetch()

        with ExitStack() as patchstack:
            crates = self.suite.deps + [self]
            [patchstack.enter_context(c.repo.patch(self.profile)) for c in crates]
            self._cargo_build()
            target_bin = self.build_dir / "release" / super().name
            if not target_bin.exists():
                print(target_bin)
                raise BuildError(f"Build target does not exist")
            logging.info(f"Symlinking {target_bin} -> {self.path}")
            os.symlink(target_bin, self.path)


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
        for e in self.experiments:
            e.process()

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

        identical = [GCVS.GC, PremOpt.OPT, Elision.OPT]

        executors = set()
        for e in self.experiments:
            for p in e.profiles:
                name = "default" if p in identical else p.full
                is_metrics = e.measurement == Metric.METRICS
                alloy = Alloy(p, metrics=is_metrics)
                executors.add(Executor(e.suite, e.measurement, name, alloy))

        return executors

    @classmethod
    def all(cls, pexecs: int = 30) -> "Experiments":
        profiles = [GCVS, PremOpt, Elision]
        return cls([e for p in profiles for e in p.experiments(pexecs)])


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
