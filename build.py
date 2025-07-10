import logging
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from xvfbwrapper import Xvfb

import benchmarks
from artefacts import *
from util import command_runner


class DoubleQuotedStr(str):
    pass


def double_quoted_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(DoubleQuotedStr, double_quoted_str_representer)


REBENCH_EXEC = "rebench"
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
            if not set(cls).isdisjoint(set(suite.profiles))
        ]

    def __lt__(self, other):
        return self.value < other.value


class GCVS(ExperimentProfile):
    GC = ("gc", "Alloy")
    RC = ("rc", r"RC \texttt{[gcmalloc]}")
    ARC = ("arc", r"RC \texttt{[gcmalloc]}")
    BASELINE = (
        "baseline",
        r"RC \texttt{[jemalloc]}",
        {"gc-default-allocator": False},
    )
    TYPED_ARENA = ("typed-arena", "Typed Arena")
    RUST_GC = ("rust-gc", "Rust-GC")

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.GC
        return None

    @property
    def baseline(cls):
        return cls.BASELINE


class PremOpt(ExperimentProfile):
    NAIVE = (
        "naive",
        "All Barriers",
        {"premature-finalizer-prevention-optimize": False},
    )
    OPT = ("opt", "Optimized Barriers")
    NONE = (
        "none",
        "No Barriers ",
        {
            "premature-finalizer-prevention": False,
            "premature-finalizer-prevention-optimize": False,
        },
    )

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF, Metric.METRICS]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.OPT
        return None

    @property
    def baseline(cls):
        return cls.NONE


class Elision(ExperimentProfile):
    NAIVE = ("naive", "No Elision", {"finalizer-elision": False})
    OPT = ("opt", "Elision")

    @classmethod
    @property
    def measurements(cls):
        return [Metric.PERF, Metric.METRICS]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.OPT
        return None

    @property
    def baseline(cls):
        return cls.NAIVE


def cargo_build(c, src, outdir, install_dir, build_artefact, bin, env=None):
    src = Path(src)
    outdir = Path(outdir)
    install_dir = Path(install_dir)

    install_dir.mkdir(parents=True, exist_ok=True)
    target_bin_dir = outdir / "release"

    install_path = install_dir / bin
    if install_path.exists():
        logging.info(f"{install_path} already exists. Skipping...")
        return

    logging.info(f"Building {install_path} from {src})")

    build_cmd = [
        "cargo",
        "build",
        "--release",
        "--target-dir",
        str(outdir),
        "--bin",
        build_artefact,
    ]

    with c.cd(str(src)):
        try:
            c.run(" ".join(build_cmd), env=env)
        except Exception as e:
            logging.error(f"Failed to build {src}: {e}")
            raise

    # Create symlinks for newly built binaries
    source_path = target_bin_dir / build_artefact
    target_path = install_dir / bin

    if not source_path.exists():
        logging.warning(f"Built binary {source_path} not found after build")
        return

    try:
        if target_path.is_symlink():
            target_path.unlink()

        os.symlink(source_path.absolute(), target_path)
        logging.info(f"Symlinked {source_path} -> {target_path}")
    except OSError as e:
        logging.error(f"Failed to create symlink for {bin}: {e}")
        raise

    logging.info(f"Successfully installed {source_path} --> {target_path}")


def has_unstaged_changes(c, directory):
    with c.cd(directory):
        result = c.run("git diff --quiet", warn=True)
        return result.exited != 0


@contextmanager
def xvfb_display(width=1280, height=720, colordepth=24, display=99):
    xvfb = Xvfb(width=width, height=height, colordepth=colordepth, display=display)
    try:
        xvfb.start()
        yield xvfb
    finally:
        xvfb.stop()


def patch_repo(c, repo_path, patch_file):
    repo_path = Path(repo_path)
    patch_file = Path(patch_file)
    logging.info(f"Attempting to patch: {repo_path}")

    with c.cd(str(repo_path)):
        # Reset to clean state
        logging.info("Resetting repository to HEAD...")
        c.run("git reset --hard HEAD")

        if not patch_file.exists():
            logging.info(f"No patch file exists: {patch_file}. Skipping..")
        else:
            # Apply the patch
            logging.info(f"Applying patch: {patch_file}")
            c.run(f"git apply {patch_file.absolute()}")

            logging.info("Patch applied successfully")


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    crate: Crate
    remove: str
    benchmarks: Tuple["Benchmark"]
    gcvs: Optional[Tuple[ExperimentProfile]] = None
    deps: Optional[Tuple[Crate]] = ()

    @property
    def path(self) -> Path:
        pass

    def __repr__(self):
        return self.name

    @property
    def latex(self) -> str:
        return f"\\{self.name.replace('-','')}"

    @property
    def results(self):
        from results import SuiteData

        results = SuiteData(self)
        return results

    def raw_data(self, measurement) -> Path:
        return RESULTS_DIR / self.name / measurement.value / "results.data"

    @property
    def args(self) -> str:
        return str(self.cmd_args)

    @property
    def cmd_args(self) -> str:
        return ""

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
            Grmtools(
                "grmtools", "", "", benchmarks=benchmarks.GRMTOOLS, gcvs=(GCVS.RC,)
            ),
            BinaryTrees(
                "binary_trees",
                "",
                "",
                benchmarks.BINARY_TREES,
                (GCVS.ARC, GCVS.TYPED_ARENA, GCVS.BASELINE, GCVS.RUST_GC),
            ),
            RegexRedux(
                "regex_redux",
                "",
                "",
                benchmarks.REGEX_REDUX,
                (GCVS.ARC,),
            ),
            RipGrep(
                "ripgrep",
                "",
                "",
                benchmarks.RIPGREP,
                (GCVS.ARC,),
            ),
            Alacritty(
                "alacritty",
                "",
                "",
                benchmarks.ALACRITTY,
                (GCVS.ARC,),
            ),
            FdFind("fd", "", "", benchmarks.FD, (GCVS.ARC,)),
            SomrsAST(
                "som-rs-ast",
                "",
                "",
                benchmarks.SOM,
                (GCVS.RC,),
            ),
            SomrsBC("som-rs-bc", "", "", benchmarks.SOM, (GCVS.RC,)),
            YkSOM(
                "yksom",
                "",
                "",
                benchmarks.SOM,
            ),
        }


class Grmtools(BenchmarkSuite):
    JAVA_SRCS = (
        Repo(
            name="jenkins",
            shallow_clone=True,
            url="https://github.com/jenkinsci/jenkins",
            version="master",
        ),
        Repo(
            name="spring",
            shallow_clone=True,
            url="https://github.com/spring-projects/spring-framework",
            version="master",
        ),
        Repo(
            name="hadoop",
            shallow_clone=True,
            url="https://github.com/apache/hadoop",
            version="master",
        ),
        Repo(
            name="eclipse",
            shallow_clone=True,
            url="https://github.com/eclipse-platform/eclipse.platform",
            version="master",
        ),
    )

    GRMTOOLS = Repo(
        name="grmtools",
        url="https://github.com/softdevteam/grmtools",
        version="a0972be0777e599a3dbca710fb0a595c39560b69",
    )

    ERRORGEN = Path(SRC_DIR / "errorgen")

    PARSERBENCH = Path(SRC_DIR / "parserbench")

    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    CACTUS = Repo(
        name="cactus",
        url="https://github.com/softdevteam/cactus",
        version="8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad",
    )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        for repo in self.JAVA_SRCS:
            repo.fetch()

        self.GRMTOOLS.fetch()
        self.REGEX.fetch()
        self.CACTUS.fetch()

        cargo_build(
            c,
            self.ERRORGEN,
            BUILD_DIR / "aux/errorgen",
            BIN_DIR / "aux/errorgen",
            "errorgen",
            "errorgen",
        )

        for repo in self.JAVA_SRCS:
            if has_unstaged_changes(c, repo.src):
                logging.info(
                    f"{repo.name} sources have already had parser-errors introduced. Skipping..."
                )
                continue
            with c.cd(repo.src):
                c.run(str(BIN_DIR / "aux" / "errorgen" / "errorgen") + " " + repo.name)

        patch_repo(c, self.GRMTOOLS.src, PATCH_DIR / "grmtools.diff")

        gc_cfgs = [
            "default",
            "premopt-opt",
            "premopt-none",
            "premopt-naive",
            "elision-naive",
            "elision-none",
            "default-metrics",
            "premopt-opt-metrics",
            "premopt-none-metrics",
            "premopt-naive-metrics",
            "elision-naive-metrics",
            "elision-none-metrics",
        ]
        if any(profile in s for s in gc_cfgs):
            patch_repo(c, self.REGEX.src, PATCH_DIR / "regex.gc.diff")
            patch_repo(c, self.CACTUS.src, PATCH_DIR / "cactus.gc.diff")
        else:
            patch = profile.split("-")[-1]
            patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{patch}.diff")
            patch_repo(c, self.CACTUS.src, PATCH_DIR / f"cactus.{patch}.diff")

        cargo_build(
            c,
            self.PARSERBENCH,
            target_dir,
            install_dir,
            build_artefact="parserbench",
            bin=bench_cfg_bin,
            env=env,
        )


class BinaryTrees(BenchmarkSuite):
    BINARY_TREES = Path(SRC_DIR / "binary_trees")

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        if profile == "baseline":
            profile = "arc"
        elif profile not in {"gc", "typed-arena", "rust-gc", "arc"}:
            profile = "gc"
        cargo_build(
            c,
            self.BINARY_TREES,
            target_dir,
            install_dir,
            profile,
            bench_cfg_bin,
            env,
        )


class RegexRedux(BenchmarkSuite):
    REGEX_REDUX = Path(SRC_DIR / "regex_redux")
    FASTA = Path(SRC_DIR / "fasta")

    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.REGEX.fetch()

        cargo_build(
            c,
            self.FASTA,
            BUILD_DIR / "aux/fasta",
            BIN_DIR / "aux/fasta",
            "fasta",
            "fasta",
        )

        redux_input = BIN_DIR / "aux" / "fasta" / "redux_input.txt"
        if not redux_input.exists():
            with c.cd(str(redux_input.parent)):
                c.run(f"{str(redux_input.parent / 'fasta')} 2500000 > redux_input.txt")

        patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{profile}.diff")

        cargo_build(
            c,
            self.REGEX_REDUX,
            target_dir,
            install_dir,
            "regex_redux",
            bench_cfg_bin,
            env,
        )

        target_path = install_dir / "redux_input.txt"
        if not target_path.exists():
            os.symlink(redux_input.absolute(), target_path)
            logging.info(f"Symlinked {redux_input} -> {target_path}")
        else:
            logging.warning(f"Symlink {target_path} already exists. Skipping...")


class RipGrep(BenchmarkSuite):
    RIPGREP = Repo(
        name="ripgrep",
        url="https://github.com/burntsushi/ripgrep",
        version="de4baa10024f2cb62d438596274b9b710e01c59b",
    )

    LINUX = Repo(
        name="linux",
        url="https://github.com/BurntSushi/linux",
        version="master",
        shallow_clone=True,
        post_checkout=(("make", "defconfig"), ("make", "-j100")),
    )

    @property
    def cmd_args(self):
        return f"-j1 $(cat {str(Path('aux/ripgrep_args').resolve())}/%(benchmark)s) {str(RipGrep.LINUX.src)}"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.LINUX.fetch()
        self.RIPGREP.fetch()

        if not (self.LINUX.src / ".config").exists():
            with c.cd(str(self.LINUX.src)):
                c.run(f"make defconfig")
                c.run(f"make -j100")

        patch_repo(c, self.RIPGREP.src, PATCH_DIR / f"ripgrep.{profile}.diff")

        cargo_build(
            c,
            self.RIPGREP.src,
            target_dir,
            install_dir,
            "rg",
            bench_cfg_bin,
            env,
        )


class SomrsAST(BenchmarkSuite):

    SOMRS = Repo(
        name="som-rs",
        url="https://github.com/Hirevo/som-rs",
        version="35b780cbee765cca24201fe063d3f1055ec7f608",
        recursive=True,
    )

    BINARY_NAME = "som-interpreter-ast"

    @property
    def cmd_args(self):
        return (
            f"-c {self.SOMRS.src}/core-lib/Smalltalk "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/Richards "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/DeltaBlue "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/NBody "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/Json "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/GraphSearch "
            f"{self.SOMRS.src}/core-lib/Examples/Benchmarks/LanguageFeatures "
            f"-- BenchmarkHarness %(benchmark)s %(iterations)s"
        )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.SOMRS.fetch()

        patch_repo(c, self.SOMRS.src, PATCH_DIR / f"som-rs.{profile}.diff")

        cargo_build(
            c,
            self.SOMRS.src,
            target_dir,
            install_dir,
            self.BINARY_NAME,
            bench_cfg_bin,
            env,
        )


class SomrsBC(SomrsAST):
    BINARY_NAME = "som-interpreter-bc"


class YkSOM(BenchmarkSuite):
    YKSOM = Repo(
        name="yksom",
        url="https://github.com/softdevteam/yksom",
        version="master",
        recursive=True,
    )

    @property
    def cmd_args(self):
        return (
            f"--cp {self.YKSOM.src}/SOM/Smalltalk:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/Richards:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/DeltaBlue:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/NBody:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/Json:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/GraphSearch:"
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/LanguageFeatures "
            f"{self.YKSOM.src}/SOM/Examples/Benchmarks/BenchmarkHarness.som "
            f"%(benchmark)s %(iterations)s"
        )

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.YKSOM.fetch()
        cargo_build(
            c,
            self.YKSOM.src,
            target_dir,
            install_dir,
            "yksom",
            bench_cfg_bin,
            env,
        )


class FdFind(BenchmarkSuite):
    FD = Repo(
        name="fd",
        url="https://github.com/sharkdp/fd",
        version="a4fdad6ff781b5b496c837fde24001b0e46973d6",
    )

    @property
    def cmd_args(self):
        return "-j1"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.FD.fetch()
        ALLOY.repo.fetch()
        cargo_build(
            c,
            self.FD.src,
            target_dir,
            install_dir,
            "fd",
            bench_cfg_bin,
            env,
        )


class Alacritty(BenchmarkSuite):
    ALACRITTY = Repo(
        name="alacritty",
        url="https://github.com/alacritty/alacritty.git",
        version="1063706f8e8a84139e5d2b464a4978e9d840ea17",
    )

    VTE_BENCH = Repo(
        name="vtebench",
        url="https://github.com/alacritty/vtebench.git",
        version="c75155bfc252227c0efc101c1971df3e327c71c4",
    )

    @property
    def cfg_args(self):
        return (
            f"-e bash -c \"[ ! -f {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} ] || "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} && "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'benchmark'}\""
        )

    # @property
    # def cfg_args(self):
    #     return f"""-e bash -c \"[ ! -f {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} ] || {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} && {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'benchmark'}\""""

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.VTE_BENCH.fetch()
        self.ALACRITTY.fetch()

        patch_repo(c, self.ALACRITTY.src, PATCH_DIR / f"alacritty.{profile}.diff")

        cargo_build(
            c,
            self.ALACRITTY.src,
            target_dir,
            install_dir,
            "alacritty",
            bench_cfg_bin,
            env,
        )


@dataclass(frozen=True)
class Experiment:
    profiles: Tuple[ExperimentProfile]
    measurement: Metric
    suite: "BenchmarkSuite"
    pexecs: int = 0

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    def run(self, c, pexecs, config):
        self.results.parent.mkdir(parents=True, exist_ok=True)
        rebench_cmd = [
            str(REBENCH_EXEC),
            "-R",
            "-D",
            "--invocations",
            str(pexecs),
            "--iterations",
            "1",
            str(config),
            self.name,
        ]

        if self.suite.name == "alacritty":
            with Xvfb():
                c.run(" ".join(rebench_cmd), warn=True, pty=True)
        else:
            c.run(" ".join(rebench_cmd), warn=True, pty=True)

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
    def baseline(self):
        # Helper to safely get config or None
        def get_config(config_id):
            return next(
                (e for e in self.experiment.configurations() if config_id in e.id), None
            )

        if self.experiment.experiment == "premopt":
            return get_config("premopt-none")
        elif self.experiment.experiment == "elision":
            if "elision-naive" not in self.id:
                return get_config("elision-naive")
            else:
                return get_config("default")
        else:
            if "default" not in self.id:
                return get_config("default")
            else:
                arc = get_config("gcvs-arc")
                rc = get_config("gcvs-rc")
                return arc or rc

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
        if self.alloy.profile in [
            GCVS.GC,
            PremOpt.NAIVE,
            PremOpt.NONE,
            PremOpt.OPT,
            Elision.OPT,
            Elision.NAIVE,
        ]:
            return "gc"
        else:
            return self.alloy.profile.value

    @property
    def env(self):
        return {
            "RUSTC": self.alloy.path,
            "LD_PRELOAD": str(self.alloy.install_prefix / "lib" / "libgc.so"),
        }

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
        return 10
        # return self.suite.crate.steps

    def build(self, c=None):
        env = self.env.copy()
        env.update({"GC_DONT_GC": "true"})
        self.suite.build(
            c,
            self.build_dir,
            self.install_prefix,
            self.name,
            self.patch_suffix,
            env=env,
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

        from results import Criterea, SuiteData

        # results = SuiteData.concat([s.results for s in suites])
        # premopt = results.premopt
        # print(premopt)

        for s in suites:
            s.results.premopt.plot_bar(
                "benchmark", x="configuration", y=Criterea.WALLCLOCK
            )
            s.results.premopt.plot_bar("benchmark", x="configuration", y=Criterea.USER)

            s.results.gcvs.plot_bar(
                "benchmark", x="configuration", y=Criterea.WALLCLOCK
            )
            s.results.gcvs.plot_bar("benchmark", x="configuration", y=Criterea.USER)
            s.results.elision.plot_bar(
                "benchmark", x="configuration", y=Criterea.WALLCLOCK
            )
            s.results.elision.plot_bar("benchmark", x="configuration", y=Criterea.USER)

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
            libgc = str(cfg.alloy.install_prefix / "lib" / "libgc.so")
            exec_part[cfg.name] = {
                "path": str(cfg.install_prefix),
                "executable": cfg.path.name,
            }
            if cfg.metric == Metric.METRICS:
                exec_part[cfg.name].update(
                    {
                        "env": {
                            "GC_LOG_DIR": str(cfg.metrics_data),
                            "LD_PRELOAD": libgc,
                        }
                    }
                )
            else:
                exec_part[cfg.name].update(
                    {
                        "env": {
                            "LD_PRELOAD": libgc,
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
                "command": e.suite.args,
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
