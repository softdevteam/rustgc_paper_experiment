import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from xvfbwrapper import Xvfb

import benchmarks
from artefacts import *

REBENCH_DATA = "results.data"

HS_MAP = {
    "ripgrep": {
        "heapsize-s": "32M",
        "heapsize-l": "64M",
        "heapsize-xl": "128M",
    },
    "binary_trees": {
        "heapsize-s": "4M",
        "heapsize-l": "8M",
        "heapsize-xl": "16M",
    },
    "fd": {
        "heapsize-s": "16M",
        "heapsize-l": "32M",
        "heapsize-xl": "64M",
    },
    "alacritty": {
        "heapsize-s": "16M",
        "heapsize-l": "32M",
        "heapsize-xl": "64M",
    },
    "regex_redux": {
        "heapsize-s": "256M",
        "heapsize-l": "512M",
        "heapsize-xl": "1024M",
    },
    "som-rs-bc": {
        "heapsize-s": "32M",
        "heapsize-l": "64M",
        "heapsize-xl": "128M",
    },
    "som-rs-ast": {
        "heapsize-s": "64M",
        "heapsize-l": "96M",
        "heapsize-xl": "128M",
    },
    "grmtools": {
        "heapsize-s": "1024M",
        "heapsize-l": "2048M",
        "heapsize-xl": "4096M",
    },
}


class Aggregation(Enum):
    INDIVIDUAL = "individual"
    SUITE = "suite"
    OVERALL = "overall"
    SUITE_ARITH = "suite_arith"
    SUITE_GEO = "suite_geo"
    GLOBAL_ARITH = "global_arith"
    GLOBAL_GEO = "global_geo"
    SNAPSHOT = "snapshot"


class Agg(Enum):
    ARITH = 1
    GEO = 2


class Metric(Enum):
    def __init__(self, value, latex=None):
        self._value_ = value
        self.latex = latex

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None

    COLLECTION_NUMBER = "collection_number"
    TOTAL_COLLECTIONS = (
        "total_collections",
        r"\# collections",
    )
    MINOR_COLLECTIONS = "minor_collections"
    MAJOR_COLLECTIONS = "major_collections"
    PHASE = "kind"
    MEM_HSIZE_ENTRY = "mem_hsize_entry"
    MEM_HSIZE_EXIT = "mem_hsize_exit"
    MEM_ALLOCD_ENTRY = "mem_allocd_entry"
    MEM_ALLOCD_EXIT = "mem_allocd_exit"
    MEM_ALLOCD_FLZQ = "mem_allocd_flzq"
    MEM_FREED_EXPLICIT_ENTRY = "mem_freed_explicit_entry"
    MEM_FREED_EXPLICIT_EXIT = "mem_freed_explicit_exit"
    MEM_FREED_SWEPT = "mem_freed_swept"
    MEM_FREED_FLZ = "mem_freed_flz"
    TIME_MARKING = "time_marking"
    TIME_FIN_Q = "time_fin_q"
    TIME_SWEEPING = "time_sweeping"
    TIME_TOTAL = "time_total"
    FLZ_REGISTERED = "flz_registered"
    FLZ_ELIDED = "flz_elided"
    FLZ_RUN = "flz_run"
    OBJ_ALLOCD_ARC = "obj_allocd_arc"
    OBJ_ALLOCD_BOX = "obj_allocd_box"
    OBJ_ALLOCD_FLZQ = "obj_allocd_flzq"
    OBJ_ALLOCD_GC = "obj_allocd_gc"
    OBJ_ALLOCD_RC = "obj_allocd_rc"
    OBJ_FREED_SWEPT = "obj_freed_swept"
    OBJ_FREED_EXPLICIT = "obj_freed_explicit"

    MEM_HSIZE_AVG = (
        "mem_hsize_avg",
        "Avg. heap size",
    )
    MAX_MEMORY = "max_heap_size"
    PCT_ELIDED = (
        "pct_elided",
        r"\% Fin. elided",
    )
    GC_TIME = (
        "gc_time",
        "Total GC pause time",
    )

    WALLCLOCK = (
        "total",
        "Wall-clock time",
    )
    USER = (
        "usr",
        "User time",
    )

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return self.value

    @property
    def pathname(self):
        if self == Metric.WALLCLOCK:
            return "wallclock"

        if self == Metric.USER:
            return "user"

        return self.value


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


class Measurement(Enum):
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
    GC = ("gc", r"\texttt{Gc<T>} (Alloy)")
    RC = ("rc", r"\texttt{Arc<T>}/\texttt{Rc<T>}")
    ARC = ("arc", r"\texttt{Arc<T>}/\texttt{Rc<T>}")
    BASELINE = (
        "baseline",
        r"RC \texttt{[jemalloc]}",
        {"gc-default-allocator": False},
    )
    TYPED_ARENA = ("typed-arena", r"\texttt{Arena<T>")
    RUST_GC = ("rust-gc", r"\texttt{Gc<T>} (Rust-GC)")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF, Measurement.HEAPTRACK]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"rc" in value:
                return cls.RC
            elif f"arc" in value:
                return cls.RC
            elif f"{member.value}" in value:
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
        return [Measurement.PERF, Measurement.METRICS, Measurement.HEAPTRACK]

    @classmethod
    def _missing_(cls, value):
        if value == "default":
            return cls.OPT
        else:
            for member in cls:
                if member.value in value:
                    return member
        return None

    @property
    def baseline(cls):
        return cls.NONE


class Elision(ExperimentProfile):
    NAIVE = ("naive", "Before", {"finalizer-elision": False})
    OPT = ("opt", "After")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF, Measurement.METRICS]

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

    def __repr__(self):
        return self.value


class HeapSize(ExperimentProfile):
    DEFAULT = ("default", r"Default")
    S = ("s", "1")
    L = ("l", "2")
    XL = ("xl", "3")

    @classmethod
    @property
    def measurements(cls):
        return [Measurement.PERF]

    @classmethod
    def _missing_(cls, value):
        exp = cls.__name__.lower()

        for member in cls:
            if f"{exp}-{member.value}" in value:
                return member
            elif "default" in value:
                return cls.DEFAULT
        return None

    @property
    def baseline(cls):
        return cls.DEFAULT

    def __repr__(self):
        return self.value


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
    crate: str
    remove: str
    benchmarks: Tuple["Benchmark"]
    gcvs: Optional[Tuple[ExperimentProfile]] = None

    @property
    def path(self) -> Path:
        pass

    def __repr__(self):
        return self.name

    @property
    def latex(self) -> str:
        return f"\\{self.name.replace('-','')}"

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
        p = tuple(PremOpt) + tuple(Elision) + tuple(HeapSize)
        if self.gcvs:
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

    @property
    def cmd_args(self):
        return f"-j1 $(cat {str(Path('aux/ripgrep_args').resolve())}/%(benchmark)s) {str(LINUX.src)}"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        LINUX.fetch()
        self.RIPGREP.fetch()

        if not (LINUX.src / ".config").exists():
            with c.cd(str(LINUX.src)):
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
    REGEX = Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    )

    @property
    def cmd_args(self):
        return "-j1"

    def build(self, c, target_dir, install_dir, bench_cfg_bin, profile, env):
        self.FD.fetch()
        LINUX.fetch()
        self.REGEX.fetch()

        patch_repo(c, self.FD.src, PATCH_DIR / f"fd.{profile}.diff")
        patch_repo(c, self.REGEX.src, PATCH_DIR / f"regex.{profile}.diff")

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
    def cmd_args(self):
        return (
            f"-e bash -c \"[ ! -f {self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} ] || "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} && "
            f"{self.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'benchmark'}\""
        )

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
    measurement: Measurement
    suite: "BenchmarkSuite"
    pexecs: int = 0

    @property
    def gauge_adapter(self) -> str:
        return {"AlloyAdapter": "alloy_adapter.py"}

    def run(self, c, pexecs, config):
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
            with Xvfb(display="99"):
                c.run(" ".join(rebench_cmd), warn=True, pty=True)
        else:
            c.run(" ".join(rebench_cmd), warn=True, pty=True)

    def configurations(self, only_installed=False, only_missing=False):
        if only_installed and only_missing:
            raise ValueError("Can't select both only_installed and only_missing")

        identical = [
            GCVS.GC,
            PremOpt.OPT,
            Elision.OPT,
            HeapSize.DEFAULT,
        ]
        executors = set()

        for p in self.profiles:
            name = "default" if p in identical else p.full
            is_metrics = self.measurement == Measurement.METRICS
            alloy = Alloy(p, metrics=is_metrics)
            executors.add(Executor(self, self.suite, self.measurement, name, alloy))

        if only_installed:
            executors = [self for self in executors if self.installed]
        elif only_missing:
            executors = [self for self in executors if not self.installed]
        return executors

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.suite.name / self.measurement.value

    @property
    def results(self) -> [Path]:

        if self.measurement == Measurement.PERF:
            return {self.results_dir / REBENCH_DATA}
        else:
            return set(
                f
                for f in self.results_dir.rglob("*")
                if f.name != REBENCH_DATA and not f.is_dir()
            )

    @property
    def experiment(self):
        expected = self.profiles[0].experiment

        assert all([x.experiment == expected for x in self.profiles])
        return self.profiles[0].experiment

    @property
    def profile_class(self):
        return self.profiles[0].__class__

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
    metric: "Measurement"
    id: str
    alloy: Alloy

    @property
    def name(self):
        is_metrics = self.metric == Measurement.METRICS
        base = f"{self.suite.name}-{self.id}"
        return f"{base}-metrics" if is_metrics else base

    @property
    def rebench_name(self):
        is_perf = self.metric == Measurement.PERF
        base = f"{self.suite.name}-{self.id}"
        return base if is_perf else f"{base}-{self.metric.value}"

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
    def bin(self) -> str:
        return str(self.name)

    @property
    def run_env(self):
        libgc = Path("/home/jake/research/bdwgc/out/libgc.so")
        env = {
            "DISPLAY": ":99",
            "LD_PRELOAD": str(libgc),
        }
        env["RESULTS_DIR"] = str(self.results_dir)
        env["HT_PATH"] = str(HEAPTRACK.path)
        if self.metric == Measurement.HEAPTRACK:
            env["USE_HT"] = "true"
        if self.metric == Measurement.METRICS:
            env["USE_MT"] = "true"

        if self.id == "heapsize-s":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-s"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-s"]
        elif self.id == "heapsize-l":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-l"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-l"]
        elif self.id == "heapsize-xl":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xl"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xl"]
        elif self.id == "heapsize-xxl":
            env["GC_INITIAL_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xxl"]
            env["GC_MAXIMUM_HEAP_SIZE"] = HS_MAP[self.suite.name]["heapsize-xxl"]

        return env

    @property
    def results_dir(self) -> Path:
        dir = self.experiment.results_dir / self.id
        dir.mkdir(parents=True, exist_ok=True)
        return dir

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
            HeapSize.DEFAULT,
            HeapSize.XL,
            HeapSize.L,
            HeapSize.S,
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

    def remove(self):
        for e in self.experiments:
            e.results_dir.unlink(missing_ok=True)

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
            exec_part[cfg.rebench_name] = {
                "path": str(cfg.install_prefix),
                "executable": cfg.bin,
                "env": cfg.run_env,
            }

        for e in self.experiments:
            exp_part[e.name] = {
                "suites": [e.suite.name],
                "executions": [
                    cfg.rebench_name for cfg in executors if cfg.experiment == e
                ],
                "data_file": str(e.results_dir / REBENCH_DATA),
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
        profiles = [GCVS, PremOpt, Elision, HeapSize]
        return cls(pexecs, [e for p in profiles for e in p.experiments(pexecs)])

    @property
    def results(self):
        results = set()
        for e in self.experiments:
            results.update(e.results)
        return results


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
