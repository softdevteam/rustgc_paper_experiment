import logging
import os
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple

import toml

from util import command_runner

DRY_RUN = os.getenv("DRY_RUN", False)

BIN_DIR = Path("artefacts/bin").resolve()
LIB_DIR = Path("artefacts/lib").resolve()
BUILD_DIR = Path("artefacts/build").resolve()
SRC_DIR = Path("src").resolve()
PATCH_DIR = Path("patch").resolve()


class BuildError(Exception):
    def __str__(self):
        return f"Build error: {self.message}"


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    version: str
    recursive: bool = False
    shallow_clone: bool = False
    post_checkout: Optional[Tuple[str]] = None

    @property
    def src(self):
        return SRC_DIR / self.name

    @command_runner(description="Cloning", steps=300)
    def _clone(self):
        clone = ["git", "clone", "--progress"]
        if self.recursive:
            clone.append("--recursive")

        if self.shallow_clone:
            clone.extend(["--depth", "1"])

        clone.extend([self.url, self.src])
        return clone

    @command_runner(description="Checking out", steps=300)
    def _checkout(self):
        return ["git", "-C", self.src, "checkout", self.version]

    @command_runner(description="Patching", steps=1, write_progress=False)
    def _patch(self, diff):
        return ["git", "-C", self.src, "apply", diff]

    @command_runner(
        description="Resetting working tree for", steps=1, write_progress=False
    )
    def _reset(self):
        return ["git", "-C", self.src, "reset", "--hard"]

    def fetch(self):
        if self.src.exists():
            return

        self._clone()
        self._checkout()

    @contextmanager
    def patch(self, profile: "ExperimentProfile"):
        from build import GCVS

        self._reset()
        profile = GCVS.GC.value if not isinstance(profile, GCVS) else profile.value
        diff = PATCH_DIR / f"{self.name}.{profile}.diff"
        if diff.exists():
            self._patch(diff)
        else:
            logging.info(f"No patch applied for {self.name} ({profile})")
        yield
        self._reset()

    def remove(self):
        shutil.rmtree(self.src)
        logging.info(f"-> removed: {self.src}")


def prepare_build(method):
    def wrapper(self, *args, **kwargs):
        if self.installed:
            logging.info(
                f"Skipping {self.name}: {os.path.relpath(self.path)} already exists"
            )
            return

        if not self.build_dir:
            raise BuildError(f" No build directory specified for {self.name}")

        if not self.install_prefix:
            raise BuildError(f" No install prefix specified for {self.name}")

        for dep in self.deps:
            dep.build()

        self.repo.fetch()
        logging.info(f"Starting build: {self.name}")
        self.install_prefix.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        method(self, *args, **kwargs)
        logging.info(
            f"Build finished: {self.name}, installed at '{os.path.relpath(self.path)}'"
        )

    return wrapper


@dataclass(frozen=True)
class Artefact:
    _name: Optional[str] = None
    repo: Optional[Repo] = None
    deps: Tuple["Artefact"] = ()
    steps: Optional[int] = 0
    _src: Optional[Path] = None

    @property
    def name(self) -> str:
        return self._name or self.repo.name

    @property
    def debug_name(self) -> str:
        return self.name

    @property
    def src(self) -> Path:
        return self._src or (SRC_DIR / self.repo.name)

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / self.repo.name

    @property
    def path(self) -> Path:
        return self.install_prefix / self.name

    @property
    def installed(self) -> Path:
        return os.path.lexists(self.path)

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / self.repo.name

    def remove(self, including_src=False):
        logging.info(f"Removing {self.name}:")
        self.path.unlink(missing_ok=True)
        logging.info(f"-> removed: {self.path}")
        shutil.rmtree(self.build_dir)
        logging.info(f"-> removed: {self.build_dir}")
        if including_src:
            self.repo.remove()


class Crate(Artefact):
    @property
    def cargo_toml(self):
        return f"{self.src}/Cargo.toml"

    @command_runner(description="Building", dry_run=DRY_RUN)
    def _cargo_build(self):
        return [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            self.cargo_toml,
            "--target-dir",
            self.build_dir,
        ]

    @prepare_build
    def build(self):
        self._cargo_build()
        for f in (self.target / "release").glob("*"):
            if (
                f.is_file()
                and not f.suffix in [".d", ".rlib"]
                and not f.name.startswith(".")
            ):
                os.symlink(f, self.bin / f.name)


ALLOY = Artefact(
    steps=5907,
    repo=Repo(
        name="alloy",
        url="https://github.com/jacob-hughes/alloy",
        version="quickfix-stats",
    ),
)


class Alloy(Artefact):
    DEFAULT_FLAGS: ClassVar[Dict[str, bool]] = {
        "gc-metrics": False,
        "finalizer-safety-analysis": False,
        "finalizer-elision": True,
        "premature-finalizer-prevention": True,
        "premature-finalizer-prevention-optimize": True,
        "gc-default-allocator": True,
    }

    def __init__(self, profile: "ExperimentProfile", metrics=False):
        self.__dict__.update(ALLOY.__dict__)
        self.profile = profile
        self.metrics = metrics
        self._config = None

    @property
    def config(self) -> Path:
        if self._config and self._config.exists():
            return self._config

        config = {
            "alloy": self.flags,
            "rust": {
                "codegen-units": 0,
                "optimize": True,
                "debug": False,
            },
            "build": {
                "install-stage": 2,
            },
            "llvm": {
                "download-ci-llvm": True,
            },
            "install": {
                "sysconfdir": "etc",
            },
        }
        file = self.src / f"{self.name.replace('-','.')}.config.toml"
        with open(file, "w") as f:
            toml.dump(config, f)
        return file

    @property
    def flags(self):
        return self.DEFAULT_FLAGS.copy() | (self.profile.alloy_flags or {})

    @property
    def name(self) -> str:
        if not self.flags["gc-default-allocator"]:
            base = "rustc-upstream"
        elif self.flags == self.DEFAULT_FLAGS:
            base = "default"
        else:
            base = self.profile.full
        return f"{base}-metrics" if self.metrics else base

    @property
    def install_prefix(self) -> Path:
        return BIN_DIR / self.repo.name / self.name

    @property
    def build_dir(self) -> Path:
        return BUILD_DIR / self.repo.name / self.name

    @property
    def path(self) -> Path:
        return self.install_prefix / "bin" / "rustc"

    @property
    def installed(self) -> bool:
        return self.path.exists()

    @command_runner(dry_run=DRY_RUN)
    def _xpy_install(self):
        return [
            f"{self.src}/x.py",
            "install",
            "--config",
            self.config,
            "--stage",
            "2",
            "--build-dir",
            self.build_dir,
            "--set",
            "build.docs=false",
            "--set",
            f"install.prefix={self.install_prefix}",
            "--set",
            "install.sysconfdir=etc",
        ]

    @prepare_build
    def build(self):
        self._xpy_install()


@dataclass
class CustomExecutor:
    experiment: "Experiment"
    suite: "BenchmarkSuite"
    name: str
    path: Path
    _baseline: Optional[str] = None

    @property
    def install_prefix(self) -> Path:
        return self.path.parent

    @property
    def baseline(self) -> str:
        if self._baseline:
            return self._baseline

        if len(self.experiment.configurations()) == 2:
            return next(
                cfg.name
                for cfg in self.experiment.configurations()
                if cfg.name != self.name
            )
        return None


# BENCHMARKS

SOM_REPO = Repo(
    name="som-rs",
    url="https://github.com/Hirevo/som-rs",
    version="35b780cbee765cca24201fe063d3f1055ec7f608",
    recursive=True,
)
SOMRS_AST = Crate("som-interpreter-ast", steps=71, repo=SOM_REPO)
SOMRS_BC = Crate("som-interpreter-bc", steps=71, repo=SOM_REPO)

YKSOM = Crate(
    steps=102,
    repo=Repo(
        name="yksom",
        url="https://github.com/softdevteam/yksom",
        version="master",
        recursive=True,
    ),
)

GRMTOOLS = Crate(
    repo=Repo(
        name="grmtools",
        url="https://github.com/softdevteam/grmtools",
        version="a0972be0777e599a3dbca710fb0a595c39560b69",
    ),
)

PARSER_BENCH = Crate(
    "parserbench",
    _src=SRC_DIR / "parserbench",
)

RIPGREP = Crate(
    "rg",
    repo=Repo(
        name="ripgrep",
        url="https://github.com/burntsushi/ripgrep",
        version="de4baa10024f2cb62d438596274b9b710e01c59b",
    ),
)

ALACRITTY = Crate(
    steps=152,
    repo=Repo(
        name="alacritty",
        url="https://github.com/alacritty/alacritty.git",
        version="1063706f8e8a84139e5d2b464a4978e9d840ea17",
    ),
)

FD = Crate(
    steps=70,
    repo=Repo(
        name="fd",
        url="https://github.com/sharkdp/fd",
        version="a4fdad6ff781b5b496c837fde24001b0e46973d6",
    ),
)

# benchmark extras

CACTUS = Artefact(
    repo=Repo(
        name="cactus",
        url="https://github.com/softdevteam/cactus",
        version="8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad",
    ),
)

REGEX = Artefact(
    repo=Repo(
        name="regex",
        url="https://github.com/rust-lang/regex",
        version="bcbe40342628b15ab2543d386c745f7f0811b791",
    ),
)

HADOOP = Artefact(
    repo=Repo(
        name="hadoop",
        shallow_clone=True,
        url="https://github.com/apache/hadoop",
        version="master",
    ),
)

ECLIPSE = Artefact(
    repo=Repo(
        name="eclipse",
        shallow_clone=True,
        url="https://github.com/eclipse-platform/eclipse.platform",
        version="master",
    ),
)

SPRING = Artefact(
    repo=Repo(
        name="spring",
        shallow_clone=True,
        url="https://github.com/spring-projects/spring-framework",
        version="master",
    ),
)

JENKINS = Artefact(
    repo=Repo(
        name="jenkins",
        shallow_clone=True,
        url="https://github.com/jenkinsci/jenkins",
        version="master",
    ),
)

VTE_BENCH = Artefact(
    repo=Repo(
        name="vtebench",
        url="https://github.com/alacritty/vtebench.git",
        version="c75155bfc252227c0efc101c1971df3e327c71c4",
    ),
)

LINUX = Artefact(
    repo=Repo(
        name="linux",
        url="https://github.com/BurntSushi/linux",
        version="master",
        shallow_clone=True,
        post_checkout=(("make", "defconfig"), ("make", "-j100")),
    )
)
