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
    def patch(self, suffix: Optional[str]):
        self._reset()
        if suffix:
            patch = PATCH_DIR / f"{self.name}.{suffix}.diff"
            self._patch(patch)
        else:
            logging.info(f"No patch applied for {self.name}")
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

LINUX = Repo(
    name="linux",
    url="https://github.com/BurntSushi/linux",
    version="master",
    shallow_clone=True,
    post_checkout=(("make", "defconfig"), ("make", "-j100")),
)


class Alloy(Artefact):
    DEFAULT_FLAGS: ClassVar[Dict[str, bool]] = {
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
        self.flags = self.DEFAULT_FLAGS.copy() | (self.profile.alloy_flags or {})

    @property
    def config(self) -> Path:
        if self._config:
            return self._config

        flags = self.flags.copy()
        if self.metrics:
            flags.update({"gc-metrics": True})

        config = {
            "alloy": flags,
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
