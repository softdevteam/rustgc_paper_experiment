import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from invoke import task
from tqdm import tqdm

# --- Configuration ---
ALLOY_REPO = "https://github.com/softdevteam/alloy"
ALLOY_COMMIT = "master"
LIBGC_PATH = "bdwgc/lib"

EXPERIMENTS = {
    "gcvs": {
        "metrics": ["mem", "perf"],
        "config_template": "gcvs.{metric}.config.toml",  # Now uses metrics
    },
    "premopt": {
        "configs": ["none", "opt", "naive"],
        "metrics": ["mem", "perf"],
        "config_template": "premopt.{config}.{metric}.config.toml",
    },
    "elision": {
        "configs": ["naive", "opt"],
        "metrics": ["mem", "perf"],
        "config_template": "elision.{config}.{metric}.config.toml",
    },
}

BENCHMARKS = {
    "alacritty": {
        "alacritty": {
            "url": "https://github.com/alacritty/alacritty.git",
            "rev": "1063706f8e8a84139e5d2b464a4978e9d840ea17",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "fd": {
        "fd": {
            "url": "https://github.com/sharkdp/fd",
            "rev": "a4fdad6ff781b5b496c837fde24001b0e46973d6",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "grmtools": {
        "grmtools": {
            "url": "https://github.com/softdevteam/grmtools",
            "rev": "a0972be0777e599a3dbca710fb0a595c39560b69",
            "gcvs_variants": ["baseline", "arc", "gc"],
        }
    },
    "som": {
        "som-rs": {
            "url": "https://github.com/Hirevo/som-rs",
            "rev": "35b780cbee765cca24201fe063d3f1055ec7f608",
            "gcvs_configs": ["baseline", "gc", "rc"],
            "bins": ["som-interpreter-ast", "som-interpreter-bc"],
        },
        "yksom": {
            "url": "https://github.com/softdevteam/yksom",
            "rev": "master",
        },
    },
    "ripgrep": {
        "ripgrep": {
            "url": "https://github.com/burntsushi/ripgrep",
            "rev": "de4baa10024f2cb62d438596274b9b710e01c59b",
            "gcvs_variants": ["baseline", "gc", "arc"],
        }
    },
    # "binary_trees": {
    #     "binary_trees": {
    #         "path": "binary_trees",
    #         "gcvs_variants": ["baseline", "gc", "arc"],
    #     }
    # },
}


# --- Path Configuration ---
ALLOY_DIR = Path("alloy").resolve()
BIN_DIR = Path("bin").resolve()
BENCHMARK_BIN_DIR = BIN_DIR / "benchmarks"
BUILD_DIR = Path("build").resolve()
BENCHMARK_BUILD_DIR = BUILD_DIR / "benchmarks"
SRC_DIR = Path("src").resolve()
CONFIG_DIR = Path("configs")
BENCHMARKS_DIR = Path("benchmarks").resolve()


LIBGC_REPO = "https://github.com/softdevteam/bdwgc"
LIBGC_COMMIT = "e49b178f892d8e4b65785029c4fba3480850ce62"
LIBGC_DIR = Path("bdwgc").absolute()
LIBGC_BUILD_DIR = LIBGC_DIR / "build"
LIBGC_LIB_DIR = LIBGC_DIR / "lib"

BUILD_ENV = os.environ.copy()
BUILD_ENV["LD_LIBRARY_PATH"] = f"{LIBGC_LIB_DIR}"
BUILD_ENV["RUSTFLAGS"] = f"-L {LIBGC_LIB_DIR}"


@task
def build_libgc(c):
    """Clone and build BDWGC (libgc)"""
    if LIBGC_DIR.exists():
        print(f"Skipping BDWGC. ({LIBGC_LIB_DIR} exists)...")
        return

    print("\n=== Building BDWGC (libgc) ===")
    if not LIBGC_DIR.exists():
        print(f"Cloning BDWGC repository ({LIBGC_COMMIT})...")
        c.run(f"git clone {LIBGC_REPO} {LIBGC_DIR}")
    else:
        print("BDWGC repository already exists")

    with c.cd(str(LIBGC_DIR)):
        c.run(f"git fetch origin")
        c.run(f"git checkout {LIBGC_COMMIT}")
        c.run("git submodule update --init --recursive")
    if not LIBGC_BUILD_DIR.exists():
        LIBGC_BUILD_DIR.mkdir()
    with c.cd(str(LIBGC_BUILD_DIR)):
        print("Running cmake...")
        c.run(
            f'cmake -DCMAKE_INSTALL_PREFIX={LIBGC_DIR} -DCMAKE_BUILD_TYPE=Release DCMAKE_C_FLAGS="-DGC_ALWAYS_MULTITHREADED -DVALGRIND_TRACKING" ..'
        )
        print("Building libgc...")
        c.run("make -j$(nproc) install")
    print("BDWGC build complete")


@task
def clone(c):
    """Clone all required repositories"""
    print("=== Cloning repositories ===")

    # Clone Alloy
    if not ALLOY_DIR.exists():
        print(f"Cloning Alloy ({ALLOY_COMMIT})...")
        c.run(f"git clone {ALLOY_REPO} {ALLOY_DIR}")
    else:
        print("Alloy repository exists")

    # Checkout Alloy commit
    with c.cd(str(ALLOY_DIR)):
        c.run(f"git checkout {ALLOY_COMMIT}")

    # Clone benchmarks
    for bench_name, suites in BENCHMARKS.items():
        print(f"\n--- {bench_name} ---")
        repos = SRC_DIR / bench_name
        repos.mkdir(parents=True, exist_ok=True)
        for suite_name, suite_cfg in suites.items():
            print(suite_name)
            print(suite_cfg)
            repo_path = SRC_DIR / bench_name / suite_name
            if not repo_path.exists():
                print(f"Cloning {suite_cfg['url']}")
                c.run(f"git clone --recursive {suite_cfg['url']} {repo_path}")
            else:
                print(f"Exists: {suite_name}")

            print(repo_path)
            c.run(f"git -C {repo_path} checkout {suite_cfg['rev']}")


@task
def build_all(c):
    """Full build process: clone -> build Alloy -> build benchmarks"""
    clone(c)
    build_libgc(c)
    build_alloy(c)
    build_benchmarks(c)


@task
def build_alloy(c):
    """Build Alloy configurations for all experiments/metrics (DRY, with build_if_missing decorator)"""

    @build_if_missing("prefix")
    def build_single_config(c, config_file, prefix):
        cmd = [
            f"{ALLOY_DIR}/x",
            "install",
            "--config",
            str(config_file),
            "--stage",
            "1",
            "--set",
            "build.docs=false",
            "--set",
            f"install.prefix={prefix}",
            "--set",
            "install.sysconfdir=etc",
        ]
        c.run(" ".join(cmd), env=BUILD_ENV)

    # GCVS experiment
    for metric in EXPERIMENTS["gcvs"]["metrics"]:
        config_file = CONFIG_DIR / EXPERIMENTS["gcvs"]["config_template"].format(
            metric=metric
        )
        prefix = f"{BIN_DIR}/alloy/gcvs/{metric}/"
        build_single_config(c, config_file, prefix=prefix)

    # Premopt/elision experiments
    for exp in ["premopt", "elision"]:
        exp_cfg = EXPERIMENTS[exp]
        for config in exp_cfg["configs"]:
            for metric in exp_cfg["metrics"]:
                config_file = CONFIG_DIR / exp_cfg["config_template"].format(
                    config=config, metric=metric
                )
                prefix = f"{BIN_DIR}/alloy/{exp}/{config}/{metric}/"
                build_single_config(c, config_file, prefix=prefix)


def patch_repo(src_dir: Path, patch_file: Path) -> None:
    try:
        src_dir.relative_to(SRC_DIR)
    except ValueError as e:
        print(f"[ABORT] '{src_dir}' is not a subdirectory of {SRC_DIR}")
        sys.exit(1)

    if not patch_file.exists():
        tqdm.write(
            f"    [patching]: no patch file found for {src_dir.name}. skipping..."
        )
        return

    tqdm.write(f"    [patching]: {patch_file.name} -> {src_dir.name}")

    try:
        subprocess.run(
            ["git", "-C", str(src_dir), "reset", "--hard"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(src_dir), "apply", str(patch_file)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to apply patch: {e.stderr}")
        sys.exit(1)


def run_command(
    cmd,
    env,
    dry_run=False,
    verbose=False,
    progress=None,
    desc="Compiling",
    unit=" crates",
):
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        ) as proc, tqdm(total=progress, desc=desc, unit=unit, leave=True) as pbar:
            for line in proc.stdout:
                if verbose:
                    tqdm.write(line.rstrip())
                pbar.update(1)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception as e:
        print(f"\n[ERROR] {desc} failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cargo_build(
    prefix,
    src_dir,
    build_dir,
):
    return [
        "cargo",
        "build",
        "--release",
        f"--manifest-path={src_dir}/Cargo.toml",
        f"--target-dir={build_dir}",
    ]


@task
def build_benchmarks(c):
    """Build all benchmarks for all configurations"""

    def get_rustc_path(experiment, cfg, metric):
        rustc_path = BIN_DIR / "alloy" / experiment
        if experiment != "gcvs":
            rustc_path = rustc_path / cfg
        rustc_path = rustc_path / f"{metric}" / "bin" / "rustc"
        return rustc_path

    def get_patch_file(bench_name, suite_name, cfg, experiment):
        if experiment != "gcvs":
            return BENCHMARKS_DIR / bench_name / "patches" / f"{suite_name}.alloy.diff"
        else:
            if cfg != "baseline":
                return (
                    BENCHMARKS_DIR / bench_name / "patches" / f"{suite_name}.{cfg}.diff"
                )
            else:
                return None

    for bench_name, bench_cfg in BENCHMARKS.items():
        tqdm.write(f"\033[34m[building {bench_name} benchmarks]\033[0m")
        for suite_name, suite_cfg in bench_cfg.items():
            todo = len(EXPERIMENTS.items()) * len(BENCHMARKS.items())
            with tqdm(total=todo, colour="blue") as pbar:
                for experiment, experiment_cfg in EXPERIMENTS.items():
                    alloy_cfgs = (
                        suite_cfg.get("gcvs_configs", [])
                        if experiment == "gcvs"
                        else experiment_cfg["configs"]
                    )
                    for cfg in alloy_cfgs:
                        for metric in experiment_cfg["metrics"]:
                            prefix = (
                                BENCHMARK_BIN_DIR
                                / experiment
                                / bench_name
                                / suite_name
                                / cfg
                                / metric
                            )
                            build_dir = (
                                BENCHMARK_BUILD_DIR
                                / experiment
                                / bench_name
                                / suite_name
                                / cfg
                                / metric
                            )

                            if prefix.exists() and os.listdir(prefix):
                                tqdm.write(
                                    f"    [skipping]: 'bin/benchmarks/{experiment}/{bench_name}/{suite_name}/{metric}/' already exists"
                                )
                                pbar.update(1)
                                continue

                            rustc = get_rustc_path(experiment, cfg, metric)
                            patch_file = get_patch_file(
                                bench_name, suite_name, cfg, experiment
                            )
                            src_dir = SRC_DIR / bench_name / suite_name
                            if patch_file:
                                patch_repo(src_dir, patch_file)

                            env = deepcopy(BUILD_ENV)
                            env["RUSTC"] = str(rustc)
                            prefix.mkdir(parents=True, exist_ok=True)
                            build_dir.mkdir(parents=True, exist_ok=True)
                            # run_command(
                            #     cargo_build(prefix, src_dir, build_dir),
                            #     env,
                            #     desc=f"    [compiling]",
                            # )
                            for f in (build_dir / "release").glob("*"):
                                if (
                                    f.is_file()
                                    and not f.suffix in [".d", ".rlib"]
                                    and not f.name.startswith(".")
                                ):
                                    shutil.copy2(f, prefix / f.name)
                            pbar.update(1)


@task
def list_configs(c):
    """List all build configurations"""
    print("=== Alloy Configurations ===")
    for metric in EXPERIMENTS["gcvs"]["metrics"]:
        print(f"gcvs/{metric}")
    for exp in ["premopt", "elision"]:
        exp_cfg = EXPERIMENTS[exp]
        for config in exp_cfg["configs"]:
            for metric in exp_cfg["metrics"]:
                print(f"{exp}/{config}-{metric}")

    print("\n=== Benchmark Configurations ===")
    for bench_name, bench_cfg in BENCHMARKS.items():
        print(f"\n{bench_name}:")
        print("  GCVS variants:")
        for variant in bench_cfg["gcvs_variants"]:
            for metric in EXPERIMENTS["gcvs"]["metrics"]:
                print(f"    - {variant}/{metric}")
        print("  Experiments:")
        for exp in ["premopt", "elision"]:
            exp_cfg = EXPERIMENTS[exp]
            print(f"    {exp}: {', '.join(exp_cfg['configs'])}")


@task
def clean(c):
    """Clean all build artifacts"""
    print("=== Cleaning ===")
    shutil.rmtree(ALLOY_DIR / "bin", ignore_errors=True)
    shutil.rmtree(BIN_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print("Removed Alloy builds, binaries, and build directories")
