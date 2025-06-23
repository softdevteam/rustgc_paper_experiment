from dataclasses import dataclass
from typing import Any

import artefacts


@dataclass(frozen=True)
class Benchmark:
    name: str
    extra_args: Any = None

    def __repr__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


ALACRITTY = [
    Benchmark("cursor_motion"),
    Benchmark("dense_cells"),
    Benchmark("light_cells"),
    Benchmark("scrolling"),
    Benchmark("scrolling_bottom_region"),
    Benchmark("scrolling_bottom_small_region"),
    Benchmark("scrolling_fullscreen"),
    Benchmark("scrolling_top_region"),
    Benchmark("scrolling_top_small_region"),
    Benchmark("unicode"),
]

FD = [
    Benchmark(
        name="no-pattern", extra_args=f"--hidden --no-ignore '' '{artefacts.LINUX.src}'"
    ),
    Benchmark(
        name="simple-pattern", extra_args=f"'.*[0-9]\\.jpg$' '{artefacts.LINUX.src}'"
    ),
    Benchmark(
        name="simple-pattern-HI",
        extra_args=f"-HI '.*[0-9]\\.jpg$' '{artefacts.LINUX.src}'",
    ),
    Benchmark(
        name="file-extension", extra_args=f"-HI --extension jpg '{artefacts.LINUX.src}'"
    ),
    Benchmark(name="file-type", extra_args=f"-HI --type l '{artefacts.LINUX.src}'"),
    Benchmark(
        name="command-execution", extra_args=f"'ab' '{artefacts.LINUX.src}' --exec echo"
    ),
    Benchmark(
        name="command-execution-large-output",
        extra_args=f"-tf 'ab' '{artefacts.LINUX.src}' --exec echo",
    ),
]

SOM = (
    Benchmark("Richards", 1),
    Benchmark("DeltaBlue", 400),
    Benchmark("NBody", 1000),
    Benchmark("JsonSmall", 7),
    Benchmark("GraphSearch", 7),
    Benchmark("PageRank", 50),
    Benchmark("Fannkuch", 7),
    Benchmark("Fibonacci", "10"),
    Benchmark("Dispatch", 10),
    Benchmark("Bounce", "10"),
    Benchmark("Loop", 10),
    Benchmark("Permute", "10"),
    Benchmark("Queens", "10"),
    Benchmark("List", "5"),
    Benchmark("Recurse", "10"),
    Benchmark("Storage", 10),
    Benchmark("Sieve", 10),
    Benchmark("BubbleSort", "10"),
    Benchmark("QuickSort", 20),
    Benchmark("Sum", 10),
    Benchmark("Towers", "3"),
    Benchmark("TreeSort", "3"),
    Benchmark("IntegerLoop", 5),
    Benchmark("FieldLoop", 5),
    Benchmark("WhileLoop", 20),
    Benchmark("Mandelbrot", 50),
)

ALACRITTY_ARGS = f"""-e bash -c \"[ ! -f {artefacts.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} ] || {artefacts.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'setup'} && {artefacts.VTE_BENCH.src / 'benchmarks' / '%(benchmark)s' / 'benchmark'}\""""

FD_ARGS = ""


SOMRS_ARGS = f"""-c {artefacts.SOMRS_AST.src}/core-lib/Smalltalk {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/Richards {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/DeltaBlue {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/NBody {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/Json {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/GraphSearch {artefacts.SOMRS_AST.src}/core-lib/Examples/Benchmarks/LanguageFeatures -- BenchmarkHarness %(benchmark)s %(iterations)s"""

YKSOM_ARGS = f"""--cp {artefacts.YKSOM.src}/SOM/Smalltalk:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/Richards:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/DeltaBlue:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/NBody:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/Json:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/GraphSearch:{artefacts.YKSOM.src}/SOM/Examples/Benchmarks/LanguageFeatures {artefacts.YKSOM.src}/SOM/Examples/Benchmarks/BenchmarkHarness.som %(benchmark)s %(iterations)s"""
