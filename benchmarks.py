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


ALACRITTY = (
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
)

FD = (
    Benchmark(
        name="no-pattern",
        extra_args=f"--hidden --no-ignore 'alloy'",
    ),
    Benchmark(name="simple-pattern", extra_args=f"'.*[0-9]\\.jpg$' . 'alloy'"),
    Benchmark(
        name="simple-pattern-HI",
        extra_args=f"-HI '.*[0-9]\\.jpg$' 'alloy'",
    ),
    Benchmark(
        name="file-extension",
        extra_args=f"-HI --extension jpg . 'alloy'",
    ),
    Benchmark(name="file-type", extra_args=f"-HI --type l . 'alloy'"),
    Benchmark(name="command-execution", extra_args=f"'ab' 'alloy' --exec echo"),
    Benchmark(
        name="command-execution-large-output",
        extra_args=f"-tf 'ab' 'alloy' --exec echo",
    ),
)

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

GRMTOOLS = (
    Benchmark("eclipse", str(artefacts.SRC_DIR / "eclipse")),
    Benchmark("hadoop", str(artefacts.SRC_DIR / "hadoop")),
    Benchmark("spring", str(artefacts.SRC_DIR / "spring")),
    Benchmark("jenkins", str(artefacts.SRC_DIR / "jenkins")),
)

RIPGREP = (
    Benchmark(
        "linux_alternates",
    ),
    Benchmark("linux_alternates_casei"),
    Benchmark("linux_literal"),
    Benchmark("linux_literal_casei"),
    Benchmark("linux_literal_casei_mmap"),
    Benchmark("linux_literal_default"),
    Benchmark("linux_literal_mmap"),
    Benchmark("linux_re_literal_suffix"),
    Benchmark("linux_unicode_greek"),
    Benchmark("linux_unicode_greek_casei"),
    Benchmark("linux_unicode_word_1"),
    Benchmark("linux_unicode_word_2"),
    Benchmark("linux_word"),
)


BINARY_TREES = (Benchmark("binary_trees", 14),)
REGEX_REDUX = (Benchmark("regex_redux", "0 < redux_input.txt"),)
