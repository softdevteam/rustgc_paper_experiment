#! /usr/bin/env python

import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats

from build import Experiment

warnings.simplefilter(action="ignore", category=FutureWarning)


def bytes_formatter(max_value):
    units = [
        ("B", 1),
        ("KiB", 1024),
        ("MiB", 1024 * 1024),
        ("GiB", 1024 * 1024 * 1024),
    ]

    for unit, factor in reversed(units):
        if max_value >= factor:
            break

    def format_func(x, pos):
        return f"{x/factor:.2f}"

    return FuncFormatter(format_func), unit


# ============== PLOT FORMATTING =================

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # LaTeX and font settings
        "text.usetex": False,
        "svg.fonttype": "none",
        "text.latex.preamble": r"\usepackage{sfmath}",
        "font.family": "sans-serif",
        # Basic graph axes styling
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        # Grid and line settings
        "lines.linewidth": 1,
        "grid.linewidth": 0.25,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        # Tick settings
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        # Legend and figure settings
        # "legend.title_fontsize": 0,
        "errorbar.capsize": 1,
    }
)


class Criterea(Enum):
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
    WALLCLOCK = "total"
    SYS = "sys"

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class Results:
    experiment: Experiment
    data: pd.DataFrame

    @classmethod
    def from_raw_data(cls, experiment):
        # def to_executor(name):
        #     for cfg in self.experiment.configurations():
        #         if cfg.name == name:
        #             return cfg
        #     raise ValueError(f"Executor for {name} not found.")

        def to_benchmark(name):
            for b in experiment.suite.benchmarks:
                if b.name == name:
                    return b
            raise ValueError(f"Benchmark for {name} not found.")

        def to_criterion(name):
            for b in experiment.suite.benchmarks:
                if b.name == name:
                    return b
            raise ValueError(f"Benchmark for {name} not found.")

        raw = pd.read_csv(
            experiment.results,
            sep="\t",
            comment="#",
            index_col="suite",
            converters={"criterion": Criterea, "benchmark": to_benchmark},
        )
        raw = raw.rename(columns={"executor": "configuration"}).reset_index()[
            ["benchmark", "configuration", "value", "criterion"]
        ]
        return cls(experiment, raw)

    def summary(self):
        return Summary(self)

    def geometric_mean(self) -> "Results":
        def with_99_cis(series):
            clean_vals = series.dropna()
            n = len(clean_vals)

            if n == 0 or (clean_vals <= 0).any():
                return pd.Series([0] * 3, index=["value", "lower", "upper"])

            log_vals = np.log(clean_vals)
            mean_log = np.mean(log_vals)
            std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

            if n == 1:
                return pd.Series(
                    [np.exp(mean_log), np.nan, np.nan],
                    index=["value", "lower", "upper"],
                )

            sem_log = std_log / np.sqrt(n)
            t_crit = stats.t.ppf((1 + 0.99) / 2, df=n - 1)

            ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

            # Convert back to original scale
            return pd.Series(
                [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
                index=["value", "lower", "upper"],
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)

    def arithmetic_mean(self) -> "Results":
        def with_99_cis(series):
            n = len(series)
            mean = series.mean()
            std_err = series.std(ddof=1) / (n**0.5)  # Standard error
            margin_of_error = (
                stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err
            )  # t-score * SE
            return pd.Series(
                {
                    "mean": mean,
                    "ci": margin_of_error,
                    "lower": mean - margin_of_error,
                    "upper": mean + margin_of_error,
                }
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "benchmark", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)


def plot_line(
    df,
    metrics=None,
    x_column="collection number",
    figsize=(12, 8),
    title=None,
    xlabel="Collection Number",
    ylabel="Value",
):
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    df_pivoted = df.pivot(
        index="collection number", columns="criterion", values="value"
    )

    formatter, unit = bytes_formatter(np.max(df["value"].max()))

    # # Create bins of 10 collections and take the mean
    # df_pivoted = df_pivoted.groupby(df_pivoted.index // 10 * 10).mean()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    mem = ["allocated_kib", "freed_kib"]

    criteria_data = [df_pivoted[col] for col in df_pivoted.columns if col in mem]

    plt.stackplot(
        df_pivoted.index, *criteria_data, labels=df_pivoted.columns, alpha=0.8
    )
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))

    plt.xlabel("Collection Number")
    plt.ylabel("Time (ms)")
    plt.title("GC Time Composition by Collection")
    plt.legend(loc="upper left")
    plt.savefig("test.svg", format="svg", bbox_inches="tight")


class CollectionStats(Results):
    @classmethod
    def from_raw_data(cls, experiment):
        with open(Path("gc.log"), "r") as f:
            data = []
            blocks = re.split(r"--> Marking for collection", f.read())

            for block in blocks:
                if not block.strip() or "#" not in block:
                    continue

                row = {}

                if match := re.search(r"#(\d+) after (\d+) allocated bytes", block):
                    row["collection"] = int(match.group(1))
                    row["allocated_kib"] = int(match.group(2)) * 1024

                if match := re.search(
                    r"World-stopped marking took (\d+) ms (\d+) ns", block
                ):
                    ms = int(match.group(1))
                    ns = int(match.group(2))
                    row["marking_time_ms"] = ms + (ns / 1_000_000)

                if match := re.search(
                    r"freed (-?\d+) bytes, heap (\d+) KiB \(\+ (\d+) KiB unmapped \+ (\d+) KiB internal\)",
                    block,
                ):
                    row["freed_kib"] = int(match.group(1)) * 1024
                    row["heap_kib"] = int(match.group(2))
                    row["unmapped_kib"] = int(match.group(3))
                    row["internal_kib"] = int(match.group(4))

                if match := re.search(
                    r"In-use heap: (\d+)% \((\d+) KiB pointers \+ (\d+) KiB other\)",
                    block,
                ):
                    row["heap_usage_percent"] = int(match.group(1))

                if match := re.search(r"(\d+) finalization entries", block):
                    row["fin_entry"] = int(match.group(1))

                if match := re.search(r"(\d+) finalization-ready objects", block):
                    row["moved_to_finq"] = int(match.group(1))

                if match := re.search(
                    r"Finalize and initiate sweep took (\d+) ms (\d+) ns \+ (\d+) ms (\d+) ns",
                    block,
                ):
                    ms1 = int(match.group(1))
                    ns1 = int(match.group(2))
                    ms2 = int(match.group(3))
                    ns2 = int(match.group(4))
                    row["finalize_time"] = ms1 + (ns1 / 1_000_000)
                    row["sweep_initiate_time"] = ms2 + (ns2 / 1_000_000)
                    row["total_finalize_sweep"] = (
                        row["finalize_time"] + row["sweep_initiate_time"]
                    )

                if match := re.search(
                    r"Complete collection took (\d+) ms (\d+) ns", block
                ):
                    ms = int(match.group(1))
                    ns = int(match.group(2))
                    row["total_collection_time"] = ms + (ns / 1_000_000)

                if match := re.search(r"Grew fo table to (\d+) entries", block):
                    row["fo_table_size"] = int(match.group(1))

                row["is_full_collection"] = "full world-stop" in block

                if row:
                    data.append(row)

            df = pd.DataFrame(data)

            # Now make this the same layout as the existing perf/mem dataframes
            # so we can re-use the same stats code on this.
            long_rows = []
            for _, row in df.iterrows():
                collection_num = row["collection"]
                is_full = row["is_full_collection"]

                for col in df.columns:
                    if col in ["collection", "is_full_collection"]:
                        continue

                    value = row[col]
                    if pd.isna(value):
                        continue

                    if "bytes" in col:
                        unit = "KiB"
                    elif "kib" in col:
                        unit = "KiB"
                    elif "time" in col or col.endswith("_ms"):
                        unit = "ms"
                    elif "percent" in col:
                        unit = "%"
                    elif any(
                        word in col for word in ["entries", "objects", "links", "size"]
                    ):
                        unit = "count"
                    else:
                        unit = ""

                    long_rows.append(
                        {
                            "collection number": collection_num,
                            "criterion": col,
                            "value": value,
                            "unit": unit,
                            "is_full_collection": is_full,
                        }
                    )
            return cls(data=pd.DataFrame(long_rows))

    @property
    def raw(self):
        if self._raw is None:
            self._parse_raw_data()
        return self._raw

    @property
    def full_collections(self):
        return self.raw.loc[self.raw["is_full_collection"]].drop(
            "is_full_collection", axis=1
        )

    @property
    def time_measurements(self):
        df = self.full_collections
        return df[df["unit"] == "ms"].drop("unit", axis=1)

    @property
    def mem_measurements(self):
        df = self.full_collections
        return df[df["unit"] != "ms"].drop("unit", axis=1)

    def plot_mem_measurements(self):
        self.mem_measurements
        plot_line(self.mem_measurements)

    def plot_time_measurements(self):
        self.time_measurements.set_index("criterion")
        plot_line(self.time_measurements)


class Summary(Results):
    data: pd.DataFrame

    def __init__(self, results: Results, base="static"):
        self.raw_data = results.data
        self.means = results.arithmetic_mean().data
        self.gmeans = results.geometric_mean().data
        self.config_map = self._create_config_map(results)

        # Build the summary data
        best_worst = self._calculate_best_worst()
        gmean_ratios = self._calculate_gmean_ratios()

        # Final merge
        self.data = self.gmeans.merge(
            best_worst, on=["configuration", "criterion"], how="outer"
        ).merge(gmean_ratios, on=["configuration", "criterion"], how="outer")
        self.data["suite"] = results.experiment.suite.name
        self.data["experiment"] = results.experiment.name

    def _create_config_map(self, results):
        return pd.DataFrame(
            [
                {"configuration": cfg.name, "baseline": cfg.baseline.name}
                for cfg in results.experiment.configurations()
            ]
        )

    def _bootstrap_ci(self, config_vals, baseline_vals, stat_func):
        res = stats.bootstrap(
            (config_vals, baseline_vals),
            statistic=stat_func,
            n_resamples=1000,
            confidence_level=0.99,
            method="percentile",
            vectorized=True,
        )
        return res.confidence_interval.low, res.confidence_interval.high

    def _prepare_comparison_data(self):
        means_with_baseline = self.means.merge(self.config_map, on="configuration")
        baseline_data = self.means.rename(
            columns={
                "configuration": "baseline",
                "mean": "mean_b",
                "lower": "lower_b",
                "upper": "upper_b",
            }
        )

        m = means_with_baseline.merge(
            baseline_data, on=["benchmark", "criterion", "baseline"]
        )
        m["pct"] = (m["mean"] - m["mean_b"]) / m["mean_b"] * 100
        m["ratio"] = m["mean"] / m["mean_b"]
        m["significant"] = ~(
            (m["lower"] <= m["upper_b"]) & (m["upper"] >= m["lower_b"])
        )

        return m

    def _bootstrap_benchmark_cis(
        self, config_name, baseline_name, criterion, benchmark
    ):
        config_data = self.raw_data[
            (self.raw_data["configuration"] == config_name)
            & (self.raw_data["criterion"] == criterion)
            & (self.raw_data["benchmark"] == benchmark)
        ]
        baseline_data = self.raw_data[
            (self.raw_data["configuration"] == baseline_name)
            & (self.raw_data["criterion"] == criterion)
            & (self.raw_data["benchmark"] == benchmark)
        ]

        config_vals = config_data["value"].values
        baseline_vals = baseline_data["value"].values

        pct_low, pct_high = self._bootstrap_ci(
            config_vals,
            baseline_vals,
            lambda c, b, axis=None: (np.mean(c, axis=axis) - np.mean(b, axis=axis))
            / np.mean(b, axis=axis)
            * 100,
        )
        ratio_low, ratio_high = self._bootstrap_ci(
            config_vals,
            baseline_vals,
            lambda c, b, axis=None: np.mean(c, axis=axis) / np.mean(b, axis=axis),
        )

        return pct_low, pct_high, ratio_low, ratio_high

    def _add_bootstrap_cis(self, sig):
        ci_data = []
        for _, row in sig.iterrows():
            pct_low, pct_high, ratio_low, ratio_high = self._bootstrap_benchmark_cis(
                row["configuration"],
                row["baseline"],
                row["criterion"],
                row["benchmark"],
            )

            ci_data.append(
                {
                    "pct_ci_low": pct_low,
                    "pct_ci_high": pct_high,
                    "ratio_ci_low": ratio_low,
                    "ratio_ci_high": ratio_high,
                }
            )

        return pd.concat([sig.reset_index(drop=True), pd.DataFrame(ci_data)], axis=1)

    def _extremes(self, group):
        if len(group) == 0:
            return pd.Series(
                {
                    "worst": "None",
                    "best": "None",
                    "worst_pct": 0,
                    "worst_ratio": 1.0,
                    "worst_pct_ci_low": np.nan,
                    "worst_pct_ci_high": np.nan,
                    "worst_ratio_ci_low": np.nan,
                    "worst_ratio_ci_high": np.nan,
                    "best_pct": 0,
                    "best_ratio": 1.0,
                    "best_pct_ci_low": np.nan,
                    "best_pct_ci_high": np.nan,
                    "best_ratio_ci_low": np.nan,
                    "best_ratio_ci_high": np.nan,
                }
            )

        worst_row = group.loc[group["pct"].idxmax()]
        best_row = group.loc[group["pct"].idxmin()]

        return pd.Series(
            {
                "worst": worst_row["benchmark"],
                "worst_pct": worst_row["pct"],
                "worst_ratio": worst_row["ratio"],
                "worst_pct_ci_low": worst_row["pct_ci_low"],
                "worst_pct_ci_high": worst_row["pct_ci_high"],
                "worst_ratio_ci_low": worst_row["ratio_ci_low"],
                "worst_ratio_ci_high": worst_row["ratio_ci_high"],
                "best": best_row["benchmark"],
                "best_pct": -best_row["pct"],
                "best_ratio": best_row["ratio"],
                "best_pct_ci_low": best_row["pct_ci_low"],
                "best_pct_ci_high": best_row["pct_ci_high"],
                "best_ratio_ci_low": best_row["ratio_ci_low"],
                "best_ratio_ci_high": best_row["ratio_ci_high"],
            }
        )

    def _calculate_best_worst(self):
        m = self._prepare_comparison_data()
        sig = m[m["significant"]].copy()

        if not sig.empty:
            sig = self._add_bootstrap_cis(sig)

        return (
            sig.groupby(["criterion", "configuration"])
            .apply(self._extremes)
            .reset_index()
            .round(3)
        )

    def _calculate_gmean_ratios(self):
        gmean_with_baseline = self.gmeans.merge(self.config_map, on="configuration")
        gmean_baseline = self.gmeans.rename(
            columns={"configuration": "baseline", "value": "value_b"}
        )
        gmean_merged = gmean_with_baseline.merge(
            gmean_baseline, on=["criterion", "baseline"]
        )

        gmean_merged["gmean_ratio"] = gmean_merged["value"] / gmean_merged["value_b"]
        gmean_merged["gmean_pct"] = (
            (gmean_merged["value"] - gmean_merged["value_b"])
            / gmean_merged["value_b"]
            * 100
        )

        gmean_cis = []
        for _, row in gmean_merged.iterrows():
            config_data = self.raw_data[
                (self.raw_data["configuration"] == row["configuration"])
                & (self.raw_data["criterion"] == row["criterion"])
            ]
            baseline_data = self.raw_data[
                (self.raw_data["configuration"] == row["baseline"])
                & (self.raw_data["criterion"] == row["criterion"])
            ]

            config_vals = config_data["value"].values
            baseline_vals = baseline_data["value"].values

            pct_low, pct_high = self._bootstrap_ci(
                config_vals,
                baseline_vals,
                lambda c, b, axis=None: (
                    stats.gmean(c, axis=axis) - stats.gmean(b, axis=axis)
                )
                / stats.gmean(b, axis=axis)
                * 100,
            )
            ratio_low, ratio_high = self._bootstrap_ci(
                config_vals,
                baseline_vals,
                lambda c, b, axis=None: stats.gmean(c, axis=axis)
                / stats.gmean(b, axis=axis),
            )
            gmean_cis.append(
                {
                    "criterion": row["criterion"],
                    "configuration": row["configuration"],
                    "gmean_ratio": row["gmean_ratio"],
                    "gmean_pct": row["gmean_pct"],
                    "gmean_ratio_ci_low": ratio_low,
                    "gmean_ratio_ci_high": ratio_high,
                    "gmean_pct_ci_low": pct_low,
                    "gmean_pct_ci_high": pct_high,
                    "gmean_significant": not (ratio_low <= 1.0 <= ratio_high),
                }
            )

        return pd.DataFrame(gmean_cis).round(3)

    def without_errs(self):
        error_cols = [
            "worst_pct_ci_low",
            "worst_pct_ci_high",
            "worst_ratio_ci_low",
            "worst_ratio_ci_high",
            "best_pct_ci_low",
            "best_pct_ci_high",
            "best_ratio_ci_low",
            "best_ratio_ci_high",
            "gmean_pct_ci_low",
            "gmean_pct_ci_high",
            "gmean_ratio_ci_high",
            "gmean_ratio_ci_low",
        ]

        df = self.data.copy()
        df = df.drop(error_cols, axis=1)
        summary = object.__new__(Summary)
        summary.data = df
        return summary

    def __repr__(self):
        return repr(self.data)


@dataclass
class SuiteData:
    suite: BenchmarkSuite
    data: pd.DataFrame

    @classmethod
    def from_raw_data(cls, suite, measurement):
        # def to_executor(name):
        #     for cfg in self.experiment.configurations():
        #         if cfg.name == name:
        #             return cfg
        #     raise ValueError(f"Executor for {name} not found.")

        def to_benchmark(name):
            for b in suite.benchmarks:
                if b.name == name:
                    return b
            raise ValueError(f"Benchmark for {name} not found.")

        raw = pd.read_csv(
            suite.raw_data(measurement),
            sep="\t",
            comment="#",
            index_col="suite",
            converters={"criterion": Criterea, "benchmark": to_benchmark},
        )
        raw = raw.rename(columns={"executor": "configuration"}).reset_index()[
            ["benchmark", "configuration", "value", "criterion"]
        ]
        return cls(suite, raw)

    @classmethod
    def for_measurements(cls, suite, measurements):
        dfs = []
        for m in measurements:
            print(suite.raw_data(m))
            if not suite.raw_data(m).exists():
                continue
            df = cls.from_raw_data(suite, m).data
            dfs.append(df)

        if not dfs:
            return None
        return SuiteData(suite, pd.concat(dfs, ignore_index=True))

    def summary(self):
        return Summary(self)

    def for_experiment(self, experiment) -> "ExperimentData":
        df = self.data.copy()
        df = df[
            df["configuration"].isin([cfg.name for cfg in experiment.configurations()])
        ]
        return ExperimentData(experiment, df)

    def geometric_mean(self) -> "Results":
        def with_99_cis(series):
            clean_vals = series.dropna()
            n = len(clean_vals)

            if n == 0 or (clean_vals <= 0).any():
                return pd.Series([0] * 3, index=["value", "lower", "upper"])

            log_vals = np.log(clean_vals)
            mean_log = np.mean(log_vals)
            std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

            if n == 1:
                return pd.Series(
                    [np.exp(mean_log), np.nan, np.nan],
                    index=["value", "lower", "upper"],
                )

            sem_log = std_log / np.sqrt(n)
            t_crit = stats.t.ppf((1 + 0.99) / 2, df=n - 1)

            ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

            # Convert back to original scale
            return pd.Series(
                [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
                index=["value", "lower", "upper"],
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)

    def arithmetic_mean(self) -> "Results":
        def with_99_cis(series):
            n = len(series)
            mean = series.mean()
            std_err = series.std(ddof=1) / (n**0.5)  # Standard error
            margin_of_error = (
                stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err
            )  # t-score * SE
            return pd.Series(
                {
                    "mean": mean,
                    "ci": margin_of_error,
                    "lower": mean - margin_of_error,
                    "upper": mean + margin_of_error,
                }
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "benchmark", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)


@dataclass
class ExperimentData:
    experiment: Experiment
    data: pd.DataFrame

    def summary(self):
        return Summary(self)

    def geometric_mean(self) -> "Results":
        def with_99_cis(series):
            clean_vals = series.dropna()
            n = len(clean_vals)

            if n == 0 or (clean_vals <= 0).any():
                return pd.Series([0] * 3, index=["value", "lower", "upper"])

            log_vals = np.log(clean_vals)
            mean_log = np.mean(log_vals)
            std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

            if n == 1:
                return pd.Series(
                    [np.exp(mean_log), np.nan, np.nan],
                    index=["value", "lower", "upper"],
                )

            sem_log = std_log / np.sqrt(n)
            t_crit = stats.t.ppf((1 + 0.99) / 2, df=n - 1)

            ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

            # Convert back to original scale
            return pd.Series(
                [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
                index=["value", "lower", "upper"],
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)

    def arithmetic_mean(self) -> "Results":
        def with_99_cis(series):
            n = len(series)
            mean = series.mean()
            std_err = series.std(ddof=1) / (n**0.5)  # Standard error
            margin_of_error = (
                stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err
            )  # t-score * SE
            return pd.Series(
                {
                    "mean": mean,
                    "ci": margin_of_error,
                    "lower": mean - margin_of_error,
                    "upper": mean + margin_of_error,
                }
            )

        data = (
            self.data.copy()
            .groupby(["configuration", "benchmark", "criterion"])["value"]
            .apply(with_99_cis)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)


@dataclass
class Overall:
    data: pd.DataFrame

    def __init__(self, dfs):
        self.data = pd.concat(dfs, ignore_index=True)

    def mk_perf_table(self):
        df = self.data[self.data["experiment"].str.endswith("perf", na=False)].copy()
        baseline = "none"
        all_configs = df["configuration"].unique()
        baseline_configs = sorted([c for c in all_configs if baseline in c])
        other_configs = sorted([c for c in all_configs if baseline not in c])
        config_order = baseline_configs + other_configs

        # Convert to categorical with custom order
        df["configuration"] = pd.Categorical(
            df["configuration"], categories=config_order, ordered=True
        )

        df = df.sort_values(["suite", "configuration"])
        self._mk_table(df.copy(), [Criterea.WALLCLOCK, Criterea.USER])

    def _mk_table(self, df, criteria_list):
        def fmt_float(x, digits=2, bold=False):
            s = "-" if x is None else f"{x:.{digits}f}"
            return f"\\textbf{{{s}}}" if bold else s

        def fmt_ci(low, high):
            if low is None or high is None:
                return "-"
            return f"\\scriptsize\\textcolor{{gray!80}}{{[{fmt_float(low)}, {fmt_float(high)}]}}"

        # Generate table columns dynamically
        col_spec = "l l" + " r@{\hspace{0.5em}}l" * len(criteria_list)
        header = "Suite & Configuration"
        for crit in criteria_list:
            header += f" & \\multicolumn{{2}}{{c}}{{{crit.name}}}"

        lines = [
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]

        for s, suites in df.groupby("suite", sort=False):
            # Find best config for each criterion
            best_configs = {}
            for crit in criteria_list:
                crit_df = suites[suites["criterion"] == crit]
                best_configs[crit] = crit_df.loc[crit_df["value"].idxmin(), "configuration"]

            cfgs = suites["configuration"].drop_duplicates().to_list()

            for i, c in enumerate(cfgs):
                config_rows = suites[suites["configuration"] == c]
                row_data = [s if i == 0 else "", c]

                for crit in criteria_list:
                    row = config_rows[config_rows["criterion"] == crit].iloc[0]

                    if i == 0:  # Baseline
                        val = fmt_float(row["value"], bold=(c == best_configs[crit]))
                        ci = fmt_ci(row["lower"], row["upper"])
                    else:  # Other rows
                        val = fmt_float(row["gmean_ratio"], bold=(c == best_configs[crit])) + "×"
                        if not row["gmean_significant"]:
                            val += r"\textsuperscript{\dag}"
                        ci = fmt_ci(row["gmean_ratio_ci_low"], row["gmean_ratio_ci_high"])

                    row_data.extend([val, ci])

                lines.append(" & ".join(row_data) + r" \\")

            lines.append(r"\midrule")

        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")

        with open("plots/suites.tex", "w") as f:
            f.write("\n".join(lines))

