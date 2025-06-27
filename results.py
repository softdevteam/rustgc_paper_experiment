#! /usr/bin/env python

import glob
import os
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

from build import BenchmarkSuite, Experiment

PLOT_DIR = Path("plots")

warnings.simplefilter(action="ignore", category=FutureWarning)


def arithmean(series):
    n = len(series)
    mean = series.mean()
    std_err = series.std(ddof=1) / (n**0.5)  # Standard error
    margin_of_error = stats.t.ppf((1 + 0.99) / 2, df=n - 1) * std_err  # t-score * SE
    return pd.Series(
        {
            "arithmean": mean,
            "arithmean_ci": margin_of_error,
            "arithmean_lower": mean - margin_of_error,
            "arithmean_upper": mean + margin_of_error,
        }
    )


def geomean(series):
    clean_vals = series.dropna()
    n = len(clean_vals)

    if n == 0 or (clean_vals <= 0).any():
        return pd.Series([0] * 3, index=["gmean", "gmean_lower", "gmean_upper"])

        log_vals = np.log(clean_vals)
        mean_log = np.mean(log_vals)
        std_log = np.std(log_vals, ddof=1)  # Sample standard deviation

        if n == 1:
            return pd.Series(
                [np.exp(mean_log), np.nan, np.nan],
                index=["gmean", "gmean_lower", "gmean_upper"],
            )

        sem_log = std_log / np.sqrt(n)
        t_crit = stats.t.ppf((1 + 0.99) / 2, df=n - 1)

        ci_log = (mean_log - t_crit * sem_log, mean_log + t_crit * sem_log)

        # Convert back to original scale
        return pd.Series(
            [np.exp(mean_log), np.exp(ci_log[0]), np.exp(ci_log[1])],
            index=["gmean", "gmean_lower", "gmean_upper"],
        )


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


class SummarizeWith(Enum):
    GMEAN = "gmean"
    ARITHMEAN = "arithmean"

    @property
    def lower(self):
        return f"{self.value}_lower"

    @property
    def upper(self):
        return f"{self.value}_upper"


class Criterea(Enum):
    def __init__(self, value, mean_kind=SummarizeWith.GMEAN, latex=None, desc=None):
        self._value_ = value
        self.mean_kind = mean_kind
        self.latex = latex
        self.desc = desc

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None

    @property
    def lower(self):
        return f"{self.mean_kind.value}_lower"

    @property
    def upper(self):
        return f"{self.mean_kind.value}_upper"

    @property
    def mean(self):
        return f"{self.mean_kind.value}"

    @property
    def ratio(self):
        return f"{self.mean_kind.value}_ratio"

    @property
    def ratio_lower(self):
        return f"{self.mean_kind.value}_ratio_ci_high"

    @property
    def ratio_upper(self):
        return f"{self.mean_kind.value}_ratio_ci_low"

    @property
    def is_significant(self):
        return f"{self.mean_kind.value}_significant"

    COLLECTION_NUMBER = "collection_number"
    TOTAL_COLLECTIONS = "total_collections"
    PHASE = "kind"
    HEAP_SIZE_ON_ENTRY = "heap_size_on_entry"
    TIME_MARKING = "time_marking"
    BYTES_FREED = "bytes_freed"
    HEAP_GREW = "heap_grew"
    LIVE_OBJECTS_WITH_FINALIZERS = "live_objects_with_finalizers"
    OBJECTS_IN_FINALIZER_QUEUE = "objects_in_finalizer_queue"
    TIME_FINALIZER_QUEUE = "time_fin_q"
    TIME_SWEEPING = "time_sweeping"
    TIME_TOTAL = "time_total"
    FINALIZERS_RUN = "finalizers_run"
    FINALIZERS_REGISTERED = "finalizers_registered"
    ALLOCATED_GC = "allocated_gc"
    ALLOCATED_ARC = "allocated_arc"
    ALLOCATED_RC = "allocated_rc"
    ALLOCATED_BOXED = "allocated_boxed"

    TOTAL_FINALIZERS_RUN = "total_finalizers_run"
    TOTAL_FINALIZERS_REGISTERED = "total_finalizers_registered"
    TOTAL_ALLOCATED_GC = "total_allocated_gc"
    TOTAL_ALLOCATED_ARC = "total_allocated_arc"
    TOTAL_ALLOCATED_RC = "total_allocated_rc"
    TOTAL_ALLOCATED_BOXED = "total_allocated_boxed"
    TOTAL_LIVE_OBJECTS_WITH_FINALIZERS = "total_live_objects_with_finalizers"
    TOTAL_OBJECTS_IN_FINALIZER_QUEUE = "total_objects_in_finalizer_queue"

    WALLCLOCK = (
        "total",
        SummarizeWith.GMEAN,
        "Wall-clock Time",
        "Wall-clock time (ms). Lower is better.",
    )
    USER = (
        "usr",
        SummarizeWith.GMEAN,
        "User Time",
        "User time (ms). Lower is better.",
    )

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class Results:
    data: pd.DataFrame
    _ameans: pd.DataFrame = None
    _gmeans: pd.DataFrame = None

    def __repr__(self):
        return self.data.__repr__()

    @classmethod
    def concat(cls, results):
        data = [r.data for r in results]
        res = cls.__new__(cls)
        res.data = pd.concat(data, ignore_index=True)
        return res

    @property
    def ameans(self):
        if not self._ameans:
            self._ameans = self._arithmetic_mean(per_benchmark=True)

        return self._ameans

    @property
    def gmeans(self):
        if self._gmeans:
            return self._gmeans

        self._gmeans = self._geometric_mean(per_benchmark=True)

    def __init__(self, data):
        self.data = data

    def _arithmetic_mean(self, per_benchmark=None) -> "Results":
        if per_benchmark is None:
            raise ValueError("arg per_benchmark required")
        cols = ["configuration", "criterion"]
        if per_benchmark:
            cols += ["benchmark"]
        return (
            self.data.copy()
            .groupby(cols)["value"]
            .apply(arithmean)
            .unstack()
            .reset_index()
        )

    def _geometric_mean(self, per_benchmark=None) -> "ExperimentData":
        if per_benchmark is None:
            raise ValueError("arg per_benchmark required")
        cols = ["configuration", "criterion"]
        if per_benchmark:
            cols + ["benchmark"]

        return (
            self.data.copy()
            .groupby(cols)["value"]
            .apply(geomean)
            .unstack()
            .reset_index()
        )

    def summary(self):
        return Summary(self)

    def geometric_mean(self, per_benchmark=False) -> "ExperimentData":
        cols = ["configuration", "criterion"]
        if per_benchmark:
            cols + ["benchmark"]

        data = (
            self.data.copy()
            .groupby(cols)["value"]
            .apply(geomean)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)

    def arithmetic_mean(self, per_benchmark=True) -> "Results":
        cols = ["configuration", "criterion"]
        if per_benchmark:
            cols += ["benchmark"]
        data = (
            self.data.copy()
            .groupby(cols)["value"]
            .apply(arithmean)
            .unstack()
            .reset_index()
        )
        return Results(experiment=self.experiment, data=data)

    def plot_bar(self, index, x=None, y=None, opts=None, mean=SummarizeWith.ARITHMEAN):
        # Get the single metric name
        metric = y
        if index == "benchmark":
            suite = self.data["suite"].iloc[0].name
            filename = f"{suite}-{metric.latex.lower().replace(' ', '-')}.svg"
        else:
            filename = f"summary-{metric.latex.lower().replace(' ', '-')}.svg"

        os.makedirs(self.outdir, exist_ok=True)
        df = self.ameans.copy()
        df = df[df["criterion"] == metric]

        pivot_df = df.pivot(
            index=index, columns=x, values=[mean.value, mean.lower, mean.upper]
        ).sort_index(ascending=False)

        mean_df = pivot_df[mean.value]
        lower_df = pivot_df[mean.lower]
        upper_df = pivot_df[mean.upper]

        lower_err = mean_df - lower_df
        upper_err = upper_df - mean_df

        fig, ax = plt.subplots(figsize=(max(12, mean_df.shape[0] * 0.4), 5))

        indices = np.arange(len(mean_df.index))
        bar_width = 0.25
        col_names = mean_df.columns
        ax.margins(x=0.01)

        for i, col in enumerate(col_names):
            ax.bar(
                indices + i * bar_width,
                mean_df[col].values,
                bar_width,
                yerr=[lower_err[col].values, upper_err[col].values],
                capsize=4,
                label=col.latex,
                error_kw={"elinewidth": 1, "capthick": 1},
            )

        ax.set_ylabel(metric.desc)
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels(mean_df.index, rotation=90, ha="center", fontsize=8)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.outdir / filename, bbox_inches="tight")
        plt.close()


class PremOptResults(Results):

    @property
    def outdir(self):
        return PLOT_DIR / "premopt"


class GcvsResults(Results):

    @property
    def outdir(self):
        return PLOT_DIR / "gcvs"


class ElisionResults(Results):

    @property
    def outdir(self):
        return PLOT_DIR / "elision"


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
        self.ameans = results.arithmetic_mean().data
        self.gmeans = results.geometric_mean().data
        self.config_map = self._create_config_map(results)

        # Build the summary data
        print(self.ameans)
        best_worst = self._calculate_best_worst()
        # gmean_ratios = self._calculate_gmean_ratios()

        ratios = self._calculate_mean_ratios()
        # Final merge
        self.data = self.gmeans.merge(
            best_worst, on=["configuration", "criterion"], how="outer"
        ).merge(ratios, on=["configuration", "criterion"], how="outer")
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
        means_with_baseline = self.ameans.merge(self.config_map, on="configuration")
        baseline_data = self.ameans.rename(
            columns={
                "configuration": "baseline",
                "arithmean": "arithmean_b",
                "arithmean_lower": "arithmean_lower_b",
                "arithmean_upper": "arithmean_upper_b",
            }
        )

        m = means_with_baseline.merge(
            baseline_data, on=["benchmark", "criterion", "baseline"]
        )
        m["ratio"] = m["arithmean"] / m["arithmean_b"]
        m["significant"] = ~(
            (m["arithmean_lower"] <= m["arithmean_upper_b"])
            & (m["arithmean_upper"] >= m["arithmean_lower_b"])
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

        ratio_low, ratio_high = self._bootstrap_ci(
            config_vals,
            baseline_vals,
            lambda c, b, axis=None: np.mean(c, axis=axis) / np.mean(b, axis=axis),
        )

        return ratio_low, ratio_high

    def _add_bootstrap_cis(self, sig):
        ci_data = []
        for _, row in sig.iterrows():
            ratio_low, ratio_high = self._bootstrap_benchmark_cis(
                row["configuration"],
                row["baseline"],
                row["criterion"],
                row["benchmark"],
            )

            ci_data.append(
                {
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
                    "worst_ratio": 1.0,
                    "worst_ratio_ci_low": np.nan,
                    "worst_ratio_ci_high": np.nan,
                    "best_ratio": 1.0,
                    "best_ratio_ci_low": np.nan,
                    "best_ratio_ci_high": np.nan,
                }
            )

        worst_row = group.loc[group["ratio"].idxmax()]
        best_row = group.loc[group["ratio"].idxmin()]

        return pd.Series(
            {
                "worst": worst_row["benchmark"],
                "worst_ratio": worst_row["ratio"],
                "worst_ratio_ci_low": worst_row["ratio_ci_low"],
                "worst_ratio_ci_high": worst_row["ratio_ci_high"],
                "best": best_row["benchmark"],
                "best_ratio": best_row["ratio"],
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

    def _calculate_mean_ratios(self):
        def process_mean(
            means_df,
            mean_col,
            mean_func,
            ratio_col,
            ci_ratio_col_low,
            ci_ratio_col_high,
            sig_col,
        ):
            with_baseline = means_df.merge(self.config_map, on="configuration")
            baseline = means_df.rename(
                columns={"configuration": "baseline", mean_col: f"{mean_col}_b"}
            )
            merged = with_baseline.merge(baseline, on=["criterion", "baseline"])
            merged[ratio_col] = merged[mean_col] / merged[f"{mean_col}_b"]

            cis = []
            for _, row in merged.iterrows():
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

                ratio_low, ratio_high = self._bootstrap_ci(
                    config_vals,
                    baseline_vals,
                    lambda c, b, axis=None: mean_func(c, axis=axis)
                    / mean_func(b, axis=axis),
                )
                cis.append(
                    {
                        "criterion": row["criterion"],
                        "configuration": row["configuration"],
                        ratio_col: row[ratio_col],
                        ci_ratio_col_low: ratio_low,
                        ci_ratio_col_high: ratio_high,
                        sig_col: not (ratio_low <= 1.0 <= ratio_high),
                    }
                )
            return pd.DataFrame(cis)

        # Process geometric mean
        print(self.gmeans)
        gmeans = self.gmeans[
            self.gmeans["criterion"].apply(lambda x: x.mean_kind == SummarizeWith.GMEAN)
        ]
        ameans = self.ameans[
            self.ameans["criterion"].apply(
                lambda x: x.mean_kind == SummarizeWith.ARITHMEAN
            )
        ]
        print(len(self.ameans))
        print(len(ameans))

        gmean_df = (
            process_mean(
                means_df=gmeans,
                mean_col="gmean",
                mean_func=stats.gmean,
                ratio_col="gmean_ratio",
                ci_ratio_col_low="gmean_ratio_ci_low",
                ci_ratio_col_high="gmean_ratio_ci_high",
                sig_col="gmean_significant",
            )
            if not gmeans.empty
            else pd.DataFrame()
        )

        arithmean_df = (
            process_mean(
                means_df=ameans,
                mean_col="arithmean",
                mean_func=np.mean,
                ratio_col="arithmean_ratio",
                ci_ratio_col_low="arithmean_ratio_ci_low",
                ci_ratio_col_high="arithmean_ratio_ci_high",
                sig_col="arithmean_significant",
            )
            if not ameans.empty
            else pd.DataFrame()
        )

        # Merge results on criterion/configuration
        merged = pd.merge(
            gmean_df, arithmean_df, on=["criterion", "configuration"], how="outer"
        )
        return merged.round(3)

    def _calculate_gmean_ratios(self):
        gmean_with_baseline = self.gmeans.merge(self.config_map, on="configuration")
        gmean_baseline = self.gmeans.rename(
            columns={"configuration": "baseline", "gmean": "gmean_b"}
        )
        gmean_merged = gmean_with_baseline.merge(
            gmean_baseline, on=["criterion", "baseline"]
        )

        gmean_merged["gmean_ratio"] = gmean_merged["gmean"] / gmean_merged["gmean_b"]

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
                    "gmean_ratio_ci_low": ratio_low,
                    "gmean_ratio_ci_high": ratio_high,
                    "gmean_significant": not (ratio_low <= 1.0 <= ratio_high),
                }
            )

        return pd.DataFrame(gmean_cis).round(3)

    def without_errs(self):
        error_cols = [
            "worst_ratio_ci_low",
            "worst_ratio_ci_high",
            "best_ratio_ci_low",
            "best_ratio_ci_high",
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
class SuiteData(Results):
    def __init__(self, suite):
        self.data = self._parse_perf(suite)
        # per_gc, totals = self._parse_metrics(suite)
        #
        # self.collector_data = per_gc
        # self.data = pd.concat([self.data, totals], ignore_index=True)

    @property
    def premopt(self):
        from build import ExperimentProfile, PremOpt

        def to_premopt(val):
            try:
                return PremOpt(val)
            except ValueError:
                return np.nan

        df = self.data.copy()
        df["configuration"] = df["configuration"].map(to_premopt)
        df = df.dropna(subset="configuration")
        return PremOptResults(df)

    @property
    def gcvs(self):
        from build import GCVS, ExperimentProfile

        def to_gcvs(val):
            try:
                return GCVS(val)
            except ValueError:
                return np.nan

        df = self.data.copy()
        df["configuration"] = df["configuration"].map(to_gcvs)
        df = df.dropna(subset="configuration")
        return GcvsResults(df)

    @property
    def elision(self):
        from build import Elision, ExperimentProfile

        def to_elision(val):
            try:
                return Elision(val)
            except ValueError:
                return np.nan

        df = self.data.copy()
        df["configuration"] = df["configuration"].map(to_elision)
        df = df.dropna(subset="configuration")
        return ElisionResults(df)

    def _process_metric_csv(self, f, suite):
        path = Path(f)
        benchmark = [
            b for b in suite.benchmarks if b.name.lower() == path.stem.split("-")[-1]
        ][0]
        configuration = f"{suite.name}-{'-'.join(path.stem.split('-')[:-1])}"

        # Read CSV with only non-numeric column specified
        df = pd.read_csv(f, dtype={Criterea.PHASE: "string"})
        df = df[~(df == df.columns).all(axis=1)]
        df.columns = [Criterea._value2member_map_.get(c, c) for c in df.columns]

        # Convert all columns except non-numeric to numeric
        numeric_cols = [col for col in df.columns if col != Criterea.PHASE]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Process bytes_freed and heap_grew
        df[Criterea.HEAP_GREW] = 0
        negative_mask = df[Criterea.BYTES_FREED] < 0
        df.loc[negative_mask, Criterea.HEAP_GREW] = -df.loc[
            negative_mask, Criterea.BYTES_FREED
        ]
        df.loc[negative_mask, Criterea.BYTES_FREED] = 0

        # Process time columns
        time_cols = [
            Criterea.TIME_MARKING,
            Criterea.TIME_SWEEPING,
            Criterea.TIME_FINALIZER_QUEUE,
            Criterea.TIME_TOTAL,
        ]
        for col in time_cols:
            ms_col, ns_col = f"{col.value}_ms", f"{col.value}_ns"
            df[col] = df[ms_col] + df[ns_col] / 1_000_000
        df = df.drop(
            columns=[f"{c.value}_{u}" for c in time_cols for u in ("ms", "ns")]
        )

        # Process collection numbers
        df[Criterea.COLLECTION_NUMBER] = pd.to_numeric(
            df[Criterea.COLLECTION_NUMBER], errors="coerce"
        )

        # Create invocation counter
        sentinel = df[Criterea.COLLECTION_NUMBER] == -1
        df["invocation"] = sentinel.cumsum().shift(fill_value=0).astype(int) + 1

        # Process per-collection data
        per_collection = df[df[Criterea.COLLECTION_NUMBER] != -1].copy()
        total_collections = (
            per_collection.groupby("invocation")[Criterea.COLLECTION_NUMBER]
            .max()
            .reset_index()
            .rename(columns={Criterea.COLLECTION_NUMBER: "value"})
        )
        total_collections["criterion"] = Criterea.TOTAL_COLLECTIONS

        # Process summary metrics
        drop_cols = [
            Criterea.PHASE,
            Criterea.TIME_TOTAL,
            Criterea.TIME_MARKING,
            Criterea.TIME_SWEEPING,
            Criterea.HEAP_SIZE_ON_ENTRY,
            Criterea.COLLECTION_NUMBER,
            Criterea.TIME_FINALIZER_QUEUE,
            Criterea.BYTES_FREED,
            Criterea.HEAP_GREW,
        ]
        # Filter out potentially missing columns
        drop_cols = [col for col in drop_cols if col in df.columns]

        totals = (
            df[sentinel]
            .drop(columns=drop_cols)
            .rename(
                columns={
                    Criterea.ALLOCATED_ARC: Criterea.TOTAL_ALLOCATED_ARC,
                    Criterea.ALLOCATED_BOXED: Criterea.TOTAL_ALLOCATED_BOXED,
                    Criterea.ALLOCATED_RC: Criterea.TOTAL_ALLOCATED_RC,
                    Criterea.ALLOCATED_GC: Criterea.TOTAL_ALLOCATED_GC,
                    Criterea.FINALIZERS_REGISTERED: Criterea.TOTAL_FINALIZERS_REGISTERED,
                    Criterea.FINALIZERS_RUN: Criterea.TOTAL_FINALIZERS_RUN,
                    Criterea.OBJECTS_IN_FINALIZER_QUEUE: Criterea.TOTAL_OBJECTS_IN_FINALIZER_QUEUE,
                    Criterea.LIVE_OBJECTS_WITH_FINALIZERS: Criterea.TOTAL_LIVE_OBJECTS_WITH_FINALIZERS,
                }
            )
            .melt(id_vars=["invocation"], var_name="criterion", value_name="value")
        )
        totals = pd.concat([totals, total_collections], ignore_index=True)

        # Final per-collection processing
        per_collection = per_collection.drop(columns=[Criterea.PHASE]).melt(
            id_vars=[Criterea.COLLECTION_NUMBER, "invocation"],
            var_name="criterion",
            value_name="value",
        )
        per_collection["value"] = pd.to_numeric(
            per_collection["value"], errors="coerce"
        )

        # Add metadata
        for df_part in [per_collection, totals]:
            df_part["benchmark"] = benchmark
            df_part["configuration"] = configuration
            df_part["suite"] = suite

        return per_collection, totals

    def _parse_perf(self, suite):
        from build import Metric

        def to_benchmark(name):
            for b in suite.benchmarks:
                if b.name.lower() == name.lower():
                    return b
            raise ValueError(f"Benchmark for {name} not found.")

        file = suite.raw_data(Metric.PERF)
        if not file.exists():
            return pd.DataFrame()
        df = pd.read_csv(
            file,
            sep="\t",
            comment="#",
            index_col="suite",
            converters={
                "criterion": Criterea,
                "benchmark": to_benchmark,
            },
        )
        df = df.rename(columns={"executor": "configuration"}).reset_index()[
            ["benchmark", "configuration", "value", "criterion", "invocation"]
        ]
        df["suite"] = suite
        return df

    def _parse_metrics(self, suite):
        from build import Metric

        csvs = glob.glob(str(suite.raw_data(Metric.METRICS).parent / "*.csv"))
        gc_metrics, summary_metrics = [], []
        for f in csvs:
            per_collection, totals = self._process_metric_csv(f, suite)
            gc_metrics.append(per_collection)
            summary_metrics.append(totals)
        return pd.concat(gc_metrics, ignore_index=True), pd.concat(
            summary_metrics, ignore_index=True
        )

    def summary(self):
        return Summary(self)


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
                best_configs[crit] = crit_df.loc[
                    crit_df[crit.mean].idxmin(), "configuration"
                ]

            cfgs = suites["configuration"].drop_duplicates().to_list()

            for i, c in enumerate(cfgs):
                config_rows = suites[suites["configuration"] == c]
                row_data = [s if i == 0 else "", c]

                for crit in criteria_list:
                    row = config_rows[config_rows["criterion"] == crit].iloc[0]

                    if i == 0:  # Baseline
                        val = fmt_float(row[crit.mean], bold=(c == best_configs[crit]))
                        ci = fmt_ci(row[crit.lower], row[crit.upper])
                    else:  # Other rows
                        val = (
                            fmt_float(row[crit.ratio], bold=(c == best_configs[crit]))
                            + "×"
                        )
                        if not row[crit.is_significant]:
                            val += r"\textsuperscript{\dag}"
                        ci = fmt_ci(row[crit.ratio_lower], row[crit.ratio_upper])

                    row_data.extend([val, ci])

                lines.append(" & ".join(row_data) + r" \\")

            lines.append(r"\midrule")

        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")

        with open("plots/suites.tex", "w") as f:
            f.write("\n".join(lines))
