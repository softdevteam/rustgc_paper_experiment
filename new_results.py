import glob
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from build import GCVS, Elision, PremOpt

plt.rcParams.update(
    {
        "axes.edgecolor": "#b0b0b0",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.4,
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "axes.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.columnspacing": 1.0,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": True,
        "text.latex.preamble": r"""
                \usepackage{contour}
                \contourlength{0.5pt}
                \renewcommand{\familydefault}{\sfdefault}
            """,
    }
)

CONFIDENCE_LEVEL = 0.99
BOOTSTRAP_SAMPLES = 100

BENCHMARK_SUITES = [
    "som-rs-bc",
    "som-rs-ast",
    "grmtools",
    "alacritty",
    "ripgrep",
    "fd",
    "binary-trees",
    "regex-redux",
]


class Aggregation(Enum):
    INDIVIDUAL = "individual"
    SUITE_ARITH = "suite_arith"
    SUITE_GEO = "suite_geo"
    GLOBAL_ARITH = "global_arith"
    GLOBAL_GEO = "global_geo"


class Metric(Enum):
    def __init__(self, value, latex=None, desc=None):
        self._value_ = value
        self.latex = latex
        self.desc = desc

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None

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

    WALLCLOCK = (
        "total",
        "Wall-clock time",
        "Wall-clock time (ms). Lower is better.",
    )
    USER = (
        "usr",
        "User time",
        "User time (ms). Lower is better.",
    )

    def __lt__(self, other):
        return self.value < other.value


ALL_CFGS = [GCVS, Elision, PremOpt]


@dataclass
class MetricColumn:
    metric: Metric
    agg_type: Aggregation
    header_name: str
    decimal_places: int
    overall: Aggregation
    value_type: str


class Results:
    def __init__(self, df):
        self.df = df.copy()
        self._results = None

    def _add_baseline_comparisons(
        self, df, group_keys, stat_cols, main_col, lower_col, upper_col
    ):
        def process_row(row):
            baseline_config = row["configuration"].baseline
            mask = [df[k] == row[k] for k in group_keys if k != "configuration"]
            mask.append(df["configuration"] == baseline_config)
            mask = np.logical_and.reduce(mask)

            baseline_row = df[mask]
            if baseline_row.empty:
                baseline_stats = {f"baseline_{col}": np.nan for col in stat_cols}
                baseline_stats.update(
                    {
                        "ratio": np.nan,
                        "ratio_upper": np.nan,
                        "ratio_lower": np.nan,
                        "distinguishable": False,
                    }
                )
            else:
                base = baseline_row.iloc[0]
                baseline_stats = {f"baseline_{col}": base[col] for col in stat_cols}

                if not np.isnan(baseline_stats[f"baseline_{main_col}"]):
                    ratio = row[main_col] / baseline_stats[f"baseline_{main_col}"]
                    ratio_upper = (
                        row[upper_col] / baseline_stats[f"baseline_{lower_col}"]
                    )
                    ratio_lower = (
                        row[lower_col] / baseline_stats[f"baseline_{upper_col}"]
                    )

                    distinguishable = not (
                        np.isnan(row[lower_col])
                        or np.isnan(baseline_stats[f"baseline_{upper_col}"])
                    ) and (
                        (row[lower_col] > baseline_stats[f"baseline_{upper_col}"])
                        or (row[upper_col] < baseline_stats[f"baseline_{lower_col}"])
                    )

                    baseline_stats.update(
                        {
                            "ratio": ratio,
                            "ratio_upper": ratio_upper,
                            "ratio_lower": ratio_lower,
                            "distinguishable": distinguishable,
                        }
                    )
                else:
                    baseline_stats.update(
                        {
                            "ratio": np.nan,
                            "ratio_upper": np.nan,
                            "ratio_lower": np.nan,
                            "distinguishable": False,
                        }
                    )

            return pd.Series(baseline_stats)

        processed = df.apply(process_row, axis=1)
        return pd.concat([df, processed], axis=1)

    @property
    def results(self):
        if self._results is not None:
            return self._results

        results = []

        def _amean(series):
            mean = series.mean()
            n = len(series)
            if n < 2:
                return pd.Series({"mean": mean, "ci_lower": mean, "ci_upper": mean})
            sem = stats.sem(series)
            alpha = 1 - CONFIDENCE_LEVEL
            h = sem * stats.t.ppf(1 - alpha / 2, n - 1)
            return pd.Series({"mean": mean, "ci_lower": mean - h, "ci_upper": mean + h})

        individual = (
            self.df.groupby(
                ["experiment", "benchmark", "configuration", "suite", "metric"]
            )["value"]
            .apply(_amean)
            .unstack()
            .reset_index()
        )
        individual = self._add_baseline_comparisons(
            individual,
            ["experiment", "benchmark", "configuration", "suite", "metric"],
            ["mean", "ci_lower", "ci_upper"],
            "mean",
            "ci_lower",
            "ci_upper",
        )
        individual["agg_type"] = Aggregation.INDIVIDUAL
        results.append(individual)

        def suite_arith_mean(series):
            arr = np.array(series)
            if len(arr) == 0:
                return pd.Series(
                    {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                )
            if len(arr) == 1:
                val = arr[0]
                return pd.Series({"mean": val, "ci_lower": val, "ci_upper": val})
            mean = arr.mean()
            sem = stats.sem(arr)
            alpha = 1 - CONFIDENCE_LEVEL
            h = sem * stats.t.ppf(1 - alpha / 2, len(arr) - 1)
            return pd.Series({"mean": mean, "ci_lower": mean - h, "ci_upper": mean + h})

        suite_arith = (
            self.df.groupby(["experiment", "suite", "configuration", "metric"])["value"]
            .apply(suite_arith_mean)
            .unstack()
            .reset_index()
        )
        suite_arith = self._add_baseline_comparisons(
            suite_arith,
            ["experiment", "suite", "configuration", "metric"],
            ["mean", "ci_lower", "ci_upper"],
            "mean",
            "ci_lower",
            "ci_upper",
        )
        suite_arith["agg_type"] = Aggregation.SUITE_ARITH
        results.append(suite_arith)

        def suite_geo_mean(series):
            arr = np.array(series[series > 0])
            if len(arr) == 0:
                return pd.Series(
                    {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                )
            if len(arr) == 1:
                val = arr[0]
                return pd.Series({"mean": val, "ci_lower": val, "ci_upper": val})
            res = stats.bootstrap(
                (arr,),
                statistic=lambda y: stats.gmean(y, axis=0),
                n_resamples=BOOTSTRAP_SAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                method="percentile",
                axis=0,
            )
            return pd.Series(
                {
                    "mean": stats.gmean(arr),
                    "ci_lower": res.confidence_interval.low,
                    "ci_upper": res.confidence_interval.high,
                }
            )

        suite_geo = (
            self.df.groupby(["experiment", "suite", "configuration", "metric"])["value"]
            .apply(suite_geo_mean)
            .unstack()
            .reset_index()
        )
        suite_geo = self._add_baseline_comparisons(
            suite_geo,
            ["experiment", "suite", "configuration", "metric"],
            ["mean", "ci_lower", "ci_upper"],
            "mean",
            "ci_lower",
            "ci_upper",
        )
        suite_geo["agg_type"] = Aggregation.SUITE_GEO
        results.append(suite_geo)

        global_arith = (
            self.df.groupby(["experiment", "configuration", "metric"])["value"]
            .apply(suite_arith_mean)
            .unstack()
            .reset_index()
        )
        global_arith = self._add_baseline_comparisons(
            global_arith,
            ["experiment", "configuration", "metric"],
            ["mean", "ci_lower", "ci_upper"],
            "mean",
            "ci_lower",
            "ci_upper",
        )
        global_arith["agg_type"] = Aggregation.GLOBAL_ARITH
        results.append(global_arith)

        global_geo = (
            self.df.groupby(["experiment", "configuration", "metric"])["value"]
            .apply(suite_geo_mean)
            .unstack()
            .reset_index()
        )
        global_geo = self._add_baseline_comparisons(
            global_geo,
            ["experiment", "configuration", "metric"],
            ["mean", "ci_lower", "ci_upper"],
            "mean",
            "ci_lower",
            "ci_upper",
        )
        global_geo["agg_type"] = Aggregation.GLOBAL_GEO
        results.append(global_geo)

        self._results = pd.concat(results, ignore_index=True)
        return self._results

    @property
    def arith(self):
        return self.results[self.results["agg_type"] == Aggregation.INDIVIDUAL]

    @property
    def geo(self):
        return self.results[self.results["agg_type"] == Aggregation.SUITE_GEO]

    @property
    def global_geo(self):
        return self.results[self.results["agg_type"] == Aggregation.GLOBAL_GEO]

    @property
    def global_arith(self):
        return self.results[self.results["agg_type"] == Aggregation.GLOBAL_ARITH]

    @property
    def suite_arith(self):
        return self.results[self.results["agg_type"] == Aggregation.SUITE_ARITH]

    @property
    def elision(self):
        return Results(self.df[self.df["experiment"] == "elision"])

    @property
    def gcvs(self):
        return Results(self.df[self.df["experiment"] == "gcvs"])

    @property
    def premopt(self):
        return Results(self.df[self.df["experiment"] == "premopt"])

    @property
    def experiments(self):
        return self.df["experiment"].unique().tolist()

    def mk_table(
        self,
        columns,
        output_file=None,
    ):
        def format_value(val, low, high, dp):
            if pd.isna(val):
                return "", ""
            if abs(val - 1.0) < 1e-6:
                val_fmt = f"\\textbf{{{val:.{dp}f}}}"
            else:
                val_fmt = f"{val:.{dp}f}$\\times$"
            ci_fmt = (
                f"\\scriptsize\\textcolor{{gray!80}}{{[{low:.{dp}f}, {high:.{dp}f}]}}"
            )
            return val_fmt, ci_fmt

        col_format = "l l " + " ".join(["r@{\\hspace{0.5em}}l" for _ in columns])
        header_line1 = ["Suite", "Configuration"] + [
            f"\\multicolumn{{2}}{{c}}{{{col.header_name}}}" for col in columns
        ]

        lines = []
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")
        lines.append(" & ".join(header_line1) + " \\\\")
        lines.append("\\midrule")

        # Main data by suite
        for suite, group in self.results.groupby("suite"):
            first_row = True
            for config in sorted(group["configuration"].unique()):
                suite_name = suite if first_row else ""
                first_row = False
                row_cells = [suite_name, config.latex]

                for col in columns:
                    match = self.results[
                        (self.results["agg_type"] == col.agg_type)
                        & (self.results["metric"] == col.metric)
                        & (self.results["suite"] == suite)
                        & (self.results["configuration"] == config)
                    ]

                    if match.empty:
                        val_fmt, ci_fmt = "", ""
                    else:
                        row = match.iloc[0]
                        val_col = (
                            col.value_type
                            if col.value_type in match.columns
                            else "mean"
                        )
                        low_col = (
                            f"{col.value_type}_lower"
                            if f"{col.value_type}_lower" in match.columns
                            else "ci_lower"
                        )
                        high_col = (
                            f"{col.value_type}_upper"
                            if f"{col.value_type}_upper" in match.columns
                            else "ci_upper"
                        )
                        val_fmt, ci_fmt = format_value(
                            row[val_col],
                            row[low_col],
                            row[high_col],
                            col.decimal_places,
                        )

                    row_cells.extend([val_fmt, ci_fmt])

                lines.append(" & ".join(row_cells) + " \\\\")
            lines.append("\\midrule")

        # Overall section
        if any(col.overall is not None for col in columns):
            first_overall_row = True
            for config in sorted(self.results["configuration"].unique()):
                suite_name = "overall" if first_overall_row else ""
                first_overall_row = False
                row_cells = [suite_name, config.latex]

                for col in columns:
                    if col.overall is None:
                        val_fmt, ci_fmt = "", ""
                    else:
                        match = self.results[
                            (self.results["agg_type"] == col.overall)
                            & (self.results["metric"] == col.metric)
                            & (self.results["configuration"] == config)
                        ]

                        if match.empty:
                            val_fmt, ci_fmt = "", ""
                        else:
                            row = match.iloc[0]
                            val_col = (
                                col.value_type
                                if col.value_type in match.columns
                                else "mean"
                            )
                            low_col = (
                                f"{col.value_type}_lower"
                                if f"{col.value_type}_lower" in match.columns
                                else "ci_lower"
                            )
                            high_col = (
                                f"{col.value_type}_upper"
                                if f"{col.value_type}_upper" in match.columns
                                else "ci_upper"
                            )
                            val_fmt, ci_fmt = format_value(
                                row[val_col],
                                row[low_col],
                                row[high_col],
                                col.decimal_places,
                            )

                    row_cells.extend([val_fmt, ci_fmt])

                lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        latex_str = "\n".join(lines)

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(latex_str)

        return latex_str

    class TableGen:
        def fmt_float(self, x, digits=2, bold=False):
            if pd.isna(x):
                return "-"
            s = f"{x:.{digits}f}"
            return f"\\textbf{{{s}}}" if bold else s

        def fmt_ratio(self, x, digits=2, distinguishable=True, use_times=True):
            if pd.isna(x):
                return "-"
            symbol = "$\\times$" if use_times else "×"
            s = f"{x:.{digits}f}{symbol}"
            return f"{s}\\dag" if not distinguishable else s

        def fmt_ci_interval(self, lower, upper, digits=2):
            if pd.isna(lower) or pd.isna(upper):
                return ""
            return f"\\scriptsize\\textcolor{{gray!80}}{{[{lower:.{digits}f}, {upper:.{digits}f}]}}"

        def fmt_ci_symmetric(self, mean_val, ci_lower, ci_upper, digits=1):
            if pd.isna(mean_val) or pd.isna(ci_lower) or pd.isna(ci_upper):
                return ""
            ci_range = (ci_upper - ci_lower) / 2
            return f"\\scriptsize\\color{{gray!80}}{{±{ci_range:.{digits}f}}}"

        def mk_summary_table(
            self,
            results,
            metrics,
            agg_using,
            header_names,
            decimal_places=None,
            output_file=None,
        ):
            if decimal_places is None:
                decimal_places = [2] * len(metrics)

            # Filter results by aggregation types
            agg_values = [agg.value for agg in agg_using]
            df = results.results[
                (results.results["metric"].isin(metrics))
                & (results.results["aggregation_type"].isin(agg_values))
            ]

            df_unique = (
                df.groupby(["suite", "configuration", "metric", "aggregation_type"])
                .first()
                .reset_index()
            )
            baseline = df_unique["configuration"].iloc[0].baseline

            # Table setup
            col_spec = "l l " + "r@{\\hspace{0.5em}}l " * len(metrics)
            header_cols = [
                f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in header_names
            ]
            header = "Suite & Configuration & " + " & ".join(header_cols) + " \\\\"

            lines = [
                f"\\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                header,
                r"\midrule",
            ]

            # Suite groups
            for suite_name, suite_data in df_unique.groupby("suite"):
                configs = sorted(
                    suite_data["configuration"].unique(),
                    key=lambda x: (x != baseline, x),
                )

                for i, config in enumerate(configs):
                    row_data = [suite_name if i == 0 else "", config.latex]

                    for metric, agg_type, digits in zip(
                        metrics, agg_using, decimal_places
                    ):
                        metric_data = suite_data[
                            (suite_data["metric"] == metric)
                            & (suite_data["configuration"] == config)
                            & (suite_data["aggregation_type"] == agg_type.value)
                        ]

                        if metric_data.empty:
                            row_data.extend(["-", ""])
                            continue

                        row = metric_data.iloc[0]
                        is_baseline = config == baseline

                        # Use standardized column names
                        mean_val = (
                            self.fmt_float(
                                row["baseline_mean"], digits=digits, bold=True
                            )
                            if is_baseline
                            else self.fmt_ratio(
                                row["ratio"],
                                digits=digits,
                                distinguishable=row["distinguishable"],
                            )
                        )

                        # Use symmetric CI for arithmetic means, interval CI for geometric means
                        is_arithmetic = agg_type in [
                            Aggregation.INDIVIDUAL,
                            Aggregation.SUITE_ARITH,
                            Aggregation.GLOBAL_ARITH,
                        ]
                        if is_arithmetic:
                            ci_col = "baseline_" if is_baseline else "ratio_"
                            ci_val = self.fmt_ci_symmetric(
                                row[f"{ci_col}mean" if is_baseline else "ratio"],
                                row[f"{ci_col}ci_lower"],
                                row[f"{ci_col}ci_upper"],
                                digits=digits,
                            )
                        else:
                            ci_col = "baseline_" if is_baseline else "ratio_"
                            ci_val = self.fmt_ci_interval(
                                row[f"{ci_col}ci_lower"],
                                row[f"{ci_col}ci_upper"],
                                digits=digits,
                            )

                        row_data.extend([mean_val, ci_val])

                    lines.append(" & ".join(row_data) + r" \\")
                lines.append(r"\midrule")

            # Overall results (using global aggregations)
            configs = sorted(
                df_unique["configuration"].unique(), key=lambda x: (x != baseline, x)
            )

            for i, config in enumerate(configs):
                row_data = ["overall" if i == 0 else "", config.latex]

                for metric, agg_type, digits in zip(metrics, agg_using, decimal_places):
                    # Map to global equivalent
                    global_agg = {
                        Aggregation.SUITE_GEO: Aggregation.GLOBAL_GEO,
                        Aggregation.INDIVIDUAL: Aggregation.GLOBAL_ARITH,
                        Aggregation.SUITE_ARITH: Aggregation.GLOBAL_ARITH,
                        Aggregation.GLOBAL_GEO: Aggregation.GLOBAL_GEO,
                        Aggregation.GLOBAL_ARITH: Aggregation.GLOBAL_ARITH,
                    }.get(agg_type, Aggregation.GLOBAL_ARITH)

                    metric_data = df_unique[
                        (df_unique["metric"] == metric)
                        & (df_unique["configuration"] == config)
                        & (df_unique["aggregation_type"] == global_agg.value)
                    ]

                    if metric_data.empty:
                        row_data.extend(["-", ""])
                        continue

                    row = metric_data.iloc[0]
                    is_baseline = config == baseline

                    mean_val = (
                        self.fmt_float(row["baseline_mean"], digits=digits, bold=True)
                        if is_baseline
                        else self.fmt_ratio(
                            row["ratio"], digits=digits, distinguishable=True
                        )
                    )

                    is_arithmetic = global_agg == Aggregation.GLOBAL_ARITH
                    if is_arithmetic:
                        ci_col = "baseline_" if is_baseline else "ratio_"
                        ci_val = self.fmt_ci_symmetric(
                            row[f"{ci_col}mean" if is_baseline else "ratio"],
                            row[f"{ci_col}ci_lower"],
                            row[f"{ci_col}ci_upper"],
                            digits=digits,
                        )
                    else:
                        ci_col = "baseline_" if is_baseline else "ratio_"
                        ci_val = self.fmt_ci_interval(
                            row[f"{ci_col}ci_lower"],
                            row[f"{ci_col}ci_upper"],
                            digits=digits,
                        )

                    row_data.extend([mean_val, ci_val])

                lines.append(" & ".join(row_data) + r" \\")

            lines.extend([r"\bottomrule", r"\end{tabular}"])

            content = "\n".join(lines)
            if output_file:
                with open(output_file, "w") as f:
                    f.write(content)
            return content


class Plotter:
    EXPERIMENTS = {
        "gcvs": {
            "colors": ["#3A87D9", "#FF8F2E"],
            "outline": ["#1A5C85", "#D66000"],
            "geomean": ["#1A5C85", "#994400"],
            "alphas": [0.1, 0.15],
        },
        "premopt": {
            "colors": ["#34495E", "#3A87D9"],
            "outline": ["#2C3E50", "#1A5C85"],
            "geomean": ["#1B2631", "#1A5C85"],
            "alphas": [0.1, 0.1],
        },
        "elision": {
            "colors": ["#3A87D9", "#E74C3C"],
            "outline": ["#1A5C85", "#C0392B"],
            "geomean": ["#1A5C85", "#922B21"],
            "alphas": [0.1, 0.1],
        },
    }

    def __init__(
        self,
        figsize=(2, 2),
        title=None,
        ylabel="",
        savepath="plots/test.svg",
        group_width=0.6,
        margin_ratio=1,
        ci_lower="ratio_lower",
        ci_upper="ratio_upper",
    ):
        self.figsize = figsize
        self.title = title
        self.ylabel = ylabel
        self.savepath = savepath
        self.group_width = group_width
        self.margin_ratio = margin_ratio
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper

    def plot(
        self,
        df,
        y="mean",
        metrics=None,
        savepath=None,
        experiment=None,
        xlim=None,
        ncols=1,
    ):
        if savepath:
            self.savepath = savepath

        baseline_config = df["configuration"].iloc[0].baseline
        baseline = df[df["configuration"] == baseline_config]
        data = df[df["configuration"] != baseline_config]

        if metrics:
            data = data[data["metric"].isin(metrics)]
            baseline = baseline[baseline["metric"].isin(metrics)]

        benchmarks = sorted(data["benchmark"].unique())
        configs = list(data["configuration"].unique())

        if ncols == 1:
            return self._plot_single(
                data,
                baseline,
                benchmarks,
                configs,
                experiment,
                xlim,
                baseline_config,
                y,
            )
        else:
            return self._plot_multi(
                data,
                baseline,
                benchmarks,
                configs,
                experiment,
                xlim,
                ncols,
                baseline_config,
                y,
            )

    def _plot_single(
        self, df, baseline, benchmarks, configs, experiment, xlim, baseline_config, y
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw(
            ax, df, baseline, benchmarks, configs, experiment, xlim, baseline_config, y
        )

        if self.title:
            ax.set_title(self.title, fontsize=8, pad=10)

        plt.tight_layout(pad=0.2)
        plt.savefig(self.savepath, bbox_inches="tight", dpi=300)
        plt.close()
        return self.savepath

    def _plot_multi(
        self,
        df,
        baseline,
        benchmarks,
        configs,
        experiment,
        xlim,
        ncols,
        baseline_config,
        y,
    ):
        chunk_size = int(np.ceil(len(benchmarks) / ncols))
        chunks = [
            benchmarks[i : i + chunk_size]
            for i in range(0, len(benchmarks), chunk_size)
        ]
        max_per_col = max(len(chunk) for chunk in chunks)

        if not xlim:
            xlim = self._calc_xlim(df, baseline)

        fig, axes = plt.subplots(
            1, ncols, figsize=(self.figsize[0] * ncols, self.figsize[1])
        )
        if ncols == 1:
            axes = [axes]

        for col, (ax, bench_subset) in enumerate(zip(axes, chunks)):
            subset_df = df[df["benchmark"].isin(bench_subset)]
            subset_baseline = baseline[baseline["benchmark"].isin(bench_subset)]

            self._draw(
                ax,
                subset_df,
                subset_baseline,
                bench_subset,
                configs,
                experiment,
                xlim,
                baseline_config,
                y,
                show_legend=False,
            )
            ax.set_ylim(max_per_col - 0.5, -0.5)

            if col > 0:
                ax.set_ylabel("")

        if self.title:
            fig.suptitle(self.title, fontsize=12, y=0.98)

        ylabel = f"{self.ylabel} to \\contour{{black}}{{{baseline_config.latex}}} (lower is better)"
        fig.text(0.5, 0.0, ylabel, ha="center", va="bottom", fontsize=6)

        self._make_legend(fig, baseline_config, configs, experiment)
        plt.savefig(self.savepath, bbox_inches="tight", dpi=300)
        plt.close()
        return self.savepath

    def _calc_xlim(self, df, baseline):
        values = []
        for _, row in df.iterrows():
            values.extend([row["mean"], row[self.ci_lower], row[self.ci_upper]])
        for _, row in baseline.iterrows():
            values.extend(
                [row["ratio_lower_gmean"], row["ratio_upper_gmean"], row["ratio_gmean"]]
            )

        x_min, x_max = min(values), max(values)
        padding = (x_max - x_min) * 0.05
        return (x_min - padding, x_max + padding)

    def _make_legend(self, fig, baseline_config, configs, experiment):
        exp = self.EXPERIMENTS[experiment]
        handles, labels = [], []

        handles.append(plt.Line2D([], [], color="none", linestyle="None"))
        labels.append(
            f"\\hspace{{-20pt}}\\makebox[0pt][l]{{\\vrule width 11pt height 1.5ex depth 0pt}}"
            f"\\hspace{{0.6em}} {baseline_config.latex}"
        )

        for i, cfg in enumerate(configs):
            color_idx = i % len(exp["colors"])
            handles.append(plt.Line2D([], [], color=exp["colors"][color_idx], lw=4))
            labels.append(str(cfg.latex))

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            fontsize=6,
            bbox_to_anchor=(0.49, 1.3),
            columnspacing=1.2,
            handletextpad=1.2,
            handlelength=1.0,
        )

    def _draw(
        self,
        ax,
        df,
        baseline,
        benchmarks,
        configs,
        experiment,
        xlim,
        baseline_config,
        y,
        show_legend=True,
    ):
        exp = self.EXPERIMENTS[experiment]

        ax.axvline(x=1, color="black", linewidth=0.5, zorder=30)
        ax.tick_params(axis="y", pad=1)

        self._draw_ci(ax, baseline, df, configs, exp)
        self._draw_bars(ax, df, benchmarks, configs, exp, show_legend, y)

        ax.grid(True, axis="x", zorder=5)
        ax.grid(False, axis="y")
        ax.set_yticks(range(len(benchmarks)))
        ax.set_yticklabels(benchmarks)

        if xlim:
            ax.set_xlim(xlim)

        if show_legend:
            legend = ax.legend(loc="best", frameon=True)
            legend.get_frame().set_linewidth(0.4)
            legend.set_zorder(100)

        ax.minorticks_off()

        n = len(benchmarks)
        margin = 0.5 * self.margin_ratio / (n - 1) if n > 1 else 0.02
        ax.margins(x=margin, y=margin)

    def _draw_ci(self, ax, baseline, df, configs, exp):
        ranges = []

        if not baseline.empty:
            row = baseline.iloc[0]
            r_min, r_max = row["ratio_lower_gmean"], row["ratio_upper_gmean"]
            ranges.append((r_min, r_max))

            ax.axvspan(
                r_min, r_max, facecolor="gray", alpha=0.01, edgecolor="none", zorder=2
            )

            for edge in [r_min, r_max]:
                ax.axvline(
                    edge,
                    color="#666666",
                    linewidth=0.5,
                    linestyle="--",
                    alpha=0.5,
                    zorder=3,
                )

        for i, cfg in enumerate(configs):
            cfg_data = df[df["configuration"] == cfg]
            row = cfg_data.iloc[0]
            r_min, r_max = row["ratio_lower_gmean"], row["ratio_upper_gmean"]
            ranges.append((r_min, r_max))

            color_idx = i % len(exp["colors"])

            ax.axvspan(
                r_min,
                r_max,
                facecolor=exp["colors"][color_idx],
                alpha=exp["alphas"][color_idx],
                edgecolor="none",
                zorder=2,
            )

            for edge in [r_min, r_max]:
                ax.axvline(
                    edge,
                    color=exp["outline"][color_idx],
                    linewidth=0.5,
                    linestyle="--",
                    alpha=0.5,
                    zorder=3,
                )

            ax.axvline(
                row["ratio_gmean"],
                color=exp["geomean"][color_idx],
                linewidth=0.5,
                zorder=5,
            )

        self._draw_overlaps(ax, ranges)

    def _draw_overlaps(self, ax, ranges):
        boundaries = sorted(set(b for r in ranges for b in r))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            overlap_count = sum(1 for r in ranges if r[0] < end and r[1] > start)

            if overlap_count > 1:
                alpha = min(0.05 * (overlap_count - 1), 0.1)
                ax.axvspan(
                    start,
                    end,
                    facecolor="lightgray",
                    alpha=alpha,
                    edgecolor="none",
                    zorder=4,
                )

    def _draw_bars(self, ax, df, benchmarks, configs, exp, show_legend, y):
        bar_width = self.group_width / len(configs)
        grouped = df.groupby(["benchmark", "configuration"])

        for i_bench, bench in enumerate(benchmarks):
            group_start = i_bench - self.group_width / 2

            for i_cfg, cfg in enumerate(configs):
                row = grouped.get_group((bench, cfg)).iloc[0]
                x_pos = group_start + i_cfg * bar_width
                color_idx = i_cfg % len(exp["colors"])
                label = str(cfg.latex) if (show_legend and i_bench == 0) else None

                ax.barh(
                    x_pos,
                    row[y],
                    height=bar_width,
                    color=exp["colors"][color_idx],
                    label=label,
                    alpha=0.9,
                    edgecolor=exp["outline"][color_idx],
                    linewidth=0.5,
                    zorder=10,
                    align="edge",
                )

                err_low = row[y] - row[self.ci_lower]
                err_high = row[self.ci_upper] - row[y]
                ax.errorbar(
                    row[y],
                    x_pos + bar_width / 2,
                    xerr=[[err_low], [err_high]],
                    fmt="none",
                    color="#2C2C2C",
                    capsize=0.8,
                    linewidth=0.5,
                    zorder=15,
                )


def to_cfg(name):
    s = name.partition("-")[2]
    if s == "baseline":
        return GCVS.GC
    for c in ALL_CFGS:
        try:
            return c(s)
        except ValueError:
            continue
    raise ValueError(f"Configuration for {name} not found.")


def parse_all_benchmark_data(suite):
    def _parse_metrics(suite):
        def _process_metric_csv(f):
            path = Path(f)
            benchmark = path.stem.split("-")[-1]
            configuration = to_cfg(f"{suite}-{'-'.join(path.stem.split('-')[:-1])}")

            df = pd.read_csv(f, dtype={Metric.PHASE: "string"})
            df = df[~(df == df.columns).all(axis=1)]
            df.columns = [Metric._value2member_map_.get(c, c) for c in df.columns]

            numeric_cols = [col for col in df.columns if col != Metric.PHASE]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

def process_gcvs(perf):
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="Wall-clock time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_gcvs_wallclock_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="Wall-clock time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_gcvs_wallclock_2.tex",
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="User time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_gcvs_user_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="User time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_gcvs_user_2.tex",
    )

    perf.mk_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.SUITE_GEO,
                header_name="Time (seconds)",
                group_header="Wallclock-time",
                overall=Aggregation.GLOBAL_GEO,
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.SUITE_GEO,
                header_name="Ratio",
                group_header="Wallclock-time",
                overall=Aggregation.GLOBAL_GEO,
                value_type="ratio",
            ),
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.SUITE_GEO,
                header_name="Time (seconds)",
                group_header="User-time",
                overall=Aggregation.GLOBAL_GEO,
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.SUITE_GEO,
                header_name="Ratio",
                group_header="User-time",
                overall=Aggregation.GLOBAL_GEO,
                value_type="ratio",
            ),
        ],
        output_file="tables/gcvs_perf_summary.tex",
    )

    config = PlotConfig(
        # Visual settings
        figsize=(3, 12),
        savepath="plots/performance.pdf",
        max_benchmarks_per_plot=22,
        group_width=0.8,
        # Plot settings
        metric="ratio",
        agg_type=Aggregation.INDIVIDUAL,
        experiment="gcvs",
        xlim=(0.5, 2.0),
        ncols=4,
    )

    plotter = Plotter(config)
    plotter.plot(perf.results, output_file="plots/gcvs_perf.svg")


def process_elision(perf):
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.PCT_ELIDED,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\% Fin. elided",
                value_type="mean",
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_pct_1.tex",
        baseline=False,
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.PCT_ELIDED,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\% Fin. elided",
                value_type="mean",
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_pct_2.tex",
        baseline=False,
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="Wall-clock time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_wallclock_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="Wall-clock time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_wallclock_2.tex",
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="User time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_user_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.INDIVIDUAL,
                header_name="User time (seconds)",
                value_type="mean",
                format_func=fmt_ms_to_s,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_user_2.tex",
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.TOTAL_COLLECTIONS,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\# collections",
                value_type="mean",
                format_func=fmt_big_num,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_numgcs_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.TOTAL_COLLECTIONS,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\# collections",
                value_type="mean",
                format_func=fmt_big_num,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_numgcs_2.tex",
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.ALLOCATED_GC,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\# GC objects",
                value_type="mean",
                format_func=fmt_big_num,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_numgcallocs_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.ALLOCATED_GC,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"\# GC objects",
                value_type="mean",
                format_func=fmt_big_num,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_numgcallocs_2.tex",
    )

    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.AVG_MEMORY,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"Avg. Heap Size (MiB)",
                value_type="mean",
                format_func=fmt_bytes_to_mb,
            ),
        ],
        suites=["alacritty", "fd", "som-rs-ast"],
        output_file="tables/appendix_elision_heap_1.tex",
    )
    perf.mk_individual_table(
        columns=[
            MetricColumn(
                metric=Metric.AVG_MEMORY,
                agg_type=Aggregation.INDIVIDUAL,
                header_name=r"Avg. Heap Size (MiB)",
                value_type="mean",
                format_func=fmt_bytes_to_mb,
            ),
        ],
        suites=["ripgrep", "grmtools", "som-rs-bc"],
        output_file="tables/appendix_elision_heap_2.tex",
    )
    perf.mk_table(
        columns=[
            MetricColumn(
                metric=Metric.WALLCLOCK,
                agg_type=Aggregation.SUITE_GEO,
                header_name="Wall-clock",
                overall=Aggregation.GLOBAL_GEO,
                value_type="ratio",
            ),
            MetricColumn(
                metric=Metric.USER,
                agg_type=Aggregation.SUITE_GEO,
                header_name="User-time",
                overall=Aggregation.GLOBAL_GEO,
                value_type="ratio",
            ),
        ],
        baseline=False,
        output_file="tables/elision_perf.tex",
    )

    perf.mk_table(
        columns=[
            MetricColumn(
                metric=Metric.PCT_ELIDED,
                agg_type=Aggregation.SUITE_ARITH,
                header_name=r"Fin. Elided (\%)",
                value_type="mean",
            ),
        ],
        baseline=False,
        output_file="tables/elision_metrics.tex",
    )

    mini_cfg = PlotConfig(
        figsize=(3, 2),
        max_benchmarks_per_plot=6,
        group_width=0.8,
        metric="ratio",
        agg_type=Aggregation.SUITE_GEO,
        ncols=1,
    )

    mini_plot = Plotter(mini_cfg)
    mini_plot.plot(
        perf.results,
        metric=Metric.WALLCLOCK,
        xlim=(0, 1.1),
        output_file="plots/elision_wallclock.svg",
    )

    mini_plot.plot(
        perf.results,
        metric=Metric.USER,
        xlim=(0, 1.1),
        output_file="plots/elision_user.svg",
    )

    mini_plot.plot(
        perf.results,
        metric=Metric.AVG_MEMORY,
        xlim=(0.4, 6.5),
        output_file="plots/elision_heap.svg",
    )

    mini_plot.plot(
        perf.results,
        metric=Metric.GC_TIME,
        xlim=(0, 1.7),
        output_file="plots/elision_pause.svg",
    )

    mini_plot.plot(
        perf.results,
        metric=Metric.TOTAL_COLLECTIONS,
        xlim=(0.1, 1.3),
        output_file="plots/elision_numgcs.svg",
    )

    acfg = PlotConfig(
        figsize=(3, 7),
        max_benchmarks_per_plot=22,
        group_width=0.8,
        metric="ratio",
        agg_type=Aggregation.INDIVIDUAL,
        ncols=4,
    )

    appendix = Plotter(acfg)
    appendix.plot(
        perf.results,
        metric=Metric.WALLCLOCK,
        title="Wall-clock Time",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0, 1.2),
        output_file="plots/appendix_elision_wallclock.svg",
    )

    appendix.plot(
        perf.results,
        metric=Metric.USER,
        title="User Time",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0, 1.2),
        output_file="plots/appendix_elision_user.svg",
    )

    appendix.plot(
        perf.results,
        metric=Metric.AVG_MEMORY,
        title="Avg. Heap Size",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0.5, 6.5),
        output_file="plots/appendix_elision_heap.svg",
    )

    appendix.plot(
        perf.results,
        metric=Metric.GC_TIME,
        title="Total GC Pause Time",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0, 1.2),
        output_file="plots/appendix_elision_pause.svg",
    )

    appendix.plot(
        perf.results,
        metric=Metric.TOTAL_COLLECTIONS,
        title="Number of Collections",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0.1, 3),
        output_file="plots/appendix_elision_numgcs.svg",
    )

    appendix.plot(
        perf.results,
        metric=Metric.ALLOCATED_GC,
        title="Gc Objects Allocated",
        ylabel="Relative performance of finalizer elision (lower is better).",
        xlim=(0.3, 1.1),
        output_file="plots/appendix_elision_gcallocs.svg",
    )


def process_premopt(results):
    r = results.premopt
    if r.df.empty:
        return

    plotter = Plotter(
        max_benchmarks_per_plot=15, savepath="plots/premopt_individual.svg"
    )
    df = r.combined

    if not df.empty:
        plotter.plot(
            df,
            y="ratio",
            metrics=[Metric.WALLCLOCK],
            experiment="premopt",
            xlim=(0.8, 1.4),
        )


Path("plots").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

collections, totals, perf = load_suites(BENCHMARK_SUITES)


Path("plots").mkdir(exist_ok=True)

collections_data, totals_data, perf_data = load_suites(BENCHMARK_SUITES)
results = Results(totals_data)

process_gcvs(results)
process_elision(results)
process_premopt(results)
