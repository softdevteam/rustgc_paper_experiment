import glob
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from build import GCVS, Aggregation, Elision, HeapSize, Metric, PremOpt
from helpers import (
    BASELINES,
    BENCHMARK_SUITES,
    BMS,
    SUITES,
    arithmetic_mean,
    geometric_mean,
    ratio_agg,
    to_cfg,
)
from ht_parser import parse_heaptrack
from plots import *

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
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "legend.columnspacing": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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


def parse_all_benchmark_data(suite):
    def _parse_metrics(suite):
        def _process_metric_json(f):
            path = Path(f)
            benchmark = path.stem.split("-")[-1]
            configuration = to_cfg(f"{suite}-{'-'.join(path.stem.split('-')[:-1])}")
            records = []
            with open(f, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    new_row = {}
                    for key, value in data.items():
                        try:
                            metric = Metric(key)
                            new_row[metric] = value
                        except ValueError:
                            pass
                    records.append(new_row)

            df = pd.DataFrame(records)

            sentinel = df[Metric.COLLECTION_NUMBER] == -1
            metadata = {
                "invocation": sentinel.cumsum().shift(fill_value=0) + 1,
                "benchmark": benchmark,
                "configuration": configuration,
            }
            df = df.assign(**metadata)

            per_collection = df[~sentinel]
            summary_rows = df[sentinel]

            totals = [
                Metric.OBJ_ALLOCD_ARC,
                Metric.OBJ_ALLOCD_BOX,
                Metric.OBJ_ALLOCD_RC,
                Metric.OBJ_ALLOCD_GC,
                Metric.FLZ_REGISTERED,
                Metric.FLZ_RUN,
                Metric.FLZ_ELIDED,
                Metric.OBJ_FREED_EXPLICIT,
                Metric.OBJ_FREED_SWEPT,
                Metric.OBJ_ALLOCD_FLZQ,
                Metric.TIME_MARKING,
                Metric.TIME_SWEEPING,
                Metric.TIME_TOTAL,
            ]

            metadata_cols = list(metadata.keys())
            totals = summary_rows[metadata_cols + totals]
            totals[Metric.PCT_ELIDED] = 100 - (
                totals[Metric.FLZ_REGISTERED] / totals[Metric.OBJ_ALLOCD_GC] * 100
            )
            metadata = {
                "benchmark": benchmark,
                "configuration": configuration,
                "suite": suite,
            }

            avg_heap = (
                per_collection.groupby("invocation", as_index=False)
                .agg(value=(Metric.MEM_HSIZE_EXIT, "mean"))
                .assign(metric=Metric.MEM_HSIZE_AVG, **metadata)
            )

            max_heap = (
                per_collection.groupby("invocation", as_index=False)
                .agg(value=(Metric.MEM_HSIZE_EXIT, "max"))
                .assign(metric=Metric.MAX_MEMORY, **metadata)
            )

            gc_time = (
                per_collection.groupby("invocation", as_index=False)
                .agg(value=(Metric.TIME_TOTAL, "sum"))
                .assign(metric=Metric.GC_TIME, **metadata)
            )

            minor_count = (
                per_collection[per_collection[Metric.PHASE] == 1]
                .groupby("invocation", as_index=False)
                .size()
                .rename(columns={"size": "value"})
                .assign(metric=Metric.MINOR_COLLECTIONS, **metadata)
            )

            major_count = (
                per_collection[per_collection[Metric.PHASE] == 2]
                .groupby("invocation", as_index=False)
                .size()
                .rename(columns={"size": "value"})
                .assign(metric=Metric.MAJOR_COLLECTIONS, **metadata)
            )

            gc_count = (
                per_collection.groupby("invocation", as_index=False)
                .size()
                .rename(columns={"size": "value"})
                .assign(metric=Metric.TOTAL_COLLECTIONS, **metadata)
            )

            totals = pd.concat(
                [
                    totals.melt(
                        id_vars=metadata_cols, var_name="metric", value_name="value"
                    ),
                    avg_heap,
                    max_heap,
                    gc_time,
                    gc_count,
                    minor_count,
                    major_count,
                ],
                ignore_index=True,
            )

            per_collection = per_collection.melt(
                id_vars=[Metric.COLLECTION_NUMBER] + metadata_cols,
                var_name="metric",
                value_name="value",
            )

            return per_collection, totals

        path = Path(f"results/{suite}/metrics")
        csvs = glob.glob(str(path / "*.csv"))

        results = [_process_metric_json(f) for f in csvs]
        gc_metrics, summary_metrics = zip(*results)

        return pd.concat(gc_metrics, ignore_index=True), pd.concat(
            summary_metrics, ignore_index=True
        )

    def _parse_perf(suite):
        def to_benchmark(name):
            for b in suite.benchmarks:
                if b.name.lower() == name.lower():
                    return b
            raise ValueError(f"Benchmark for {name} not found.")

        file = Path(f"results/{suite}/perf/results.data")

        if not file.exists():
            return pd.DataFrame()

        df = pd.read_csv(
            file,
            sep="\t",
            comment="#",
            index_col="suite",
            converters={
                "criterion": Metric,
                "executor": to_cfg,
            },
        )

        return df.rename(
            columns={"executor": "configuration", "criterion": "metric"}
        ).reset_index()[["benchmark", "configuration", "value", "metric", "invocation"]]

    metrics_per_collection, metrics_summary = _parse_metrics(suite)
    perf_data = _parse_perf(suite)

    datasets = {
        "metrics_per_collection": metrics_per_collection,
        "metrics_summary": metrics_summary,
        "perf_data": perf_data,
    }

    explode_to = [GCVS.GC, PremOpt.OPT, Elision.OPT, HeapSize.DEFAULT]

    for name, df in datasets.items():
        if df.empty:
            continue

        mask = df["configuration"] == GCVS.GC
        if mask.any():
            gc_rows = df[mask].loc[df[mask].index.repeat(len(explode_to))].copy()
            gc_rows["configuration"] = explode_to * mask.sum()
            other_rows = df[~mask]
            datasets[name] = pd.concat([gc_rows, other_rows], ignore_index=True)

        datasets[name]["suite"] = suite
        datasets[name]["experiment"] = datasets[name]["configuration"].apply(
            lambda x: x.experiment
        )

    return (
        datasets["metrics_per_collection"],
        datasets["metrics_summary"],
        datasets["perf_data"],
    )


ALL_CFGS = [GCVS, Elision, PremOpt]


def fmt_ms_to_s(value):
    seconds = value / 1000.0
    return f"{seconds:.2f}"


def fmt_big_num(value):
    return f"{round(value):,}"


def fmt_bytes_to_mb(value):
    return f"{(value / 1024 / 1024):.2f}"


@dataclass
class MetricColumn:
    metric: Metric = None
    agg_type: Aggregation = None
    header_name: str = None
    group_header: str = None
    overall: Aggregation = None
    value_type: str = None
    format_func: callable = None


class TableMixin:
    def mk_individual_table(
        self, columns, output_file=None, baseline=True, suites=None
    ):
        metrics = list(set(c.metric for c in columns))
        configs = sorted(self.results["configuration"].unique())
        if not baseline:
            configs = [c for c in configs if c.baseline != c]

        df = self.results[
            (self.results["agg_type"] == Aggregation.INDIVIDUAL)
            & (self.results["metric"].isin(metrics))
            & (self.results["configuration"].isin(configs))
        ].copy()

        if suites:
            df = df[df["suite"].isin(suites)]

        def format_row(row, value_type, agg_type, format_func=None):
            is_ratio = value_type == "ratio"
            val_col, low_col, high_col = (
                value_type,
                f"{value_type}_lower",
                f"{value_type}_upper",
            )
            if value_type == "mean":
                low_col, high_col = "ci_lower", "ci_upper"

            val, low, high = row.get(val_col), row.get(low_col), row.get(high_col)
            if pd.isna(val):
                return "", ""
            if is_ratio and abs(val - 1.0) < 1e-6:
                return "-", ""

            if format_func:
                if abs(val - low) < 1e-6 and abs(val - high) < 1e-6:
                    return format_func(val), ""

                val_fmt = format_func(val)
                if agg_type in [
                    Aggregation.INDIVIDUAL,
                    Aggregation.SUITE_ARITH,
                    Aggregation.GLOBAL_ARITH,
                ]:
                    margin = format_func((high - low) / 2)
                    ci_fmt = f"\\scriptsize\\textcolor{{gray!60}}{{$\\pm${margin}}}"
                else:
                    low_fmt = format_func(low)
                    high_fmt = format_func(high)
                    ci_fmt = (
                        f"\\scriptsize\\textcolor{{gray!60}}{{[{low_fmt},{high_fmt}]}}"
                    )
                return val_fmt, ci_fmt

            if abs(val - low) < 1e-6 and abs(val - high) < 1e-6:
                return f"{val:.2f}" + ("\\times" if is_ratio else ""), ""
            if agg_type in [
                Aggregation.INDIVIDUAL,
                Aggregation.SUITE_ARITH,
                Aggregation.GLOBAL_ARITH,
            ]:
                margin = (high - low) / 2
                return (
                    f"{val:.2f}" + ("\\times" if is_ratio else ""),
                    f"\\scriptsize\\textcolor{{gray!60}}{{$\\pm${margin:.2f}}}",
                )
            return (
                f"{val:.2f}" + ("\\times" if is_ratio else ""),
                f"\\scriptsize\\textcolor{{gray!60}}{{[{low:.2f},{high:.2f}]}}",
            )

        formatted_cols = []
        full_index = (
            df[["suite", "benchmark"]]
            .drop_duplicates()
            .set_index(["suite", "benchmark"])
        )

        for col_def in columns:
            for config in configs:
                subset = df[
                    (df["metric"] == col_def.metric) & (df["configuration"] == config)
                ]
                group_header = col_def.group_header or col_def.header_name

                if subset.empty:
                    empty_series = pd.Series("", index=full_index.index)
                    formatted_cols.append(
                        empty_series.rename(
                            (group_header, col_def.header_name, config.latex, "Value")
                        )
                    )
                    formatted_cols.append(
                        empty_series.rename(
                            (group_header, col_def.header_name, config.latex, "CI")
                        )
                    )
                    continue

                formatted = subset.apply(
                    lambda r: format_row(
                        r, col_def.value_type, col_def.agg_type, col_def.format_func
                    ),
                    axis=1,
                    result_type="expand",
                ).set_index(pd.MultiIndex.from_frame(subset[["suite", "benchmark"]]))
                formatted.columns = ["value", "ci"]

                formatted_cols.append(
                    formatted["value"]
                    .reindex(full_index.index, fill_value="")
                    .rename((group_header, col_def.header_name, config.latex, "Value"))
                )
                formatted_cols.append(
                    formatted["ci"]
                    .reindex(full_index.index, fill_value="")
                    .rename((group_header, col_def.header_name, config.latex, "CI"))
                )

        if not formatted_cols:
            return "\\begin{tabular}{}\nNo data to display.\n\\end{tabular}"

        table_df = pd.concat(formatted_cols, axis=1)
        table_df = table_df.rename(index=BMS, level=1)

        table_df = table_df.sort_index()

        lines = []
        col_format = "ll" + "".join(
            ["@{\\hspace{6pt}}r@{\\hspace{3pt}}l"] * (len(table_df.columns) // 2)
        )
        lines.append(f"\\begin{{tabular}}{{{col_format}}}")
        lines.append("\\toprule")

        row1 = ["Suite", "Benchmark"]
        for group_name, group in groupby(table_df.columns, key=lambda x: x[0]):
            row1.append(f"\\multicolumn{{{len(list(group))}}}{{c}}{{{group_name}}}")
        lines.append(" & ".join(row1) + " \\\\")

        if any(c[0] != c[1] for c in table_df.columns):
            row2 = ["", ""]
            for (group_name, metric_name), group in groupby(
                table_df.columns, key=lambda x: (x[0], x[1])
            ):
                row2.append(
                    f"\\multicolumn{{{len(list(group))}}}{{c}}{{{metric_name}}}"
                )
            lines.append(" & ".join(row2) + " \\\\")

        if len(configs) > 1:
            row3 = ["", ""]
            for (g, m, config_name), group in groupby(
                table_df.columns, key=lambda x: (x[0], x[1], x[2])
            ):
                row3.append(
                    f"\\multicolumn{{{len(list(group))}}}{{c}}{{{config_name}}}"
                )
            lines.append(" & ".join(row3) + " \\\\")

        lines.append("\\midrule")

        suites_to_render = table_df.index.get_level_values("suite").unique()
        for i, suite_name in enumerate(suites_to_render):
            suite_group = table_df.loc[suite_name]
            for j, (benchmark_name, row) in enumerate(suite_group.iterrows()):
                suite_cell = ""
                if j == 0:
                    suite_cell = f"\\multirow{{{len(suite_group)}}}{{*}}{{\\rotatebox{{90}}{{{suite_name}}}}}"

                row_cells = [suite_cell, benchmark_name] + list(row.values)
                lines.append(" & ".join(row_cells) + " \\\\")

            if i < len(suites_to_render) - 1:
                lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        latex_str = "\n".join(lines)

        with open(output_file, "w") as f:
            f.write(latex_str)


class Results(TableMixin):
    def __init__(self, df):
        self.df = df.copy()
        self._results = None

    @property
    def results(self):
        if self._results is not None:
            return self._results

        df = self.df.copy()
        df["is_baseline"] = df["configuration"].isin(BASELINES)
        group_cols = ["experiment", "suite", "benchmark", "invocation", "metric"]

        # Baseline merge and ratio computation
        baseline_part = df[df["is_baseline"]][group_cols + ["value"]]
        baseline_part = baseline_part.rename(columns={"value": "baseline_value"})
        df = df.merge(baseline_part, on=group_cols, how="left")
        df["ratio"] = df["value"] / df["baseline_value"]

        def agg_block(group_cols, val, fn):
            d = (
                df.groupby(group_cols + ["metric"])[val]
                .apply(fn)
                .unstack()
                .reset_index()
            )
            return d

        results = []

        # INDIVIDUAL
        cols = ["experiment", "benchmark", "configuration", "suite"]
        indiv = agg_block(cols, "ratio", ratio_agg).merge(
            agg_block(cols, "value", arithmetic_mean), on=cols + ["metric"], how="outer"
        )
        indiv["agg_type"] = Aggregation.INDIVIDUAL
        results.append(indiv)

        # SUITE
        cols = ["experiment", "suite", "configuration"]
        suite = agg_block(cols, "value", geometric_mean).merge(
            agg_block(cols, "ratio", ratio_agg), on=cols + ["metric"], how="outer"
        )
        suite["agg_type"] = Aggregation.SUITE_GEO
        results.append(suite)

        # GLOBAL
        cols = ["experiment", "configuration"]
        global_ = agg_block(cols, "value", geometric_mean).merge(
            agg_block(cols, "ratio", ratio_agg), on=cols + ["metric"], how="outer"
        )
        global_["agg_type"] = Aggregation.GLOBAL_GEO
        results.append(global_)

        res = pd.concat(results, ignore_index=True)
        self._results = res
        return res

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
    def heapsizes(self):
        return Results(self.df[self.df["experiment"] == "heapsize"])


def load_suites(suite_names):
    all_collections = []
    all_totals = []
    all_perf = []

    for suite_name in suite_names:
        collections, totals, perf = parse_all_benchmark_data(suite_name)

        if not collections.empty:
            collections = collections.drop(Metric.COLLECTION_NUMBER, axis=1)
            num_collections = totals[totals["metric"] == Metric.TOTAL_COLLECTIONS]
            collections = (
                collections.groupby(
                    [
                        "suite",
                        "experiment",
                        "benchmark",
                        "configuration",
                        "metric",
                        "invocation",
                    ]
                )["value"]
                .sum()
                .reset_index()
            )
            collections = collections.merge(
                num_collections,
                on=[
                    "suite",
                    "experiment",
                    "benchmark",
                    "configuration",
                    "metric",
                    "invocation",
                    "value",
                ],
                how="outer",
            )
            all_collections.append(collections)

        if not totals.empty:
            all_totals.append(totals)

        if not perf.empty:
            all_perf.append(perf)

    collections_df = (
        pd.concat(all_collections, ignore_index=True)
        if all_collections
        else pd.DataFrame()
    )
    totals_df = (
        pd.concat(all_totals, ignore_index=True) if all_totals else pd.DataFrame()
    )
    perf_df = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

    return collections_df, totals_df, perf_df


@dataclass
class ExperimentConfig:
    colors: List[str]
    outline: List[str]
    geomean: List[str]
    alphas: List[float]


@dataclass
class SimplePlotConfig:
    figsize: Tuple[int, int] = (10, 8)
    group_width: float = 0.8
    show_legend: bool = True


class SimplePlotter:
    EXPERIMENTS = {
        "gcvs": ExperimentConfig(
            colors=["#3A87D9", "#FF8F2E"],
            outline=["#1A5C85", "#D66000"],
            geomean=["#1A5C85", "#994400"],
            alphas=[0.1, 0.15],
        ),
        "heapsizes": ExperimentConfig(
            colors=["#3A87D9", "#FF8F2E", "#228B22", "#800080", "#DC143C"],
            outline=["#1A5C85", "#D66000", "#145214", "#4B004B", "#8B0A1E"],
            geomean=["#1A5C85", "#994400", "#0F3A10", "#330033", "#5C0B16"],
            alphas=[0.1, 0.15, 0.1, 0.1],
        ),
        "premopt": ExperimentConfig(
            colors=["#34495E", "#3A87D9"],
            outline=["#2C3E50", "#1A5C85"],
            geomean=["#1B2631", "#1A5C85"],
            alphas=[0.1, 0.1],
        ),
        "elision": ExperimentConfig(
            colors=["#3A87D9", "#E74C3C"],
            outline=["#1A5C85", "#C0392B"],
            geomean=["#1A5C85", "#922B21"],
            alphas=[0.1, 0.1],
        ),
    }

    def __init__(self, config: SimplePlotConfig = None):
        self.config = config

    def plot(
        self,
        df: pd.DataFrame,
        experiment: str,
        value_col: str,
        output_file: str = "plot.png",
        xlim: Optional[Tuple] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
    ):
        exp_config = self.EXPERIMENTS[experiment]

        suite_data = df[df["agg_type"] == Aggregation.SUITE_GEO]
        suites = sorted(suite_data["suite"].unique())
        configs = sorted(suite_data["configuration"].unique())
        print(df)

        fig, ax = plt.subplots(figsize=self.config.figsize)
        self._draw_ci_shadows(ax, df, value_col, configs, exp_config)
        ax.axvline(x=1, color="black", linewidth=0.5, zorder=30)
        bar_width = self.config.group_width / len(configs)

        for i_suite, suite in enumerate(suites):
            group_start = i_suite - self.config.group_width / 2

            for i_config, config in enumerate(configs):
                row_data = suite_data[
                    (suite_data["suite"] == suite)
                    & (suite_data["configuration"] == config)
                ]

                if row_data.empty:
                    continue

                row = row_data.iloc[0]
                y_pos = group_start + i_config * bar_width
                color_idx = i_config % len(exp_config.colors)

                # Bar
                ax.barh(
                    y_pos,
                    row[value_col],
                    height=bar_width,
                    color=exp_config.colors[color_idx],
                    alpha=0.9,
                    edgecolor=exp_config.outline[color_idx],
                    linewidth=0.5,
                    zorder=10,
                    align="edge",
                )

                # Error bars
                ci_lower_col = f"{value_col}_lower"
                ci_upper_col = f"{value_col}_upper"

                if (
                    ci_lower_col in row
                    and ci_upper_col in row
                    and not pd.isna(row[ci_lower_col])
                    and not pd.isna(row[ci_upper_col])
                ):

                    err_low = max(0, row[value_col] - row[ci_lower_col])
                    err_high = max(0, row[ci_upper_col] - row[value_col])

                    ax.errorbar(
                        row[value_col],
                        y_pos + bar_width / 2,
                        xerr=[[err_low], [err_high]],
                        fmt="none",
                        color="#2C2C2C",
                        capsize=0.8,
                        linewidth=0.5,
                        zorder=15,
                    )

        # Format
        ax.set_ylim(len(suites) - 0.5, -0.5)
        ax.set_yticks(range(len(suites)))
        ax.set_yticklabels([SUITES[s] for s in suites])
        ax.grid(True, axis="x", zorder=5)
        ax.grid(False, axis="y")

        if xlim:
            ax.set_xlim(xlim)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)

        if self.config.show_legend:
            self._add_legend(plt, configs, exp_config)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

        return output_file

    def _draw_ci_shadows(
        self,
        ax,
        df: pd.DataFrame,
        value_col: str,
        configs: List[str],
        exp_config: ExperimentConfig,
    ):
        global_data = df[df["agg_type"] == Aggregation.GLOBAL_GEO]
        if global_data.empty:
            return

        ranges = []

        for i, config in enumerate(configs):
            config_data = global_data[global_data["configuration"] == config]
            if config_data.empty:
                continue

            row = config_data.iloc[0]
            ci_lower_col = f"{value_col}_lower"
            ci_upper_col = f"{value_col}_upper"

            if (
                ci_lower_col in row
                and ci_upper_col in row
                and not pd.isna(row[ci_lower_col])
                and not pd.isna(row[ci_upper_col])
            ):

                r_min, r_max = row[ci_lower_col], row[ci_upper_col]
                ranges.append((r_min, r_max))
                color_idx = i % len(exp_config.colors)

                # CI shadow
                ax.axvspan(
                    r_min,
                    r_max,
                    facecolor=exp_config.colors[color_idx],
                    alpha=exp_config.alphas[color_idx],
                    edgecolor="none",
                    zorder=2,
                )

                # CI edges
                for edge in [r_min, r_max]:
                    ax.axvline(
                        edge,
                        color=exp_config.outline[color_idx],
                        linewidth=0.5,
                        linestyle="--",
                        alpha=0.5,
                        zorder=3,
                    )

                # Geomean line
                if value_col in row and not pd.isna(row[value_col]):
                    ax.axvline(
                        row[value_col],
                        color=exp_config.geomean[color_idx],
                        linewidth=0.5,
                        zorder=5,
                    )

        # Overlaps
        if len(ranges) >= 2:
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

    def _add_legend(self, ax, configs: List[str], exp_config: ExperimentConfig):
        handles = []
        for i, config in enumerate(configs):
            color_idx = i % len(exp_config.colors)
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=exp_config.colors[color_idx],
                    lw=4,
                    label=str(config.latex),
                )
            )

        ax.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            ncols=2,
            handlelength=1.0,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.15),
        )


def parse_all_ht_data():
    paths = []
    for s in BENCHMARK_SUITES:
        paths.append(Path(f"results/{s}/heaptrack/default"))
        paths.append(Path(f"results/{s}/heaptrack/gcvs-arc"))
        paths.append(Path(f"results/{s}/heaptrack/gcvs-rc"))
        paths.append(Path(f"results/{s}/heaptrack/gcvs-typed-arena"))
        paths.append(Path(f"results/{s}/heaptrack/gcvs-rust-gc"))

    files = sum([glob.glob(str(p / "*.zst")) for p in paths], [])
    df = parse_heaptrack(files)
    df["experiment"] = "gcvs"
    df["metric"] = Metric.MEM_HSIZE_AVG
    return df


def process_gcvs(perf):
    results = perf.results
    results = results[results["configuration"].isin([GCVS.GC])]
    mini_cfg = SimplePlotConfig(figsize=(3.2, 2.5), group_width=0.8, show_legend=False)
    mini = SimplePlotter(mini_cfg)

    for metric, xlim in GCVS_MINI_PLOTS:
        df = results[results["metric"] == metric]
        mini.plot(
            df,
            "gcvs",
            "ratio",
            title=metric.latex,
            xlim=xlim,
            xlabel=r"Relative to \texttt{Arc<T>}/\texttt{Rc<T>}",
            output_file=f"plots/gcvs_{metric.pathname}.svg",
        )


def process_heapsizes(perf):
    results = perf.results
    results = results[~results["configuration"].isin([HeapSize.DEFAULT])]
    mini_cfg = SimplePlotConfig(figsize=(3.2, 2.5), group_width=0.8, show_legend=True)
    mini = SimplePlotter(mini_cfg)

    for metric, xlim in HEAPSIZES_MINI_PLOTS:
        df = results[results["metric"] == metric]
        mini.plot(
            df,
            "heapsizes",
            "ratio",
            title=metric.latex,
            xlim=xlim,
            xlabel=r"",
            output_file=f"plots/heapsizes_{metric.pathname}.svg",
        )


def process_elision(perf):
    results = perf.results
    results = results[results["configuration"].isin([Elision.OPT])]
    mini_cfg = SimplePlotConfig(figsize=(3.2, 2.5), group_width=0.8, show_legend=False)
    mini = SimplePlotter(mini_cfg)

    for metric, xlim in ELISION_MINI_PLOTS:
        df = results[results["metric"] == metric]
        mini.plot(
            df,
            "elision",
            "ratio",
            title=metric.latex,
            xlim=xlim,
            xlabel=r"",
            output_file=f"plots/elision_{metric.pathname}.svg",
        )

    for metric in GCVS_APPENDIX_TABLES:
        col = MetricColumn(
            metric=metric,
            agg_type=Aggregation.INDIVIDUAL,
            header_name=metric.latex,
            value_type="mean",
        )

        perf.mk_individual_table(
            columns=[col],
            suites=["alacritty", "fd", "som-rs-ast"],
            output_file=f"tables/appendix_gcvs_{metric.pathname}_1.tex",
        )

        perf.mk_individual_table(
            columns=[col],
            suites=["ripgrep", "grmtools", "som-rs-bc"],
            output_file=f"tables/appendix_gcvs_{metric.pathname}_1.tex",
        )

    acfg = PlotConfig(
        figsize=(3, 12),
        max_benchmarks_per_plot=22,
        group_width=0.8,
        metric="ratio",
        agg_type=Aggregation.INDIVIDUAL,
        ncols=4,
    )

    appendix = Plotter(acfg)
    for metric, xlim in GCVS_APPENDIX_PLOTS:
        ylabel = (
            f"Relative performance compared to {GCVS.BASELINE.latex} (lower is better)."
        )
        f = f"plots/appendix_gcvs_{metric.pathname}.svg"
        appendix.plot(
            perf.results,
            metric=metric,
            xlim=xlim,
            title=metric.latex,
            ylabel=ylabel,
            output_file=f,
        )


def process_premopt(perf):

    for metric, xlim in PREMOPT_MINI_PLOTS:
        df = results[results["metric"] == metric]
        mini.plot(
            df,
            "premopt",
            "ratio",
            title=metric.latex,
            xlim=xlim,
            xlabel=r"Relative to None",
            output_file=f"plots/premopt_{metric.pathname}.svg",
        )


def process_results():
    Path("plots").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)

    collections, totals, perf = load_suites(BENCHMARK_SUITES)

    htdata = parse_all_ht_data()
    df = pd.concat([totals, perf, htdata], ignore_index=True)
    # Filter out jemalloc results
    # df = df[df["configuration"] != GCVS.BASELINE]
    # df = Results(df).heapsizes.results

    process_gcvs(results.gcvs)
    process_elision(results.elision)
    process_heapsizes(results.heapsizes)
    process_premopt(results.premopt)
