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


class Results:
    def __init__(self, df):
        self.df = df.copy()
        self._arith = None
        self._geo = None
        self._combined = None

    def _arith_mean(self):
        def _amean(self, series):
            mean = series.mean()
            n = len(series)

            if n < 2:
                return pd.Series({"mean": mean, "ci_lower": mean, "ci_upper": mean})

            sem = stats.sem(series)
            alpha = 1 - CONFIDENCE_LEVEL
            h = sem * stats.t.ppf(1 - alpha / 2, n - 1)

            return pd.Series({"mean": mean, "ci_lower": mean - h, "ci_upper": mean + h})

        group_cols = ["experiment", "benchmark", "configuration", "suite", "metric"]
        return (
            self.df.groupby(group_cols)["value"]
            .apply(self._amean)
            .unstack()
            .reset_index()
        )

    def _geo_mean(self):
        def mean(series):
            arr = np.array(series[series > 0])

            if len(arr) == 0:
                return pd.Series(
                    {
                        "geomean": np.nan,
                        "ci_lower_gmean": np.nan,
                        "ci_upper_gmean": np.nan,
                    }
                )

            if len(arr) == 1:
                val = arr[0]
                return pd.Series(
                    {"geomean": val, "ci_lower_gmean": val, "ci_upper_gmean": val}
                )

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
                    "geomean": stats.gmean(arr),
                    "ci_lower_gmean": res.confidence_interval.low,
                    "ci_upper_gmean": res.confidence_interval.high,
                }
            )

        return (
            self.df.groupby(["experiment", "suite", "configuration", "metric"])["value"]
            .apply(mean)
            .unstack()
            .reset_index()
        )

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
    def arith(self):
        if self._arith is None:
            raw = self._arith_mean()
            self._arith = self._add_baseline_comparisons(
                raw,
                ["experiment", "benchmark", "configuration", "suite", "metric"],
                ["mean", "ci_lower", "ci_upper"],
                "mean",
                "ci_lower",
                "ci_upper",
            )
        return self._arith

    @property
    def geo(self):
        if self._geo is None:
            raw = self._geo_mean()
            with_baselines = self._add_baseline_comparisons(
                raw,
                ["experiment", "suite", "configuration", "metric"],
                ["geomean", "ci_lower_gmean", "ci_upper_gmean"],
                "geomean",
                "ci_lower_gmean",
                "ci_upper_gmean",
            )

            distinguishable = self.arith[self.arith["distinguishable"]].copy()

            if not distinguishable.empty:
                group_cols = ["suite", "configuration", "metric"]

                best_idx = distinguishable.groupby(group_cols)["ratio"].idxmin()
                worst_idx = distinguishable.groupby(group_cols)["ratio"].idxmax()

                best = distinguishable.loc[
                    best_idx, group_cols + ["benchmark", "ratio"]
                ].rename(
                    columns={
                        "benchmark": "best_benchmark",
                        "ratio": "best_benchmark_ratio",
                    }
                )
                worst = distinguishable.loc[
                    worst_idx, group_cols + ["benchmark", "ratio"]
                ].rename(
                    columns={
                        "benchmark": "worst_benchmark",
                        "ratio": "worst_benchmark_ratio",
                    }
                )

                with_baselines = with_baselines.merge(
                    best, on=group_cols, how="left"
                ).merge(worst, on=group_cols, how="left")

            self._geo = with_baselines
        return self._geo

    @property
    def combined(self):
        if self._combined is None:
            merge_keys = ["suite", "experiment", "configuration", "metric"]
            geo_renamed = self.geo.rename(
                columns={
                    col: f"{col}_gmean"
                    for col in self.geo.columns
                    if col not in merge_keys
                }
            )

            self._combined = self.arith.merge(
                geo_renamed, on=merge_keys, how="outer", suffixes=("", "_gmean")
            )
        return self._combined

    @property
    def elision(self):
        filtered = self.df[self.df["experiment"] == "elision"].copy()
        return Results(filtered)

    @property
    def gcvs(self):
        filtered = self.df[self.df["experiment"] == "gcvs"].copy()
        return Results(filtered)

    @property
    def premopt(self):
        filtered = self.df[self.df["experiment"] == "premopt"].copy()
        return Results(filtered)

    @property
    def experiments(self):
        return self.df["experiment"].unique().tolist()


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

            df[Metric.HEAP_GREW] = (-df[Metric.BYTES_FREED]).clip(lower=0)
            df[Metric.BYTES_FREED] = df[Metric.BYTES_FREED].clip(lower=0)

            time_cols = [
                Metric.TIME_MARKING,
                Metric.TIME_SWEEPING,
                Metric.TIME_FINALIZER_QUEUE,
                Metric.TIME_TOTAL,
            ]
            time_ms_cols = [f"{col.value}_ms" for col in time_cols]
            time_ns_cols = [f"{col.value}_ns" for col in time_cols]

            for i, col in enumerate(time_cols):
                df[col] = df[time_ms_cols[i]] + df[time_ns_cols[i]] / 1_000_000
            df = df.drop(columns=time_ms_cols + time_ns_cols)

            sentinel = df[Metric.COLLECTION_NUMBER] == -1
            metadata = {
                "invocation": sentinel.cumsum().shift(fill_value=0) + 1,
                "benchmark": benchmark,
                "configuration": configuration,
            }
            df = df.assign(**metadata)

            per_collection = df[~sentinel].drop(columns=[Metric.PHASE])
            summary_rows = df[sentinel]

            totals = [
                Metric.ALLOCATED_ARC,
                Metric.ALLOCATED_BOXED,
                Metric.ALLOCATED_RC,
                Metric.ALLOCATED_GC,
                Metric.FINALIZERS_REGISTERED,
                Metric.FINALIZERS_RUN,
                Metric.OBJECTS_IN_FINALIZER_QUEUE,
                Metric.LIVE_OBJECTS_WITH_FINALIZERS,
                Metric.TIME_MARKING,
                Metric.TIME_SWEEPING,
                Metric.TIME_FINALIZER_QUEUE,
                Metric.TIME_TOTAL,
            ]

            metadata_cols = list(metadata.keys())
            totals = summary_rows[metadata_cols + totals]

            max_collections = per_collection.groupby("invocation", as_index=False)[
                Metric.COLLECTION_NUMBER
            ].max()
            max_collections = max_collections.assign(
                metric=Metric.TOTAL_COLLECTIONS,
                benchmark=benchmark,
                configuration=configuration,
                suite=suite,
            ).rename(columns={Metric.COLLECTION_NUMBER: "value"})

            totals = pd.concat(
                [
                    totals.melt(
                        id_vars=metadata_cols, var_name="metric", value_name="value"
                    ),
                    max_collections,
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

        results = [_process_metric_csv(f) for f in csvs]
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

    explode_to = [GCVS.GC, PremOpt.OPT, Elision.OPT]

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


def mk_gc_breakdown_table(results, metrics=None, output_file="plots/gc_breakdown.tex"):
    if metrics is None:
        metrics = [
            Metric.FINALIZERS_REGISTERED,
            Metric.TOTAL_COLLECTIONS,
            Metric.TIME_MARKING,
            Metric.TIME_TOTAL,
        ]

    df = results.arith

    def fmt_float(x, digits=1, bold=False):
        s = "-" if pd.isna(x) else f"{x:.{digits}f}"
        return f"\\textbf{{{s}}}" if bold else s

    def fmt_ratio(x, digits=2, distinguishable=True):
        s = "-" if pd.isna(x) else f"{x:.{digits}f}×"
        if not distinguishable:
            return f"{s}^\\dag"
        else:
            return s

    def fmt_ci_symmetric(mean_val, ci_lower, ci_upper, digits=1):
        if pd.isna(mean_val) or pd.isna(ci_lower) or pd.isna(ci_upper):
            return ""
        ci_range = (ci_upper - ci_lower) / 2
        return f"\\scriptsize\\color{{gray!80}}{{±{ci_range:.{digits}f}}}"

    col_spec = "c l " + "r@{\\hspace{0.5em}}l " * len(metrics)

    header1 = (
        "\\multirow{2}{*}{Suite} & \\multirow{2}{*}{Benchmark} & "
        "\\multicolumn{2}{c}{Finalizers Registered} & "
        "\\multicolumn{2}{c}{Collections} & "
        "\\multicolumn{2}{c}{Mark Phase} & "
        "\\multicolumn{2}{c}{Total}"
    )

    header2 = " &  & Before & Ratio & Baseline & Current "
    for metric in metrics[2:]:
        header2 += " & Before (s) & Ratio"

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header1 + r" \\",
        header2 + r" \\",
        r"\midrule",
    ]

    baseline = df["configuration"].iloc[0].baseline
    df = df[df["configuration"] != baseline]

    suite_groups = list(df.groupby("suite"))

    for suite_idx, (suite_name, suite_data) in enumerate(suite_groups):
        suite_benchmarks = suite_data["benchmark"].unique()

        for i, benchmark in enumerate(suite_benchmarks):
            benchmark_data = suite_data[suite_data["benchmark"] == benchmark]

            if i == 0:
                suite_cell = f"\\multirow{{{len(suite_benchmarks)}}}{{*}}[0pt]{{\\rotatebox{{90}}{{{suite_name}}}}}"
            else:
                suite_cell = ""

            row_data = [suite_cell, benchmark]

            finalizers_data = benchmark_data[benchmark_data["metric"] == metrics[0]]
            if not finalizers_data.empty:
                row = finalizers_data.iloc[0]
                baseline_val = fmt_float(row["baseline_mean"])
                baseline_ci = fmt_ci_symmetric(
                    row["baseline_mean"],
                    row["baseline_ci_lower"],
                    row["baseline_ci_upper"],
                )
                baseline_cell = f"{baseline_val}{baseline_ci}"

                is_distinguishable = row["distinguishable"]
                ratio_val = fmt_ratio(row["ratio"], distinguishable=is_distinguishable)
                ratio_ci = fmt_ci_symmetric(
                    row["ratio"], row["ratio_lower"], row["ratio_upper"], digits=2
                )
                ratio_cell = f"{ratio_val}{ratio_ci}"

                row_data.extend([baseline_cell, ratio_cell])
            else:
                row_data.extend(["-", "-"])

            collections_data = benchmark_data[benchmark_data["metric"] == metrics[1]]
            if not collections_data.empty:
                row = collections_data.iloc[0]
                baseline_collections = (
                    f"{int(row['baseline_mean'])}"
                    if not pd.isna(row["baseline_mean"])
                    else "-"
                )
                current_collections = (
                    f"{int(row['mean'])}" if not pd.isna(row["mean"]) else "-"
                )
                row_data.extend([baseline_collections, current_collections])
            else:
                row_data.extend(["-", "-"])

            for metric in metrics[2:]:
                metric_data = benchmark_data[benchmark_data["metric"] == metric]

                if metric_data.empty:
                    row_data.extend(["-", "-"])
                    continue

                row = metric_data.iloc[0]

                baseline_mean_sec = (
                    row["baseline_mean"] / 1000
                    if not pd.isna(row["baseline_mean"])
                    else None
                )
                baseline_ci_lower_sec = (
                    row["baseline_ci_lower"] / 1000
                    if not pd.isna(row["baseline_ci_lower"])
                    else None
                )
                baseline_ci_upper_sec = (
                    row["baseline_ci_upper"] / 1000
                    if not pd.isna(row["baseline_ci_upper"])
                    else None
                )

                baseline_val = fmt_float(baseline_mean_sec, digits=2)
                baseline_ci = fmt_ci_symmetric(
                    baseline_mean_sec,
                    baseline_ci_lower_sec,
                    baseline_ci_upper_sec,
                    digits=2,
                )
                baseline_cell = f"{baseline_val}{baseline_ci}"

                is_distinguishable = row["distinguishable"]
                ratio_val = fmt_ratio(row["ratio"], distinguishable=is_distinguishable)
                ratio_ci = fmt_ci_symmetric(
                    row["ratio"], row["ratio_lower"], row["ratio_upper"], digits=2
                )
                ratio_cell = f"{ratio_val}{ratio_ci}"

                row_data.extend([baseline_cell, ratio_cell])

            lines.append(" & ".join(row_data) + r" \\")

        if suite_idx < len(suite_groups) - 1:
            total_cols = 2 + 2 * len(metrics)
            lines.append(f"\\cmidrule{{1-{total_cols}}}")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


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


def process_gcvs(results):
    gcvs_results = results.gcvs
    if gcvs_results.df.empty:
        return

    mk_gc_breakdown_table(gcvs_results, output_file="plots/gcvs_table.tex")

    plotter = Plotter(savepath="plots/gcvs_plot.svg")
    gc_metrics = [Metric.TIME_MARKING, Metric.TIME_SWEEPING, Metric.TIME_TOTAL]
    gc_data = gcvs_results.arith[gcvs_results.arith["metric"].isin(gc_metrics)]

    if not gc_data.empty:
        plotter.plot(gc_data, y="ratio", experiment="gcvs")


def process_elision(results):
    elision_results = results.elision
    if elision_results.df.empty:
        return

    mk_gc_breakdown_table(elision_results, output_file="plots/elision_table.tex")

    plotter = Plotter(savepath="plots/elision_plot.svg")
    if not elision_results.arith.empty:
        plotter.plot(elision_results.arith, y="ratio", experiment="elision")


def process_premopt(results):
    premopt_results = results.premopt
    if premopt_results.df.empty:
        return

    mk_gc_breakdown_table(premopt_results, output_file="plots/premopt_table.tex")

    plotter = Plotter(savepath="plots/premopt_plot.svg")
    if not premopt_results.arith.empty:
        plotter.plot(premopt_results.arith, y="ratio", experiment="premopt")


Path("plots").mkdir(exist_ok=True)

collections_data, totals_data, perf_data = load_suites(BENCHMARK_SUITES)
results = Results(totals_data)

process_gcvs(results)
process_elision(results)
process_premopt(results)
