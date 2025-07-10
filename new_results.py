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
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._arith = None
        self._geo = None
        self._combined = None

    def _arith_mean(self) -> pd.DataFrame:
        def mean(series: pd.Series) -> pd.Series:
            mean = series.mean()
            n = len(series)

            if n < 2:
                return pd.Series({"mean": mean, "ci_lower": mean, "ci_upper": mean})

            sem = stats.sem(series)
            alpha = 1 - CONFIDENCE_LEVEL
            h = sem * stats.t.ppf(1 - alpha / 2, n - 1)

            return pd.Series({"mean": mean, "ci_lower": mean - h, "ci_upper": mean + h})

        return (
            self.df.groupby(
                ["experiment", "benchmark", "configuration", "suite", "metric"]
            )["value"]
            .apply(mean)
            .unstack()
            .reset_index()
        )

    def _geo_mean(self) -> pd.DataFrame:
        def mean(series: pd.Series) -> pd.Series:
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
                method="BCa",
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

    def _attach_baselines_and_ratios(
        self,
        df: pd.DataFrame,
        group_keys: list,
        stat_cols: list,
        main_col: str,
        lower_col: str,
        upper_col: str,
    ) -> pd.DataFrame:
        def process_row(row):
            baseline_config = row["configuration"].baseline

            mask = [df[k] == row[k] for k in group_keys if k != "configuration"]
            mask.append(df["configuration"] == baseline_config)
            mask = np.logical_and.reduce(mask)

            baseline_row = df[mask]
            if baseline_row.empty:
                baseline_stats = {f"baseline_{col}": np.nan for col in stat_cols}
            else:
                base = baseline_row.iloc[0]
                baseline_stats = {f"baseline_{col}": base[col] for col in stat_cols}

            if not np.isnan(baseline_stats[f"baseline_{main_col}"]):
                ratio = row[main_col] / baseline_stats[f"baseline_{main_col}"]
                ratio_upper = row[upper_col] / baseline_stats[f"baseline_{lower_col}"]
                ratio_lower = row[lower_col] / baseline_stats[f"baseline_{upper_col}"]

                distinguishable = not (
                    np.isnan(row[lower_col])
                    or np.isnan(baseline_stats[f"baseline_{upper_col}"])
                ) and (
                    (row[lower_col] > baseline_stats[f"baseline_{upper_col}"])
                    or (row[upper_col] < baseline_stats[f"baseline_{lower_col}"])
                )
            else:
                ratio = ratio_upper = ratio_lower = np.nan
                distinguishable = False

            return pd.Series(
                {
                    **baseline_stats,
                    "ratio": ratio,
                    "ratio_upper": ratio_upper,
                    "ratio_lower": ratio_lower,
                    "distinguishable": distinguishable,
                }
            )

        processed = df.apply(process_row, axis=1)
        return pd.concat([df, processed], axis=1)

    @property
    def arith(self) -> pd.DataFrame:
        if self._arith is None:
            raw = self._arith_mean()
            self._arith = self._attach_baselines_and_ratios(
                raw,
                ["experiment", "benchmark", "configuration", "suite", "metric"],
                ["mean", "ci_lower", "ci_upper"],
                "mean",
                "ci_lower",
                "ci_upper",
            )
        return self._arith

    @property
    def geo(self) -> pd.DataFrame:
        if self._geo is None:
            raw = self._geo_mean()
            with_baselines = self._attach_baselines_and_ratios(
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
    def combined(self) -> pd.DataFrame:
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
    def elision(self) -> "Results":
        filtered = self.df[self.df["experiment"] == "elision"].copy()
        return Results(filtered)

    @property
    def gcvs(self) -> "Results":
        filtered = self.df[self.df["experiment"] == "gcvs"].copy()
        return Results(filtered)

    @property
    def premopt(self) -> "Results":
        filtered = self.df[self.df["experiment"] == "premopt"].copy()
        return Results(filtered)

    @property
    def experiments(self) -> list:
        return self.df["experiment"].unique().tolist()


class Plotter:

    EXPERIMENT = {
        "elision": {
            "bar_cols": ["#27AE60", "#36d278"],
            "outline_cols": ["#186d3c", "#229854"],
            "ci_alphas": [0.08, 0.16],
        }
    }

    # General styling bits to play around with
    GEOLINE_WIDTH = 0.5
    GEOLINE_STYLE = "--"
    GEOCI_ALPHA = 0.13
    ERROR_BAR_COLOR = "black"
    ERROR_BAR_CAP_SIZE = 1
    ERROR_BAR_LINE_WIDTH = 0.5
    BAR_ALPHA = 0.8
    LEGEND_NCOLS = 2
    LEGEND_LOCATION = "upper center"
    LEGEND_BBOX_ANCHOR = (0.5, 1.05)

    def __init__(
        self,
        ci_lower="ratio_lower",
        ci_upper="ratio_upper",
        figsize=(2, 2),
        title=None,
        ylabel=None,
        xlabel=None,
        metric_gap=0.3,
        group_width=0.6,
        savepath="plots/test.svg",
    ):
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.figsize = figsize
        self.title = title
        self.ylabel = ylabel or ""
        self.xlabel = xlabel
        self.metric_gap = metric_gap
        self.group_width = group_width
        self.savepath = savepath

    def plot(self, df, y="mean", metrics=None, savepath=None, experiment=None):
        colours = self.EXPERIMENT[experiment]["bar_cols"]
        outlines = self.EXPERIMENT[experiment]["outline_cols"]
        ci_alphas = self.EXPERIMENT[experiment]["ci_alphas"]
        if savepath is not None:
            self.savepath = savepath
        baseline_config = df["configuration"].iloc[0].baseline
        baseline = df[df["configuration"] == baseline_config]
        df = df[df["configuration"] != baseline_config]

        if metrics is not None:
            df = df[df["metric"].isin(metrics)]
            baseline = baseline[baseline["metric"].isin(metrics)]

        benchmarks = list(df["benchmark"].unique())
        configs = list(df["configuration"].unique())
        metrics = list(df["metric"].unique())

        n_metrics = len(metrics)
        metric_width = self.group_width / (
            n_metrics + (n_metrics - 1) * self.metric_gap
        )
        gap_width = self.metric_gap * metric_width
        bar_width = metric_width / len(configs)

        fig, ax = plt.subplots(figsize=self.figsize)
        grouped = df.groupby(["metric", "benchmark", "configuration"])

        ax.axhline(
            y=1,
            color="black",
            linewidth=self.GEOLINE_WIDTH,
        )

        for i_crit, crit in enumerate(metrics):
            color = colours[i_crit]
            outline_color = outlines[i_crit]
            outline_color_rgb = mcolors.to_rgb(outline_color)
            darker_color = tuple(c * 0.7 for c in outline_color_rgb) + (1.0,)
            alpha = ci_alphas[i_crit]

            bl = baseline[baseline["metric"] == crit]
            bl_row = bl.iloc[0]

            ax.axhspan(
                ymin=bl_row["ratio_lower_gmean"],
                ymax=bl_row["ratio_upper_gmean"],
                color=darker_color,
                alpha=alpha,
            )

            crit_data = df[df["metric"] == crit]
            gmean_row = crit_data.iloc[0]

            ax.axhspan(
                ymin=gmean_row["ratio_lower_gmean"],
                ymax=gmean_row["ratio_upper_gmean"],
                color=darker_color,
                alpha=self.GEOCI_ALPHA,
            )
            ax.axhline(
                y=gmean_row["ratio_gmean"],
                color=darker_color,
                linestyle=self.GEOLINE_STYLE,
                linewidth=self.GEOLINE_WIDTH,
            )

            for i_bench, bench in enumerate(benchmarks):
                x_start = i_bench - self.group_width / 2
                current_x = x_start + i_crit * (metric_width + gap_width)

                for i_cfg, cfg in enumerate(configs):
                    row = grouped.get_group((crit, bench, cfg)).iloc[0]

                    bar_height = row[y]
                    err_low = bar_height - row[self.ci_lower]
                    err_high = row[self.ci_upper] - bar_height
                    x_pos = current_x + i_cfg * bar_width

                    label = crit.latex if (i_cfg == 0 and i_bench == 0) else None

                    ax.bar(
                        x_pos,
                        bar_height,
                        width=bar_width,
                        color=color,
                        label=label,
                        align="edge",
                        alpha=self.BAR_ALPHA,
                        edgecolor=outline_color,
                    )

                    ax.errorbar(
                        x_pos + bar_width / 2,
                        bar_height,
                        yerr=[[err_low], [err_high]],
                        fmt="none",
                        c=self.ERROR_BAR_COLOR,
                        capsize=self.ERROR_BAR_CAP_SIZE,
                        lw=self.ERROR_BAR_LINE_WIDTH,
                    )

        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels(benchmarks)
        ax.set_ylabel(self.ylabel)

        legend = ax.legend(
            ncols=self.LEGEND_NCOLS,
            loc=self.LEGEND_LOCATION,
            bbox_to_anchor=self.LEGEND_BBOX_ANCHOR,
        )
        legend.set_zorder(20)

        ax.grid(True, axis="y", which="major")
        ax.grid(False, axis="x")
        ax.minorticks_off()

        plt.tight_layout(pad=0.2)
        plt.savefig(self.savepath)
        plt.close(fig)

        return self.savepath


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


def parse_all_benchmark_data():
    """Parse both metrics and perf data with unified processing."""

    def _parse_metrics():
        def _process_metric_csv(f):
            path = Path(f)
            benchmark = path.stem.split("-")[-1]
            configuration = to_cfg(f"grmtools-{'-'.join(path.stem.split('-')[:-1])}")

            df = pd.read_csv(f, dtype={Metric.PHASE: "string"})
            df = df[~(df == df.columns).all(axis=1)]
            df.columns = [Metric._value2member_map_.get(c, c) for c in df.columns]

            numeric_cols = [col for col in df.columns if col != Metric.PHASE]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            # Fix negative bytes_freed -> heap_grew
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
                suite="grmtools",
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

        path = Path("results/grmtools/metrics")
        csvs = glob.glob(str(path / "*.csv"))

        results = [_process_metric_csv(f) for f in csvs]
        gc_metrics, summary_metrics = zip(*results)

        return pd.concat(gc_metrics, ignore_index=True), pd.concat(
            summary_metrics, ignore_index=True
        )

    def _parse_perf():

        suite = "grmtools"

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

    metrics_per_collection, metrics_summary = _parse_metrics()
    perf_data = _parse_perf()

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

        datasets[name]["suite"] = "grmtools"
        datasets[name]["experiment"] = datasets[name]["configuration"].apply(
            lambda x: x.experiment
        )

        elision = datasets[name]
        elision = elision[elision["configuration"] == Elision.OPT]

    return (
        datasets["metrics_per_collection"],
        datasets["metrics_summary"],
        datasets["perf_data"],
    )


collections, totals, perf = parse_all_benchmark_data()
elision = Results(perf).elision.combined

plotter = Plotter(
    figsize=(3.5, 3),
    ylabel="Performance Ratio (lower is better)",
)

plotter.plot(
    df=elision,
    y="ratio",
    metrics=[Metric.WALLCLOCK, Metric.USER],
    experiment="elision",
)
