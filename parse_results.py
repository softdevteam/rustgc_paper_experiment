import hashlib
import inspect
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

from build import Measurement, Metric

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def hash_object(obj):
    """Hash DataFrames, arrays, files (Path or str), or anything pickle-able."""
    if (isinstance(obj, (Path, str))) and Path(obj).exists():
        p = Path(obj)
        return f"{p.resolve()}:{p.stat().st_mtime}"
    if isinstance(obj, pd.DataFrame):
        return hashlib.md5(
            pd.util.hash_pandas_object(obj, index=True).values
        ).hexdigest()
    if isinstance(obj, pd.Series):
        return hashlib.md5(
            pd.util.hash_pandas_object(obj, index=True).values
        ).hexdigest()
    if isinstance(obj, np.ndarray):
        return hashlib.md5(obj.tobytes()).hexdigest()
    try:
        return hashlib.md5(pickle.dumps(obj)).hexdigest()
    except Exception:
        return hashlib.md5(str(obj).encode()).hexdigest()


class Cache:
    @staticmethod
    def make_key(static_deps, args, kwargs, func=None):
        key_parts = [hash_object(x) for x in static_deps]
        key_parts += [hash_object(x) for x in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={hash_object(v)}")
        if func is not None:
            try:
                source = inspect.getsource(func)
                key_parts.append(hashlib.md5(source.encode()).hexdigest())
            except Exception:
                pass
        key_str = "|".join(map(str, key_parts))
        return hashlib.md5(key_str.encode()).hexdigest()

    def __call__(self, *static_deps):
        def decorator(func):
            def wrapper(*args, **kwargs):
                key = self.make_key(static_deps, args, kwargs, func=func)
                file = CACHE_DIR / (key + ".pkl")
                if file.exists():
                    print(f"[Cache hit] {func.__name__} → {file.name}")
                    with open(file, "rb") as f:
                        return pickle.load(f)
                print(f"[Cache miss] {func.__name__} → {file.name}")
                result = func(*args, **kwargs)
                with open(file, "wb") as f:
                    pickle.dump(result, f)
                return result

            return wrapper

        return decorator


cache = Cache()


def parse(files, experiment, suite, measurement):
    if measurement == Measurement.PERF:
        return parse_perf(files.pop(), experiment, suite)
    if measurement == Measurement.METRICS:
        df = parse_parallel(parse_metrics, files, experiment, suite)
        print(df)
        return df
    if measurement == Measurement.HEAPTRACK:
        df = parse_parallel(parse_heaptrack, files, experiment, suite)
        print(df)
        return df


def parse_parallel(fn, files, experiment, suite):
    results = []
    with ProcessPoolExecutor(max_workers=24) as executor:
        future_to_file = {
            executor.submit(fn, fp, experiment, suite): fp for fp in files
        }

        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return pd.concat(results, ignore_index=True)


def parse_metrics(f, experiment, suite):
    return parse_metric_file(f, experiment, suite)


def parse_heaptrack(f, experiment, suite):
    # Needed so we can cache the inner call
    return parse_heaptrack_profile(f, experiment, suite)


@cache()
def parse_metric_file(f, experiment, suite):
    path = Path(f)
    benchmark, invocation = path.stem.rsplit("-", 1)
    configuration = experiment(path.parent.name)
    records = []
    expname = experiment.__name__.lower()
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
    gcs = df[df[Metric.COLLECTION_NUMBER] != -1].copy()
    on_exit = df[df[Metric.COLLECTION_NUMBER] == -1].copy()[
        [
            Metric.OBJ_ALLOCD_ARC,
            Metric.OBJ_ALLOCD_BOX,
            Metric.OBJ_ALLOCD_GC,
            Metric.OBJ_ALLOCD_RC,
            Metric.FLZ_REGISTERED,
            Metric.FLZ_RUN,
            Metric.FLZ_ELIDED,
        ]
    ]

    gcs[Metric.MEM_HSIZE_AVG] = gcs[Metric.MEM_HSIZE_EXIT]
    gcs[Metric.TOTAL_COLLECTIONS] = gcs[Metric.COLLECTION_NUMBER]
    gcs = gcs[[Metric.MEM_HSIZE_AVG, Metric.TIME_TOTAL, Metric.TOTAL_COLLECTIONS]].agg(
        {
            Metric.MEM_HSIZE_AVG: "mean",
            Metric.TIME_TOTAL: "sum",
            Metric.TOTAL_COLLECTIONS: "count",
        }
    )

    merged = pd.DataFrame([{**on_exit.iloc[0].to_dict(), **gcs.to_dict()}])
    merged = merged.melt(var_name="metric", value_name="value")

    merged["benchmark"] = benchmark
    merged["configuration"] = configuration
    merged["experiment"] = expname
    merged["invocation"] = invocation

    return merged


@cache()
def parse_perf(file, experiment, suite):

    def to_cfg(name):
        s = name.split("-")[-2:]
        try:
            return experiment(s)
        except:
            return None

    def to_benchmark(name):
        for b in suite.benchmarks:
            if b.name.lower() == name.lower():
                return b
        raise ValueError(f"Benchmark for {name} not found.")

    df = pd.read_csv(
        file,
        sep="\t",
        comment="#",
        index_col="suite",
        converters={
            "criterion": Metric,
            "executor": to_cfg,
        },
    ).dropna(subset="executor")

    df = df.rename(
        columns={"executor": "configuration", "criterion": "metric"}
    ).reset_index()[["benchmark", "configuration", "value", "metric", "invocation"]]

    df["experiment"] = experiment
    return df


class AllocationInfo:
    def __init__(self, size: int, trace_id: int):
        self.size = size
        self.trace_id = trace_id


def parse_heaptrack_profile(file, experiment, suite, snapshot_interval=1000):
    path = Path(file)
    benchmark, invocation = path.stem.rsplit("-", 1)
    configuration = experiment(path.parent.name)
    expname = experiment.__name__.lower()
    with open(file, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            chunks = []
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        decompressed_data = b"".join(chunks)

    content = decompressed_data.decode("utf-8")
    lines = content.strip().split("\n")

    timestamps = []
    heap_sizes = []
    num_allocations_list = []
    allocation_infos = []

    allocation_counts = {}
    allocation_total_size = {}

    current_allocations = 0
    current_total_size = 0
    total_allocations_made = 0
    total_deallocations_made = 0
    timestamp = 0
    event_count = 0

    for line_num, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue

        parts = line.split()
        if not parts:
            continue

        op = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        try:
            if op == "v":
                # Version info - skip
                continue
            elif op == "X":
                # Executable info - skip
                continue
            elif op == "I":
                # System info - skip
                continue
            elif op == "s":
                # String definition - skip
                continue
            elif op == "t":
                # Trace definition - skip
                continue
            elif op == "i":
                # Instruction pointer - skip
                continue
            elif op == "a":
                # Allocation info: a <size> <trace_id>
                if len(args) >= 2:
                    size = int(args[0], 16)
                    trace_id = int(args[1], 16)
                    allocation_infos.append(AllocationInfo(size, trace_id))

            elif op == "+":
                if len(args) >= 1:
                    alloc_info_index = int(args[0], 16)
                    if alloc_info_index < len(allocation_infos):
                        info = allocation_infos[alloc_info_index]

                        allocation_counts[alloc_info_index] = (
                            allocation_counts.get(alloc_info_index, 0) + 1
                        )
                        allocation_total_size[alloc_info_index] = (
                            allocation_total_size.get(alloc_info_index, 0) + info.size
                        )

                        current_allocations += 1
                        current_total_size += info.size
                        total_allocations_made += 1
                        event_count += 1

            elif op == "-":
                if len(args) >= 1:
                    alloc_info_index = int(args[0], 16)
                    if (
                        alloc_info_index in allocation_counts
                        and allocation_counts[alloc_info_index] > 0
                    ):

                        info = allocation_infos[alloc_info_index]
                        allocation_counts[alloc_info_index] -= 1
                        allocation_total_size[alloc_info_index] -= info.size

                        if allocation_counts[alloc_info_index] == 0:
                            del allocation_counts[alloc_info_index]
                            del allocation_total_size[alloc_info_index]

                        current_allocations -= 1
                        current_total_size -= info.size
                        total_deallocations_made += 1
                        event_count += 1

            elif op == "c":
                # Timestamp: c <timestamp>
                if len(args) >= 1:
                    timestamp = int(args[0], 16)

            elif op == "R":
                # RSS info - skip
                continue
            elif op == "A":
                # Attached mode - reset counters
                current_allocations = 0
                current_total_size = 0
                total_allocations_made = 0
                total_deallocations_made = 0
                allocation_counts.clear()
                allocation_total_size.clear()
                event_count = 0
                continue
            else:
                continue

            if event_count > 0 and event_count % snapshot_interval == 0:
                timestamps.append(timestamp)
                heap_sizes.append(current_total_size)
                num_allocations_list.append(current_allocations)

        except (ValueError, IndexError) as e:
            continue

    needs_final_snapshot = event_count > 0 and (
        not timestamps
        or event_count % snapshot_interval != 0
        or heap_sizes[-1] != current_total_size
        or num_allocations_list[-1] != current_allocations
    )

    if needs_final_snapshot:
        timestamps.append(timestamp)
        heap_sizes.append(current_total_size)
        num_allocations_list.append(current_allocations)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heap_size": heap_sizes,
            "num_allocations": num_allocations_list,
            "benchmark": benchmark,
            "invocation": invocation,
            "configuration": configuration,
        }
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    leaked_bytes = sum(allocation_total_size.values())
    num_leaks = sum(allocation_counts.values())

    summary_stats = {
        "num_leaks": num_leaks,
        "leaked_bytes": leaked_bytes,
        "total_allocations": total_allocations_made,
        "total_deallocations": total_deallocations_made,
        "peak_heap_size": df["heap_size"].max() if not df.empty else 0,
        "peak_num_allocations": df["num_allocations"].max() if not df.empty else 0,
        "final_heap_size": current_total_size,
        "final_num_allocations": current_allocations,
    }

    return df
