#!/bin/sh

CFG=$1
INVOCATION=$2
BENCHMARK_SUITE=$3
BIN=$4
BENCHMARK=$5
shift 5

OUTDIR="results/$EXPERIMENT/$BENCHMARK_SUITE/metrics/runtime"

if [ "$EXPERIMENT" == "gcvs" ] && [ "$CFG" != "gc" ]; then
    export GC_DONT_GC=true
fi

LOGFILE="$BIN.$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"
ALLOY_LOG="$OUTDIR/$LOGFILE" "benchmarks/$BENCHMARK_SUITE/bin/$EXPERIMENT/$CFG/perf/bin/$BIN" "$@"
