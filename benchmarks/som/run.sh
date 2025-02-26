#!/bin/sh

CFG=$1
INVOCATION=$2
shift 2

BENCHMARK=${@: -3:1}
OUTDIR="../../results/$EXPERIMENT/$BIN"
if [ "$BIN" = "yksom" ]; then
    BIN="$(pwd)/$BIN/$EXPERIMENT/$CFG/$EXPTYPE/bin/$BIN $@"
else
    BIN="$(pwd)/som-rs/$EXPERIMENT/$CFG/$EXPTYPE/bin/$BIN $@"
fi;

export LD_LIBRARY_PATH="../../bdwgc/lib"

if [ "$EXPTYPE" = "perf" ]; then
    METRICS_LOGFILE="$OUTDIR/metrics/$INVOCATION.$CFG.$BENCHMARK.csv"
    ALLOY_LOG="$METRICS_LOGFILE" $BIN
else
    HTPATH="../../heaptrack/bin"
    SAMPLER="../../venv/bin/python ../../sample_memory.py"
    HTDATA="$OUTDIR/heaptrack/$INVOCATION.$CFG.$BENCHMARK"
    SAMPLERDATA="$OUTDIR/samples/$INVOCATION.$CFG.$BENCHMARK.csv"
    $HTPATH/heaptrack --record-only -o $HTDATA $BIN
    $HTPATH/heaptrack_print -M $HTDATA.massif $HTDATA.zst
    $SAMPLER -o $SAMPLERDATA $BIN
fi;

