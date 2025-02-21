#!/bin/sh

HPATH="../alloy/heaptrack/bin/heaptrack"
EXPERIMENT=$1
CFG=$2
DATAPOINT=$3
BIN=$4
INVOCATION=$5
BENCHMARK=${@: -3:1}
OUTDIR="../../results/$EXPERIMENT/$BIN/"

shift 5

EXEC="$(pwd)/$EXPERIMENT/$CFG/$DATAPOINT/bin/$BIN $@"

if [ "$DATAPOINT" == "perf" ]; then
   ALLOY_LOG="$OUTDIR/metrics/$CFG-$BENCHMARK-$INVOCATION.log" \
       $(pwd)/$EXPERIMENT/$CFG/$DATAPOINT/bin/$BIN $@
else
    htpath="../alloy/heaptrack/bin"
    outfile="$OUTDIR/heaptrack/$CFG.$BENCHMARK"
    export LD_LIBRARY_PATH="../alloy/bdwgc/lib"
    $htpath/heaptrack --record-only -o $outfile $EXEC
    $htpath/heaptrack_print -M $outfile.massif $outfile.zst
fi

