#!/bin/sh
BIN="$(dirname $0)/parserbench"
GRMTOOLS_DIR="benchmarks/grmtools"

if [ -z "${HT}" ]; then
    $BIN $GRMTOOLS_DIR/$@
else
    $HT $BIN $GRMTOOLS_DIR/$@
fi
