#!/bin/sh
BIN="$(dirname $0)/alacritty"
BM_DIR="benchmarks/alacritty/vtebench/benchmarks/$@"

if [ -e "$BM_DIR/setup" ]; then
    CMD="($BM_DIR/setup;$BM_DIR/benchmark)"
else
    CMD=$BM_DIR/benchmark
fi

Xvfb :99 -ac -screen 0 1024x268x24 &
xvfb_pid=$!
#
trap "kill -9 $xvfb_pid; exit" EXIT
#
export DISPLAY=":99"

if [ -z "${HT}" ]; then
    $BIN -e /bin/bash -c $CMD
else
    $HT $BIN -e /bin/bash -c $CMD
fi
