#!/bin/sh
BIN="$(dirname $0)/regex_redux"

if [ -z "${HT}" ]; then
    $BIN 0 <benchmarks/regex-redux/redux_input.txt
else
    $HT $BIN 0 <benchmarks/regex-redux/redux_input.txt
fi
