#!/bin/sh
BIN="$(dirname $0)/fd"

if [ -z "${HT}" ]; then
    $BIN $@
else
    $HT $BIN $@
fi
