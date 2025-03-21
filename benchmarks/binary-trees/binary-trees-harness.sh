#!/bin/sh
BIN="$(dirname $0)/binary_trees"

if [ -z "${HT}" ]; then
    $BIN $@
else
    $HT $BIN $@
fi
