#!/bin/sh

# Rebench does not let us templatize environment variable values, so this hack
# lets us forward onto the actual command while setting an environment.
bin="$1"
cfg="$2"
benchmark="$3"
invocation="$4"
shift 4

logfile="../../results/premopt/yksom/metrics/$cfg-$benchmark-$invocation.log"

echo $bin $@ > test.log

ALLOY_LOG=${logfile} $bin $@


