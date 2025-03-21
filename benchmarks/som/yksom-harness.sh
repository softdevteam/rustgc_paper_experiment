#!/bin/sh
SOM="benchmarks/som/SOM"
VM="$(dirname $0)/yksom"
CLASSPATH="--cp $SOM/Smalltalk:$SOM/Examples/Benchmarks/Richards:$SOM/Examples/Benchmarks/DeltaBlue:$SOM/Examples/Benchmarks/NBody:$SOM/Examples/Benchmarks/Json:$SOM/Examples/Benchmarks/GraphSearch:$SOM/Examples/Benchmarks/LanguageFeatures $SOM/Examples/Benchmarks/BenchmarkHarness.som"

if [ -z "${HT}" ]; then
    $VM $CLASSPATH $@
else
    $HT $VM $CLASSPATH $@
fi
