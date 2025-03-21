#!/bin/sh
SOM="benchmarks/som/SOM"
VM="$(dirname $0)/som-interpreter-bc"
CLASSPATH="-c $SOM/Smalltalk $SOM/Examples/Benchmarks $SOM/Examples/Benchmarks/Richards $SOM/Examples/Benchmarks/DeltaBlue $SOM/Examples/Benchmarks/NBody $SOM/Examples/Benchmarks/Json $SOM/Examples/Benchmarks/GraphSearch $SOM/Examples/Benchmarks/LanguageFeatures"

if [ -z "${HT}" ]; then
    $VM $CLASSPATH $@
else
    $HT $VM $CLASSPATH $@
fi
