#!/bin/env bash

SERVER="$(dirname $0)/static-web-server"
WWW="$(dirname $0)/www"

CONNECTIONS=100
THREADS=4
DURATION="3s"
WRK="benchmarks/static-web-server/wrk/wrk"
URL="http://localhost:8787"
BM_COLLECTOR="benchmarks/static-web-server/benchmark_collector.lua"

mkdir -p "$OUTDIR/wrk"

$SERVER -p 8787 -d $WWW &
http_pid=$!

sleep 1
$WRK -c $CONNECTIONS -t $THREADS -d $DURATION --latency -s $BM_COLLECTOR $URL
# exit 1

kill $http_pid
