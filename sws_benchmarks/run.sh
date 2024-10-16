#!/bin/sh
for num in $(seq 1 $PEXECS)
do
	sws/arc/release/static-web-server -p 8787 -d www/ &
	http_pid=$!

	sleep 1
	$1 -c $2 -t $3 -d $4 --latency -s benchmark_collector.lua $5
	kill $http_pid
done
