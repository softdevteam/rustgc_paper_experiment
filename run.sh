#! /bin/sh

set -e

mkdir -p raw_data
# =========== Run som-rs experiments ========
cd som-rs
# Perf
git checkout -- .
git apply ../configs/som-rs/rebench_perf.patch
rebench --experiment all rebench.conf all || true
mv rebench.data ../raw_data/som-rs-perf.data

# Finaliser elision
git checkout -- .
git apply ../configs/som-rs/rebench_finaliser_elision.patch
mkdir -p naive_counts
mkdir -p elision_counts
rebench --experiment all rebench.conf all || true
mv rebench.data ../raw_data/som-rs-finaliser_elision.data
mv naive_counts ../raw_data/
mv elision_counts ../raw_data/

# Earlier finalisation / barriers
git checkout -- .
git apply ../configs/som-rs/rebench_barriers.patch
rebench --experiment all rebench.conf all || true
mv rebench.data ../raw_data/som-rs-barriers.data
