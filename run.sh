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
cd ..

# =========== Run yksom experiments ========
cd yksom

# Finaliser elision
git checkout -- .
cp ../configs/yksom/rebench_finaliser_elision.conf rebench.conf
mkdir -p yksom_naive_counts
mkdir -p yksom_elision_counts
rebench --experiment all rebench.conf all || true
mv rebench.data ../raw_data/yksom_finaliser_elision.data
mv yksom_naive_counts ../raw_data/
mv yksom_elision_counts ../raw_data/

# # Earlier finalisation / barriers
git checkout -- .
cp ../configs/yksom/rebench_barriers.conf rebench.conf
rebench --experiment all rebench.conf all || true
mv rebench.data ../raw_data/yksom_barriers.data
cd ..
