#! /bin/sh

set -e

ALLOYSV=c50c1767de51e8354a90cb4e680330182e3eb921
SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
WLAMBDASV=b370e342fdffeae6f

if [ ! -d alloy ]; then
    git clone https://github.com/softdevteam/alloy
    cd alloy
    git checkout ${ALLOYSV}
    cd ..
fi

if [ ! -d som-rs ]; then
    git clone --recursive https://github.com/Hirevo/som-rs
    cd som-rs
    git checkout ${SOMRSSV}
    cd ..
fi

if [ ! -d WLambda ]; then
    git clone --recursive https://github.com/WeirdConstructor/WLambda
    cd WLambda
    git checkout ${WLAMBDASV}
    cd ..
fi

# ============= Build Alloy with each configuration =====================
if [ ! -f alloy.lock ]; then
    cd alloy
    # Build full Alloy
    git checkout -- .
    echo "===> building Alloy (full)"
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_full
    rustup toolchain link alloy_full alloy_full/x86_64-unknown-linux-gnu/stage1/
    # Build Alloy with naive finalisation
    git checkout -- .
    git apply ../configs/alloy/naive_finalisation.patch
    echo "===> building Alloy (naive finalisation)"
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_naive_finalisation
    rustup toolchain link alloy_naive_finalisation alloy_naive_finalisation/x86_64-unknown-linux-gnu/stage1/
    # Build Alloy with no barriers (unsound)
    git checkout -- .
    git apply ../configs/alloy/no_barriers.patch
    echo "===> building Alloy (no finalisation barriers)"
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_no_barriers
    rustup toolchain link alloy_no_barriers alloy_no_barriers/x86_64-unknown-linux-gnu/stage1/
    # Build Alloy with all barriers
    git checkout -- .
    git apply ../configs/alloy/all_barriers.patch
    echo "===> building Alloy (all finalisation barriers)"
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_all_barriers
    rustup toolchain link alloy_all_barriers alloy_all_barriers/x86_64-unknown-linux-gnu/stage1/
    # Once built, create a lock file to prevent us from rebuilding it.
    cd ../
    touch alloy.lock
fi

# ============= Build som-rs with each configuration =====================
cd som-rs
# Nightly rustc with reference counting (baseline)
echo "===> building som-rs (nightly + rc)"
git checkout -- .
cargo +nightly build --release -p som-interpreter-bc --target-dir=rc
# Alloy full gc
echo "===> building som-rs (alloy)"
git checkout -- .
git apply ../configs/som-rs/use_gc.patch
cargo +alloy_full build --release -p som-interpreter-bc --target-dir=gc
echo "===> building som-rs (for finalisation elision test)"
git checkout -- .
git apply ../configs/som-rs/use_gc.patch
git apply ../configs/som-rs/optimised_finalisation.patch
cargo +alloy_full build --release -p som-interpreter-bc --target-dir=finaliser_elision
# Alloy naive finalisation
echo "===> building som-rs (alloy + naive finalisation)"
git checkout -- .
git apply ../configs/som-rs/use_gc.patch
git apply ../configs/som-rs/naive_finalisation.patch
cargo +alloy_naive_finalisation build --release -p som-interpreter-bc --target-dir=naive_finalisation
# Alloy no barriers
echo "===> building som-rs (alloy + no finalisation barriers)"
git checkout -- .
git apply ../configs/som-rs/use_gc.patch
cargo +alloy_no_barriers build --release -p som-interpreter-bc --target-dir=no_barriers
# Alloy all barriers
echo "===> building som-rs (alloy + all finalisation barriers)"
git checkout -- .
git apply ../configs/som-rs/use_gc.patch
cargo +alloy_all_barriers build --release -p som-interpreter-bc --target-dir=all_barriers

