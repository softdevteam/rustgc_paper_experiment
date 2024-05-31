#! /bin/sh

set -e

ALLOYSV=c50c1767de51e8354a90cb4e680330182e3eb921
SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
YKSOMSV=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6
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

if [ ! -d yksom ]; then
    git clone --recursive https://github.com/softdevteam/yksom
    cd yksom
    git checkout ${YKSOMSV}
    cd ..
fi

build_alloy_barriers() {
    cd alloy
    echo "===> building Alloy (all finalisation barriers)"
    git checkout -- .
    git apply ../configs/alloy/all_barriers.patch
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_all_barriers
    rustup toolchain link alloy_all_barriers alloy_all_barriers/x86_64-unknown-linux-gnu/stage1/
    echo "===> building Alloy (opt finalisation barriers)"
    git checkout -- .
    git apply ../configs/alloy/opt_barriers.patch
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_opt_barriers
    rustup toolchain link alloy_opt_barriers alloy_opt_barriers/x86_64-unknown-linux-gnu/stage1/
    echo "===> building Alloy (no finalisation barriers)"
    git checkout -- .
    git apply ../configs/alloy/no_barriers.patch
    python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_no_barriers
    rustup toolchain link alloy_no_barriers alloy_no_barriers/x86_64-unknown-linux-gnu/stage1/
    cd ..
}

build_som_rs_barriers() {
    git checkout -- .
    git apply ../configs/som-rs/use_gc.patch
    echo "===> building som-rs (alloy + all finalisation barriers)"
    cargo +alloy_all_barriers build --release -p som-interpreter-bc --target-dir=all_barriers
    echo "===> building som-rs (alloy + no finalisation barriers)"
    cargo +alloy_no_barriers build --release -p som-interpreter-bc --target-dir=no_barriers
    echo "===> building som-rs (alloy + optimised finalisation barriers)"
    cargo +alloy_opt_barriers build --release -p som-interpreter-bc --target-dir=opt_barriers
    cd ..
}

# ============= Build Alloy with each configuration =====================
if [ ! -f alloy.lock ]; then
    # Build full Alloy
    # git checkout -- .
    # echo "===> building Alloy (full)"
    # python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_full
    # rustup toolchain link alloy_full alloy_full/x86_64-unknown-linux-gnu/stage1/
    # # Build Alloy with naive finalisation
    # git checkout -- .
    # git apply ../configs/alloy/naive_finalisation.patch
    # echo "===> building Alloy (naive finalisation)"
    # python3 x.py build --stage 1 --config ../benchmark.config.toml --build-dir=alloy_naive_finalisation
    # rustup toolchain link alloy_naive_finalisation alloy_naive_finalisation/x86_64-unknown-linux-gnu/stage1/
    build_alloy_barriers
    touch alloy.lock
fi

# ============= Build som-rs with each configuration =====================
if [ ! -f som-rs.lock ]; then
    build_som_rs_barriers
    # cd som-rs
    # # Nightly rustc with reference counting (baseline)
    # echo "===> building som-rs (nightly + rc)"
    # git checkout -- .
    # cargo +nightly build --release -p som-interpreter-bc --target-dir=rc
    # # Alloy full gc
    # echo "===> building som-rs (alloy)"
    # git checkout -- .
    # git apply ../configs/som-rs/use_gc.patch
    # cargo +alloy_full build --release -p som-interpreter-bc --target-dir=gc
    # echo "===> building som-rs (for finalisation elision test)"
    # git checkout -- .
    # git apply ../configs/som-rs/use_gc.patch
    # git apply ../configs/som-rs/optimised_finalisation.patch
    # cargo +alloy_full build --release -p som-interpreter-bc --target-dir=finaliser_elision
    # # Rc but with boehm allocator
    # echo "===> building som-rs (RC + bdwgcalloc)"
    # git checkout -- .
    # git apply ../configs/som-rs/bdwgc_allocator.patch
    # cargo +alloy_full build --release -p som-interpreter-bc --target-dir=boehm_rc
    # rm boehm_rc.sh
    # # Alloy naive finalisation
    # echo "===> building som-rs (alloy + naive finalisation)"
    # git checkout -- .
    # git apply ../configs/som-rs/use_gc.patch
    # git apply ../configs/som-rs/naive_finalisation.patch
    # cargo +alloy_naive_finalisation build --release -p som-interpreter-bc --target-dir=naive_finalisation
    # # Alloy no barriers
    # echo "===> building som-rs (alloy + no finalisation barriers)"
    # git checkout -- .
    # git apply ../configs/som-rs/use_gc.patch
    # cargo +alloy_no_barriers build --release -p som-interpreter-bc --target-dir=no_barriers
    # # Alloy all barriers
    # cd ..
    # touch som-rs.lock
fi
