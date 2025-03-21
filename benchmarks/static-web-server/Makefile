GCVS_CFGS := gcvs/gc gcvs/arc
include ../../common.mk

SWS_REPO = https://github.com/static-web-server/static-web-server
SWS_VERSION = 30a640961322d7f8aa2ffa0ab6f409134ed97eea
SWS_SRC = $(PWD)/sws

WRK_REPO = https://github.com/wg/wrk
WRK_VERSION = a211dd5a7050b1f9e8a9870b95513060e72ac4a0
WRK_SRC = $(PWD)/wrk

HYPER_REPO = https://github.com/hyperium/hyper
HYPER_VERSION = f8e2a831943e9a687a1b3277c806caca6e0ac8a4
HYPER_SRC = $(PWD)/hyper

URL ?= "http://localhost:8787"
THREADS ?= 4
CONNECTIONS ?= 100
DURATION ?= 3s
WRK = $(PWD)/wrk/wrk
RESULTS = $(PWD)/results
RESULTS_PROCESSOR = ../process.py

all: build

.PHONY: all build clean

build: $(SWS_SRC) $(HYPER_SRC) $(WRK_SRC) $(TARGETS)

# bench: bench-perf bench-barriers bench-elision
#
# bench-perf: $(PERF)
#
# bench-barriers: $(BARRIERS)
#
# bench-elision: $(ELISION)

plot: plot-perf plot-barriers plot-elision

plot-perf: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/raw_gc.csv \
		$(RESULTS)/raw_arc.csv

plot-barriers: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/raw_barriers_naive.csv \
		$(RESULTS)/raw_barriers_opt.csv \
		$(RESULTS)/raw_barriers_none.csv \

plot-elision: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/raw_finalise_naive.csv \
		$(RESULTS)/raw_finalise_elide.csv

plot-setup: venv
	mkdir -p plots
	$(PYTHON_EXEC) merge_results.py

$(PERF):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

$(BARRIERS):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

$(ELISION):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

$(BIN)/gcvs/%:
	$(call patch_repo,$(SWS_SRC),$(patsubst %/,%,$(dir $*)))
	$(call patch_repo,$(HYPER_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(SWS_SRC),)
	ln -s $(PWD)/sws-harness.sh $@/bin/sws-harness

$(BIN)/%:
	$(call patch_repo,$(SWS_SRC),alloy)
	$(call patch_repo,$(SWS_SRC),hyper)
	$(call build_alloy_exp,$*,$(SWS_SRC),)
	ln -s $(PWD)/sws-harness.sh $@/bin/sws-harness

# build-perf: sws hyper wrk
# 	cd sws && git reset --hard
# 	cd hyper && git reset --hard
# 	cd sws && git apply $(PATCHES)/arc/sws.patch
# 	cd sws && \
# 		RUSTC="../../bin/alloy/alloy/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/arc
# 	cd sws && git reset --hard
# 	cd sws && git apply $(PATCHES)/gc/sws.patch
# 	cd hyper && git reset --hard
# 	cd hyper && git apply $(PATCHES)/gc/hyper.patch
# 	cd sws && \
# 		RUSTC="../../bin/alloy/alloy/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/gc
# 	cd sws && git reset --hard
#
# build-barriers: sws hyper wrk
# 	cd hyper && git reset --hard
# 	cd hyper && git apply $(PATCHES)/gc/hyper.patch
# 	cd sws && git reset --hard
# 	cd sws && git apply $(PATCHES)/gc/sws.patch
# 	cd sws && \
# 		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/barriers_naive
# 	cd sws && \
# 		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/barriers_none
# 	cd sws && \
# 		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/barriers_opt
# 	cd sws && git reset --hard
# 	cd hyper && git reset --hard
#
# build-elision: sws hyper wrk
# 	cd hyper && git reset --hard
# 	cd hyper && git apply $(PATCHES)/gc/hyper.patch
# 	cd sws && git reset --hard
# 	cd sws && git apply $(PATCHES)/gc/sws.patch
# 	cd sws && \
# 		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/finalise_elide
# 	cd sws && \
# 		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
# 		cargo build --release --target-dir=$(PWD)/finalise_naive
# 	cd sws && git reset --hard
# 	cd hyper && git reset --hard

$(SWS_SRC):
	git clone $(SWS_REPO) $@
	cd $@ && git checkout $(SWS_VERSION)

$(HYPER_SRC):
	git clone $(HYPER_REPO) $@
	cd $@ && git checkout $(HYPER_VERSION)

$(WRK_SRC):
	git clone $(WRK_REPO) $@
	cd $@ && git checkout $(WRK_VERSION)
	cd $@ && make -j$(nprocs)

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: ../requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r ../requirements.txt

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean:
	rm -rf $(SWS_SRC) $(WRK_SRC) $(HYPER_SRC)
