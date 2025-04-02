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

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean:
	rm -rf $(SWS_SRC) $(WRK_SRC) $(HYPER_SRC)
