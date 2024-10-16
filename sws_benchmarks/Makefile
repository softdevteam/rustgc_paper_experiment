TODAY ?= $(shell date -u '+%B %Y')
PWD != pwd

URL ?= "http://localhost:8787"
THREADS ?= 4
CONNECTIONS ?= 100
DURATION ?= 1m
WRK = $(PWD)/wrk/wrk

SWS_VERSION = 30a640961322d7f8aa2ffa0ab6f409134ed97eea
HYPER_VERSION = f8e2a831943e9a687a1b3277c806caca6e0ac8a4
WRK_VERSION = a211dd5a7050b1f9e8a9870b95513060e72ac4a0

SWS_REPO = https://github.com/static-web-server/static-web-server
WRK_REPO = https://github.com/wg/wrk
HYPER_REPO = https://github.com/hyperium/hyper

SERVERS = arc gc
SERVER_BUILDS = $(addprefix $(PWD)/, $(SERVERS))

make all: $(SERVERS)

.PHONY: $(SERVERS)

bench: $(SERVERS)

plot:
	$(PYTHON_EXEC) process.py

clean-plots:
	- rm summary.csv

$(SERVERS):
	@if [ -f "data/benchmark_$@.csv" ]; then \
		rm -f "data/benchmark_$@.csv"; \
	fi
	mkdir -p $@/www
	SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

build: build-gc build-arc

build-gc: sws hyper wrk
	cd sws && git reset --hard
	cd sws && git apply $(PWD)/gc.patch
	cd sws && cargo build --release --target-dir=gc

build-arc: sws hyper wrk
	cd sws && git reset --hard
	cd sws && git apply $(PWD)/arc.patch
	cd sws && cargo build --release --target-dir=arc

sws:
	git clone $(SWS_REPO) sws
	cd sws && git apply $(PWD)/$(notdir $@).patch
	cd sws && git checkout $(SWS_VERSION)

hyper:
	git clone $(HYPER_REPO) hyper
	cd hyper && git checkout $(HYPER_VERSION)
	cd hyper && git apply $(PWD)/hyper.patch

wrk:
	git clone $(WRK_REPO) wrk
	cd wrk && git checkout $(WRK_VERSION)
	cd wrk && make -j$(nprocs)

clean-builds:
	rm -rf arc
	rm -rf gc

clean-src:
	rm -rf sws
	rm -rf hyper
	rm -rf wrk

clean-benchmarks:
	rm -rf data

clean: clean-src clean-builds clean-benchmarks
