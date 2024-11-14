TODAY ?= $(shell date -u '+%B %Y')
PWD != pwd

# VENV
PYTHON = python3
PYTHON_EXEC = $(VENV_DIR)/bin/python
VENV_DIR = $(PWD)/venv
PIP = $(VENV_DIR)/bin/pip

# SOURCES
ALACRITTY_REPO = https://github.com/alacritty/alacritty.git
ALACRITTY_VERSION = 1063706f8e8a84139e5d2b464a4978e9d840ea17
PATCHES = $(PWD)/patches

BENCHMARKS_REPO = https://github.com/alacritty/vtebench.git
BENCHMARKS_VERSION = c75155bfc252227c0efc101c1971df3e327c71c4
# BENCHMARK RUNNER
PEXECS ?= 30
ITERS ?= 1
BARRIERS = barriers_naive barriers_none barriers_opt
ELISION = finalise_elide finalise_naive
PERF = arc gc
REBENCH_EXEC = $(VENV_DIR)/bin/rebench

XVFB_PID = $(PWD)/xvfb.pid

make all: venv build bench plot

.PHONY: $(PERF) $(BARRIERS) $(ELISION)
.PHONY: bench bench-perf bench-barriers bench-elision
.PHONY: build build-perf build-barriers build-elision
.PHONY: plot plot-setup plot-perf plot-barriers plot-elision
.PHONY: venv

bench: bench-perf bench-barriers bench-elision

bench-perf: $(PERF)

bench-barriers: $(BARRIERS)

bench-elision: $(ELISION)

build: build-perf build-barriers build-elision

plot: plot-perf plot-barriers plot-elision

plot-perf: plot-setup
	@echo "TODO: plot-perf"

plot-barriers: plot-setup
	@echo "TODO: plot-barriers"

plot-elision: plot-setup
	@echo "TODO: plot-elision"

plot-setup: venv
	mkdir -p plots
	$(PYTHON_EXEC) merge_results.py

build-perf: alacritty
	cd alacritty && git reset --hard
	cd alacritty && git apply $(PATCHES)/alacritty/arc.patch
	cd alacritty && \
		# RUSTC="../../bin/alloy/alloy/bin/rustc" \
		RUSTC="/tmp/alloytmp/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/arc
	cd alacritty && git reset --hard
	cd alacritty && git apply $(PATCHES)/alacritty/gc.patch
	cd alacritty && \
		RUSTC="/tmp/alloytmp/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/gc
	cd alacritty && git reset --hard

bench-perf: venv benchmarks
	mkdir -p $(PWD)/results/perf
	Xvfb :99 -ac -screen 0 1024x268x24 & echo $$! > $(XVFB_PID)
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/perf/rebench.data \
		alacritty.conf perf
	kill `cat $(XVFB_PID)` && rm $(XVFB_PID)

bench-barriers: venv benchmarks
	mkdir -p $(PWD)/results/perf
	Xvfb :99 -ac -screen 0 1024x268x24 & echo $$! > $(XVFB_PID)
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/barriers/rebench.data \
		alacritty.conf barriers
	kill `cat $(XVFB_PID)` && rm $(XVFB_PID)

bench-elision: venv benchmarks
	mkdir -p $(PWD)/results/elision
	Xvfb :99 -ac -screen 0 1024x268x24 & echo $$! > $(XVFB_PID)
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/perf/rebench.data \
		alacritty.conf elision
	kill `cat $(XVFB_PID)` && rm $(XVFB_PID)

build-barriers:
	cd alacritty && git reset --hard
	cd alacritty && git apply $(PATCHES)/alacritty/gc.patch
	cd alacritty && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_naive
	cd alacritty && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_none
	cd alacritty && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_opt
	cd alacritty && git reset --hard

build-elision:
	cd alacritty && git reset --hard
	cd alacritty && git apply $(PATCHES)/alacritty/gc.patch
	cd alacritty && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_elide
	cd alacritty && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_naive
	cd alacritty && git reset --hard

alacritty:
	git clone $(ALACRITTY_REPO) alacritty
	cd alacritty && git checkout $(ALACRITTY_VERSION)

benchmarks:
	git clone $(BENCHMARKS_REPO) benchmarks
	cd benchmarks && git checkout $(BENCHMARKS_VERSION)


venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: ../requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r ../requirements.txt

clean-builds:
	rm -rf arc
	rm -rf gc
	rm -rf barriers_opt
	rm -rf barriers_naive
	rm -rf barriers_none
	rm -rf finalise_naive
	rm -rf finalise_elide

clean-src:
	rm -rf alacritty

clean-benchmarks:
	rm -rf results

clean-plots:
	rm -rf plots

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean: clean-confirm clean-src clean-builds clean-benchmarks clean-plots
	@echo "Clean"
