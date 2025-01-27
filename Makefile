export PEXECS ?= 10

REBENCH_EXEC = $(VENV)/bin/rebench
REBENCH_PROCESSOR = $(PWD)/process.py

PWD != pwd
BIN = $(PWD)/bin

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python
BUILD = $(PWD)/build

ALLOY_REPO = https://github.com/softdevteam/alloy
ALLOY_VERSION = master
ALLOY_SRC = $(BUILD)/alloy
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_CFGS = $(wildcard $(PWD)/configs/alloy/*.config.toml)
ALLOY_TARGETS := $(patsubst $(PWD)/configs/alloy/%.config.toml,$(BIN)/alloy/%,$(ALLOY_CFGS))

ALLOY_BUILD_LOG = $(PWD)/alloy.build.log

RESULTS = $(PWD)/results
SUITES = som-rs-ast som-rs-octave-ast yksom
EXPERIMENTS = gcrc premopt elision

DATA = $(foreach s, $(SUITES), $(foreach e, $(EXPERIMENTS), $(RESULTS)/test/$e/$s/perf.csv))
PLOTS = $(patsubst $(RESULTS)/test/%.csv, $(RESULTS)/test/%.svg, $(DATA))

all: build

.PHONY: venv
.PHONY: build build-alloy

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

build-alloy: $(ALLOY_CFGS) $(ALLOY_TARGETS)

$(ALLOY_SRC):
	mkdir -p $(BUILD)
	git clone $(ALLOY_REPO) $(ALLOY_SRC)
	cd $(ALLOY_SRC) && git checkout $(ALLOY_VERSION)


$(ALLOY_TARGETS): $(ALLOY_SRC)
	@echo "Building $@" 2>&1 | tee -a $(ALLOY_BUILD_LOG)
	$(PYTHON) $(ALLOY_SRC)/x.py install \
		--config $(PWD)/configs/alloy/$(notdir $@).config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC)/build \
		--set build.docs=false \
		--set install.prefix=$@ \
		--set install.sysconfdir=etc 2>&1 | tee -a $(ALLOY_BUILD_LOG)

clean:
	rm -rf $(ALLOY_SRC_DIR)
	@echo "Clean"

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean-plots:
	cd sws_benchmarks && make clean-plots
	cd awfy_benchmarks && make clean-plots
	cd grmtools_benchmarks && make clean-plots
	cd alacritty_benchmarks && make clean-plots

clean-benchmarks: clean-confirm
	cd sws_benchmarks && make clean-benchmarks
	cd awfy_benchmarks && make clean-benchmarks
	cd grmtools_benchmarks && make clean-benchmarks
	cd alacritty_benchmarks && make clean-benchmarks

clean-builds: clean-confirm
	cd sws_benchmarks && make clean-builds
	cd awfy_benchmarks && make clean-builds
	cd grmtools_benchmarks && make clean-builds
	cd alacritty_benchmarks && make clean-builds

clean-alloy: clean-confirm
	rm -rf $(BIN)
