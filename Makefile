export PEXECS ?= 10

PWD != pwd

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python

RESULTS = $(PWD)/results
SUITES = som-rs-ast som-rs-octave-ast yksom
EXPERIMENTS = gcrc premopt elision

DATA = $(foreach s, $(SUITES), $(foreach e, $(EXPERIMENTS), $(RESULTS)/test/$e/$s/perf.csv))
PLOTS = $(patsubst $(RESULTS)/test/%.csv, $(RESULTS)/test/%.svg, $(DATA))

REBENCH_EXEC = $(VENV)/bin/rebench
REBENCH_PROCESSOR = $(PWD)/process.py

all: build

.PHONY: venv
.PHONY: build build-alloy

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

build-alloy:
	cd configs/alloy && make

build: build-alloy

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt


clean:
	rm -rf $(ALLOY_SRC)
	rm -rf results/
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
