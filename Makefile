export PEXECS ?= 10

PWD != pwd

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python

RESULTS = $(PWD)/results
CONFIGS = $(PWD)/configs
SUITES = som-rs-ast som-rs-bc yksom
# SUITES = som-rs-ast
# EXPERIMENTS = gcrc premopt elision
EXPERIMENTS = premopt

PERF_DATA = $(foreach s, $(SUITES), $(foreach e, $(EXPERIMENTS), $(RESULTS)/$e/$s/perf.csv))
MEM_DATA = $(foreach s, $(SUITES), $(foreach e, $(EXPERIMENTS), $(RESULTS)/$e/$s/mem.csv))

PERF_PLOTS = $(patsubst $(RESULTS)/%, $(PWD)/plots/%, $(patsubst %.csv, %.svg, $(PERF_DATA)))

REBENCH_EXEC = $(VENV)/bin/rebench
REBENCH_PROCESSOR = $(PWD)/process.py

all: build

.PHONY: venv
.PHONY: build build-alloy plot

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

build-alloy:
	cd configs/alloy && make

build: build-alloy

plot: venv $(PERF_PLOTS)

geomean: venv
	$(PYTHON_EXEC) geomean.py


clean:
	# rm -rf $(ALLOY_SRC)
	rm -rf logs/*
	@echo "Clean"

clean-builds:
	$(foreach s, $(SUITES), cd $(CONFIGS)/$s && make clean;)

clean-results:
	rm -rf $(RESULTS)

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

bench-perf: build-alloy venv $(PERF_DATA)

$(PERF_DATA):
	mkdir -p $(dir $@)
	mkdir -p $(dir $@)/metrics
	$(REBENCH_EXEC) -R -D \
		--invocations $(PEXECS) \
		--iterations 1 \
		--build-log $(dir $@)build.log \
		-df $@ \
		-exp $(notdir $(patsubst %/,%,$(dir $@))) \
		$(CONFIGS)/$(notdir $(patsubst %/,%,$(dir $@)))/rebench.conf

$(PERF_PLOTS):
	mkdir -p $(dir $@)
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) \
		$(patsubst $(PWD)/plots/%, $(RESULTS)/%, $(patsubst %.svg, %.csv, $@)) $@
