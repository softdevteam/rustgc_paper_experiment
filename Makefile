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
.PHONY: build build-alloy
.PHONY: bench plot
.PHONY: clean clean-alloy clean-results clean-plots clean-confirm

build-alloy:
	cd configs/alloy && make

build: build-alloy
	$(foreach s, $(SUITES), cd $(CONFIGS)/$s && make build;)


venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt


bench: build venv $(PERF_DATA)

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

plot: bench venv $(PERF_PLOTS)

$(PERF_PLOTS):
	mkdir -p $(dir $@)
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) \
		$(patsubst $(PWD)/plots/%, $(RESULTS)/%, $(patsubst %.svg, %.csv, $@)) $@

clean: clean-confirm clean-alloy clean-builds clean-results clean-plots
	@echo "Clean"

clean-alloy:
	cd configs/alloy && make clean

clean-builds:
	$(foreach s, $(SUITES), cd $(CONFIGS)/$s && make clean;)

clean-results:
	rm -rf $(RESULTS)

clean-plots:
	rm -rf plots

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
