export PEXECS ?= 10

PWD != pwd

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python
REBENCH_PROCESSOR = $(PWD)/process.py

ALLOY_REPO = https://github.com/jacob-hughes/alloy
ALLOY_VERSION = experiments
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_SRC = $(PWD)/alloy
ALLOY_CFGS = premopt/naive premopt/opt premopt/none \
			 elision/naive elision/opt
ALLOY_DEFAULT = $(addprefix gcvs/, perf mem)

LIBGC_REPO = https://github.com/softdevteam/bdwgc
LIBGC_VERSION = master
LIBGC_SRC = $(PWD)/bdwgc

HEAPTRACK_REPO = https://github.com/kde/heaptrack
HEAPTRACK_VERSION = master
HEAPTRACK_SRC = $(PWD)/heaptrack
HEAPTRACK = $(HEAPTRACK_SRC)/bin/heaptrack

RESULTS = $(PWD)/results
CONFIGS = $(PWD)/configs

BENCHMARKS = $(PWD)/benchmarks/som
# MEM_DATA = $(foreach s, $(SUITES), $(foreach e, $(EXPERIMENTS), $(RESULTS)/$e/$s/mem.csv))
# PERF_PLOTS = $(patsubst $(RESULTS)/%, $(PWD)/plots/%, $(patsubst %.csv, %.svg, $(PERF_DATA)))

export ALLOY_CFGS := $(addsuffix /perf, $(ALLOY_CFGS)) $(addsuffix /mem, $(ALLOY_CFGS))
export ALLOY_PATH = $(ALLOY_SRC)/bin
export LIBGC_PATH = $(LIBGC_SRC)/lib
export REBENCH_EXEC = $(VENV)/bin/rebench
export LD_LIBRARY_PATH = $(LIBGC_PATH)
export EXPERIMENTS = $(foreach e, gcvs elision premopt, $(foreach ty, mem perf, $(e)/$(ty)))
export RESULTS_DIR = $(PWD)/results
# export RUSTFLAGS = "-L $(LIBGC_PATH)"

all: build

.PHONY: venv
.PHONY: build build-alloy
.PHONY: bench plot
.PHONY: clean clean-alloy clean-results clean-plots clean-confirm

build-alloy: $(ALLOY_SRC) $(LIBGC_PATH) $(HEAPTRACK) $(ALLOY_CFGS) $(ALLOY_DEFAULT)

$(ALLOY_CFGS) $(ALLOY_DEFAULT):
	@if [ "$(notdir $@)" = "mem" ]; then \
		export LD_LIBRARY_PATH=$(LIBGC_PATH); \
		export RUSTFLAGS="-L $(LIBGC_PATH)"; \
		export GC_LINK_DYNAMIC=true; \
	fi; \
	$(ALLOY_SRC)/x install \
		--config $(PWD)/$(subst /,.,$(dir $@))config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC)/$(@) \
		--set build.docs=false \
		--set install.prefix=$(ALLOY_SRC)/bin/$@ \
		--set install.sysconfdir=etc

$(ALLOY_SRC):
	git clone $(ALLOY_REPO) $@
	cd $@ && git checkout $(ALLOY_VERSION)

$(LIBGC_SRC):
	git clone $(LIBGC_REPO) $@
	cd $@ && git checkout $(LIBGC_VERSION)

$(LIBGC): $(LIBGC_SRC)
	mkdir -p $</build
	cd $</build && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX="$(LIBGC_SRC)" \
		-DCMAKE_C_FLAGS="-DGC_ALWAYS_MULTITHREADED -DVALGRIND_TRACKING" ../ && \
		make -j$(numproc) install

$(HEAPTRACK_SRC):
	git clone $(HEAPTRACK_REPO) $@
	cd $@ && git checkout $(HEAPTRACK_VERSION)

$(HEAPTRACK): $(HEAPTRACK_SRC)
	mkdir -p $</build
	cd $</build && \
		cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(HEAPTRACK_SRC) ../ && \
		make -j$(numproc) install


build-benchmarks:
	$(foreach b, $(BENCHMARKS), cd $(b)/ && make build;)

bench:
	$(foreach b, $(BENCHMARKS), cd $(b)/ && make bench;)

build: build-alloy build-benchmarks


venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt


# bench: build venv $(PERF_DATA)
#
# $(PERF_DATA):
# 	mkdir -p $(dir $@)
# 	mkdir -p $(dir $@)/metrics
# 	$(REBENCH_EXEC) -R -D \
# 		--invocations $(PEXECS) \
# 		--iterations 1 \
# 		--build-log $(dir $@)build.log \
# 		-df $@ \
# 		-exp $(notdir $(patsubst %/,%,$(dir $@))) \
# 		$(CONFIGS)/$(notdir $(patsubst %/,%,$(dir $@)))/rebench.conf

plot: bench venv $(PERF_PLOTS)

$(PERF_PLOTS):
	mkdir -p $(dir $@)
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) \
		$(patsubst $(PWD)/plots/%, $(RESULTS)/%, $(patsubst %.svg, %.csv, $@)) $@

clean: clean-confirm clean-alloy clean-builds clean-results clean-plots
	@echo "Clean"

clean-alloy:
	rm -rf $(LIBGC_SRC) $(HEAPTRACK_SRC) $(ALLOY_SRC)/bin

clean-benchmarks:
	$(foreach b, $(BENCHMARKS), cd $(b)/ && make clean;)

clean-results:
	rm -rf $(RESULTS)

clean-plots:
	rm -rf plots

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
