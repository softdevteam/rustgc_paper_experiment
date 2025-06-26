CURDIR ?= $(shell pwd)

IMAGE := $(if $(FULL),full:latest,quick:latest)
LOG_STAGE := log_export
RUNTIME_STAGE := runtime
RESULTS := $(CURDIR)/results
LOGFILE := experiment.log

VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PYTHON := $(VENV_PYTHON)
INVOKE := $(VENV_DIR)/bin/invoke

BIN_DIR ?= $(CURDIR)/artefacts/bin
BIN_ARCHIVE ?= artefacts-bin.tar.xz

EXP_ARG := $(if $(strip $(EXPERIMENTS)),--experiments "$(EXPERIMENTS)")
SUITE_ARG := $(if $(strip $(SUITES)),--suites $(SUITES))
MEASURE_ARG := $(if $(strip $(MEASUREMENTS)),--measurements $(MEASUREMENTS))

QUICK_PEXECS = 5
FULL_PEXECS = 30

.PHONY: run-full run-quick fetch-binaries bare-metal

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

venv: pyproject.toml | $(VENV_DIR)


define WITH_DOCKER
	docker buildx build --progress=plain \
		--build-arg FULL=$1 \
		$(if $(EXP_ARG),--build-arg EXPERIMENTS=$(EXP_ARG)) \
		$(if $(SUITE_ARG),--build-arg SUITES=$(SUITE_ARG)) \
		$(if $(MEASURE_ARG),--build-arg MEASUREMENTS=$(MEASURE_ARG)) \
		--target runtime \
		--tag $(IMAGE) \
		--load .
	@test -f $(LOGFILE) || touch $(LOGFILE)
	chmod a+w $(LOGFILE)
	docker run --rm -it \
		--mount type=bind,source="$(LOGFILE)",target=/app/experiment.log \
		--mount type=bind,source="$(RESULTS)",target=/app/results \
		$(IMAGE)
endef


run-quick:
	$(call WITH_DOCKER,false)

run-full:
	$(call WITH_DOCKER,true)

tarball:
	@echo "Compressing $(BIN_DIR) with maximum compression..."
	cd artefacts && tar -cf - bin | xz -9e > $(BIN_ARCHIVE)
	@echo "Created $(BIN_ARCHIVE) ($$(du -h artefacts/$(BIN_ARCHIVE) | cut -f1))"

fetch-binaries: $(GDRIVE_ARTEFACT)

$(GDRIVE_ARTEFACT):
	./fetch_binaries.sh

bare-metal: venv
	@$(INVOKE) build-benchmarks \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)

	@$(INVOKE) run-benchmarks \
		$(if $(PEXECS),$(PEXECS),$(FULL_PEXECS)) \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)

process: venv
	@$(INVOKE) process-benchmarks
