CURDIR ?= $(shell pwd)

IMAGE_full := full:latest
IMAGE_quick := quick:latest
LOG_STAGE := log_export
RUNTIME_STAGE := runtime
RESULTS := $(CURDIR)/results
LOGFILE := experiment.log

VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PYTHON := $(VENV_PYTHON)  # Use venv python consistently

BIN_DIR ?= $(CURDIR)/artefacts/bin
BIN_ARCHIVE ?= artefacts-bin.tar.xz
PREBUILT_DIR := $(CURDIR)/artefacts/prebuilt
PREBUILT_BIN := $(PREBUILT_DIR)/bin  # Added definition

GDRIVE_FILE_ID = 1oDkZ2RH65iq25_65AppzdLH_zzbt6oRz
GDRIVE_ARTEFACT = $(PREBUILT_DIR)/$(BIN_ARCHIVE)

.PHONY: venv run-full run-quick fetch-binaries bare-metal

venv:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install .

run-quick: fetch-binaries
	docker buildx build \
		--target runtime \
		--tag $(IMAGE_quick) \
		--build-arg BUILD_QUICK=true \
		--load .
	@test -f $(LOGFILE) || touch $(LOGFILE)
	chmod a+w $(LOGFILE)
	docker run --rm -it \
		--mount type=bind,source="$(LOGFILE)",target=/app/experiment.log \
		--mount type=bind,source="$(RESULTS)",target=/app/results \
		$(IMAGE_quick)

run-full:
	docker buildx build \
		--build-arg $(if $(EXPERIMENTS),EXPERIMENTS="--experiments $(EXPERIMENTS))" \
		--build-arg $(if $(SUITES),SUITES="--suites $(SUITES))" \
		--build-arg $(if $(MEASUREMENTS),MEASUREMENTS="--measurements $(MEASUREMENTS))" \
		--target runtime \
		--tag $(IMAGE_full) \
		--load .
	@test -f $(LOGFILE) || touch $(LOGFILE)
	chmod a+w $(LOGFILE)
	docker run --rm -it \
		--mount type=bind,source="$(LOGFILE)",target=/app/experiment.log \
		--mount type=bind,source="$(RESULTS)",target=/app/results \
		$(IMAGE_full)

tarball:
	@echo "Compressing $(BIN_DIR) with maximum compression..."
	cd artefacts && tar -cf - bin | xz -9e > $(BIN_ARCHIVE)
	@echo "Created $(BIN_ARCHIVE) ($$(du -h artefacts/$(BIN_ARCHIVE) | cut -f1))"

fetch-binaries: venv
	@mkdir -p $(PREBUILT_DIR)
	@echo "Downloading from Google Drive..."
	@gdown $(GDRIVE_FILE_ID) -O $(GDRIVE_ARTEFACT)
	@echo "Download complete. File size: $$(du -h $(GDRIVE_ARTEFACT) | cut -f1)"
	@echo "Unzipping $(GDRIVE_ARTEFACT) to $(PREBUILT_DIR)..."
	tar -xvf $(GDRIVE_ARTEFACT) -C $(PREBUILT_DIR)  # Fixed typo
	@echo "Done"

bare-metal: venv
	@./run build-alloy \
		$(if $(EXPERIMENTS),--experiments $(EXPERIMENTS)) \
		$(if $(SUITES),--suites $(SUITES)) \
		$(if $(MEASUREMENTS),--measurements $(MEASUREMENTS))

