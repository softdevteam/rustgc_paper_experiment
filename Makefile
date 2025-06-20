CURDIR ?= $(shell pwd)

export EXPERIMENTS ?= gcvs premopt elision
export MEASUREMENTS ?= perf mem metrics

IMAGE_NAME := run-full:latest
LOG_STAGE := log_export
RUNTIME_STAGE := runtime
RESULTS := $(PWD)/results

build:
	docker buildx build \
		--target $(RUNTIME_STAGE) \
		--tag $(IMAGE_NAME) \
		--load .
	docker buildx build \
		--target $(LOG_STAGE) \
		--output type=local,dest=$(CURDIR) .

run-full: build
	touch $(CURDIR)/docker-run-full.log
	chmod a+w $(CURDIR)/docker-run-full.log
	docker run --rm -it \
		--mount type=bind,source="$(CURDIR)/docker-run-full.log",target=/app/experiment.log \
		--mount type=bind,source="$(CURDIR)/results",target=/app/results \
		$(if $(PEXECS),--env PEXECS="$(PEXECS)",) \
		--env EXPERIMENTS="$(EXPERIMENTS)" \
		--env MEASUREMENTS="$(MEASUREMENTS)" \
		$(IMAGE_NAME)
