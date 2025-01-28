PWD != pwd
BIN = $(PWD)/bin

ALLOY_REPO = https://github.com/softdevteam/alloy
ALLOY_VERSION = master
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_CFGS = $(wildcard $(PWD)/*.config.toml)
ALLOY_TARGETS := $(patsubst $(PWD)/%.config.toml,$(BIN)/%,$(ALLOY_CFGS))
ALLOY_BUILD_LOG = $(PWD)/alloy.build.log

all: build

.PHONY: all build clean

build: $(ALLOY_CFGS) $(ALLOY_TARGETS)


$(ALLOY_TARGETS): $(PWD)/src
	$(foreach cfg, $(ALLOY_CFGS), \
		@echo "Building $@" 2>&1 | tee -a $(ALLOY_BUILD_LOG); \
		./src/x install \
		--config $(cfg) \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(PWD)/build/$(basename $@) \
		--set build.docs=false \
		--set install.prefix=$@ \
		--set install.sysconfdir=etc 2>&1 | tee -a $(ALLOY_BUILD_LOG))

$(PWD)/src:
	git clone $(ALLOY_REPO) $@
	cd $@ && git checkout $(ALLOY_VERSION)

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src
	rm -f alloy.build.log

