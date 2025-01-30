PWD != pwd
BIN = $(PWD)/bin

ALLOY_REPO = https://github.com/softdevteam/alloy
ALLOY_VERSION = master
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_CFGS = $(wildcard $(PWD)/*.config.toml)
ALLOY_TARGETS := $(patsubst $(PWD)/%.config.toml,$(BIN)/%,$(ALLOY_CFGS))

all: build

.PHONY: all build clean

build: $(ALLOY_TARGETS)

$(ALLOY_TARGETS): $(PWD)/src
	./src/x install \
		--config $(PWD)/$(notdir $@).config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(PWD)/build \
		--set build.docs=false \
		--set install.prefix=$@ \
		--set install.sysconfdir=etc

$(PWD)/src:
	git clone $(ALLOY_REPO) $@
	cd $@ && git checkout $(ALLOY_VERSION)

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src
	rm -f alloy.build.log

