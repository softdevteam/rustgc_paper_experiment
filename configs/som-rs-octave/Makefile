export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
BIN = $(PWD)/bin/$(ALLOY_CFG)/$(PATCH)

SOMRS_REPO = https://github.com/octavelarose/som-rs
SOMRS_VERSION = bde96ba315406117f87c3246310f6e990df557d4
export RUSTC = $(PWD)/../../bin/alloy/$(ALLOY_CFG)/bin/rustc
export ALLOY_CFG
export PATCH

all: build

.PHONY: all build clean

build: $(BIN)

$(BIN): $(PWD)/src
	mkdir -p $(PWD)/build
	cd $(PWD)/src && git reset --hard && git apply $(PWD)/$(PATCH).patch
	@echo "RUSTC=$(RUSTC)"
	cargo install \
		--path $(PWD)/src/som-interpreter-ast \
		--target-dir $(PWD)/build/$(ALLOY_CFG)/$(PATCH) \
		--root $@

$(PWD)/src:
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $@/core-lib $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src
	rm -f som-rs.build.log
