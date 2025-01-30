export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
BIN = $(PWD)/bin/$(ALLOY_CFG)/$(PATCH)

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608

export RUSTC = $(PWD)/../alloy/bin/$(ALLOY_CFG)/bin/rustc
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
		--path $(PWD)/src/som-interpreter-bc \
		--target-dir $(PWD)/build/$(ALLOY_CFG)/$(PATCH) \
		--root $@

$(PWD)/src:
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $@/core-lib $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src $(PWD)/SOM
