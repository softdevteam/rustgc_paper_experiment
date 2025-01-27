export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
BIN = $(PWD)/bin/$(ALLOY_CFG)/$(PATCH)

YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_VERSION = master

export RUSTC = $(PWD)/../../bin/alloy/$(ALLOY_CFG)/bin/rustc
export ALLOY_CFG

all: build

.PHONY: all build clean

build: $(BIN)

$(BIN): $(PWD)/src
	mkdir -p $(PWD)/build
	@echo "RUSTC=$(RUSTC)"
	cargo install \
		--path $(PWD)/src \
		--target-dir $(PWD)/build/$(ALLOY_CFG)/gc \
		--root $@
	cargo install \
		--path $(PWD)/src \
		--target-dir $(PWD)/build/$(ALLOY_CFG)/gc \
		--root $@

$(PWD)/src:
	git clone --recursive $(YKSOM_REPO) $@
	cd $@ && git checkout $(YKSOM_VERSION)
	ln -s $@/SOM $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src
	rm -f yksom.build.log
