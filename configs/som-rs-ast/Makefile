export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
TARGETS = $(addprefix $(PWD)/bin/, opt/gc bnaive/gc bnone/gc)

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608

export ALLOY_CFG
export PATCH

all: build

.PHONY: all build clean

build: $(TARGETS)

$(TARGETS): $(PWD)/src
	mkdir -p $(PWD)/build
	cd $(PWD)/src && git reset --hard && git apply $(PWD)/$(notdir $@).patch
	@echo "RUSTC=$(PWD)/../alloy/bin/$(notdir $(patsubst %/,%,$(dir $@)))/bin/rustc"
	RUSTC="$(PWD)/../alloy/bin/$(notdir $(patsubst %/,%,$(dir $@)))/bin/rustc" \
	cargo install \
		--path $(PWD)/src/som-interpreter-ast \
		--target-dir $(PWD)/build/$(@:$(PWD)/bin/%=%) \
		--root $@

$(PWD)/src:
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $@/core-lib $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src $(PWD)/SOM
