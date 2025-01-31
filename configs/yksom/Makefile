export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
TARGETS = $(addprefix $(PWD)/bin/, opt/gc bnaive/gc bnone/gc)

YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_VERSION = master

all: build

.PHONY: all build clean

build: $(TARGETS)

$(TARGETS): $(PWD)/src
	mkdir -p $(PWD)/build
	@echo "RUSTC=$(PWD)/../alloy/bin/$(notdir $(patsubst %/,%,$(dir $@)))/bin/rustc"
	RUSTC="$(PWD)/../alloy/bin/$(notdir $(patsubst %/,%,$(dir $@)))/bin/rustc" \
	cargo install \
		--path $(PWD)/src \
		--target-dir $(PWD)/build/$(@:$(PWD)/bin/%=%) \
		--root $@

$(PWD)/src:
	git clone --recursive $(YKSOM_REPO) $@
	cd $@ && git checkout $(YKSOM_VERSION)
	ln -s $@/SOM $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src $(PWD)/SOM
