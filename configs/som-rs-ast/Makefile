export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
PWD != pwd
PERF_TARGETS = $(addprefix $(PWD)/bin/perf/, opt/gc bnaive/gc bnone/gc)
MEM_TARGETS = $(addprefix $(PWD)/bin/mem/, opt/gc bnaive/gc bnone/gc)
TARGETS = $(PERF_TARGETS) $(MEM_TARGETS)

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
LIBGC = $(PWD)/../alloy/bdwgc/lib/
ALLOY_DIR=$(PWD)/../alloy

ALLOY_CFGS = premopt/naive/perf premopt/opt/perf premopt/none/perf \
			 premopt/naive/mem premopt/opt/mem premopt/none/mem

export ALLOY_CFG
export PATCH

all: build

.PHONY: all build clean

build: $(ALLOY_CFGS)

$(ALLOY_CFGS): $(PWD)/src
	cd $(PWD)/src && git reset --hard && git apply $(PWD)/gc.patch
	RUSTC="$(ALLOY_DIR)/$@/rustc" \
		  LD_LIBRARY_PATH=$(LIBGC) RUSTFLAGS="-L $(LIBGC)" \
		  cargo install \
		  --path $(PWD)/src/som-interpreter-ast \
		  --target-dir $(PWD)/build/$@ \
		  --root $@
	ln -s $(PWD)/$@/bin/som-interpreter-ast $(PWD)/$@/bin/som-rs-ast

$(PWD)/src:
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $@/core-lib $(PWD)/SOM

clean:
	rm -rf $(PWD)/bin $(PWD)/build $(PWD)/src $(PWD)/SOM
