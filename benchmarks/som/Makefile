GCVS_CFGS := gcvs/gc gcvs/rc
include ../../common.mk

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_SRC = $(PWD)/som-rs

YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_VERSION = master
YKSOM_SRC = $(PWD)/yksom

all: build

.PHONY: all build clean

build: $(SOMRS_SRC) $(YKSOM_SRC) $(TARGETS)

$(BIN)/gcvs/%:
	$(call patch_repo,$(SOMRS_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(SOMRS_SRC),som-interpreter-ast)
	$(call build_gcvs,$*,$(SOMRS_SRC),som-interpreter-bc)
	ln -s $(PWD)/som-rs-ast-harness.sh $@/bin/som-rs-ast-harness
	ln -s $(PWD)/som-rs-bc-harness.sh $@/bin/som-rs-bc-harness

$(BIN)/%:
	$(call patch_repo,$(SOMRS_SRC),alloy)
	$(call build_alloy_exp,$*,$(SOMRS_SRC),som-interpreter-ast)
	$(call build_alloy_exp,$*,$(SOMRS_SRC),som-interpreter-bc)
	$(call patch_repo,$(YKSOM_SRC),alloy)
	$(call build_alloy_exp,$*,$(YKSOM_SRC),)
	ln -s $(PWD)/som-rs-ast-harness.sh $@/bin/som-rs-ast-harness
	ln -s $(PWD)/som-rs-bc-harness.sh $@/bin/som-rs-bc-harness
	ln -s $(PWD)/yksom-harness.sh $@/bin/yksom-harness

$(SOMRS_SRC):
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $(SOMRS_SRC)/core-lib $(SOMRS_SRC)/SOM

$(YKSOM_SRC):
	git clone --recursive $(YKSOM_REPO) $@
	cd $@ && git checkout $(YKSOM_VERSION)

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean: clean-builds
	rm -rf $(SOMRS_SRC) $(YKSOM_SRC)
