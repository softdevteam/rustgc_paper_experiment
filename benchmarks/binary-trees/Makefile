GCVS_CFGS := gcvs/gc gcvs/arc gcvs/rust-gc gcvs/typed_arena
include ../../common.mk

BT_SRC = $(PWD)/binary_trees

.PHONY: all build clean

all: build

build: $(TARGETS)

$(BIN)/gcvs/%:
	$(call build_gcvs,$*,$(BT_SRC),$(patsubst %/,%,$(dir $*)))
	ln -s $(PWD)/binary-trees-harness.sh $@/bin/binary-trees-harness

$(BIN)/%:
	$(call build_alloy_exp,$*,$(BT_SRC),gc)
	ln -s $(PWD)/binary-trees-harness.sh $@/bin/binary-trees-harness

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean: clean-builds
