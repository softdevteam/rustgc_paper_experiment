GCVS_CFGS := gcvs/gc gcvs/arc
include ../../common.mk

FD_SRC = $(PWD)/fd
FD_REPO = https://github.com/sharkdp/fd
FD_VERSION = a4fdad6ff781b5b496c837fde24001b0e46973d6

all: build

.PHONY: all build clean

build: $(FD_SRC) $(FDB_SRC) $(TARGETS)

$(BIN)/gcvs/%:
	$(call patch_repo,$(FD_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(FD_SRC),)
	ln -s $(PWD)/fd-harness.sh $@/bin/fd-harness

$(BIN)/%:
	$(call patch_repo,$(FD_SRC),alloy)
	$(call build_alloy_exp,$*,$(FD_SRC),)
	ln -s $(PWD)/fd-harness.sh $@/bin/fd-harness

$(FD_SRC):
	git clone $(FD_REPO) $@
	cd $@ && git checkout $(FD_VERSION)

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean:
	rm -rf $(FD_SRC)
