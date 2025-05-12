GCVS_CFGS := gcvs/gc gcvs/arc gcvs/original
include ../../common.mk

RG_SRC = $(PWD)/ripgrep
RG_REPO = https://github.com/burntsushi/ripgrep
RG_VERSION = de4baa10024f2cb62d438596274b9b710e01c59b

LINUX_SRC = $(PWD)/linux
LINUX_REPO = 'https://github.com/BurntSushi/linux'

all: build

.PHONY: all build clean

build: $(LINUX_SRC) $(RG_SRC) $(TARGETS)

$(BIN)/gcvs/%:
	$(call patch_repo,$(RG_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(RG_SRC),)
	ln -s $(PWD)/ripgrep-harness.sh $@/bin/ripgrep-harness

$(BIN)/%:
	$(call patch_repo,$(RG_SRC),alloy)
	$(call build_alloy_exp,$*,$(RG_SRC),)
	ln -s $(PWD)/ripgrep-harness.sh $@/bin/ripgrep-harness

$(RG_SRC):
	git clone $(RG_REPO) $@
	cd $@ && git checkout $(RG_VERSION)

$(LINUX_SRC):
	git clone --depth 1 $(LINUX_REPO) $@
	cd $@ && make defconfig
	cd $@ && make -j4

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean:
	rm -rf $(RG_SRC)
