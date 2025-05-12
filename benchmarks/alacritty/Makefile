GCVS_CFGS := gcvs/gc gcvs/arc gcvs/original
include ../../common.mk

ALACRITTY_REPO = https://github.com/alacritty/alacritty.git
ALACRITTY_VERSION = 1063706f8e8a84139e5d2b464a4978e9d840ea17
ALACRITTY_SRC = $(PWD)/alacritty

VTEBENCH_REPO = https://github.com/alacritty/vtebench.git
VTEBENCH_VERSION = c75155bfc252227c0efc101c1971df3e327c71c4
VTEBENCH_SRC = $(PWD)/vtebench

make all: build

.PHONY: build clean

build: $(ALACRITTY_SRC) $(VTEBENCH_SRC) $(TARGETS)

$(BIN)/gcvs/%:
	$(call patch_repo,$(ALACRITTY_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(ALACRITTY_SRC),alacritty)
	ln -s $(PWD)/alacritty-harness.sh $@/bin/alacritty-harness

$(BIN)/%:
	$(call patch_repo,$(ALACRITTY_SRC),alloy)
	$(call build_alloy_exp,$*,$(ALACRITTY_SRC),alacritty)
	ln -s $(PWD)/alacritty-harness.sh $@/bin/alacritty-harness

$(ALACRITTY_SRC):
	git clone $(ALACRITTY_REPO) $@
	cd $@ && git checkout $(ALACRITTY_VERSION)

$(VTEBENCH_SRC):
	git clone $(VTEBENCH_REPO) $@
	cd $@ && git checkout $(VTEBENCH_VERSION)

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean: clean-builds
	rm -rf $(ALACRITTY_SRC) $(VTEBENCH_SRC)
