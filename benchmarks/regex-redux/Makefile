GCVS_CFGS := gcvs/gc gcvs/arc
include ../../common.mk

REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791
REGEX_REPO = https://github.com/rust-lang/regex
REGEX_SRC = $(PWD)/regex
RR_SRC = $(PWD)/regex_redux

.PHONY: all build clean

build: $(REGEX_SRC) $(TARGETS) $(PWD)/redux-input.txt

$(PWD)/redux-input.txt:
	$(BIN)/gcvs/arc/perf/bin/fasta 2500000 > redux_input.txt

$(BIN)/gcvs/%:
	$(call patch_repo,$(REGEX_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(RR_SRC),)
	ln -s $(PWD)/regex-redux-harness.sh $@/bin/regex-redux-harness

$(BIN)/%:
	$(call patch_repo,$(REGEX_SRC),alloy)
	$(call build_alloy_exp,$*,$(RR_SRC),)
	ln -s $(PWD)/regex-redux-harness.sh $@/bin/regex-redux-harness

$(REGEX_SRC):
	git clone $(REGEX_REPO) $@
	cd $@ && git checkout $(REGEX_VERSION)

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean: clean-builds
	rm -rf $(REGEX_SRC)
