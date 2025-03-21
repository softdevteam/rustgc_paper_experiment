PWD != pwd
PATCH = $(PWD)/patches
BUILD = $(PWD)/build
BIN = $(PWD)/bin

GCVS_CFGS := $(addsuffix /perf, $(GCVS_CFGS)) $(addsuffix /mem, $(GCVS_CFGS))
TARGETS = $(addprefix $(BIN)/, $(GCVS_CFGS) $(ALLOY_CFGS))

define patch_repo
 	cd $1 && git reset --hard
	echo $(PATCH)/$(notdir $1).$2.diff
 	$(eval P := $(wildcard $(PATCH)/$(notdir $1).$2.diff))
 	$(if $(P),cd $1 && git apply $(P),)
endef

define build_alloy_exp
	mkdir -p $(BIN)/$1
	ALLOY_RUSTC_LOG="$(BIN)/$1/metrics.csv" RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$(ALLOY_PATH)/$1/bin/rustc" cargo install \
		 --path $2/$3 \
		--target-dir $(BUILD)/$1/$(notdir $2) \
		--root $(BIN)/$1
endef

define build_gcvs
	mkdir -p $(BIN)/gcvs/$1
	ALLOY_RUSTC_LOG="$(BIN)/gcvs/$1/metrics.csv" RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$(ALLOY_PATH)/gcvs/$(notdir $1)/bin/rustc" cargo install --path $2/$3 \
		--target-dir $(BUILD)/gcvs/$1/$(notdir $2) \
		--root $(BIN)/gcvs/$1
endef
