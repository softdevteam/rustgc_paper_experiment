PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/patches

SOMRS_REPO = https://github.com/Hirevo/som-rs
YKSOM_REPO = https://github.com/softdevteam/yksom

SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
YKSOM_VERSION=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6

SOMRS_CFGS = $(addprefix $(BIN)/som-rs/, finalise_elide finalise_naive \
	     barriers_naive barriers_opt barriers_none \
	     perf_rc perf_gc)
YKSOM_CFGS = $(addprefix $(BIN)/yksom/, finalise_elide finalise_naive barriers_naive \
	     barriers_opt barriers_none)

SOMRSBC_EXPS = $(addprefix results/, som_rs_bc_elision som_rs_bc_perf som_rs_bc_barriers)
SOMRSAST_EXPS = $(addprefix results/, som_rs_ast_elision som_rs_ast_perf som_rs_ast_barriers)
YKSOM_EXPS = $(addprefix results/, yksom_elision yksom_barriers)
EXPERIMENTS = $(SOMRSBC_EXPS) $(SOMRSAST_EXPS) $(YKSOM_EXPS)

all: bench

.PHONY: build bench
.PHONY: clean clean-benchmarks clean-builds

plot:
	mkdir -p plots
	# $(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary som_rs_bc_perf som_rs_ast_perf
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) elision som_rs_bc_elision som_rs_ast_elision
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) barriers som_rs_bc_barriers som_rs_ast_barriers
	# $(PYTHON_EXEC) $(REBENCH_PROCESSOR) barriers som_rs_ast_barriers

clean-plots:
	rm -rf plots
	- rm summary.csv

bench: $(EXPERIMENTS)

$(YKSOM_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		yksom.conf $(notdir $@)

$(SOMRSAST_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-ast.conf $(notdir $@)

$(SOMRSBC_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-bc.conf $(notdir $@)

build: $(YKSOM_CFGS) $(SOMRS_CFGS)

$(YKSOM_CFGS): yksom/Cargo.toml
	cd yksom && git diff-index --quiet HEAD --
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	@if [ -n "$(filter $(ALLOY_CFGS), $(notdir $@))" ]; then \
		cd yksom && \
		RUSTC="../../bin/alloy/$(notdir $@)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	else \
		cd yksom && \
		RUSTC="../../bin/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	fi
	ln -s $(PWD)/yksom/SOM $@/SOM
	ln -s $@/release/yksom $@/yksom
	cd yksom && git reset --hard

som-rs-barriers-naive:
	- rm -rf $(BIN)/som-rs/barriers_naive
	mkdir -p $(BIN)/som-rs/barriers_naive
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=barriers_naive
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/barriers_naive/core-lib
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-ast $(BIN)/som-rs/barriers_naive/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-bc $(BIN)/som-rs/barriers_naive/som-rs-bc

som-rs-barriers-opt:
	- rm -rf $(BIN)/som-rs/barriers_opt
	mkdir -p $(BIN)/som-rs/barriers_opt
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=barriers_opt
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/barriers_opt/core-lib
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-ast $(BIN)/som-rs/barriers_opt/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-bc $(BIN)/som-rs/barriers_opt/som-rs-bc

som-rs-barriers-none:
	- rm -rf $(BIN)/som-rs/barriers_none
	mkdir -p $(BIN)/som-rs/barriers_none
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=barriers_none
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/barriers_none/core-lib
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-ast $(BIN)/som-rs/barriers_none/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-bc $(BIN)/som-rs/barriers_none/som-rs-bc

som-rs-finalise-elide:
	- rm -rf $(BIN)/som-rs/finalise_elide
	mkdir -p $(BIN)/som-rs/finalise_elide
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=finalise_elide
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_elide/core-lib
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-ast $(BIN)/som-rs/finalise_elide/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-bc $(BIN)/som-rs/finalise_elide/som-rs-bc

som-rs-finalise-naive:
	- rm -rf $(BIN)/som-rs/finalise_naive
	mkdir -p $(BIN)/som-rs/finalise_naive
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_naive/core-lib
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-ast $(BIN)/som-rs/finalise_naive/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-bc $(BIN)/som-rs/finalise_naive/som-rs-bc

build-som-rs: som-rs/Cargo.lock som-rs-finalise-naive som-rs-finalise-elide som-rs-barriers-naive som-rs-barriers-none som-rs-barriers-opt

clean-builds:
	rm -rf som-rs
	rm -rf yksom
	rm -rf bin

som-rs/Cargo.lock:
	git clone --recursive $(SOMRS_REPO) som-rs
	cd som-rs && git checkout $(SOMRS_VERSION)

yksom/Cargo.toml:
	git clone --recursive $(YKSOM_REPO) yksom
	cd yksom && git checkout $(YKSOM_VERSION)

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf yksom
	rm -rf som-rs
	rm -rf $(BIN)

clean: clean-benchmarks clean-builds
