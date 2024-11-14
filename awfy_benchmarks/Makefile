PWD != pwd
BIN = $(PWD)/bin
PATCHES  = $(PWD)/patches

# VENV
PYTHON = python3
PYTHON_EXEC = $(VENV_DIR)/bin/python
VENV_DIR = $(PWD)/venv
PIP = $(VENV_DIR)/bin/pip

# SOURCES
SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_VERSION=master


SOM_RS_AST_PERF = $(addprefix results/, som_rs_ast_perf)
SOM_RS_BC_PERF = $(addprefix results/, som_rs_bc_perf)
SOM_RS_AST_BARRIERS = $(addprefix results/, som_rs_ast_barriers)
SOM_RS_BC_BARRIERS = $(addprefix results/, som_rs_bc_barriers)
SOM_RS_AST_ELISION = $(addprefix results/, som_rs_ast_elision)
SOM_RS_BC_ELISION = $(addprefix results/, som_rs_bc_elision)
SOM_RS_AST_EXPS = $(SOM_RS_AST_PERF) $(SOM_RS_AST_ELISION) $(SOM_RS_AST_BARRIERS)
SOM_RS_BC_EXPS = $(SOM_RS_BC_PERF) $(SOM_RS_BC_ELISION) $(SOM_RS_BC_BARRIERS)

YKSOM_ELISION = $(addprefix results/, yksom_elision)
YKSOM_BARRIERS = $(addprefix results/, yksom_barriers)
YKSOM_EXPS = $(YKSOM_ELISION) $(YKSOM_BARRIERS)

REBENCH_DATA = rebench.data
RESULTS = $(PWD)/results
RESULTS_PROCESSOR = ../process.py

all: bench

plot: plot-perf plot-barriers plot-elision

plot-perf: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/som_rs_ast_perf/rebench.data

plot-barriers: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/som_rs_ast_barriers/rebench.data
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/som_rs_bc_barriers/rebench.data
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/yksom_barriers/rebench.data

plot-elision: plot-setup
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/som_rs_ast_elision/rebench.data
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/som_rs_bc_elision/rebench.data
	$(PYTHON_EXEC) $(RESULTS_PROCESSOR) \
		$(RESULTS)/yksom_elision/rebench.data

plot-setup: venv
	mkdir -p plots

bench-perf: $(SOM_RS_AST_PERF) $(SOM_RS_BC_PERF)

bench-elision: $(SOM_RS_AST_ELISION) $(SOM_RS_BC_ELISION) $(YKSOM_ELISION)

bench-barriers: $(SOM_RS_AST_BARRIERS) $(SOM_RS_BC_BARRIERS) $(YKSOM_BARRIERS)

$(SOM_RS_AST_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-ast.conf $(notdir $@)

$(SOM_RS_BC_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-bc.conf $(notdir $@)


$(YKSOM_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		yksom.conf $(notdir $@)



build: sources build-perf build-barriers build-elision

build-perf: sources som-rs-perf-gc som-rs-perf-rc

build-barriers: sources yksom-barriers-none yksom-barriers-naive yksom-barriers-opt \
	som-rs-barriers

build-elision: sources yksom-finalise-elide yksom-finalise-naive \
	som-rs-elision

sources: som-rs/Cargo.lock yksom/Cargo.toml

som-rs-barriers:
	mkdir -p $(BIN)/som-rs/barriers_naive
	mkdir -p $(BIN)/som-rs/barriers_opt
	mkdir -p $(BIN)/som-rs/barriers_none
	cd $(PWD)/som-rs && git reset --hard
	cd som-rs && git apply $(PATCHES)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCHES)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=barriers_naive
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=barriers_none
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=barriers_opt
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/core-lib
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-ast $(BIN)/som-rs/barriers_naive/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-ast $(BIN)/som-rs/barriers_none/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-ast $(BIN)/som-rs/barriers_opt/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-bc $(BIN)/som-rs/barriers_naive/som-rs-bc
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-bc $(BIN)/som-rs/barriers_none/som-rs-bc
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-bc $(BIN)/som-rs/barriers_opt/som-rs-bc

som-rs-elision:
	mkdir -p $(BIN)/som-rs/finalise_elide
	mkdir -p $(BIN)/som-rs/finalise_naive
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCHES)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCHES)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=finalise_elide
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_elide/core-lib
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-ast $(BIN)/som-rs/finalise_elide/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-bc $(BIN)/som-rs/finalise_elide/som-rs-bc
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_naive/core-lib
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-ast $(BIN)/som-rs/finalise_naive/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-bc $(BIN)/som-rs/finalise_naive/som-rs-bc

som-rs-finalise-naive:
	- rm -rf $(BIN)/som-rs/finalise_naive
	mkdir -p $(BIN)/som-rs/finalise_naive
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCHES)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCHES)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_naive/core-lib
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-ast $(BIN)/som-rs/finalise_naive/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-bc $(BIN)/som-rs/finalise_naive/som-rs-bc

som-rs-perf-rc:
	- rm -rf $(BIN)/som-rs/perf_rc
	mkdir -p $(BIN)/som-rs/perf_rc
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCHES)/som-rs/bdwgc_allocator.patch
	cd som-rs && git apply $(PATCHES)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=perf_rc
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/perf_rc/core-lib
	ln -s $(PWD)/som-rs/perf_rc/release/som-interpreter-ast $(BIN)/som-rs/perf_rc/som-rs-ast
	ln -s $(PWD)/som-rs/perf_rc/release/som-interpreter-bc $(BIN)/som-rs/perf_rc/som-rs-bc

som-rs-perf-gc:
	- rm -rf $(BIN)/som-rs/perf_gc
	mkdir -p $(BIN)/som-rs/perf_gc
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCHES)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCHES)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=perf_gc
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/perf_gc/core-lib
	ln -s $(PWD)/som-rs/perf_gc/release/som-interpreter-ast $(BIN)/som-rs/perf_gc/som-rs-ast
	ln -s $(PWD)/som-rs/perf_gc/release/som-interpreter-bc $(BIN)/som-rs/perf_gc/som-rs-bc

yksom-finalise-naive:
	- rm -rf $(BIN)/yksom/finalise_naive
	mkdir -p $(BIN)/yksom/finalise_naive
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCHES)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
	      cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/finalise_naive/SOM
	ln -s $(PWD)/yksom/finalise_naive/release/yksom $(BIN)/yksom/finalise_naive/yksom

yksom-finalise-elide:
	- rm -rf $(BIN)/yksom/finalise_elide
	mkdir -p $(BIN)/yksom/finalise_elide
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCHES)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
	      cargo build --release --target-dir=finalise_elide
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/finalise_elide/SOM
	ln -s $(PWD)/yksom/finalise_elide/release/yksom $(BIN)/yksom/finalise_elide/yksom

yksom-barriers-naive:
	- rm -rf $(BIN)/yksom/barriers_naive
	mkdir -p $(BIN)/yksom/barriers_naive
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCHES)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
	      cargo build --release --target-dir=barriers_naive
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_naive/SOM
	ln -s $(PWD)/yksom/barriers_naive/release/yksom $(BIN)/yksom/barriers_naive/yksom

yksom-barriers-opt:
	- rm -rf $(BIN)/yksom/barriers_opt
	mkdir -p $(BIN)/yksom/barriers_opt
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCHES)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
	      cargo build --release --target-dir=barriers_opt
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_opt/SOM
	ln -s $(PWD)/yksom/barriers_opt/release/yksom $(BIN)/yksom/barriers_opt/yksom

yksom-barriers-none:
	- rm -rf $(BIN)/yksom/barriers_none
	mkdir -p $(BIN)/yksom/barriers_none
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCHES)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
	      cargo build --release --target-dir=barriers_none
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_none/SOM
	ln -s $(PWD)/yksom/barriers_none/release/yksom $(BIN)/yksom/barriers_none/yksom

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

clean-plots:
	rm -rf plots

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean: clean-confirm clean-builds clean-benchmarks clean-plots
	@echo "Clean"
