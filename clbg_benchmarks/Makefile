PWD != pwd
PYTHON = python3
BIN = $(PWD)/bin

EXPERIMENTS = binary_trees regex_redux
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791
REGEX_REPO = https://github.com/rust-lang/regex

.PHONY: build
.PHONY: clean clean-builds
.PHONY: $(EXPERIMENTS)

all: $(EXPERIMENTS)

build: $(BIN)/regex_redux $(BIN)/binary_trees

plot:
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary binary_trees regex_redux

clean-plots:
	- rm summary.csv

bench: $(EXPERIMENTS)

$(EXPERIMENTS): build
	mkdir -p $(PWD)/results/$@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/$(REBENCH_DATA) \
		clbg.conf $@

$(BIN)/regex_redux: regex
	mkdir -p $@
	cd $(PWD)/regex && git apply $(PWD)/regex_gc.patch
	cd $(PWD)/regex_redux && cargo build --release --bin alloy
	cd $(PWD)/regex && git reset --hard
	cd $(PWD)/regex_redux && cargo build --release --bin rc
	cd $(PWD)/regex_redux && cargo build --release --bin fasta
	$(PWD)/regex_redux/target/release/fasta 2500000 > redux_input.txt
	ln -s $(PWD)/$(notdir $@)/target/release/* $@/

$(BIN)/binary_trees:
	mkdir -p $@
	cd $(PWD)/binary_trees && cargo build --release
	@echo "Built with RUSTC=$$RUSTC"
	ln -s $(PWD)/$(notdir $@)/target/release/* $@/

regex:
	git clone $(REGEX_REPO) regex
	cd regex && git checkout $(REGEX_VERSION)

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf $(PWD)/binary_trees/target
	rm -rf $(PWD)/regex_redux/target
	rm -rf $(PWD)/regex
	rm -rf $(PWD)/regex
	rm -rf $(PWD)/bin
	rm -rf redux_input.txt

clean: clean-builds clean-benchmarks
