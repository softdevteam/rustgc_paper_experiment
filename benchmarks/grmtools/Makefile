GCVS_CFGS := gcvs/gc gcvs/rc
include ../../common.mk

GRMTOOLS_SRC = $(PWD)/grmtools
GRMTOOLS_REPO = https://github.com/softdevteam/grmtools
GRMTOOLS_VERSION = a0972be0777e599a3dbca710fb0a595c39560b69

CACTUS_SRC = $(PWD)/cactus
CACTUS_REPO = https://github.com/softdevteam/cactus
CACTUS_VERSION = 8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad

REGEX_SRC = $(PWD)/regex
REGEX_REPO = https://github.com/rust-lang/regex
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791

HADOOP_SRC = $(PWD)/hadoop
ECLIPSE_SRC = $(PWD)/eclipse
SPRING_SRC = $(PWD)/spring
JENKINS_SRC = $(PWD)/jenkins

HADOOP_REPO = https://github.com/apache/hadoop
ECLIPSE_REPO = https://github.com/eclipse-platform/eclipse.platform
SPRING_REPO = https://github.com/spring-projects/spring-framework
JENKINS_REPO = https://github.com/jenkinsci/jenkins

JAVA_SRCS = $(HADOOP_SRC) $(ECLIPSE_SRC) $(SPRING_SRC) $(JENKINS_SRC)
BUILD_SRCS = $(GRMTOOLS_SRC) $(CACTUS_SRC) $(REGEX_SRC)
PB_SRC = $(PWD)/parserbench
EG_SRC = $(PWD)/errorgen

.PHONY: all build clean

all: build

build: srcs $(TARGETS)

$(EG_SRC)/target/release:
	cd $(EG_SRC) && \
		RUSTC="$(ALLOY_PATH)/gcvs/perf/bin/rustc" cargo build --release
	$@/errorgen $(HADOOP_SRC)
	$@/errorgen $(ECLIPSE_SRC)
	$@/errorgen $(SPRING_SRC)
	$@/errorgen $(JENKINS_SRC)

$(BIN)/gcvs/%:
	$(call patch_repo,$(REGEX_SRC),$(patsubst %/,%,$(dir $*)))
	$(call patch_repo,$(CACTUS_SRC),$(patsubst %/,%,$(dir $*)))
	$(call build_gcvs,$*,$(PB_SRC),)
	ln -s $(PWD)/grmtools-harness.sh $@/bin/grmtools-harness

$(BIN)/%:
	$(call patch_repo,$(REGEX_SRC),alloy)
	$(call patch_repo,$(CACTUS_SRC),alloy)
	$(call build_alloy_exp,$*,$(PB_SRC),)
	ln -s $(PWD)/grmtools-harness.sh $@/bin/grmtools-harness

srcs: $(JAVA_SRCS) $(BUILD_SRCS) $(EG_SRC)/target/release

$(HADOOP_SRC):
	git clone $(HADOOP_REPO) $@ --depth 1

$(ECLIPSE_SRC):
	git clone $(ECLIPSE_REPO) $@ --depth 1

$(SPRING_SRC):
	git clone $(SPRING_REPO) $@ --depth 1

$(JENKINS_SRC):
	git clone $(JENKINS_REPO) $@ --depth 1

$(GRMTOOLS_SRC):
	git clone $(GRMTOOLS_REPO) $@
	cd $@ && git checkout $(GRMTOOLS_VERSION)
	cd $@ && git apply $(PATCH)/grmtools.diff

$(CACTUS_SRC):
	git clone $(CACTUS_REPO) $@
	cd $@ && git checkout $(CACTUS_VERSION)

$(REGEX_SRC):
	git clone $(REGEX_REPO) $@
	cd $@ && git checkout $(REGEX_VERSION)

clean-builds:
	rm -rf $(BIN) $(BUILD)

clean: clean-builds
	rm -rf $(BUILD_SRCS)
