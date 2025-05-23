export PEXECS ?= 30
export BENCHMARKS ?= som grmtools alacritty fd regex-redux binary-trees ripgrep
export EXPERIMENTS ?= gcvs premopt elision
export METRICS ?= perf mem

PWD != pwd

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python

.PHONY: venv
.PHONY: build
.PHONY: bench
.PHONY: clean

# Artefacts
SRCS = $(PWD)/src

LINUX_SRC = $(SRCS)/artefacts/linux
LINUX_REPO = 'https://github.com/BurntSushi/linux'

HADOOP_SRC = $(SRCS)/artefacts/hadoop
ECLIPSE_SRC = $(SRCS)/artefacts/eclipse
SPRING_SRC = $(SRCS)/artefacts/spring
JENKINS_SRC = $(SRCS)/artefacts/jenkins

HADOOP_REPO = https://github.com/apache/hadoop
ECLIPSE_REPO = https://github.com/eclipse-platform/eclipse.platform
SPRING_REPO = https://github.com/spring-projects/spring-framework
JENKINS_REPO = https://github.com/jenkinsci/jenkins

CACTUS_SRC = $(SRCS)/cactus
CACTUS_REPO = https://github.com/softdevteam/cactus
CACTUS_VERSION = 8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad

REGEX_SRC = $(SRCS)/regex
REGEX_REPO = https://github.com/rust-lang/regex
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791

GRMTOOLS_BUILD_SRCS = $(CACTUS_SRC) $(REGEX_SRC)
GRMTOOLS_PARSE_SRCS = $(JENKINS_SRC) $(HADOOP_SRC) $(SPRING_SRC) $(ECLIPSE_SRC)

$(LINUX_SRC):
	git clone --depth 1 $(LINUX_REPO) $@
	cd $@ && make defconfig
	cd $@ && make -j100

$(HADOOP_SRC):
	git clone $(HADOOP_REPO) $@ --depth 1

$(ECLIPSE_SRC):
	git clone $(ECLIPSE_REPO) $@ --depth 1

$(SPRING_SRC):
	git clone $(SPRING_REPO) $@ --depth 1

$(JENKINS_SRC):
	git clone $(JENKINS_REPO) $@ --depth 1

$(CACTUS_SRC):
	git clone $(CACTUS_REPO) $@
	cd $@ && git checkout $(CACTUS_VERSION)

$(REGEX_SRC):
	git clone $(REGEX_REPO) $@
	cd $@ && git checkout $(REGEX_VERSION)


grmtools-extra: $(GRMTOOLS_PARSE_SRCS) $(GRMTOOLS_BUILD_SRCS)
	cargo install \
		--target-dir=$(PWD)/build/errorgen \
		--path=$(PWD)/benchmarks/grmtools/errorgen \
		--root $(PWD)/bin/artefacts/errorgen
		$(PWD)/bin/artefacts/errorgen/bin/errorgen $(HADOOP_SRC)
		$(PWD)/bin/artefacts/errorgen/bin/errorgen $(ECLIPSE_SRC)
		$(PWD)/bin/artefacts/errorgen/bin/errorgen $(SPRING_SRC)
		$(PWD)/bin/artefacts/errorgen/bin/errorgen $(JENKINS_SRC)




grep-extras: $(LINUX_SRC)

archive-results:
	mkdir -p archive
	tar --exclude='*/elision' -czvf archive/alloy_experiment_data.tar.gz results

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
