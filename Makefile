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
SRCS = $(PWD)/srcs

LINUX_SRC = $(SRCS)/artefacts/linux
LINUX_REPO = 'https://github.com/BurntSushi/linux'

HADOOP_SRC = $(SRCS)/hadoop
ECLIPSE_SRC = $(SRCS)/eclipse
SPRING_SRC = $(SRCS)/spring
JENKINS_SRC = $(SRCS)/jenkins

HADOOP_REPO = https://github.com/apache/hadoop
ECLIPSE_REPO = https://github.com/eclipse-platform/eclipse.platform
SPRING_REPO = https://github.com/spring-projects/spring-framework
JENKINS_REPO = https://github.com/jenkinsci/jenkins

$(LINUX_SRC):
	git clone --depth 1 $(LINUX_REPO) $@
	cd $@ && make defconfig
	cd $@ && make -j4

$(HADOOP_SRC):
	git clone $(HADOOP_REPO) $@ --depth 1

$(ECLIPSE_SRC):
	git clone $(ECLIPSE_REPO) $@ --depth 1

$(SPRING_SRC):
	git clone $(SPRING_REPO) $@ --depth 1

$(JENKINS_SRC):
	git clone $(JENKINS_REPO) $@ --depth 1

GRMTOOLS_SRCS = $(LINUX_SRC) $(HADOOP_REPO) $(ECLIPSE_SRC) \
	$(SPRING_SRC) $(JENKINS_SRC)

grmtools-extras: $(GRMTOOLS_SRCS)
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
