ifeq ($(OS), Windows_NT)
	MAKE_OS:=Windows
	RMDIR_CMD=rmdir /Q /S
	SHELL:=cmd
else
	MAKE_OS:=Linux
	RMDIR_CMD=rm -rf
	SHELL:=/bin/sh
endif

PYTHON_VERSION=3.11
VENV_NAME=.venv
PIP_PREREQUISITES="pip>=24" wheel

ifeq ($(MAKE_OS), Windows)
	CREATE_ENV_CMD=py -$(PYTHON_VERSION) -m venv $(VENV_NAME)
	PYTHON=$(VENV_NAME)\Scripts\python
	ACTIVATE=$(VENV_NAME)\Scripts\activate
	CMDSEP=&
else
	CREATE_ENV_CMD=python$(PYTHON_VERSION) -m venv $(VENV_NAME)
	PYTHON=$(VENV_NAME)/bin/python
	ACTIVATE=source $(VENV_NAME)/bin/activate
	CMDSEP=;
endif

RUN_MODULE=$(PYTHON) -m
PIP=$(RUN_MODULE) pip

help:
	$(info MAKE: Showing all available make subcommands ('make help' or 'make info'):)
	@echo -------------------------------
	@echo Main development setup commands
	@echo -------------------------------
	@echo   - make install: Setup dev environment from fresh
	@echo   - make install-requirements: Install all requirements for dev environment in existing venv
	@echo   - make create-env: Initialize new virtual environment
	@echo   - make install-pre-commit: Install pre-commit hooks
	@echo -------------------------------
	@echo Main development commands
	@echo -------------------------------
	@echo   - make pre-commit: Run pre-commit hooks on all files
	@echo -------------------------------

info: help

install: create-env install-requirements install-pre-commit

create-env:
	$(info MAKE: Initializing environment in .venv ...)
	$(CREATE_ENV_CMD)
	$(PIP) install --upgrade $(PIP_PREREQUISITES)

install-build-requirements:
	$(info MAKE: Installing build requirements ...)
	$(PIP) install -r requirements-ci.txt

install-requirements:
	$(info MAKE: Installing development requirements ...)
	$(PIP) install -r requirements.txt

install-pre-commit:
	$(info MAKE: Installing pre-commit hooks...)
	$(RUN_MODULE) pre_commit install

pre-commit:
	$(info MAKE: Pre-commit hooks check over all files...)
	$(RUN_MODULE) pre_commit run --all-files
