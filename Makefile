.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Please use 'make <target>' where target is one of:"
	@echo ""
	@echo "	format-check		runs the formatting tools, only checking for errors"
	@echo "	format-fix		runs the formatting tools, fixing errors where possible"
	@echo "	lint			runs the linting tools, only checking for errors"
	@echo "	typing			runs the static type checker, only checking for errors"
	@echo "	static-check		runs the following targets: format-check, lint and typing"
	@echo "	static-fix		runs the following targets: format-fix, lint and typing"
	@echo "	unit-test		runs the unit tests"
	@echo ""


.PHONY: format-check
format-check:
	black --check src/ tests/
	isort --check-only --profile black src/ tests/

.PHONY: format-fix
format-fix:
	black src/ tests/
	isort --profile black src/ tests/

.PHONY: lint
lint:
	flake8 --statistics src/ tests/

.PHONY: typing
typing:
	mypy src/ tests/

.PHONY: static-check
static-check: format-check lint typing

.PHONY: static-fix
static-fix: format-fix lint typing

.PHONY: unit-test
unit-test:
	pytest --cov-report term-missing --cov=src/ -v -W ignore::DeprecationWarning
