.PHONY: install dev test lint format audit clean help

PYTHON ?= python3
UV     ?= uv

help:
	@echo "Targets:"
	@echo "  install      install runtime deps via uv"
	@echo "  dev          install runtime + dev + ml deps"
	@echo "  test         run pytest (excludes integration)"
	@echo "  lint         ruff + mypy"
	@echo "  format       ruff format"
	@echo "  audit        run Phase 0 BQ data audit"
	@echo "  clean        remove caches and build artifacts"

install:
	$(UV) sync

dev:
	$(UV) sync --extra dev --extra ml

test:
	$(UV) run pytest -m "not integration and not slow"

lint:
	$(UV) run ruff check daity tests scripts
	$(UV) run mypy daity

format:
	$(UV) run ruff format daity tests scripts
	$(UV) run ruff check --fix daity tests scripts

audit:
	$(UV) run python -m daity.scripts.phase0_audit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
