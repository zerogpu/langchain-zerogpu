.PHONY: help install lint format mypy test integration_test check all

help:
	@echo "Targets:"
	@echo "  install           Sync the venv with all dependency groups (uv)"
	@echo "  lint              Run ruff lint + format checks"
	@echo "  format            Auto-fix lint issues and format code"
	@echo "  mypy              Type-check with mypy (disallow_untyped_defs)"
	@echo "  test              Run unit tests (sockets disabled)"
	@echo "  integration_test  Run integration tests (needs ZeroGPU creds)"
	@echo "  check             Run lint + mypy + unit tests"

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

mypy:
	uv run mypy langchain_zerogpu

test:
	uv run pytest --disable-socket tests/unit_tests

integration_test:
	uv run pytest tests/integration_tests

check: lint mypy test

all: check
