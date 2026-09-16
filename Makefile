.PHONY: help install lint format mypy test integration_test check all bump bump-minor bump-major

help:
	@echo "Targets:"
	@echo "  install           Sync the venv with all dependency groups (uv)"
	@echo "  lint              Run ruff lint + format checks"
	@echo "  format            Auto-fix lint issues and format code"
	@echo "  mypy              Type-check with mypy (disallow_untyped_defs)"
	@echo "  test              Run unit tests (sockets disabled)"
	@echo "  integration_test  Run integration tests (needs ZeroGPU creds)"
	@echo "  check             Run lint + mypy + unit tests"
	@echo "  bump              Bump the patch version (bug fix)     — see RELEASING.md"
	@echo "  bump-minor        Bump the minor version (new feature)"
	@echo "  bump-major        Bump the major version (breaking change)"

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

# Updates pyproject.toml and uv.lock only — no commit, no tag. Merging the PR
# to main releases it.
bump:
	uv version --bump patch

bump-minor:
	uv version --bump minor

bump-major:
	uv version --bump major
