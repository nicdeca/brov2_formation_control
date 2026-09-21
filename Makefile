.PHONY: test lint format check public-check

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
	uv run ruff check .
	uv run ruff format --check .

public-check:
	uv run python scripts/check_public_repo.py
