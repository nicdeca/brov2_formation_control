.PHONY: test lint format check

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
