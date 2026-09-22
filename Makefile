.PHONY: install lint fmt run

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

run:
	uv run uvicorn dota_winprob.app:app --reload --port 8000
