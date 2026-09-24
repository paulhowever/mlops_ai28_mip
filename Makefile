.PHONY: install hooks lint fmt run up down logs release

install:
	uv sync

hooks:
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

run:
	uv run uvicorn dota_winprob.app:app --reload --port 8000

up:
	GIT_SHA=$$(git rev-parse --short HEAD) BUILT_AT=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
		docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

release:
	@test -n "$(VERSION)" || { echo "использование: make release VERSION=1.2.3"; exit 1; }
	@python3 -c "import pathlib, re; p = pathlib.Path('pyproject.toml'); p.write_text(re.sub(r'^version = \".*\"$$', 'version = \"$(VERSION)\"', p.read_text(), count=1, flags=re.M))"
	uv lock
	git add pyproject.toml uv.lock
	git commit -m "chore: релиз $(VERSION)"
	git tag -a v$(VERSION) -m "v$(VERSION)"
	git push --follow-tags
