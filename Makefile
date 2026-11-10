.PHONY: install ml-install hooks lint fmt run up down logs release data eda train \
	tasks-file source workers worker-logs worker-health experiment \
	reconfigure prefetch poison poison-retry kill-worker \
	obs-up bootstrap obs-down obs-wipe obs-logs obs-stats logs-up traces-up metrics-up

WORKERS ?= 2

install:
	uv sync

ml-install:
	uv sync --group ml

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

BUILD_ENV = GIT_SHA=$$(git rev-parse --short HEAD) BUILT_AT=$$(date -u +%Y-%m-%dT%H:%M:%SZ)

up:
	$(BUILD_ENV) docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

release:
	@test -n "$(VERSION)" || { echo "использование: make release VERSION=1.2.3"; exit 1; }
	uv version $(VERSION)
	git add pyproject.toml uv.lock
	git commit -m "chore: релиз $(VERSION)"
	git tag -a v$(VERSION) -m "v$(VERSION)"
	git push --follow-tags

test:
	uv run pytest --cov --cov-report=term-missing

data:
	mkdir -p data/raw
	uv run --group ml kaggle datasets download devinanzelmo/dota-2-matches -f match.csv -p data/raw
	uv run --group ml kaggle datasets download devinanzelmo/dota-2-matches -f player_time.csv -p data/raw
	cd data/raw && for z in *.zip; do unzip -o -q "$$z" && rm -f "$$z"; done

eda:
	uv run --group ml python -m ml.eda

train:
	uv run --group ml python -m ml.train

tasks-file:
	uv run --group ml python -m ml.tasks_file

source:
	docker compose --profile batch run --rm source

workers:
	docker compose up -d --no-recreate --scale worker=$(WORKERS) worker

worker-logs:
	docker compose logs -f worker

experiment:
	uv run --group ml python -m experiments.run

worker-health:
	@for index in $$(seq 1 $$(docker compose ps -q worker | wc -l)); do \
		port=$$(docker compose port --index $$index worker 8001 | cut -d: -f2); \
		curl -s "localhost:$$port/health"; echo; \
	done

RABBITMQ_API = http://localhost:$${RABBITMQ_MANAGEMENT_PORT:-15672}/api
RABBITMQ_AUTH = $${RABBITMQ_USER:-dota}:$${RABBITMQ_PASSWORD:-dota}

reconfigure:
	@for queue in tasks.score tasks.retry tasks.dlq; do \
		curl -s -o /dev/null -u "$(RABBITMQ_AUTH)" -X DELETE "$(RABBITMQ_API)/queues/%2F/$$queue"; \
	done
	docker compose up -d --no-deps --force-recreate app worker

prefetch:
	docker compose up -d --no-deps --force-recreate worker

poison:
	curl -s -u "$(RABBITMQ_AUTH)" -X POST "$(RABBITMQ_API)/exchanges/%2F/tasks/publish" \
		-H 'content-type: application/json' \
		-d '{"properties": {"delivery_mode": 2}, "routing_key": "score", "payload": "not a task", "payload_encoding": "string"}'
	@echo

POISON_FEATURES = {\"radiant_gold\": 13500, \"dire_gold\": 12400, \"gold_advantage\": 1100, \"radiant_xp\": 14200, \"dire_xp\": 13600, \"xp_advantage\": 600, \"radiant_lh\": 125, \"dire_lh\": 116, \"lh_advantage\": 9}

poison-retry:
	curl -s -u "$(RABBITMQ_AUTH)" -X POST "$(RABBITMQ_API)/exchanges/%2F/tasks/publish" \
		-H 'content-type: application/json' \
		-d '{"properties": {"delivery_mode": 2, "message_id": "poison-retry", "content_type": "application/json"}, "routing_key": "score", "payload_encoding": "string", "payload": "{\"task_id\": \"poison-retry\\u0000\", \"source\": \"batch\", \"published_at\": \"'"$$(date -u +%Y-%m-%dT%H:%M:%SZ)"'\", \"features\": $(POISON_FEATURES)}"}'
	@echo

kill-worker:
	docker kill $$(docker compose ps -q worker | head -1)

OBS_SERVICES = otel-collector alloy loki tempo prometheus grafana

obs-up:
	$(BUILD_ENV) docker compose --profile observability up -d --build

bootstrap: obs-up
	$(MAKE) train
	docker compose up -d --no-deps --force-recreate worker
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' $$(docker compose ps -q worker | head -1))" = healthy ]; do sleep 3; done
	@echo "воркер готов"

obs-down:
	docker compose --profile observability down

obs-wipe:
	docker compose --profile observability down --volumes

logs-up:
	docker compose --profile logs up -d

traces-up:
	docker compose --profile traces up -d

metrics-up:
	docker compose --profile metrics up -d

obs-logs:
	docker compose --profile observability logs -f $(OBS_SERVICES)

obs-stats:
	docker stats --no-stream $$(docker compose --profile observability ps -q $(OBS_SERVICES))
