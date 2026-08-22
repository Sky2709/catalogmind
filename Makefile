.DEFAULT_GOAL := help

COMPOSE := docker compose -f infra/docker-compose.yml
# Call the venv directly rather than through `uv run`. `uv` installs to ~/.local/bin,
# which non-interactive shells (make recipes, CI steps, cron) do not have on PATH -
# depending on it made every target fail with "uv: No such file or directory".
PY      := .venv/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

## ---- environment ----
install: ## Create the venv (Python 3.12) and install all deps
	uv python install 3.12
	uv venv --python 3.12
	UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
	UV_INDEX_STRATEGY=unsafe-best-match \
	uv pip install -e ".[dev]"

up: ## Start the 4 datastores and wait until they actually serve
	$(COMPOSE) up -d
	@$(MAKE) --no-print-directory wait

wait: ## Block until every datastore answers (weaviate has no container healthcheck)
	@echo "waiting for datastores..."
	@for i in $$(seq 1 60); do \
	  ready=$$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
	    http://localhost:8080/v1/.well-known/ready || true); \
	  if [ "$$ready" = "200" ]; then echo "  weaviate ready"; break; fi; \
	  if [ $$i -eq 60 ]; then echo "  weaviate NOT ready after 120s"; exit 1; fi; \
	  sleep 2; \
	done
	@$(COMPOSE) ps --format "table {{.Service}}\t{{.Status}}"

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

nuke: ## Stop the stack and DELETE all data volumes
	$(COMPOSE) down -v

logs: ## Tail stack logs
	$(COMPOSE) logs -f

ps: ## Show stack status
	$(COMPOSE) ps

psql: ## Open a psql shell on the app database
	$(COMPOSE) exec postgres psql -U catalogmind -d catalogmind

## ---- app ----
dev: ## Run the API with reload -> http://localhost:8000/docs
	$(PY) -m uvicorn app.main:app --reload --port 8000

migrate: ## Apply database migrations
	$(PY) -m alembic upgrade head

revision: ## Autogenerate a migration:  make revision m="what changed"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

seed: ## Provision the 3 demo merchants and ingest their catalogs
	$(PY) -m scripts.seed

## ---- quality ----
lint: ## Ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt: ## Auto-format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck: ## mypy
	$(PY) -m mypy app eval

helm-lint: ## Helm chart sanity - schema/template check, no cluster needed
	helm lint infra/helm
	helm template infra/helm > /dev/null

test: ## Unit tests only (no stack needed)
	$(PY) -m pytest -m "not integration" -q

test-all: ## All tests (requires `make up`)
	$(PY) -m pytest -q

check: lint typecheck test ## Everything CI runs

## ---- evaluation ----
eval: ## Run the retrieval eval suite and regenerate README numbers + charts
	$(PY) -m eval.retrieval_eval
	$(PY) -m eval.report
	# eval.generation_eval (groundedness/hallucination) joins this target on Day 6,
	# once the chat agent it evaluates exists - see PROGRESS.md.

sweep: ## Run the alpha sweep experiment (the headline chart)
	$(PY) -m eval.sweep_alpha

.PHONY: help install up wait down nuke logs ps psql dev migrate revision seed \
        lint fmt typecheck test test-all check eval sweep
