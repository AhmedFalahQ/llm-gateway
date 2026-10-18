.PHONY: help install test lint demo bench build deploy seed-key usage destroy \
        local-up local-init local-test local-demo local-smoke local-usage local-reset \
        local-down local-all

STACK ?= llm-gateway
REGION ?= us-east-1

# --- local development against DynamoDB Local -------------------------------
# One variable switches every target below from moto to a real DynamoDB over
# HTTP. Credentials are dummies: DynamoDB Local checks the shape of a signature,
# not its contents, and pinning them here means a stray AWS profile in your
# shell can never reach a real account from a local target.
LOCAL_ENDPOINT ?= http://localhost:8000
LOCAL_BUDGET_TABLE ?= llm-gateway-budgets
LOCAL_CACHE_TABLE ?= llm-gateway-cache
LOCAL_ENV = AWS_ENDPOINT_URL_DYNAMODB=$(LOCAL_ENDPOINT) \
            AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local \
            AWS_SESSION_TOKEN= AWS_DEFAULT_REGION=us-east-1 \
            BUDGET_TABLE=$(LOCAL_BUDGET_TABLE) CACHE_TABLE=$(LOCAL_CACHE_TABLE)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies
	python -m pip install -e ".[dev]"

test: ## Run the test suite against moto (no AWS account or API key needed)
	python -m pytest

lint: ## Lint and format-check
	python -m ruff check .

demo: ## Run the gateway end to end against a fake provider, no cloud, no spend
	python bench/demo.py

bench: ## Replay the sample query set and report cache hit rate and $ saved
	python bench/replay.py --queries bench/queries.sample.jsonl

# --- local stack ------------------------------------------------------------

local-up: ## Start DynamoDB Local in Docker
	docker compose up -d
	@echo "DynamoDB Local starting on $(LOCAL_ENDPOINT) — run 'make local-init' next"

local-init: ## Create the local tables and mint a dev key
	$(LOCAL_ENV) python scripts/local_setup.py --endpoint $(LOCAL_ENDPOINT) \
		--budget-table $(LOCAL_BUDGET_TABLE) --cache-table $(LOCAL_CACHE_TABLE) \
		--seed-key $(or $(LABEL),dev) --limit $(or $(LIMIT),5)

local-test: ## Run the SAME suite against real DynamoDB instead of moto
	$(LOCAL_ENV) python -m pytest

local-demo: ## Run the end-to-end demo against real DynamoDB
	$(LOCAL_ENV) python bench/demo.py

local-smoke: ## Send a few requests through the gateway into the persistent local tables
	$(LOCAL_ENV) python scripts/local_smoke.py

local-usage: ## Per-key spend from the local tables
	$(LOCAL_ENV) python scripts/report_usage.py --table $(LOCAL_BUDGET_TABLE)

local-reset: ## Drop and recreate the local tables
	$(LOCAL_ENV) python scripts/local_setup.py --endpoint $(LOCAL_ENDPOINT) \
		--budget-table $(LOCAL_BUDGET_TABLE) --cache-table $(LOCAL_CACHE_TABLE) --reset

local-down: ## Stop DynamoDB Local
	docker compose down

local-all: local-init local-test local-demo local-smoke local-usage ## The whole local loop

# --- AWS --------------------------------------------------------------------

build: ## Package the Lambda with SAM
	sam build

deploy: build ## Deploy the stack (first run: adds --guided)
	sam deploy --stack-name $(STACK) --region $(REGION) \
		--capabilities CAPABILITY_IAM --resolve-s3 --no-confirm-changeset

seed-key: ## Create an API key with a monthly budget. make seed-key LIMIT=5 LABEL=dev
	python scripts/seed_key.py --stack $(STACK) --region $(REGION) \
		--limit $(or $(LIMIT),5) --label $(or $(LABEL),dev)

usage: ## Print this month's spend for every key in the deployed stack
	python scripts/report_usage.py --stack $(STACK) --region $(REGION)

destroy: ## Delete the stack and both tables
	sam delete --stack-name $(STACK) --region $(REGION) --no-prompts
