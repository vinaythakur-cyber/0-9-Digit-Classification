# Common tasks. `make help` lists them.
.PHONY: help setup test train evaluate serve samples docker clean
.DEFAULT_GOAL := help

PY := PYTHONPATH=src python3

help:  ## show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## install dependencies (CPU-only torch)
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

train:  ## train the model (downloads MNIST on first run, ~20 min on CPU)
	$(PY) -m digitood.training.train --config configs/default.yaml

evaluate:  ## calibrate thresholds, write reports/metrics.json and figures
	$(PY) -m digitood.evaluation.evaluate

samples:  ## regenerate the negative-families contact sheet
	$(PY) scripts/make_samples.py

serve:  ## run the web app at http://localhost:8000
	$(PY) -m uvicorn app.main:app --reload --port 8000

docker:  ## build and run the container
	docker build -t digit-ood .
	docker run --rm -p 8000:8000 digit-ood

clean:  ## remove caches (keeps datasets and the trained model)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
