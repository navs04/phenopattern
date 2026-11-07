.PHONY: data test sanity detect benchmark benchmark-full benchmark-smoke backend-data backend-run backend-test
PY ?= python

data:        ## download HPO files and build ml/data/processed/*
	cd ml && $(PY) scripts/prepare_data.py

test:        ## run all ML unit tests
	cd ml && $(PY) -m pytest -q

sanity:      ## print a few similarity values on REAL data to eyeball
	cd ml && $(PY) scripts/sanity_check.py

detect:      ## simulate a cohort from real HPO, run detection, write backend/data/*.json
	cd ml && $(PY) scripts/run_detection.py --n-jobs 4 $(DETECT_ARGS)   # e.g. DETECT_ARGS='--method agglomerative --tau 0.90'

benchmark:   ## ONE command: quick benchmark on real HPO data -> benchmarks/reports/benchmark.md + plots
	$(PY) benchmarks/run_benchmark.py --preset quick

benchmark-full:  ## the long run behind the README numbers (leave it running; add --resume after an interruption)
	$(PY) benchmarks/run_benchmark.py --preset full

benchmark-smoke: ## sanity check on a synthetic ontology, no downloads (used in CI); numbers are NOT results
	$(PY) benchmarks/run_benchmark.py --preset smoke --synthetic --workers 1

backend-data: ## copy benchmark results + plots next to the API's data (run `make detect` first)
	mkdir -p backend/data/plots
	@if [ -f benchmarks/reports/benchmark.json ]; then cp benchmarks/reports/benchmark.json backend/data/; else echo "no benchmark.json yet (skipping)"; fi
	@if ls benchmarks/reports/plots/*.png >/dev/null 2>&1; then cp benchmarks/reports/plots/*.png backend/data/plots/; else echo "no plots yet (skipping)"; fi

backend-run: ## start the API on http://localhost:8000 (docs at /docs)
	cd backend && uvicorn app.main:app --reload --port 8000

backend-test: ## backend tests (HTTP tests need fastapi + httpx)
	cd backend && $(PY) -m pytest -q
