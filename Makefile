.PHONY: data test sanity detect benchmark benchmark-full benchmark-smoke
PY ?= python

data:        ## download HPO files and build ml/data/processed/*
	cd ml && $(PY) scripts/prepare_data.py

test:        ## run all ML unit tests
	cd ml && $(PY) -m pytest -q

sanity:      ## print a few similarity values on REAL data to eyeball
	cd ml && $(PY) scripts/sanity_check.py

detect:      ## simulate a cohort from real HPO, run detection, write backend/data/*.json
	cd ml && $(PY) scripts/run_detection.py --n-jobs 4

benchmark:   ## ONE command: quick benchmark on real HPO data -> benchmarks/reports/benchmark.md + plots
	$(PY) benchmarks/run_benchmark.py --preset quick

benchmark-full:  ## the long run behind the README numbers (leave it running; add --resume after an interruption)
	$(PY) benchmarks/run_benchmark.py --preset full

benchmark-smoke: ## sanity check on a synthetic ontology, no downloads (used in CI); numbers are NOT results
	$(PY) benchmarks/run_benchmark.py --preset smoke --synthetic --workers 1
