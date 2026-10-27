.PHONY: data test sanity detect
PY ?= python

data:        ## download HPO files and build ml/data/processed/*
	cd ml && $(PY) scripts/prepare_data.py

test:        ## run all ML unit tests
	cd ml && $(PY) -m pytest -q

sanity:      ## print a few similarity values on REAL data to eyeball
	cd ml && $(PY) scripts/sanity_check.py

detect:      ## simulate a cohort from real HPO, run detection, write backend/data/*.json
	cd ml && $(PY) scripts/run_detection.py --n-jobs 4
