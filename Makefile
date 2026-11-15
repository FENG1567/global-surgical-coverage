PYTHON ?= python
.PHONY: verify figures tables smoke
verify:
	$(PYTHON) scripts/verify_release.py
figures:
	$(PYTHON) scripts/reproduce_figures.py
tables:
	$(PYTHON) scripts/reproduce_tables.py
smoke: verify tables
	$(PYTHON) -m unittest discover -s tests -v
