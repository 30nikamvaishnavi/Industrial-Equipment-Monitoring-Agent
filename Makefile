PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)

.PHONY: install check_install lint data bench test eval readme all all-with-agents

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check_install:
	$(PYTHON) scripts/check_install.py

lint:
	$(PYTHON) -m ruff check .

data:
	$(PYTHON) -m data.generate_data

bench:
	$(PYTHON) -m forecasting.forecaster
	$(PYTHON) -m anomaly.detector

test:
	$(PYTHON) -m pytest -v

eval:
	$(PYTHON) -m evaluation.run_eval

readme:
	$(PYTHON) scripts/update_readme.py

all: install check_install lint data bench test readme

all-with-agents: all eval readme
