# Windows: run from Git Bash. PY points at the project venv.
PY ?= .venv/Scripts/python
CFG ?= config/default.yaml

.PHONY: all setup test lint character data clean
all:
	$(PY) -m retarget_lab run --config $(CFG)

setup:
	python -m venv .venv && $(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m retarget_lab fetch-data --config $(CFG)

character:
	$(PY) -m retarget_lab character --config $(CFG)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

clean:
	rm -rf outputs/*/ .pytest_cache
