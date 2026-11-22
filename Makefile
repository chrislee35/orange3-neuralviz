# Makefile for building Sphinx docs and the Python package

.PHONY: all docs build clean test ruff ruff-fix

all: docs build

docs:
	sphinx-build -b html doc/ doc/_build/html

build:
	python -m build

# Push package to pypi.org
release: build
	twine upload dist/*

clean:
	rm -rf build/ dist/ *.egg-info doc/_build

test:
	QT_QPA_PLATFORM=offscreen python -m pytest tests -q

ruff:
	ruff check orangecontrib/neuralviz

ruff-fix:
	ruff check --fix orangecontrib/neuralviz
