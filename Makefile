.PHONY: install test quick journal figures clean

install:
	python -m pip install -e '.[experiments,test]'

test:
	python -m pytest

quick: test
	bash scripts/reproduce.sh configs/quick.json results/quick

journal: test
	bash scripts/reproduce.sh configs/journal.json results/journal

figures:
	python experiments/make_figures.py --results results/journal --output results/journal/figures

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info src/*.egg-info
