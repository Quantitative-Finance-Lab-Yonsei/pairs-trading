clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-test-dep set-git set-dev
# test: set-test-dep set-test


##### basic #####
set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip install ruff==0.6.8

set-test-dep:
	pip install pytest==8.3.2

set-precommit:
	pip install pre-commit==3.8.0
	pre-commit install

set-dev:
	pip install -r requirements.txt

set-test:
	python3 -m pytest tests/

set-style:
	ruff check --fix .
	ruff format .

check-quality:
	ruff check .
	ruff format --check .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
