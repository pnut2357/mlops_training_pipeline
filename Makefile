pippoetry:
	pip install --upgrade pip && pip install poetry
install:
	poetry install --with dev
test:
	poetry run python -m pytest -vv --cov=src --cov-config=.coveragerc
format:
	poetry run black ./src/
lint:
	poetry run pylint --disable=R,C ./src/pipeline.py
all: pippoetry install test