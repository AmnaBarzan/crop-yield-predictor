install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

format:
	black .

lint:
	pylint app/ model/

all: install format lint test