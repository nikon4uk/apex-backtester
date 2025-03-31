.PHONY: test test-unit test-integration install lint

install:
	pip install -r requirements.txt

test:
	pytest

test-unit:
	pytest -s tests/unit --asyncio-mode=auto

test-integration:
	pytest tests/integration

lint:
	flake8 .
