.PHONY: install test lint doctor ask eval serve baseline clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/
	python -m ruff format --check src/ tests/

format:
	python -m ruff format src/ tests/
	python -m ruff check --fix src/ tests/

doctor:
	python -m agent.cli doctor

ask:
	python -m agent.cli ask "$(Q)"

baseline:
	python -m agent.cli baseline "$(Q)"

eval:
	python evals/run_evals.py

serve:
	python -m uvicorn agent.api:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .cache/ runs/ dist/ build/ *.egg-info/

