.PHONY: test coverage coverage-html coverage-report clean install

# Run tests with coverage
test:
	python -m pytest -n 3 --cov=scale --cov-report=term-missing

# Generate coverage report
coverage:
	python -m pytest -n 3 --cov=scale --cov-report=xml --cov-report=term-missing

# Generate HTML coverage report
coverage-html:
	python -m pytest -n 3 --cov=scale --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Show coverage report
coverage-report:
	python -m coverage report --show-missing

# Clean coverage files
clean:
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Install development dependencies
install:
	python -m pip install --upgrade pip
	python -m pip install pytest pytest-cov pytest-xdist
	python -m pip install -r requirements.txt
	python -m pip install -e . 