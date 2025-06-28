# Makefile for PlanetScope-py development workflow
.PHONY: help install install-dev test test-quick lint format clean build docs

# Default target
help:
	@echo "PlanetScope-py Development Commands"
	@echo "==================================="
	@echo "install      Install package in development mode"
	@echo "install-dev  Install with development dependencies"
	@echo "test         Run all tests with coverage"
	@echo "test-quick   Run quick tests (unit tests only)"
	@echo "lint         Run code linting (flake8)"
	@echo "format       Format code with black"
	@echo "format-check Check code formatting without changing"
	@echo "clean        Clean build artifacts"
	@echo "build        Build package for distribution"
	@echo "docs         Build documentation"
	@echo "ci-setup     Set up CI/CD environment"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -e ".[docs]"
	pre-commit install

# Testing commands
test:
	pytest tests/ --cov=planetscope_py --cov-report=html --cov-report=term-missing -v

test-quick:
	pytest tests/ -m "not slow" -x -v

test-unit:
	pytest tests/ -m "unit" -v

test-integration:
	pytest tests/ -m "integration" -v

# Code quality commands
lint:
	flake8 planetscope_py/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black planetscope_py/ tests/

format-check:
	black --check planetscope_py/ tests/

type-check:
	mypy planetscope_py/

# Security checks
security:
	safety check
	bandit -r planetscope_py/

# Build commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

build-check: build
	twine check dist/*

# Documentation commands
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# CI/CD setup
ci-setup:
	@echo "Setting up CI/CD environment..."
	@echo "Creating .github/workflows directory..."
	mkdir -p .github/workflows
	@echo "CI/CD setup complete!"

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

# Quick development cycle
dev-cycle: format lint test-quick
	@echo "Development cycle complete!"

# Pre-commit workflow
pre-commit: format-check lint test-quick
	@echo "Pre-commit checks passed!"

# Release workflow
release-check: clean format-check lint test security build-check
	@echo "Release checks complete!"

# Docker commands (if using Docker)
docker-build:
	docker build -t planetscope-py:latest .

docker-test:
	docker run --rm planetscope-py:latest pytest tests/

# Environment info
env-info:
	@echo "Python version:"
	@python --version
	@echo "Pip version:"
	@pip --version
	@echo "Installed packages:"
	@pip list | grep -E "(planetscope|pytest|black|flake8)"