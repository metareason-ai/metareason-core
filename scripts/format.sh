#!/bin/bash
# Code formatting and linting script

set -e

echo "Running code formatters and linters..."

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi

echo "Running Black formatter..."
black src tests

echo "Running isort..."
isort src tests

echo "Running flake8..."
flake8 src tests || true

echo "Running mypy..."
mypy src || true

echo "Running bandit security checks..."
bandit -r src || true

echo "Code formatting complete!"
