#!/bin/bash
# Clean up temporary files and caches

echo "Cleaning up temporary files..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove test and coverage files
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
rm -rf .mypy_cache
rm -rf .ruff_cache

# Remove build artifacts
rm -rf build
rm -rf dist
rm -rf .eggs

# Remove cache directories
rm -rf .cache

echo "Cleanup complete!"
