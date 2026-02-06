#!/usr/bin/env bash
set -e

echo "Running black..."
black src tests

echo "Running isort..."
isort src tests

echo "Formatting complete."
