#!/bin/bash
# Test runner script with various options

set -e

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi

# Default test command
TEST_CMD="pytest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            TEST_CMD="$TEST_CMD --cov=metareason --cov-report=term-missing --cov-report=html"
            shift
            ;;
        --parallel)
            TEST_CMD="$TEST_CMD -n auto"
            shift
            ;;
        --verbose)
            TEST_CMD="$TEST_CMD -vv"
            shift
            ;;
        --fast)
            TEST_CMD="$TEST_CMD -m 'not slow'"
            shift
            ;;
        --integration)
            TEST_CMD="$TEST_CMD -m integration"
            shift
            ;;
        --unit)
            TEST_CMD="$TEST_CMD -m unit"
            shift
            ;;
        *)
            TEST_CMD="$TEST_CMD $1"
            shift
            ;;
    esac
done

echo "Running tests: $TEST_CMD"
$TEST_CMD
