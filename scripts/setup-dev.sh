#!/bin/bash
# Development environment setup script

set -e

echo "Setting up MetaReason development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev,docs]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please update .env with your API keys and configuration"
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p reports
mkdir -p .cache
mkdir -p config

echo "Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run linting and formatting:"
echo "  ./scripts/format.sh"
