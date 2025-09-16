# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a fresh start for MetaReason Core, an open-source tool for quantitative measurement of LLM and agentic AI systems. The previous implementation has been removed and the project is being rebuilt from scratch.

## Project Vision

The intended architecture includes:
- YAML-based specifications for evaluation configurations
- Jinja2 templating for prompt generation
- Latin Hypercube Sampling for parameter space exploration
- LLM Judges for evaluation
- Bayesian analysis for statistical rigor
- Pipeline-based execution model
- CLI interface
- Report generation

## Development Environment

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Code Quality Commands
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linting
flake8 src tests

# Security checks
bandit -r src

# Run all formatting
./scripts/format.sh

# Run tests (when implemented)
./scripts/test.sh
```

## Project Structure

Currently, the project is essentially empty:
- `src/metareason/` - Main package directory (currently only __init__.py)
- `tests/` - Test directory (currently empty)
- `scripts/` - Development scripts for formatting and testing
- `pyproject.toml` - Project configuration with dependencies

## Claude's Role

**Advisory Role Only**: Claude should act as an advisor, reviewer, and helper - NOT as the primary coding agent. The user writes all implementation code themselves.

- **Review and suggest**: Provide code reviews, answer technical questions, and suggest best practices when asked
- **No proactive coding**: Do not write implementation code unless explicitly requested
- **Explain and guide**: Focus on explaining concepts, pointing out issues, and providing guidance
- **Keyboard-first workflow**: Support the user's goal of staying on the keyboard and minimizing mouse usage

## Important Notes

- **Python 3.13+ required**: The project targets Python 3.13 or higher
- **Test coverage requirement**: 80% coverage is configured but not currently enforced since there's no code

## Dependencies

Key dependencies specified in pyproject.toml:
- **pymc** - Bayesian analysis
- **pydantic** - Configuration validation
- **click** - CLI framework
- **jinja2** - Template rendering
- **httpx** - Async HTTP client
- **openai/anthropic/google-genai** - LLM API clients

## Development Philosophy

Given the fresh restart, focus on:
1. Simple, working implementations over complex abstractions
2. Incremental development with working code at each step
3. Real integration tests over mocked unit tests
4. Clear separation of concerns without over-engineering

- Always use venv to run any commands or python
