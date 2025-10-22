# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

MetaReason Core is in active development with core features implemented and working. The project provides a quantitative framework for measuring LLM and agentic AI systems using Latin Hypercube Sampling and LLM-based evaluation judges.

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

The project has the following structure:

- `src/metareason/` - Main package directory
  - `config/` - Pydantic models for YAML specification validation
  - `adapters/` - LLM adapter implementations (currently: Ollama)
  - `oracles/` - Evaluation oracles (currently: LLM Judge)
  - `sampling/` - Sampling strategies (currently: Latin Hypercube)
  - `pipeline/` - Template rendering and execution pipeline
  - `cli/` - Click-based command-line interface
- `tests/` - Test directory (currently: LHS sampler tests with 11 test cases)
- `test_suite/` - Example specifications for testing
- `scripts/` - Development scripts for formatting and testing
- `pyproject.toml` - Project configuration with dependencies

## Implementation Status

### ✅ Completed Features
- YAML-based specification system with Pydantic validation
- Jinja2 template rendering for prompts
- Latin Hypercube Sampling with multiple distributions (uniform, normal, truncnorm, beta)
- LLM Judge oracle for evaluations
- Async pipeline execution with multi-stage support
- CLI commands: `metareason run` and `metareason validate`
- Ollama adapter for local models

### 🚧 In Progress
- Bayesian analysis with PyMC (dependency installed, not yet implemented)
- Report generation (currently JSON only, HTML/PDF planned)
- Additional adapters (OpenAI, Anthropic, Google GenAI)
- Comprehensive test coverage (currently ~20%, target 80%)

### 📋 Planned
- Additional oracle types (regex, keyword, statistical)
- Additional sampling methods beyond LHS
- Visualization and statistical reporting

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
