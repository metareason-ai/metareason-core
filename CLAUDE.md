# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Development Guidelines

**Always activate the virtual environment before executing any Python commands or scripts. Use `source venv/bin/activate` before running pytest, pip, python, or any Python-related tools.**

## Common Development Commands

### Environment Setup
```bash
# Initial development setup
./scripts/setup-dev.sh

# Activate virtual environment
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs]"
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
./scripts/test.sh --coverage

# Run tests in parallel
./scripts/test.sh --parallel

# Run only fast tests (exclude slow tests)
./scripts/test.sh --fast

# Run specific test types
./scripts/test.sh --unit      # Unit tests only
./scripts/test.sh --integration  # Integration tests only

# Run specific test file
pytest tests/test_config_loader.py

# Run single test
pytest tests/test_config_loader.py::test_all_distribution_types -v
```

### Code Quality
```bash
# Run all formatters and linters
./scripts/format.sh

# Individual tools
black src tests           # Code formatting
isort src tests          # Import sorting
flake8 src tests         # Linting
bandit -r src           # Security checks
```

### CLI Usage
```bash
# Show system info
metareason info

# Validate configuration
metareason config validate examples/simple_evaluation.yaml

# Run evaluation (dry run)
metareason run examples/simple_evaluation.yaml --dry-run

# Run evaluation with output
metareason run examples/simple_evaluation.yaml --output results.json
```

## Project Architecture

MetaReason Core is a statistical evaluation framework for Large Language Models that uses Latin Hypercube Sampling and dual-oracle evaluation to provide rigorous confidence intervals.

### Core Architecture
- **CLI Module** (`metareason.cli`): Command-line interface using Click framework
- **Config Module** (`metareason.config`): YAML-based configuration with Pydantic validation
- **Sampling Module** (`metareason.sampling`): Latin Hypercube Sampling and optimization strategies
- **Oracles Module** (`metareason.oracles`): Evaluation criteria (accuracy, explainability, confidence calibration)
- **Analysis Module** (`metareason.analysis`): Bayesian statistical analysis with PyMC
- **Adapters Module** (`metareason.adapters`): LLM provider integrations
- **Utils Module** (`metareason.utils`): Shared utilities and helpers

### Key Design Patterns
- **Plugin Architecture**: Extensible via entry points for adapters, oracles, and sampling strategies
- **Async-First**: Core operations use async/await for performance
- **Configuration-Driven**: YAML specifications define entire evaluation workflows
- **Statistical Rigor**: Built-in Bayesian confidence intervals and uncertainty quantification

### YAML Configuration Schema
The project uses a declarative YAML format for evaluation specifications:
- **Prompt Templates**: Jinja2-compatible with variable substitution
- **Schema Definition**: Categorical and continuous parameter distributions
- **Sampling Configuration**: Latin Hypercube with optimization criteria
- **Oracle Definitions**: Multiple evaluation criteria (embedding similarity, LLM-as-judge, custom)
- **Statistical Configuration**: Bayesian inference settings

### Development Workflow
1. **Configuration Management**: All evaluations start with YAML configuration files
2. **Sampling Strategy**: Latin Hypercube Sampling generates parameter space coverage
3. **Oracle Evaluation**: Multiple oracles provide different quality assessments
4. **Statistical Analysis**: Bayesian models quantify confidence and uncertainty
5. **Results Output**: Rich console output and structured data formats

### Testing Strategy
- **Unit Tests**: Individual component testing with pytest
- **Integration Tests**: End-to-end workflow testing
- **Coverage**: Maintained at 80%+ with branch coverage
- **Test Markers**: `slow`, `integration`, `unit` for selective test execution

### Code Quality Standards
- **Type Hints**: Python type hints for better code clarity
- **Code Formatting**: Black (88 char line length) + isort
- **Linting**: flake8 with docstring and bugbear plugins
- **Security**: bandit security scanning
- **Pre-commit**: Automated checks before commits

### Current Development Status
The project is in active development (v0.1.0) with the following features in progress:
- YAML-based evaluation specifications ✅
- Latin Hypercube Sampling implementation ✅ (on feat/latin-hypercube-sampling branch)
- Configuration validation and loading ✅
- CLI interface foundation ✅
- Oracle evaluation framework (in development)
- Bayesian analysis integration (planned)
