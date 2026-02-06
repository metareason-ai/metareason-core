# Contributing to MetaReason Core

Thank you for your interest in contributing to MetaReason Core! This guide covers everything you need to get started.

## Development Environment Setup

### Prerequisites

- **Python 3.13+** (required)
- **git** with pre-commit hook support
- A virtual environment tool (built-in `venv` recommended)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/metareason-ai/metareason-core.git
cd metareason-core

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Your Setup

```bash
# Run the test suite
./scripts/test.sh

# Run formatting checks
./scripts/format.sh

# Validate an example spec
metareason validate examples/quantum_entanglement_eval.yml
```

## Code Style and Formatting

We use automated tools enforced via pre-commit hooks:

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Code formatting (line length 88) | `pyproject.toml [tool.black]` |
| **isort** | Import sorting (black-compatible profile) | `pyproject.toml [tool.isort]` |
| **flake8** | Linting | `.flake8` |
| **bandit** | Security checks | `pyproject.toml [tool.bandit]` |

### Running Manually

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Security scan
bandit -r src

# Or run all formatting at once
./scripts/format.sh
```

Pre-commit hooks run automatically on `git commit` and will auto-fix formatting issues. If a hook modifies files, simply re-stage and commit again.

## Testing

### Requirements

- **80% minimum coverage** (enforced by pytest-cov in CI)
- All tests must pass before merging

### Running Tests

```bash
# Run full test suite with coverage
./scripts/test.sh

# Run specific test file
pytest tests/test_adapters.py -v

# Run a specific test class or method
pytest tests/test_cli.py::TestReportCommand -v

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

- Place tests in `tests/` with filenames matching `test_*.py`
- Use class-based grouping: `class TestFeatureName:`
- Use `pytest` fixtures and `unittest.mock` for mocking external dependencies (LLM APIs, file I/O)
- Mock LLM adapters and BayesianAnalyzer in CLI tests to avoid real API calls
- Use `tmp_path` fixture for tests that create files

## Branch and Issue Naming

### Branch Names

Use the format: `<type>/<short-description>`

| Type | Use For |
|------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `test/` | Test additions/improvements |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |

Examples: `feat/report-cli`, `fix/adapter-timeout`, `test/oracle-coverage`

### Commit Messages

Use conventional commit format:

```
<type>: <short description>

<optional longer description>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/my-feature main
   ```

2. **Make your changes** following the code style guidelines above.

3. **Run tests and linting** before committing:
   ```bash
   ./scripts/test.sh
   flake8 src tests
   ```

4. **Commit and push**:
   ```bash
   git add <specific-files>
   git commit -m "feat: add my feature"
   git push -u origin feat/my-feature
   ```

5. **Open a Pull Request** against `main` with:
   - A clear title describing the change
   - A summary of what changed and why
   - A test plan describing how to verify the changes

6. **Address review feedback** and ensure all checks pass.

## Project Structure

```
src/metareason/
  adapters/      # LLM provider adapters (Ollama, OpenAI, Google, Anthropic)
  analysis/      # Bayesian statistical analysis with PyMC
  cli/           # Click-based CLI (run, validate, analyze, report)
  config/        # Pydantic models for YAML spec validation
  oracles/       # Evaluation oracles (LLM Judge)
  pipeline/      # Template rendering and async execution
  reporting/     # HTML report generation with visualizations
  sampling/      # Latin Hypercube Sampling
tests/           # Test suite (pytest)
examples/        # Example YAML specifications
scripts/         # Dev scripts (format.sh, test.sh)
```

## Questions?

Open an issue on GitHub if you have questions or run into problems.
