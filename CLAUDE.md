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

# Generate templates from configuration
metareason templates generate examples/simple_evaluation.yaml --output templates.json

# Run evaluation (dry run)
metareason run examples/simple_evaluation.yaml --dry-run

# Run evaluation with output
metareason run examples/simple_evaluation.yaml --output results.json

# Quick start with local models
metareason run examples/ollama_quickstart.yaml --dry-run
```

## Project Architecture

MetaReason Core is a statistical evaluation framework for Large Language Models that uses Latin Hypercube Sampling and dual-oracle evaluation to provide rigorous confidence intervals.

### Core Architecture
- **CLI Module** (`metareason.cli`): Command-line interface using Click framework
- **Config Module** (`metareason.config`): YAML-based configuration with Pydantic validation
- **Sampling Module** (`metareason.sampling`): Latin Hypercube Sampling and optimization strategies
- **Oracles Module** (`metareason.oracles`): Evaluation criteria including LLM-as-Judge, quality assurance, and custom metrics
- **Analysis Module** (`metareason.analysis`): Bayesian statistical analysis with PyMC
- **Adapters Module** (`metareason.adapters`): LLM provider integrations (OpenAI, Anthropic, Ollama)
- **Templates Module** (`metareason.templates`): Jinja2-based prompt generation and rendering
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
- **Oracle Definitions**: Multiple evaluation criteria (LLM-as-judge, quality assurance, embedding similarity, custom)
- **Statistical Configuration**: Bayesian inference settings

### Adapter System
The project includes a comprehensive adapter system for LLM providers:

#### Supported Providers
- **OpenAI**: Full API integration with rate limiting and retry logic
- **Anthropic**: Complete Claude API support with async operations
- **Ollama**: Local model serving with support for Llama, Mistral, Gemma, and other open-source models
- **Azure OpenAI**: Configuration ready (implementation planned)
- **HuggingFace**: Configuration ready (implementation planned)

#### Key Features
- **Plugin Architecture**: Registry-based adapter discovery and loading
- **Rate Limiting**: Configurable rate limits with automatic backoff
- **Retry Logic**: Exponential backoff with configurable retry strategies
- **Error Handling**: Comprehensive error types for different failure modes
- **Async Support**: Full async/await support for performance
- **Privacy Protection**: Built-in data sanitization and privacy-safe logging
- **Local Model Support**: Full support for local model serving via Ollama

#### Configuration
```yaml
adapters:
  # Cloud provider example
  cloud:
    type: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 80000
    retry:
      max_attempts: 3
      backoff_factor: 2.0

  # Local model example
  local:
    type: ollama
    base_url: http://localhost:11434
    default_model: llama3
    pull_missing_models: true
```

### Privacy and Local Model Support

The framework includes comprehensive privacy protection and local model evaluation capabilities:

#### Privacy Features
- **Data Sanitization**: Automatic detection and redaction of sensitive information (emails, SSNs, API keys, etc.)
- **Privacy-Safe Logging**: Request logging that sanitizes sensitive content while preserving debugging information
- **Privacy Level Assessment**: Automatic evaluation of adapter privacy levels (NONE, BASIC, ENHANCED, MAXIMUM)
- **Compliance Reporting**: GDPR and HIPAA considerations with recommendations

#### Local Model Integration
- **Ollama Support**: Full integration with Ollama for local model serving
- **Model Management**: Automatic model pulling and availability checking
- **Popular Models**: Pre-configured support for Llama, Mistral, Gemma, CodeLlama, and other open-source models
- **Privacy-First**: Local processing ensures maximum data privacy and sovereignty

#### Example Configurations
```yaml
# Privacy-focused local evaluation
adapters:
  local:
    type: ollama
    base_url: http://localhost:11434
    default_model: llama3
    pull_missing_models: true

oracles:
  judge:
    type: llm_judge
    adapter: local
    model: llama3
    system_prompt: "Rate response quality from 1-5."
```

### Development Workflow
1. **Configuration Management**: All evaluations start with YAML configuration files
2. **Template Generation**: Jinja2 templates with variable substitution generate prompts
3. **Sampling Strategy**: Latin Hypercube Sampling generates parameter space coverage
4. **LLM Interaction**: Adapters handle API calls with rate limiting and retries
5. **Oracle Evaluation**: Multiple oracles provide different quality assessments
6. **Statistical Analysis**: Bayesian models quantify confidence and uncertainty
7. **Results Output**: Rich console output and structured data formats

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

### Git Commit Guidelines
- **Never use "authored by Claude Code" in commit messages**: Commit messages should be written naturally without Claude attribution
- **Commit Format**: Use conventional commits format: `type: description`
- **Types**: feat, fix, docs, style, refactor, test, chore
- **Examples**:
  - `feat: add ollama adapter support`
  - `fix: handle connection errors in openai adapter`
  - `docs: update configuration examples`
  - `test: add privacy adapter test coverage`

### Current Development Status
The project is in active development (v0.1.0) with the following features completed and in progress:

#### ✅ Completed Features
- **YAML-based evaluation specifications**: Complete configuration system with Pydantic validation
- **Latin Hypercube Sampling implementation**: Full LHS with optimization strategies and metrics
- **Configuration validation and loading**: Comprehensive YAML loading with includes, environment variables, and caching
- **CLI interface foundation**: Complete command-line interface with config validation and templating commands
- **Adapter system**: Full adapter architecture with OpenAI, Anthropic, and Ollama implementations
- **Template system**: Jinja2-based prompt generation with custom filters and batch rendering
- **Oracle framework**: LLM-as-Judge oracle with structured evaluation responses
- **Privacy utilities**: Data sanitization and privacy-safe logging for sensitive information
- **Local model support**: Complete Ollama integration for privacy-focused local evaluation
- **Comprehensive testing**: 30+ test files with 80%+ coverage requirement

#### 🚧 In Development
- **Advanced oracle implementations**: Embedding similarity and custom evaluation metrics
- **Bayesian analysis integration**: Framework ready, PyMC integration pending
- **End-to-end evaluation pipeline**: Complete workflow from configuration to results

#### 📁 Project Structure
- **Core Modules**: All foundational modules implemented and tested
- **Configuration Schema**: Complete support for adapters, distributions, sampling, and statistical analysis
- **LLM Integrations**: OpenAI, Anthropic, and Ollama adapters with rate limiting, retry logic, and privacy protection
- **Development Workflow**: Full CI/CD setup with testing, formatting, and security checks
