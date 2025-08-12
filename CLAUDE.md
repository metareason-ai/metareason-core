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
pytest tests/test_config_loader.py::test_load_valid_yaml_config -v
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
metareason run --spec-file examples/simple_evaluation.yaml --dry-run

# Run evaluation with output
metareason run --spec-file examples/simple_evaluation.yaml --output results.json

# Quick start with local models
metareason run --spec-file examples/google_quickstart.yaml --dry-run
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
- **Pipeline Steps**: Self-contained stages with templates, model configs, and variable axes
- **Template per Step**: Jinja2-compatible with variable substitution for each pipeline step
- **Axes per Step**: Categorical and continuous parameter distributions specific to each step
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
  # Single-stage pipeline with cloud model
spec_id: cloud_evaluation
pipeline:
  - template: "Analyze {{topic}} with {{approach}}"
    adapter: openai
    model: gpt-4
    temperature: 0.7
    axes:
      topic:
        type: categorical
        values: ["AI", "ML"]
      approach:
        type: categorical
        values: ["technical", "business"]

# Multi-stage pipeline with local models
spec_id: local_multi_stage
pipeline:
  - template: "Summarize {{text}}"
    adapter: ollama
    model: llama3
    axes:
      text:
        type: categorical
        values: ["doc1.txt", "doc2.txt"]
  - template: "Analyze: {{stage_1_output}} using {{method}}"
    adapter: ollama
    model: mistral
    axes:
      method:
        type: categorical
        values: ["critical", "supportive"]
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
spec_id: privacy_evaluation
pipeline:
  - template: "Evaluate {{content}} for {{criteria}}"
    adapter: ollama
    model: llama3
    temperature: 0.5
    axes:
      content:
        type: categorical
        values: ["sample1", "sample2"]
      criteria:
        type: categorical
        values: ["accuracy", "clarity"]

oracles:
  quality:
    type: llm_judge
    rubric: "Rate response quality from 1-5."
    judge_model: llama3
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

### Test Infrastructure and Fixtures

The project includes a comprehensive test infrastructure designed to eliminate test brittleness and make configuration changes painless. This infrastructure replaces hardcoded YAML strings with programmatic builders and factories.

#### Test Builders and Factories

**ConfigBuilder**: Fluent API for creating test configurations
```python
from tests.fixtures.config_builders import ConfigBuilder

config = (ConfigBuilder()
    .test_id("my_test")
    .prompt_template("Analyze {{topic}} with {{method}}")
    .with_axis("topic", lambda a: a.categorical(["AI", "ML", "DL"]))
    .with_axis("method", lambda a: a.categorical(["technical", "business"]))
    .with_oracle("accuracy", lambda o: o.embedding_similarity(
        "Comprehensive analysis covering key concepts", threshold=0.85
    ))
    .with_oracle("explainability", lambda o: o.llm_judge(
        "Rate quality from 1-5 based on clarity and accuracy"
    ))
    .with_variants(1000)
    .build())
```

**EvaluationFactory**: Common configuration patterns
```python
from tests.factories.evaluation_factory import EvaluationFactory

# Create configurations for common scenarios
minimal_config = EvaluationFactory.minimal()
single_axis_config = EvaluationFactory.with_single_axis("topic", ["AI", "ML"])
multi_axis_config = EvaluationFactory.with_multiple_axes({
    "name": ["Alice", "Bob"],
    "style": ["formal", "casual"]
})
```

**YamlFileFactory**: Temporary YAML file creation
```python
from tests.factories.evaluation_factory import YamlFileFactory

# Create temporary YAML files for CLI testing
config = EvaluationFactory.minimal()
temp_path = YamlFileFactory.create_temp_file(config)

# Create invalid configurations for error testing
invalid_path = YamlFileFactory.create_invalid_file("empty_prompt_id")
```

#### Schema Migration Support

**Migration Utilities**: Support for schema evolution
```python
from tests.helpers.schema_migration import migrate_config_v1_to_v2

# Migrate from current format (v1) to pipeline format (v2)
v1_config = {"prompt_id": "test", "prompt_template": "Hello {{name}}"}
v2_config = migrate_config_v1_to_v2(v1_config)
# Result: {"test_id": "test", "pipeline": [{"type": "template", "template": "Hello {{name}}"}]}
```

#### Pytest Fixtures

**Available Fixtures**:
- `config_builder`: ConfigBuilder instance for fluent configuration creation
- `evaluation_factory`: EvaluationFactory class for common patterns
- `minimal_config`: Pre-built minimal valid configuration
- `comprehensive_config`: Pre-built comprehensive configuration
- `temp_config_file`: Function to create temporary config files
- `common_axes`: Dictionary of common axis configurations
- `common_oracles`: Dictionary of common oracle configurations

**Usage Example**:
```python
def test_my_feature(config_builder, temp_config_file):
    # Use builder for custom config
    config = config_builder.minimal().with_variants(500).build()

    # Create temp file for CLI testing
    temp_path = temp_config_file(lambda cb: cb.comprehensive())

    # Run your test logic
    assert config.n_variants == 500
    assert temp_path.exists()
```

#### Benefits

- **Eliminates Brittleness**: Schema changes automatically propagate through all tests
- **Reduces Duplication**: Common configurations defined once, reused everywhere
- **Improves Maintainability**: Tests focus on behavior, not configuration details
- **Supports Evolution**: Ready for pipeline-based configuration format migration
- **Type Safety**: Full Pydantic validation and IDE support

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
- **Test infrastructure**: Robust ConfigBuilder, factories, and fixtures eliminating test brittleness
- **Schema migration utilities**: Support for configuration format evolution (v1 to v2 pipeline format)
- **Comprehensive testing**: 30+ test files with 80%+ coverage requirement and maintainable test patterns

#### 🚧 In Development
- **Advanced oracle implementations**: Embedding similarity and custom evaluation metrics
- **Bayesian analysis integration**: Framework ready, PyMC integration pending
- **End-to-end evaluation pipeline**: Complete workflow from configuration to results

#### 📁 Project Structure
- **Core Modules**: All foundational modules implemented and tested
- **Configuration Schema**: Complete support for adapters, distributions, sampling, and statistical analysis
- **LLM Integrations**: OpenAI, Anthropic, and Ollama adapters with rate limiting, retry logic, and privacy protection
- **Development Workflow**: Full CI/CD setup with testing, formatting, and security checks
