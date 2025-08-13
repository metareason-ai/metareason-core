# MetaReason Core

> **Quantify AI confidence through statistically rigorous evaluation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/metareason-ai/metareason-core.svg)](https://github.com/metareason-ai/metareason-core/issues)
[![GitHub Stars](https://img.shields.io/github/stars/metareason-ai/metareason-core.svg)](https://github.com/metareason-ai/metareason-core/stargazers)

MetaReason Core is the open-source evaluation engine that transforms LLM outputs from "it seems to work" to "it works with 94% confidence ±2%." Using Latin Hypercube Sampling and dual-oracle evaluation, it provides statistically rigorous confidence scores for AI systems in production.

## 🎯 What Problem Does This Solve?

- **Quantify AI Uncertainty**: Move beyond binary pass/fail to true confidence intervals
- **Statistical Rigor**: Bayesian analysis with proper uncertainty quantification
- **Real-World Testing**: Hundreds (or thousands!) of prompt variants reflect actual usage patterns
- **Regulatory Compliance**: ISO 42001 and EU AI Act ready
- **Open Methodology**: Transparent, auditable evaluation process

## 🚀 Quick Start

```bash
# Install MetaReason Core
pip install metareason-core

# Run your first evaluation
metareason run --spec-file examples/pipeline_demo.yaml

# Generate execution plan (dry run)
metareason run --spec-file examples/pipeline_demo.yaml --dry-run

# Export results with visualization dashboard
metareason run --spec-file examples/pipeline_demo.yaml --output-dir results --format dashboard
```

## 📋 Features

### ✅ Available Now (v0.1.0)

- **Complete evaluation pipeline** - End-to-end execution with multi-step workflows and dependency management
- **Pipeline-based configuration** - Multi-step YAML specifications with stage-to-stage variable passing
- **Bayesian statistical analysis** - Full PyMC integration with credible intervals and uncertainty quantification
- **Latin Hypercube Sampling** - Optimized parameter space exploration with statistical guarantees
- **Results export & visualization** - Dashboard generation, JSON/CSV/Parquet export, HTML reports
- **Advanced oracle suite** - Embedding similarity, LLM-as-Judge, quality assurance with batch processing
- **Multi-provider LLM adapters** - OpenAI, Anthropic, Google/Gemini, Ollama with rate limiting and privacy protection
- **Jinja2 template system** - Flexible prompt generation with custom filters and security validation
- **Rich CLI interface** - Configuration validation, template rendering, execution planning, and interactive output

### 🔄 In Development (v1.1)

- Custom oracle development framework
- Multi-model comparison workflows
- Performance benchmarking suite
- Real-time monitoring integration

### 🎯 Planned Features (v2.0)

- Docker containerization
- Drift detection algorithms
- Automated remediation workflows
- Enterprise dashboard API
- Advanced model comparison analytics
- A/B testing frameworks

## 📝 Configuration Examples

### Pipeline-Based Configuration

```yaml
# Multi-step evaluation pipeline
spec_id: advanced_analysis
pipeline:
  # Stage 1: Initial response generation
  - template: "Analyze {{topic}} from a {{perspective}} perspective"
    adapter: ollama
    model: llama3
    temperature: 0.7
    axes:
      topic:
        type: categorical
        values: ["AI ethics", "quantum computing", "climate tech"]
      perspective:
        type: categorical
        values: ["technical", "business", "social"]

  # Stage 2: Follow-up analysis using previous output
  - template: "Based on: {{stage_1_output}}\n\nEvaluate {{criteria}} implications"
    adapter: openai
    model: gpt-4
    axes:
      criteria:
        type: categorical
        values: ["economic impact", "ethical considerations", "future risks"]

# Statistical analysis with Bayesian inference
statistical_config:
  model: beta_binomial
  inference:
    method: mcmc
    samples: 2000
    chains: 4

# Multiple oracle evaluation
oracles:
  similarity:
    type: embedding_similarity
    canonical_answer: "Comprehensive analysis covering technical foundations and implications"
    threshold: 0.8

  quality:
    type: llm_judge
    rubric: "Rate clarity and depth from 1-5"
    judge_model: gpt-4
```

### 📖 Documentation

- Getting Started Guide - Your first evaluation in 5 minutes
- [Templating Guide](docs/templating-guide.md) - Jinja2 templates and custom filters
- YAML Schema Reference - Complete specification format
- Statistical Methodology - Mathematical foundations
- Oracle Development - Creating custom evaluation criteria
- API Reference - Python API documentation

## 🛠️ Development Setup

### Prerequisites

- Python 3.9 or higher
- Poetry (recommended) or pip

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/metareason-ai/metareason-core.git
   cd metareason-core
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Poetry (if not already installed):**
   ```bash
   pip install poetry
   ```

4. **Install dependencies:**
   ```bash
   poetry install
   ```

5. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
metareason-core/
├── src/metareason/
│   ├── analysis/      # Bayesian statistical analysis (PyMC)
│   ├── adapters/      # LLM provider adapters (OpenAI, Anthropic, Google, Ollama)
│   ├── cli/           # Command-line interface with rich output
│   ├── config/        # Configuration management and validation
│   ├── oracles/       # Oracle implementations (embedding, LLM-judge, QA)
│   ├── pipeline/      # Evaluation pipeline execution and management
│   ├── results/       # Results export and formatting
│   ├── sampling/      # Latin Hypercube Sampling strategies
│   ├── templates/     # Jinja2 template system with security
│   ├── utils/         # Utility functions
│   └── visualization/ # Dashboard and plot generation
├── tests/             # Comprehensive test suite
├── examples/          # Example configurations and demos
├── docs/              # Documentation
└── pyproject.toml     # Project configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=metareason

# Run specific test file
pytest tests/test_version.py

# Run tests in parallel
pytest -n auto
```

### CLI Examples

```bash
# System information
metareason info

# Configuration management
metareason config validate examples/pipeline_demo.yaml
metareason config show examples/pipeline_demo.yaml
metareason config diff examples/simple_evaluation.yaml examples/advanced_evaluation.yaml

# Template operations
metareason template render examples/pipeline_demo.yaml
metareason template validate examples/pipeline_demo.yaml
metareason template filters

# Run evaluations with different outputs
metareason run --spec-file examples/pipeline_demo.yaml --dry-run
metareason run --spec-file examples/pipeline_demo.yaml --output results.json
metareason run --spec-file examples/bayesian_config_example.yaml --output-dir analysis --format dashboard

# Local model evaluation (privacy-focused)
metareason run --spec-file examples/local_ollama_evaluation.yaml --verbose

# Google/Gemini evaluation
metareason run --spec-file examples/google_quickstart.yaml --max-concurrent 5
```

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security checks
- **pre-commit**: Automated checks before commits

Run all checks manually:
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linting
flake8 src tests

# Type checking
mypy src

# Security checks
bandit -r src
```

## 🤝 Contributing

We welcome contributions! MetaReason Core is built by the community, for the community.

### Quick Contribution Steps

- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

## 🏢 Enterprise Version
Looking for enterprise features like:

- Dashboards
- Automated jobs
- VPC-native deployment
- Custom oracle development
- Compliance reporting

Check out MetaReason Enterprise or contact us at enterprise@metareason.ai

## 📜 License

Copyright 2025 MetaReason LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## 🌟 Community

GitHub Discussions: github.com/metareason/metareason-core/discussions

Discord Server: discord.gg/metareason

Twitter: @MetaReasonAI

Blog: blog.metareason.ai

Made with ❤️ by the MetaReason community
