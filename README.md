# MetaReason Core

MetaReason Core is an open source tool for quantitative measurement of LLM and agentic AI systems.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jeffgbradley2/metareason-ai/metareason-core.git
cd metareason-core

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

1. **Validate a specification file**:
```bash
metareason validate examples/quantum_entanglement_eval.yml
```

2. **Run an evaluation**:
```bash
metareason run examples/quantum_entanglement_eval.yml
```

This will:
- Generate parameter samples using Latin Hypercube Sampling
- Execute the LLM pipeline with each sample
- Evaluate outputs using configured oracles
- Save results to `reports/` directory

3. **View results**:
Results are saved as JSON in the `reports/` directory with timestamps.

### Example Specification

See [examples/quantum_entanglement_eval.yml](examples/quantum_entanglement_eval.yml) for a complete example demonstrating:
- Multiple sampling distributions (categorical, uniform, normal, truncnorm, beta)
- Jinja2 template rendering with parameter interpolation
- Dual oracle evaluation (coherence + accuracy judges)
- Latin Hypercube Sampling with 10 variants

## The Vision

### ✅ Implemented
- ✅ YAML-based specifications with Pydantic validation
- ✅ Jinja2 templating for prompt generation
- ✅ Latin Hypercube Sampling for parameter space exploration
- ✅ LLM Judge oracles for evaluation
- ✅ Pipeline-based execution model (multi-stage, async)
- ✅ CLI interface (`run`, `validate` commands)

### 🚧 Coming Soon
- 🚧 Bayesian analysis for statistical rigor (PyMC integration)
- 🚧 Rich HTML/PDF report generation with visualizations
- 🚧 Additional LLM adapters (OpenAI, Anthropic, Google)
- 🚧 Additional oracle types (regex, statistical, custom)
- 🚧 Additional sampling methods

## 🛠️ Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **bandit**: Security checks
- **pytest**: Testing framework

Run all checks manually:
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linting
flake8 src tests

# Security checks
bandit -r src

# Run tests
pytest
```

## 🤝 Contributing

We welcome contributions! MetaReason Core is built by the community, for the community.

### Quick Contribution Steps

- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request


## 📜 License

Copyright (c) 2025 MetaReason LLC

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
