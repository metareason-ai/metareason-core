# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

MetaReason Core is in active development with core features implemented and working. The project provides **statistically rigorous quantitative measurement** of LLM and agentic AI systems using:
- Latin Hypercube Sampling for parameter space exploration
- LLM-based evaluation judges (oracles)
- **Bayesian analysis with PyMC** for uncertainty quantification and High-Density Credible Intervals (HDI)

The key differentiator is the ability to make statements like: **"We are 94% confident the true quality is between 4.65 and 5.10"** rather than just providing point estimates.

## Project Vision

MetaReason Core provides statistically rigorous quantitative measurement of LLM and agentic AI systems. The core architecture is now implemented:

**Implemented:**
- ✅ YAML-based specifications for evaluation configurations
- ✅ Jinja2 templating for prompt generation
- ✅ Latin Hypercube Sampling for parameter space exploration
- ✅ LLM Judges for evaluation
- ✅ Bayesian analysis for statistical rigor (PyMC + ArviZ)
- ✅ Pipeline-based execution model (async, multi-stage)
- ✅ CLI interface with progress indicators
- ✅ LLM adapters (Ollama, OpenAI, Google, Anthropic)

**Planned:**
- 🚧 Rich HTML/PDF report generation with visualizations
- 🚧 Parameter effects analysis (Bayesian regression)

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
  - `adapters/` - LLM adapter implementations (Ollama, OpenAI, Google, Anthropic)
  - `oracles/` - Evaluation oracles (currently: LLM Judge)
  - `sampling/` - Sampling strategies (currently: Latin Hypercube)
  - `pipeline/` - Template rendering and execution pipeline
  - `analysis/` - Bayesian statistical analysis with PyMC
  - `cli/` - Click-based command-line interface with rich output
- `tests/` - Test directory (currently: LHS sampler tests with 11 test cases)
- `examples/` - Example YAML specifications demonstrating features
- `test_suite/` - Additional test specifications
- `reports/` - Output directory for evaluation results (JSON)
- `scripts/` - Development scripts for formatting and testing
- `pyproject.toml` - Project configuration with dependencies

## Implementation Status

### ✅ Completed Features
- **Bayesian analysis with PyMC and ArviZ** - Fully implemented
  - Population-level quality estimation with High-Density Credible Intervals (HDI)
  - MCMC sampling using NUTS sampler
  - Configurable HDI probability (default 94%)
  - Oracle variability quantification
  - Convergence diagnostics (R-hat, ESS)
  - Results saved as JSON with full posterior statistics
- **YAML-based specification system** with Pydantic validation
- **Jinja2 template rendering** for prompts with parameter interpolation
- **Latin Hypercube Sampling** with multiple distributions (uniform, normal, truncnorm, beta)
  - Maximin optimization for space-filling designs
  - Categorical and continuous axes
- **LLM Judge oracle** for evaluations
  - JSON-based scoring with explanations
  - Multiple oracle support
- **Async pipeline execution** with multi-stage support
- **CLI interface** with three commands:
  - `metareason run` - Execute evaluations with optional `--analyze` flag
  - `metareason validate` - Validate YAML specification files
  - `metareason analyze` - Perform Bayesian analysis on saved results
- **Progress indicators** using rich library
  - Spinner and progress bars for evaluation phase
  - MCMC sampling progress tracking per oracle
  - Real-time status updates
- **LLM adapters** for multiple providers:
  - Ollama (local models)
  - OpenAI (GPT models via modern Responses API)
  - Google (Gemini models)
  - Anthropic (Claude models via Messages API)
- **Comprehensive examples** in `examples/` directory

### 🚧 In Progress
- Report generation (currently JSON, HTML/PDF with visualizations planned)
- Comprehensive test coverage (currently focused on LHS sampler, target 80%)

### 📋 Planned
- Parameter effects analysis (Bayesian regression to identify which parameters matter)
- Additional oracle types (regex, keyword, statistical, custom)
- Additional sampling methods beyond LHS
- Rich HTML/PDF visualization reports with posterior plots

## Claude's Role

**Advisory Role Only**: Claude should act as an advisor, reviewer, and helper - NOT as the primary coding agent. The user writes all implementation code themselves.

- **Review and suggest**: Provide code reviews, answer technical questions, and suggest best practices when asked
- **No proactive coding**: Do not write implementation code unless explicitly requested
- **Explain and guide**: Focus on explaining concepts, pointing out issues, and providing guidance
- **Keyboard-first workflow**: Support the user's goal of staying on the keyboard and minimizing mouse usage

## Important Notes

- **Python 3.13+ required**: The project targets Python 3.13 or higher
- **Test coverage requirement**: 80% coverage target (currently focused on core sampling functionality)
- **Always use venv**: All commands should be run within the virtual environment
- **Statistical rigor**: The core differentiator is Bayesian analysis providing credible intervals, not just point estimates

## Dependencies

Key dependencies specified in pyproject.toml:
- **pymc** - Bayesian analysis
- **pydantic** - Configuration validation
- **click** - CLI framework
- **jinja2** - Template rendering
- **httpx** - Async HTTP client
- **openai/anthropic/google-genai** - LLM API clients

## Development Philosophy

Focus on:
1. Simple, working implementations over complex abstractions
2. Incremental development with working code at each step
3. Real integration tests over mocked unit tests
4. Clear separation of concerns without over-engineering
5. Statistical rigor and proper uncertainty quantification
