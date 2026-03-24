# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install (dev mode with all dependencies)
pip install -e ".[dev]"

# Run full test suite (80% coverage enforced)
pytest

# Run a single test file
pytest tests/test_adapters.py -v

# Run a specific test
pytest tests/test_cli.py::TestReportCommand::test_method -v

# Skip slow tests
pytest -m "not slow"

# Format code
black src tests && isort src tests

# Lint
flake8 src tests

# Security scan
bandit -r src

# Validate a spec file
metareason validate examples/quantum_entanglement_eval.yml

# Run an evaluation
metareason run examples/quantum_entanglement_eval.yml --analyze --report
```

## Architecture

MetaReason is a Bayesian evaluation framework for LLM systems. The pipeline:

```
YAML Spec → LHS Sampling → Jinja2 Templating → Async LLM Execution → Oracle Scoring → Bayesian Analysis → HTML Report
```

### Source Layout (`src/metareason/`)

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `config/models.py` | Pydantic models for YAML spec validation | `SpecConfig`, `AxisConfig`, `PipelineConfig`, `OracleConfig`, `BayesianAnalysisConfig`, `CalibrateConfig`, `CalibrateMultiConfig` |
| `adapters/` | Async LLM provider abstraction with retry | `AdapterBase` (ABC), `get_adapter(name)` factory. Providers: ollama, openai, google, anthropic |
| `pipeline/runner.py` | Main evaluation orchestrator | `run(spec_path) → List[SampleResult]` — loads spec, samples params, chains LLM stages, runs oracles |
| `pipeline/loader.py` | YAML loading with recursive `file:` prefix resolution | `load_spec()`, `load_calibrate_spec()`, `load_calibrate_multi_spec()` |
| `pipeline/renderer.py` | Jinja2 prompt templating | `TemplateRenderer.render_request(template, variables)` |
| `sampling/lhs_sampler.py` | Latin Hypercube Sampling with scipy QMC | `LhsSampler.generate_samples()` — supports uniform, normal, truncnorm, beta distributions + categorical axes |
| `oracles/` | Output evaluation | `OracleBase` (ABC). Implementations: `LLMJudge` (LLM-as-judge, returns JSON score+explanation), `RegexOracle` (pattern matching, linear score interpolation 1.0-5.0) |
| `analysis/analyzer.py` | Bayesian inference with PyMC/ArviZ | `BayesianAnalyzer` — `estimate_population_quality()`, `estimate_judge_calibration()`, `estimate_parameter_effects()` |
| `analysis/agreement.py` | Inter-rater reliability | `compute_krippendorff_alpha()`, `compute_pairwise_correlations()` |
| `calibration/` | Iterative judge rubric optimization | `AutoCalibrationLoop` orchestrates evaluate→check convergence→optimize rubric→iterate. Uses `ConvergenceChecker` and `RubricOptimizer` (LLM-driven rubric revision) |
| `reporting/` | Self-contained HTML reports with Chart.js + matplotlib | `ReportGenerator`, `CalibrationReportGenerator`, `MultiJudgeReportGenerator` |
| `storage/store.py` | SQLite persistence (WAL mode) | `RunStore` context manager — tables: runs, pipeline_stages, samples, evaluations, analysis_results |
| `cli/main.py` | Click CLI | Commands: `run`, `validate`, `analyze`, `report`, `calibrate`, `calibrate-multi` |

### Key Design Patterns

- **Async throughout**: Pipeline runner uses `asyncio.gather` for concurrent sample processing. Adapters are async.
- **File reference resolution**: Any string value prefixed with `file:` in YAML specs is recursively resolved to file contents (relative to spec directory). Works across all spec types.
- **Multi-stage pipelines**: Each pipeline stage chains to the next — previous stage's response becomes the next stage's input via Jinja2 `{{ response }}` variable.
- **Oracle scoring**: All scores are on a 1.0-5.0 scale. LLMJudge expects JSON `{"score": X, "explanation": "..."}` from the judge LLM.
- **Bayesian models**: PyMC hierarchical models with configurable priors via `BayesianAnalysisConfig`. HDI (High-Density Interval) is the primary uncertainty measure, default 94%.

## Code Style

- **Formatter**: black (line-length 88) + isort (black profile)
- **Linter**: flake8 (max-line-length 120, google docstring convention, many D-codes ignored — see `.flake8`)
- **Python**: 3.13+ required
- **Commits**: Conventional format via commitizen hook — `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`
- **Branches**: `<type>/<short-description>` (e.g., `feat/report-cli`)
- **Pre-commit hooks**: black, isort, flake8, bandit, commitizen, plus standard checks (trailing whitespace, YAML, large files, debug statements, private keys)
- **Test markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`
- **Async tests**: `asyncio_mode = "auto"` — async test functions work without explicit decorator
