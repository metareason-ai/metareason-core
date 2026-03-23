# Auto-Calibration: Iterative Judge Prompt Optimization

**Issue:** #126
**Branch:** `feature/auto-calibration`

## Context

Judge calibration (`metareason calibrate`) measures how well a judge scores against an expected score, but tuning the rubric is manual. Auto-calibration closes the loop: when the judge is off, an optimizer LLM rewrites the rubric automatically and re-runs until convergence.

The existing `calibrate` command already does the inner loop (repeated evals + Bayesian analysis). This feature wraps that in an outer optimization loop.

## Architecture

```
┌─────────────────────────────────────────────┐
│           AutoCalibrationLoop               │
│                                             │
│  for iteration in 1..max_iterations:        │
│    ┌──────────────────────────────────┐     │
│    │  _run_single_calibration()       │     │
│    │  (existing: judge evals + MCMC)  │     │
│    └──────────┬───────────────────────┘     │
│               │ cal_result                  │
│    ┌──────────▼───────────────────────┐     │
│    │  ConvergenceChecker.check()      │     │
│    └──────────┬───────────────────────┘     │
│               │ converged?                  │
│           yes ╱╲ no                         │
│          done   │                           │
│    ┌────────────▼─────────────────────┐     │
│    │  RubricOptimizer.optimize()      │     │
│    │  (LLM call → revised rubric)     │     │
│    └──────────┬───────────────────────┘     │
│               │ new rubric                  │
│               └───── loop ──────────────→   │
└─────────────────────────────────────────────┘
```

## Progress

- [x] Step 1: `AutoCalibrationConfig` model + `CalibrateConfig` wiring
- [x] Behavioral tests written for all components (`tests/test_auto_calibration.py`) — 17 tests, all Green
- [x] Step 2: `ConvergenceChecker` — `src/metareason/calibration/convergence.py`
- [x] Step 3: `RubricOptimizer` — `src/metareason/calibration/optimizer.py`
- [x] Step 4: `AutoCalibrationLoop` — `src/metareason/calibration/loop.py`
- [x] Step 5: CLI integration — `--auto-calibrate` flag on `metareason calibrate`
- [x] Step 6: Package init — `src/metareason/calibration/__init__.py`

## Implementation Steps (TDD)

### Step 1: `AutoCalibrationConfig` model — COMPLETE

**Files modified:**

- `src/metareason/config/models.py` — added `AutoCalibrationConfig`, `auto_calibration` field on `CalibrateConfig`, `model_validator` requiring `expected_score`
- `src/metareason/config/__init__.py` — exported `AutoCalibrationConfig`

---

### Step 2: `ConvergenceChecker` — NOVEL (pair program)

**Tests written** in `tests/test_auto_calibration.py`:

- Converged: HDI contains expected_score AND mean within tolerance
- Not converged: HDI excludes expected_score
- Not converged: mean outside tolerance
- Max iterations returns converged=True with reason="max_iterations"
- Edge: expected_score at HDI boundary

**Key design decision:** Convergence requires BOTH conditions. The HDI condition (`expected_score` falls within the score HDI derived from `bias_hdi + expected_score`) is the statistical guarantee. The tolerance on the mean catches wide-but-off-center distributions.

**New file:** `src/metareason/calibration/convergence.py`

```python
@dataclass
class ConvergenceResult:
    converged: bool
    reason: str  # "converged", "max_iterations", "not_converged"
    hdi_contains_target: bool
    mean_within_tolerance: bool
    current_mean: float
    current_hdi: tuple[float, float]
    iteration: int

class ConvergenceChecker:
    def __init__(self, expected_score: float, tolerance: float, max_iterations: int): ...
    def check(self, cal_result: dict, iteration: int) -> ConvergenceResult: ...
```

Consumes `estimate_judge_calibration()` return dict. Score HDI = `(expected_score + bias_hdi[0], expected_score + bias_hdi[1])`.

---

### Step 3: `RubricOptimizer` — NOVEL (pair program)

**Tests written** in `tests/test_auto_calibration.py`:

- `_build_prompt` includes rubric, scores, target, gap direction/magnitude
- `optimize()` returns new rubric string (mock adapter)
- Strips markdown fences from response
- Raises on adapter failure
- Prompt includes iteration history when provided

**New file:** `src/metareason/calibration/optimizer.py`

```python
class RubricOptimizer:
    def __init__(self, model: str, adapter: AdapterBase): ...
    async def optimize(
        self,
        current_rubric: str,
        expected_score: float,
        cal_result: dict,
        iteration_history: list[dict],
    ) -> str: ...
```

Uses `AdapterRequest` + `get_adapter()` from existing adapter system. The optimizer prompt is the critical novel piece — it communicates current rubric, actual scores, expected score, bias direction/magnitude, and iteration history to prevent oscillation.

---

### Step 4: `AutoCalibrationLoop` — NOVEL (pair program)

**Tests written** in `tests/test_auto_calibration.py`:

- Converges on first iteration (no optimization needed)
- Runs multiple iterations until convergence (mock optimizer + judge)
- Stops at max iterations
- Rubric updated between iterations
- History accumulates
- Returns best rubric (lowest abs bias) when max iterations hit
- Original rubric preserved if no improvement

**New file:** `src/metareason/calibration/loop.py`

```python
@dataclass
class AutoCalibrationResult:
    converged: bool
    iterations: int
    final_rubric: str
    original_rubric: str
    convergence_result: ConvergenceResult
    history: list[dict]
    best_rubric: str
    best_cal_result: dict

class AutoCalibrationLoop:
    def __init__(self, calibrate_config, convergence_checker, rubric_optimizer, console): ...
    async def run(self) -> AutoCalibrationResult: ...
```

Contains `_run_single_calibration(config)` — extracted from the existing CLI `calibrate` command logic. Reuses `LLMJudge`, `EvaluationContext`, `SampleResult`, `BayesianAnalyzer.estimate_judge_calibration()`.

Updates rubric between iterations via `CalibrateConfig.model_copy(update=...)` with modified `OracleConfig`.

---

### Step 5: CLI integration — BOILERPLATE (pair program anyway)

**Tests first** in `tests/test_cli.py` (extend `TestCalibrateCommand`):

- `--auto` flag triggers auto loop (mock loop)
- Auto-detected from spec when `auto_calibration` section present
- Requires expected_score
- Displays iteration progress
- Saves history to output JSON
- Shows converged/max-iterations outcome

**File to modify:** `src/metareason/cli/main.py`

- Add `--auto` flag to `calibrate` command
- Branch: if auto enabled, build components and run `AutoCalibrationLoop`, else existing flow unchanged

---

### Step 6: Package init — BOILERPLATE

**New file:** `src/metareason/calibration/__init__.py`

- Export `ConvergenceChecker`, `ConvergenceResult`, `AutoCalibrationLoop`, `AutoCalibrationResult`, `RubricOptimizer`

---

## Dependency Order

```
Step 1 (config) ──┐
                  ├──→ Step 4 (loop) ──→ Step 5 (CLI)
Step 2 (convergence)                         │
Step 3 (optimizer) ──┘                       │
Step 6 (package init) ◄─────────────────────┘
```

Steps 1, 2, 3 have no interdependencies — can be built in parallel.
Step 4 depends on 2 + 3.
Step 5 depends on 4.

## Verification

1. **Unit tests:** `pytest tests/test_auto_calibration.py tests/test_cli.py -v`
2. **Linting:** `flake8 src/metareason/calibration/ && black --check src/metareason/calibration/ && isort --check src/metareason/calibration/`
3. **Coverage:** `pytest --cov=src/metareason/calibration --cov-report=term-missing` — target 80%+
4. **Integration test with example spec:**
   ```yaml
   # examples/auto_calibration.yml
   spec_id: auto_cal_test
   type: calibrate
   prompt: "Explain quantum entanglement to a college student."
   response: "Quantum entanglement is when two particles..."
   expected_score: 4
   repeats: 10
   oracle:
     type: llm_judge
     model: gemma3:27b
     adapter:
       name: ollama
     rubric: 'Evaluate coherence on 1-5 scale. Return: {"score": X, "explanation": "..."}'
   auto_calibration:
     enabled: true
     max_iterations: 5
     tolerance: 0.3
     optimizer_model: "claude-sonnet-4-6"
     optimizer_adapter:
       name: anthropic
   analysis:
     mcmc_draws: 1000
     mcmc_chains: 2
   ```
   Run: `metareason calibrate examples/auto_calibration.yml --auto --report`
5. **Full test suite:** `./scripts/test.sh` — no regressions
