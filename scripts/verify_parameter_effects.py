"""Verify parameter effects analysis recovers known effects from synthetic data.

Generates fake evaluation results where:
- "temperature" (continuous) has a strong positive effect (+1.0)
- "style" (categorical: formal/casual/technical) has a real effect
  (casual scores ~1.5 lower than formal, technical ~0.5 higher)
- "verbosity" (continuous) has NO effect (null signal)

Then runs estimate_parameter_effects and checks recovery.
"""

import numpy as np

from metareason.analysis.analyzer import BayesianAnalyzer
from metareason.cli.main import display_parameter_effects
from metareason.config.models import (
    AdapterConfig,
    AxisConfig,
    BayesianAnalysisConfig,
    OracleConfig,
    PipelineConfig,
    SamplingConfig,
    SpecConfig,
)
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.runner import SampleResult

rng = np.random.default_rng(42)
N = 40

# --- Define axes ---
axes = [
    AxisConfig(
        name="temperature",
        type="continuous",
        distribution="uniform",
        params={"low": 0.0, "high": 1.0},
    ),
    AxisConfig(
        name="style",
        type="categorical",
        values=["formal", "casual", "technical"],
    ),
    AxisConfig(
        name="verbosity",
        type="continuous",
        distribution="uniform",
        params={"low": 1.0, "high": 10.0},
    ),
]

# --- Generate synthetic parameter values ---
temperatures = rng.uniform(0.0, 1.0, N)
styles = rng.choice(["formal", "casual", "technical"], N)
verbosities = rng.uniform(1.0, 10.0, N)

# --- Generate scores with KNOWN effects ---
# Base score = 3.0
# temperature effect: +1.0 per std dev (strong positive)
# casual effect: -1.5 vs formal (strong negative)
# technical effect: +0.5 vs formal (moderate positive)
# verbosity effect: 0.0 (null)
# noise: std 0.3

temp_z = (temperatures - temperatures.mean()) / temperatures.std()
verb_z = (verbosities - verbosities.mean()) / verbosities.std()

scores = np.zeros(N)
for i in range(N):
    base = 3.0
    temp_effect = 1.0 * temp_z[i]
    style_effect = {"formal": 0.0, "casual": -1.5, "technical": 0.5}[styles[i]]
    verb_effect = 0.0 * verb_z[i]  # intentionally zero
    noise = rng.normal(0, 0.3)
    scores[i] = np.clip(base + temp_effect + style_effect + verb_effect + noise, 1, 5)

# --- Build SampleResults ---
results = [
    SampleResult(
        sample_params={
            "temperature": float(temperatures[i]),
            "style": str(styles[i]),
            "verbosity": float(verbosities[i]),
        },
        original_prompt="synthetic",
        final_response="synthetic",
        evaluations={
            "quality": EvaluationResult(score=float(scores[i]), explanation="synthetic")
        },
    )
    for i in range(N)
]

# --- Print ground truth ---
print("=" * 60)
print("GROUND TRUTH")
print("=" * 60)
print("  temperature:          +1.0  (strong positive)")
print("  style (casual):       -1.5  (strong negative vs formal)")
print("  style (technical):    +0.5  (moderate positive vs formal)")
print("  verbosity:             0.0  (null)")
print("  noise std:             0.3")
print(f"  n_samples:             {N}")
print()

# --- Print score summary ---
print("Score summary:")
for s in ["formal", "casual", "technical"]:
    mask = styles == s
    print(
        f"  {s:12s}: mean={scores[mask].mean():.2f}  std={scores[mask].std():.2f}  n={mask.sum()}"
    )
print(f"  overall:      mean={scores.mean():.2f}  std={scores.std():.2f}")
print()

# --- Run analysis ---
spec = SpecConfig(
    spec_id="synthetic_verification",
    pipeline=[
        PipelineConfig(
            template="t",
            adapter=AdapterConfig(name="ollama"),
            model="m",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
    ],
    sampling=SamplingConfig(method="latin_hypercube", optimization="maximin"),
    oracles={
        "quality": OracleConfig(
            type="llm_judge",
            model="m",
            adapter=AdapterConfig(name="ollama"),
            rubric="test",
        )
    },
    analysis=BayesianAnalysisConfig(
        mcmc_draws=2000,
        mcmc_tune=1000,
        mcmc_chains=4,
    ),
)

print("Running Bayesian regression (4 chains x 2000 draws)...")
print()

analyzer = BayesianAnalyzer(results, spec)
result = analyzer.estimate_parameter_effects("quality", axes)

# --- Display results ---
display_parameter_effects(result, "quality")

# --- Verification ---
print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

effects_by_name = {}
for e in result["effects"]:
    key = e["parameter"]
    if e["level"]:
        key += f"_{e['level']}"
    effects_by_name[key] = e

checks = []


def check(name, condition, description):
    status = "PASS" if condition else "FAIL"
    checks.append((name, condition))
    print(f"  [{status}] {description}")


# Temperature should be positive with HDI above zero
e = effects_by_name.get("temperature", {})
check(
    "temp_positive",
    e.get("effect_mean", 0) > 0.5,
    f"temperature effect mean ({e.get('effect_mean', 0):.3f}) > 0.5",
)
check(
    "temp_hdi_above_zero",
    e.get("hdi_lower", 0) > 0,
    f"temperature HDI lower ({e.get('hdi_lower', 0):.3f}) > 0",
)

# Casual should be negative with HDI below zero
e = effects_by_name.get("style_casual", {})
check(
    "casual_negative",
    e.get("effect_mean", 0) < -1.0,
    f"casual effect mean ({e.get('effect_mean', 0):.3f}) < -1.0",
)
check(
    "casual_hdi_below_zero",
    e.get("hdi_upper", 0) < 0,
    f"casual HDI upper ({e.get('hdi_upper', 0):.3f}) < 0",
)

# Technical should be positive
e = effects_by_name.get("style_technical", {})
check(
    "technical_positive",
    e.get("effect_mean", 0) > 0,
    f"technical effect mean ({e.get('effect_mean', 0):.3f}) > 0",
)

# Verbosity should be near zero (inconclusive)
e = effects_by_name.get("verbosity", {})
check(
    "verbosity_near_zero",
    abs(e.get("effect_mean", 99)) < 0.3,
    f"verbosity effect mean ({e.get('effect_mean', 99):.3f}) near zero",
)
check(
    "verbosity_inconclusive",
    e.get("hdi_lower", -99) < 0 < e.get("hdi_upper", 99),
    f"verbosity HDI crosses zero [{e.get('hdi_lower', 0):.3f}, {e.get('hdi_upper', 0):.3f}]",
)

# Sorted by magnitude
check(
    "sorted_by_magnitude",
    abs(result["effects"][0]["effect_mean"])
    >= abs(result["effects"][-1]["effect_mean"]),
    "effects sorted by |magnitude| descending",
)

passed = sum(1 for _, ok in checks if ok)
total = len(checks)
print()
print(f"Result: {passed}/{total} checks passed")
