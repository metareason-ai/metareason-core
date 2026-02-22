# MetaReason Launch — Blog Post Source Material

## The Hook

MetaReason Core is live — an open-source tool for statistically rigorous evaluation of LLMs and agentic AI systems. Instead of getting a single number like "Average Quality: 4.2" and hoping for the best, MetaReason uses Bayesian inference to tell you: **"We are 94% confident the true quality is between 4.65 and 5.10."**

The repo is public, MIT-licensed, and ready to use today: [github.com/metareason-ai/metareason-core](https://github.com/metareason-ai/metareason-core)

---

## What's Available Now

### Website & Repo

- **GitHub**: [metareason-ai/metareason-core](https://github.com/metareason-ai/metareason-core) — MIT license, v0.1.0 (Alpha)
- **Python 3.13+**, installable via `pip install -e ".[dev]"`
- 120+ tests, 82% code coverage

### What You Can Do Today

MetaReason is a CLI tool with five core commands:

| Command | What It Does |
|---------|-------------|
| `metareason run spec.yml --analyze --report` | Run a full evaluation: sample parameters, execute LLM pipeline, judge outputs, perform Bayesian analysis, generate HTML report |
| `metareason validate spec.yml` | Validate a YAML spec before running |
| `metareason analyze results.json --spec spec.yml --report` | Run Bayesian analysis on previously saved results |
| `metareason report results.json --spec spec.yml` | Generate a standalone HTML report from saved results |
| `metareason calibrate calibration_spec.yml --report` | Measure judge reliability and consistency |

The workflow is:

1. **Write a YAML spec** defining what you want to evaluate — your prompt template, parameters to explore, which LLM to use, and how to judge the outputs
2. **Run it** — MetaReason samples the parameter space, executes the LLM pipeline, scores outputs with oracle judges, then runs Bayesian analysis
3. **Read the results** — get credible intervals, not just averages. Results come as JSON + self-contained HTML reports with embedded visualizations

### Supported LLM Providers

MetaReason works with any major LLM provider out of the box:

- **Ollama** — local/open-source models (llama, mistral, etc.)
- **OpenAI** — GPT models via the Responses API
- **Google** — Gemini models
- **Anthropic** — Claude models via the Messages API

All adapters include retry logic (via tenacity) and configurable timeouts for production resilience. Configure API keys via environment variables or a `.env` file.

### Example: A Real Evaluation Spec

Here's a simplified version of what a MetaReason YAML spec looks like:

```yaml
spec_id: "my_evaluation"

pipeline:
  - template: |
      You are a helpful assistant.
      Tone: {{ tone }}
      Complexity Level: {{ complexity_level }}
      Query: Explain quantum entanglement.
    adapter:
      name: "ollama"
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000

sampling:
  method: "latin_hypercube"
  optimization: "maximin"
  random_seed: 42

n_variants: 10

oracles:
  coherence_judge:
    type: "llm_judge"
    model: "llama3"
    adapter:
      name: "ollama"
    rubric: |
      Rate coherence 1-5:
      - 1: Incoherent
      - 5: Highly coherent and logically sound
      Respond as JSON: {"score": X, "explanation": "..."}

axes:
  - name: "tone"
    type: "categorical"
    values: ["formal", "casual", "technical", "friendly"]
  - name: "complexity_level"
    type: "continuous"
    distribution: "uniform"
    params:
      low: 1.0
      high: 10.0

analysis:
  mcmc_draws: 2000
  mcmc_tune: 1000
  mcmc_chains: 4
  hdi_probability: 0.94
```

Then run:

```bash
metareason run my_spec.yml --analyze --report
```

### Output

The console output looks like:

```
Population Quality: coherence_judge

We are 94% confident the true coherence_judge quality is between 4.65 and 5.10

Population Statistics:
  Mean: 4.88
  94% HDI: [4.65, 5.10]

Oracle Variability: 0.36 (94% HDI: [0.21, 0.54])
Based on 10 evaluations
```

HTML reports include:
- Posterior distribution plots with HDI regions highlighted
- Score distribution histograms
- Oracle variability plots
- Parameter space coverage scatter plots

Reports are self-contained (base64-embedded images) — share them with anyone, no server needed.

---

## What Makes It Different

### The Problem

Most LLM eval frameworks give you a single number: "Average Quality: 4.2." That leaves you with unanswerable questions:

- Is 4.2 statistically different from 4.1? Can I trust this difference?
- The true quality could be anywhere from 2.0 to 6.0 — or it could be tightly between 4.0 and 4.4. You don't know.
- How much of the result is noise from the AI judge itself?
- Can you ship a new model or make a risk decision based on a single noisy number?

### The Approach: Latin Hypercube Sampling + Bayesian Analysis

MetaReason combines two techniques that together provide a fundamentally better answer:

**Latin Hypercube Sampling (LHS)** generates parameter combinations that efficiently cover the space you want to explore. Unlike random sampling that can leave gaps or cluster in certain regions, LHS with maximin optimization ensures every region of your parameter space is represented. This means you get better coverage with fewer samples — faster and cheaper than brute-force grid search or random sampling.

Supported distributions for parameter axes:
- Categorical (equal or weighted)
- Continuous: uniform, normal, truncated normal, beta

**Bayesian Analysis with PyMC** takes the raw scores from your oracle judges and fits a proper statistical model:

```
true_quality ~ Normal(mu, sigma)           # What's the actual quality?
oracle_noise ~ HalfNormal(sigma_noise)     # How noisy is the judge?
observed ~ Normal(true_quality, oracle_noise)  # What we measured
```

This gives you:
- **High-Density Credible Intervals (HDI)**: "We are 94% confident the true quality is between X and Y" — a real statistical statement, not a guess
- **Oracle variability quantification**: separates signal from noise — how much of the score variation is real vs. measurement error from the judge
- **Population-level estimates**: overall quality across all parameter variants, not just per-sample
- **Convergence diagnostics**: R-hat and Effective Sample Size (ESS) confirm the analysis is reliable

### The User Benefit

- **Faster**: LHS covers more ground with fewer samples than random or grid approaches
- **Cheaper**: Fewer LLM calls needed for the same statistical power
- **More rigorous**: Credible intervals instead of point estimates — decisions backed by real statistics
- **Transparent**: Oracle noise is explicitly modeled, so you know when your judge is the problem
- **Configurable**: Inject domain knowledge via Bayesian priors, adjust HDI probability, tune MCMC parameters

---

## Deep Dive: Judge Calibration — Can You Trust Your LLM Judge?

### The Meta-Problem of LLM Evaluation

There's a fundamental problem lurking behind every LLM evaluation that uses an AI judge: **how do you know the judge itself is reliable?**

If you're using GPT-4 to score outputs from Claude, or Gemma to rate responses from Llama, you're making an implicit assumption — that the judge is consistent, unbiased, and produces meaningful scores. But LLM judges are themselves stochastic systems. Given the exact same prompt and response, a judge might score it 4/5 one time and 3/5 the next. Some judges systematically over-score. Others are unreliable on subjective criteria.

Most eval frameworks ignore this entirely. MetaReason's `calibrate` command tackles it head-on.

### How Calibration Works

The idea is simple: give the judge the **exact same prompt and response** multiple times, and use Bayesian analysis to measure how it behaves.

```bash
metareason calibrate calibration_spec.yml --report
```

The workflow:

1. **Fix the input**: You provide a single prompt and a single response — no parameter variation, no sampling. The input is held constant so that any variation in scores comes purely from the judge.
2. **Repeat evaluations**: The judge scores the same input N times (configurable, typically 10-30 repeats).
3. **Run Bayesian analysis**: The same PyMC model used for evaluation analysis is applied to the repeated scores, fitting the hierarchical model:

```
true_score ~ Normal(μ_prior, σ_prior)
judge_noise ~ HalfNormal(σ_noise_prior)
observed[i] ~ Normal(true_score, judge_noise)
```

4. **Report the results**: You get credible intervals on the true score, a direct measurement of judge noise, and (optionally) bias analysis against an expected score.

### What You Learn

Calibration answers three specific questions about your judge:

**1. Consistency — How much does the judge vary on identical input?**

The `judge_noise` parameter directly quantifies this. A noise of 0.1 means the judge is very consistent — scores barely move between runs. A noise of 1.5 means the judge is unreliable — the same response could get a 3 or a 5 depending on the run.

**2. Accuracy — Does the judge score correctly?**

If you provide an `expected_score` (your ground-truth assessment), the calibration computes **bias**: the difference between the judge's average score and the expected score. Positive bias means the judge over-scores; negative means it under-scores. The report also checks whether the expected score falls within the judge's credible interval.

**3. Confidence — How certain can you be about the judge's assessment?**

The HDI (High-Density Interval) tells you: "We are 94% confident the judge's true score for this input is between X and Y." A narrow interval means the judge is precise. A wide interval means you need more repeats or a better judge.

### Example: Calibrating a Judge on Subjective Content

Here's a calibration spec that tests a judge on deliberately ambiguous content — a haiku, where scoring is inherently subjective:

```yaml
spec_id: "ambiguous_calibration"
type: calibrate

prompt: "Write a haiku about autumn."
response: |
  Crimson leaves descend
  Whispering tales of summer
  Earth prepares to sleep

expected_score: 3.5
repeats: 15

oracle:
  type: llm_judge
  model: "gemma3:27b"
  adapter:
    name: "ollama"
  temperature: 1
  rubric: |
    Rate this haiku 1-5:
    - 1: Poor structure, no imagery, fails as haiku
    - 3: Competent but unremarkable
    - 5: Exceptional imagery, perfect form, emotionally resonant
    Respond as JSON: {"score": X, "explanation": "..."}
```

With higher temperature and subjective criteria, you'd expect to see meaningful judge noise — and the calibration quantifies exactly how much.

### Calibration Spec Structure

A calibration spec is a focused YAML file with these key fields:

| Field | Required | Description |
|-------|----------|-------------|
| `spec_id` | Yes | Unique identifier for the calibration run |
| `type` | Yes | Must be `"calibrate"` |
| `prompt` | Yes | Fixed prompt text (or `file:path/to/file.txt`) |
| `response` | Yes | Fixed response text (or `file:path/to/file.txt`) |
| `repeats` | Yes | Number of repeated evaluations (10-30 typical) |
| `oracle` | Yes | Judge configuration (same format as evaluation oracles) |
| `expected_score` | No | Ground-truth score for bias analysis (1.0-5.0) |
| `analysis` | No | MCMC parameters: draws, chains, tune, HDI probability, priors |

The `file:` reference syntax lets you load long prompts and responses from external files rather than inlining them in the YAML.

### Calibration HTML Reports

Running with `--report` generates a self-contained HTML report with:

- **Confidence assessment card**: The headline finding — "We are 94% confident the true score is between X and Y"
- **Metrics dashboard**: Mean score, median score, HDI range, judge noise at a glance
- **Bias analysis** (when expected_score provided): Measured bias value and whether the expected score falls within the credible interval, color-coded green (within) or amber (outside)
- **Posterior distribution chart**: KDE plot of the Bayesian estimate with the HDI region shaded
- **Score histogram**: Bar chart of all observed scores across repeats
- **Judge noise distribution**: KDE plot of measurement error
- **Raw scores table**: Every individual score from the repeated evaluations

### Why This Matters

Judge calibration isn't just an academic exercise — it has practical implications:

- **Before trusting evaluation results**, calibrate your judge. If the judge has high noise, your evaluation credible intervals will be wide regardless of the LLM being tested.
- **Compare judges**: Run calibration on the same prompt/response pair with different judge models. Pick the one with the lowest noise and least bias.
- **Tune rubrics**: If a judge is inconsistent, the rubric might be ambiguous. Calibrate, refine the rubric, calibrate again — measure the improvement quantitatively.
- **Set expectations**: If your judge has a measured noise of 0.5, you know that score differences smaller than ~1.0 between two LLMs might not be meaningful.

This is the kind of meta-evaluation that separates rigorous benchmarking from guesswork.

---

## Who It's For

### Primary Audiences

- **Teams evaluating LLMs for production use** — choosing between models, comparing prompt strategies, validating quality thresholds. MetaReason gives you the confidence bounds to make defensible decisions.

- **AI/ML researchers** running systematic evaluations — anyone who needs more than a leaderboard score. If you care about statistical rigor in your eval methodology, this is built for you.

- **Engineers tired of brute-force eval approaches** — if you've been running 1000 random samples and averaging the scores, MetaReason gets you better answers with 10-50 samples using LHS + Bayesian inference.

- **Teams building agentic AI systems** — evaluating multi-step agent pipelines where quality variance matters. MetaReason supports multi-stage pipelines natively.

### Use Cases

- Compare model A vs. model B with statistical confidence (not just "4.2 vs 4.1")
- Evaluate how prompt parameters (tone, complexity, detail level) affect output quality
- Measure judge reliability before trusting evaluation results (calibration command)
- Generate shareable reports with visualizations for stakeholders who don't read JSON

---

## What's Next

### Near-Term Roadmap

- **Parameter effects analysis** — Bayesian regression to identify which parameters actually matter. Not just "overall quality is 4.8" but "formality has a significant positive effect while creativity has no measurable impact."
- **Additional oracle types** — regex matching, keyword matching, statistical tests, and custom scoring functions beyond LLM judges
- **Additional sampling methods** — beyond Latin Hypercube, for specific use cases

### Current Status

- **v0.1.0 (Alpha)** — core features implemented and working
- Active development on the `main` branch
- 120+ tests, 82% coverage, pre-commit hooks for code quality
- MIT licensed, open to contributions

---

## Call to Action

- **Try it**: Clone the repo, install, run `metareason validate examples/quantum_entanglement_eval.yml` to verify your setup, then run a real eval with `--analyze --report`
- **Star the repo**: [github.com/metareason-ai/metareason-core](https://github.com/metareason-ai/metareason-core)
- **Contribute**: See [CONTRIBUTING.md](https://github.com/metareason-ai/metareason-core/blob/main/CONTRIBUTING.md) for the development workflow. We use conventional commits, Black formatting, and maintain 80%+ test coverage.
- **Give feedback**: Open an issue on GitHub. We want to know what evaluation problems you're trying to solve.

### Quick Start

```bash
git clone https://github.com/metareason-ai/metareason-core.git
cd metareason-core
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
metareason run examples/quantum_entanglement_eval.yml --analyze --report
```

---

## Key Facts & Numbers (Quick Reference)

| Fact | Detail |
|------|--------|
| Version | 0.1.0 (Alpha) |
| License | MIT |
| Python | 3.13+ |
| Tests | 120+, 82% coverage |
| LLM Providers | Ollama, OpenAI, Google, Anthropic |
| CLI Commands | `run`, `validate`, `analyze`, `report`, `calibrate` |
| Sampling | Latin Hypercube with maximin optimization |
| Analysis | PyMC Bayesian inference, NUTS sampler |
| Default HDI | 94% credible interval |
| Distributions | uniform, normal, truncated normal, beta, categorical |
| Output | JSON results + self-contained HTML reports |
| Key Dependencies | PyMC, Pydantic, Click, Jinja2, matplotlib |
