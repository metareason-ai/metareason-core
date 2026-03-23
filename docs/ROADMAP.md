# MetaReason Eval — Product Roadmap

*Last updated: 2026-03-14*

This roadmap reflects priorities informed by the current state of the codebase (v0.1.0), the competitive landscape, and academic research validating Bayesian approaches to LLM evaluation.

---

## Now — Ship the Diagnostic Core

These items build directly on the working foundation and transform MetaReason from a measurement tool into a diagnostic tool.

### Parameter Effects Analysis (Bayesian Regression)
**Status:** Planned in CLAUDE.md, not yet implemented

The single most commercially important missing feature. Use Bayesian regression to identify which parameters significantly affect quality scores, producing statements like "formality has a +0.5 effect on coherence with 94% probability." This moves the product from answering "how good is it?" to "what should I change?" — a fundamentally stronger value proposition.

- Bayesian linear regression over parameter axes → posterior effect sizes
- Interaction effects between parameters (e.g., tone × complexity)
- Ranked parameter importance with credible intervals
- Integration into HTML reports as a dedicated "What Matters" section

### Quickstart Experience
**Status:** Not started

Reduce time-to-first-result to under 5 minutes. The current onboarding requires understanding YAML spec format, configuring an LLM provider, and knowing what Bayesian analysis means. Most potential users will abandon before getting a result.

- `metareason quickstart` command that generates a working spec, runs a small eval against a local Ollama model (or a free-tier API), and produces an HTML report
- Guided prompts for provider selection and API key setup
- A "what just happened" explanation in the report for users unfamiliar with Bayesian inference

### Optional Lightweight Analysis Mode
**Status:** Not started

PyMC is the right tool for full Bayesian analysis, but its dependency chain (PyTensor, C compilation) creates installation friction, especially on Python 3.13. Offer a fallback for users who want quick results without the full posterior.

- Bootstrap confidence intervals as an optional `analysis.method: bootstrap` in the spec
- Make `pymc` an optional dependency (`pip install metareason[bayesian]`)
- Clear messaging in reports when using bootstrap vs. full Bayesian mode
- Preserve the full Bayesian path as the recommended default

### Judge Meta-Evaluation & Multi-Judge Agreement
**Status:** Single-judge calibration implemented; multi-judge not started — [#85](https://github.com/metareason-ai/metareason-core/issues/85)

Every LLM evaluation framework uses LLM-as-Judge. Almost none ask the obvious follow-up: how good is the judge? The entire evaluation industry is building on unvalidated measurement instruments. MetaReason already answers the single-judge version (`metareason calibrate`). The multi-judge version is where it becomes a lead differentiator — the feature that makes the case for the entire product.

**What's built:** Single-judge calibration with Bayesian noise estimation, bias analysis against expected scores, and HTML reporting. Works today via `metareason calibrate`.

**What to build next:**

- Hierarchical Bayesian model over multiple judges: single latent "true quality" with per-judge bias and per-judge noise parameters. Produces statements like "GPT-4o rates 0.3 points higher than Claude on average, but Claude is more consistent — when we correct for bias and weight by reliability, the true quality is 4.2 [3.9, 4.5]"
- Inter-judge agreement metrics (Krippendorff's alpha, Spearman/Pearson correlations) computed from existing multi-oracle evaluation runs
- Automatic judge weighting by posterior reliability (inverse noise)
- Judge comparison report: which judges to use, which to drop, how to combine them optimally
- Extension of `metareason calibrate` to support multi-judge specs (`metareason calibrate-multi` or multi-oracle calibration YAML)

**Why this is the wedge feature:** Someone calibrates their judges, discovers one is unreliable, and now needs the full evaluation pipeline with proper uncertainty quantification. The meta-eval is the hook; the Bayesian quality estimation is the payoff.

### Bayesian A/B Model Comparison
**Status:** Not started — [#81](https://github.com/metareason-ai/metareason-core/issues/81)

Statistically rigorous model comparison that computes the posterior probability that one model is better than another, replacing naive mean comparisons. Every LLM evaluation framework offers "comparison" features that just compare averages — MetaReason can compute P(Model A > Model B) with full uncertainty quantification.

- Posterior probability of superiority (not just "which mean is higher")
- Region of Practical Equivalence (ROPE) analysis for meaningful difference thresholds
- Multi-metric comparison across oracles
- Visualization of overlapping posterior distributions

---

## Next — Expand Reach and Capability

These items widen the addressable market and add capabilities that users of the core product will ask for.

### Local Results Viewer
**Status:** Not started

A lightweight local web app for browsing results, comparing runs, and sharing with teammates who won't use a CLI. This is the minimum viable step toward making MetaReason useful for team leads, product managers, and other stakeholders who consume eval results.

- Simple Flask or Streamlit app served locally
- Browse and compare evaluation runs over time
- Side-by-side report comparison (e.g., model A vs. model B)
- Filterable views by oracle, parameter configuration, date
- Launchable via `metareason dashboard`

### Additional Oracle Types
**Status:** Planned in CLAUDE.md, not yet implemented

LLM-as-Judge is the most flexible oracle, but many evaluations have deterministic ground truth. Supporting simpler oracles reduces cost and latency for tasks where they suffice.

- Regex match oracle (pattern-based pass/fail)
- Keyword/phrase oracle (presence/absence scoring)
- Numeric oracle (exact match, within-tolerance)
- Custom Python function oracle (user-defined scoring logic)
- Composite oracle (weighted combination of multiple oracles)

### Additional Sampling Methods
**Status:** Planned in CLAUDE.md, not yet implemented

Latin Hypercube is excellent for space-filling, but some use cases benefit from different strategies.

- Sobol sequences (quasi-random, better uniformity in high dimensions)
- Grid sampling (exhaustive for small discrete parameter spaces)
- Random sampling (baseline comparator)
- Adaptive sampling (concentrate samples where variance is highest)

### Agentic Workflow Evaluation
**Status:** Not started — [#84](https://github.com/metareason-ai/metareason-core/issues/84)

Extend MetaReason to evaluate multi-turn, multi-tool agentic AI workflows — not just single-turn LLM responses. Agentic systems are the fastest-growing area in AI, but evaluation tooling hasn't kept up.

- Multi-turn conversation evaluation with trajectory-level scoring
- Tool-use correctness assessment (did the agent call the right tools with the right arguments?)
- Goal completion metrics with partial credit
- Workflow-level Bayesian analysis across conversation trajectories

### Built-in Metric Library
**Status:** Not started — [#86](https://github.com/metareason-ai/metareason-core/issues/86)

A library of pre-built evaluation metrics covering RAG, safety, and general quality — reducing setup friction and matching table-stakes features from RAGAS and DeepEval.

- RAG metrics: faithfulness, answer relevancy, context precision/recall
- Safety metrics: toxicity, bias, PII detection
- Quality metrics: coherence, fluency, helpfulness
- Usable via simple references in YAML specs (e.g., `oracle: builtin:faithfulness`)

### Documentation Site
**Status:** Sphinx is in dev dependencies but no docs are generated

The product's core concept — Bayesian uncertainty quantification for LLM eval — requires explanation for most users. Without documentation, adoption depends entirely on users already knowing what this is.

- Quickstart tutorial (end-to-end in 10 minutes)
- Conceptual guide: "Why Bayesian evaluation matters" with visual explanations
- YAML spec reference with all configuration options
- Interpreting results: what HDI, R-hat, ESS, and oracle noise mean in practice
- Cookbook of common evaluation patterns (model comparison, prompt optimization, regression testing)
- Hosted via GitHub Pages or Read the Docs

---

## Later — Platform and Ecosystem

These items represent the longer-term evolution from CLI tool to evaluation platform. Sequence depends on user feedback and market signals.

### Run Comparison and Regression Testing
Track evaluation quality over time. Flag regressions when a new model version or prompt change degrades quality beyond the credible interval of previous runs. This is the "CI for LLM quality" use case.

### Cost-Aware Evaluation
Optimize the tradeoff between statistical confidence and API cost. Estimate how many samples are needed to achieve a target HDI width, and stop early when sufficient confidence is reached.

### Multi-Factor Analysis
ANOVA-style analysis for identifying interaction effects across multiple parameter axes simultaneously, beyond pairwise Bayesian regression.

### CI/CD Integration
**Status:** Not started — [#87](https://github.com/metareason-ai/metareason-core/issues/87)

A pytest plugin and GitHub Action that integrates MetaReason evaluations into CI/CD pipelines, enabling automated quality gates with statistical rigor. This is the "CI for LLM quality" use case — fail a build if quality drops below a credible interval threshold.

- `pytest-metareason` plugin for running evals as test cases
- GitHub Action for evaluation in CI workflows
- Quality gate thresholds based on credible intervals (not just point estimates)
- JUnit-compatible output for CI dashboard integration

### Compliance Report Templates
**Status:** Not started — [#89](https://github.com/metareason-ai/metareason-core/issues/89)

Pre-built report templates aligned with regulatory frameworks (EU AI Act, NIST AI RMF, SOC2 AI controls), leveraging MetaReason's unique ability to provide quantified uncertainty. Regulated industries are increasingly required to demonstrate AI system quality with statistical evidence.

- EU AI Act compliance report template
- NIST AI RMF aligned assessment reports
- Uncertainty quantification mapped to regulatory language
- Audit-ready output with full methodology documentation

### Streaming and Online Evaluation
Support for continuous evaluation of production LLM systems, not just batch evaluation of static configurations. Log-based ingestion, rolling credible intervals, drift detection.

### Plugin and Extension System
Allow users to register custom adapters, oracles, samplers, and analysis methods without forking the core. A plugin registry for community contributions.

### Hosted/SaaS Option
Managed evaluation service for teams that don't want to run infrastructure. Shared dashboards, team-based access controls, historical run storage, and webhook integrations. Evaluate whether this is the right business model vs. remaining open-source with enterprise support.

---

## Competitive Context

The LLM evaluation space (DeepEval, Deepchecks, RAGAS, Inspect AI, MLflow) is crowded but converging on point estimates and LLM-as-Judge without statistical rigor. Recent academic work validates MetaReason's approach:

- Bayesian credible intervals achieve valid coverage at small sample sizes where CLT-based methods fail
- Oracle noise modeling is essential for trustworthy LLM-as-Judge evaluations
- Parameter effects analysis via Bayesian regression is an underserved capability in existing tools

The moat is statistical rigor. The roadmap prioritizes making that moat accessible.
