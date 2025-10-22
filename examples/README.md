# Example Specifications

This directory contains example MetaReason specifications demonstrating various features.

## quantum_entanglement_eval.yml

A comprehensive example demonstrating:
- **5 parameter axes**: 1 categorical + 4 continuous distributions (uniform, normal, truncnorm, beta)
- **Dual oracles**: Coherence and accuracy evaluation judges
- **Latin Hypercube Sampling**: 10 variants with maximin optimization for parameter space exploration
- **Template rendering**: Jinja2 parameter interpolation in prompts

### What it does

This spec evaluates how an LLM explains quantum entanglement with varying characteristics:
- **Tone**: formal, casual, technical, or friendly
- **Complexity Level**: 1-10 scale (uniform distribution)
- **Detail Level**: Centered around 5 (normal distribution)
- **Formality Score**: 0-10 (truncated normal distribution)
- **Creativity Factor**: Skewed toward lower values (beta distribution)

### Running the example

```bash
# Validate the specification
metareason validate examples/quantum_entanglement_eval.yml

# Run the evaluation (requires Ollama with gpt-oss:20b model)
metareason run examples/quantum_entanglement_eval.yml
```

### Requirements

This example uses the Ollama adapter with the `gpt-oss:20b` model. Make sure you have:
1. Ollama installed and running
2. The `gpt-oss:20b` model pulled (`ollama pull gpt-oss:20b`)

### Output

Results are saved to the `reports/` directory with timestamps. Each result includes:
- Sample parameters for each variant
- Generated prompt
- LLM response
- Oracle evaluation scores and explanations
