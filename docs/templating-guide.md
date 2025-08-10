# Templating Guide

MetaReason Core uses a powerful Jinja2-based templating system to generate dynamic prompts for Large Language Model evaluation. This guide covers everything you need to know about creating, validating, and using templates in your evaluations.

## Overview

The templating system allows you to:
- Create dynamic prompts with variable substitution
- Use powerful Jinja2 syntax with custom filters
- Validate templates for security and correctness
- Generate sample prompts for testing
- Batch render prompts with statistical sampling

## Basic Template Structure

Templates are defined in your YAML configuration file using the `prompt_template` field:

```yaml
prompt_template: |
  {{instruction}} the concept of {{topic}} in {{style}} terms.

  Please ensure your explanation includes:
  {{ requirements | format_list }}

  Use a {{tone}} tone and aim for {{complexity}} complexity.
```

## Variable Substitution

Variables are defined in the `schema` section of your configuration:

```yaml
schema:
  instruction:
    type: categorical
    values: ["Explain", "Describe", "Analyze"]

  topic:
    type: categorical
    values: ["machine learning", "neural networks", "deep learning"]

  style:
    type: categorical
    values: ["technical", "simple", "academic"]

  tone:
    type: categorical
    values: ["professional", "friendly", "authoritative"]

  complexity:
    type: categorical
    values: ["beginner", "intermediate", "advanced"]

  requirements:
    type: categorical
    values: [["key concepts", "examples"], ["theory", "applications"], ["history", "future trends"]]
```

### Variable Types

MetaReason supports several variable types:

#### Categorical Variables
```yaml
instruction:
  type: categorical
  values: ["Explain", "Describe", "Analyze"]
  weights: [0.4, 0.4, 0.2]  # Optional: weighted sampling
```

#### Continuous Variables
```yaml
temperature:
  type: truncated_normal
  mu: 0.7      # Mean
  sigma: 0.1   # Standard deviation
  min: 0.3     # Minimum value
  max: 0.9     # Maximum value
```

#### Uniform Distributions
```yaml
confidence_threshold:
  type: uniform
  min: 0.8
  max: 0.95
```

## Custom Filters

MetaReason provides several custom Jinja2 filters for common formatting tasks:

### Numerical Formatting

```jinja2
{{ 0.123 | format_continuous(2) }}           → 0.12
{{ 0.123 | format_continuous(2, 'percent') }} → 12.30%
{{ 3.14159 | fmt_num(2) }}                   → 3.14 (alias)
{{ 0.95 | round_to_precision(1) }}           → 1.0
```

### List Formatting

```jinja2
{{ ['A', 'B', 'C'] | format_list }}                    → A, B, and C
{{ ['red', 'green'] | format_list('; ', 'or') }}       → red; or green
{{ items | fmt_list }}                                  → formatted list (alias)
```

### Conditional Text

```jinja2
{{ premium | conditional_text('Pro features enabled', 'Basic features') }}
{{ is_admin | if_text('Admin access granted') }}       → conditional text or empty
```

### Text Manipulation

```jinja2
{{ 'hello world' | capitalize_first }}      → Hello world
{{ text | truncate(50, '...') }}            → truncated text
{{ 5 | pluralize('item', 'items') }}        → items
{{ 1 | pluralize('item', 'items') }}        → item
```

## Global Functions

In addition to filters, MetaReason provides global functions:

```jinja2
{{ include_if(premium, "Premium features available.") }}
{{ select(is_admin, "Admin Panel", "User Dashboard") }}
{{ range(1, 10, 2) }}                        → [1, 3, 5, 7, 9]
```

## Advanced Template Examples

### Educational Content Generation
```jinja2
You are a {{role}} assistant helping with {{subject}} education.

Task: {{instruction}} the concept of "{{topic}}" for a {{audience}} audience.

Requirements:
{{- if complexity == 'beginner' }}
- Use simple, non-technical language
- Provide concrete examples
- Avoid jargon and acronyms
{{- elif complexity == 'advanced' }}
- Include technical details and formal definitions
- Reference related concepts and research
- Use appropriate mathematical notation if relevant
{{- endif }}

Style Guidelines:
- Tone: {{ tone | capitalize_first }}
- Length: {{ select(audience == 'student', '2-3 paragraphs', '1-2 paragraphs') }}
- Examples: {{ include_if(include_examples, 'Include 1-2 practical examples') }}

Temperature: {{ temperature | format_continuous(1) }}
```

### Customer Support Scenarios
```jinja2
Customer Inquiry Type: {{ inquiry_type | title }}
Customer Sentiment: {{ sentiment | capitalize_first }}
Priority Level: {{ priority }}

Context:
The customer is {{ sentiment }} about {{ issue_category }}.
Their technical expertise level is {{ expertise_level }}.

Response Guidelines:
{{ response_guidelines | format_list }}

{{ include_if(escalation_needed, "⚠️ ESCALATION REQUIRED - Senior agent review needed.") }}

Suggested Response Tone: {{ response_tone }}
Estimated Resolution Time: {{ resolution_time | format_continuous(0) }} minutes
```

## CLI Commands

MetaReason provides several CLI commands for working with templates:

### Validate Templates
```bash
# Validate template syntax and security
metareason template validate config.yaml

# Set validation strictness
metareason template validate config.yaml --level strict

# JSON output for automation
metareason template validate config.yaml --format json
```

### Render Sample Prompts
```bash
# Generate 10 sample prompts
metareason template render config.yaml --samples 10

# Show variable context
metareason template render config.yaml --samples 5 --show-context

# Save to file
metareason template render config.yaml --samples 100 --output samples.json

# Different output formats
metareason template render config.yaml --format jsonl --output samples.jsonl
```

### Test Templates Directly
```bash
# Test a template string
metareason template test "Hello {{name}}!" --context '{"name": "World"}'

# With variable validation
metareason template test "{{greeting}} {{name}}" --variables "greeting,name"
```

### List Available Filters
```bash
# Show all custom filters and examples
metareason template filters
```

## Template Security

MetaReason includes template security validation:

### Security Features
- **Code Injection Prevention**: Blocks dangerous constructs like `{% raw %}`, `{% include %}`, `{% import %}`
- **Attribute Access Control**: Restricts access to private attributes and methods
- **Function Call Filtering**: Prevents execution of potentially dangerous functions
- **Length Limits**: Configurable maximum template length
- **Variable Validation**: Ensures all required variables are present

### Validation Levels
```bash
# Permissive - basic syntax checking only
metareason template validate config.yaml --level permissive

# Standard - recommended security checks (default)
metareason template validate config.yaml --level standard

# Strict - maximum security, blocks advanced Jinja2 features
metareason template validate config.yaml --level strict
```

## Best Practices

### Template Organization
- Keep templates focused and modular
- Use clear, descriptive variable names
- Include comments for complex logic
- Validate templates early and often

### Variable Design
- Choose appropriate distribution types for your use case
- Use weighted sampling for realistic distributions
- Consider correlation between variables when using stratified sampling
- Test edge cases with extreme variable values

### Performance Optimization
- Use batch rendering for large numbers of prompts
- Cache template compilation when possible
- Minimize complex filter operations in loops
- Consider template complexity vs. generation time

### Debugging Templates
```bash
# Generate a few samples to check output
metareason template render config.yaml --samples 3 --show-context

# Test specific variable combinations
metareason template test "{{instruction}} {{topic}}" --context '{"instruction": "Explain", "topic": "AI"}'

# Validate before full generation
metareason template validate config.yaml --level strict
```

## Integration with Sampling

Templates work seamlessly with MetaReason's sampling strategies:

```yaml
sampling:
  method: latin_hypercube
  optimization_criterion: maximin  # Maximize minimum distance
  random_seed: 42                  # Reproducible results
  stratified_by: ["instruction", "topic"]  # Ensure coverage
```

The sampling system ensures your templates receive well-distributed variable combinations, leading to comprehensive evaluation coverage.

## Error Handling

Common template errors and solutions:

### Undefined Variables
```
Error: 'topic' is undefined
Solution: Ensure all template variables are defined in the schema
```

### Invalid Jinja2 Syntax
```
Error: unexpected char '&' at 42
Solution: Check template syntax, especially around variable blocks
```

### Security Violations
```
Error: Forbidden attribute access: __class__
Solution: Remove attempts to access private attributes or methods
```

### Filter Errors
```
Error: No filter named 'custom_filter'
Solution: Use built-in filters or register custom filters properly
```

## Extending the System

While MetaReason comes with comprehensive templating features, you can extend it:

1. **Custom Filters**: Register additional Jinja2 filters for domain-specific formatting
2. **Global Functions**: Add helper functions available in all templates
3. **Variable Types**: Extend the schema system with custom distribution types
4. **Validation Rules**: Add domain-specific template validation

For advanced use cases, refer to the [API Reference](api-reference.md) for programmatic template usage.

## Examples Repository

Find more template examples in the `examples/` directory:
- `simple_evaluation.yaml` - Basic template usage
- `advanced_evaluation.yaml` - Complex multi-variable templates
- `environment_demo.yaml` - Environment variable integration

Each example includes detailed comments explaining template design decisions and best practices.
