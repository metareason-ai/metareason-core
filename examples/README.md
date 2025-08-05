# MetaReason Configuration Examples

This directory contains example YAML configuration files demonstrating various features of the MetaReason configuration system.

## Quick Start

Run the demo script to see all features in action:

```bash
python examples/load_config_demo.py
```

Or use the CLI to validate and explore configurations:

```bash
# Validate all example configurations
metareason config validate -d examples/

# Show a specific configuration
metareason config show examples/simple_evaluation.yaml

# Compare two configurations
metareason config diff examples/simple_evaluation.yaml examples/advanced_evaluation.yaml
```

## Example Files

### 1. `simple_evaluation.yaml`
**Basic configuration demonstrating core features**
- Simple prompt template with variables
- Categorical and continuous axes
- Multiple oracle types (embedding similarity, LLM judge)
- Statistical configuration for Bayesian inference
- Metadata and domain context

### 2. `base_config.yaml`
**Base configuration for inheritance**
- Common settings like sampling configuration
- Environment variable substitution examples
- Statistical defaults
- Reusable metadata template

### 3. `shared_oracles.yaml`
**Shared oracle configurations for includes**
- Standard oracle definitions
- Reusable across multiple evaluations
- Demonstrates the `!include` functionality

### 4. `advanced_evaluation.yaml`
**Advanced configuration showcasing all features**
- **Inheritance**: Extends `base_config.yaml`
- **Includes**: Imports oracles from `shared_oracles.yaml`
- **Environment variables**: Uses `${VAR}` substitution
- Complex multi-dimensional axes
- Comprehensive metadata

### 5. `environment_demo.yaml`
**Environment variable substitution showcase**
- Extensive use of `${VAR:default}` syntax
- Boolean, numeric, and string coercion
- Build-time and runtime variable examples
- Deployment configuration patterns

## Configuration Features Demonstrated

### 🔄 Inheritance
```yaml
# Child configuration inherits from parent
inherits: base_config.yaml

# Override specific values
n_variants: 2000

# Add new sections
custom_section:
  new_field: value
```

### 📎 Includes
```yaml
# Include shared configurations
oracles: !include shared_oracles.yaml

# Include can be used for any YAML section
shared_settings: !include common/settings.yaml
```

### 🌍 Environment Variables
```yaml
# Required variable
prompt_id: ${EVALUATION_NAME}

# Variable with default
n_variants: ${N_VARIANTS:1000}

# Bash-style default
threshold: ${ACCURACY_THRESHOLD:-0.8}

# Required with custom error
api_key: ${API_KEY:?API key is required for this evaluation}
```

### ⚡ Performance Features
- **Caching**: Configurations are cached with TTL and hot-reload detection
- **Search paths**: Automatic resolution from standard directories
- **Parallel loading**: Efficient loading of multiple configurations

## CLI Usage Examples

### Validation
```bash
# Validate single file
metareason config validate simple_evaluation.yaml

# Validate directory with strict mode
metareason config validate -d examples/ --strict

# JSON output for CI/CD
metareason config validate examples/ --format json -o results.json
```

### Display and Analysis
```bash
# Pretty-print configuration
metareason config show advanced_evaluation.yaml

# Show with includes expanded
metareason config show advanced_evaluation.yaml --expand-includes

# Show with environment variables resolved
metareason config show environment_demo.yaml --expand-env

# JSON format for processing
metareason config show simple_evaluation.yaml --format json
```

### Comparison
```bash
# Compare configurations
metareason config diff simple_evaluation.yaml advanced_evaluation.yaml

# Ignore metadata differences
metareason config diff config1.yaml config2.yaml --ignore-fields metadata.created_date

# Unified diff format
metareason config diff config1.yaml config2.yaml --format unified
```

### Cache Management
```bash
# Show cache statistics
metareason config cache --stats

# Clear cache
metareason config cache --clear

# Disable caching
metareason config cache --disable
```

## Environment Variables for Examples

Set these environment variables to see full functionality:

```bash
# Basic configuration
export EVALUATION_NAME="my_evaluation"
export N_VARIANTS="1500"
export USER="your_username"
export COMPANY="your_company"

# Advanced features
export SYSTEM_PROMPT="You are a specialized AI assistant"
export ACCURACY_THRESHOLD="0.85"
export JUDGE_MODEL="gpt-4-turbo"

# Build/deployment info
export BUILD_DATE="$(date -I)"
export GIT_COMMIT="$(git rev-parse --short HEAD)"
export BUILD_NUMBER="123"
export DEPLOYMENT_ENV="production"
```

## Best Practices Demonstrated

### 1. **Configuration Organization**
- Use inheritance for common settings
- Create shared includes for reusable components
- Separate environment-specific values

### 2. **Environment Variable Usage**
- Always provide sensible defaults
- Use descriptive variable names with prefixes
- Document required vs. optional variables

### 3. **Validation and Testing**
- Validate configurations in CI/CD pipelines
- Use strict mode for production configurations
- Test with different environment variable combinations

### 4. **Documentation**
- Include comprehensive metadata
- Use descriptive prompt IDs and descriptions
- Document expected parameter ranges and combinations

## Troubleshooting

### Common Issues

1. **Include files not found**
   ```
   Error: Include file not found: shared_oracles.yaml
   ```
   Solution: Ensure included files are in the correct relative path

2. **Environment variable missing**
   ```
   Error: Required environment variable 'API_KEY' is not set
   ```
   Solution: Set the variable or use default syntax `${API_KEY:default}`

3. **Validation errors**
   ```
   Error: Prompt ID cannot be empty
   ```
   Solution: Check the YAML spec documentation for required fields

4. **Circular dependencies**
   ```
   Error: Circular dependency detected
   ```
   Solution: Review inheritance and include chains for loops

### Getting Help

- Check validation output: `metareason config validate your_config.yaml`
- Use verbose mode: `metareason -v config show your_config.yaml`
- Compare with working examples: `metareason config diff examples/simple_evaluation.yaml your_config.yaml`

## Next Steps

1. **Create your own configuration**: Start with `simple_evaluation.yaml` as a template
2. **Set up inheritance**: Create a base configuration for your organization
3. **Use environment variables**: Configure for different deployment environments
4. **Integrate with CI/CD**: Add validation to your build pipeline
5. **Explore the CLI**: Try all the config management commands

For more information, see the main MetaReason documentation.
