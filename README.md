# MetaReason Core

MetaReason Core is indended to be an open source tool for quantiative measurement of LLM and agentic AI systems.

## The Vision

- Yaml based specifications
- Jinja templating
- Latin Hypercube Sampling (and others)
- LLM Judges (and others)
- Bayesian Analysis
- Pipeline of steps
- CLI
- Report generation

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security checks
- **pre-commit**: Automated checks before commits

Run all checks manually:
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linting
flake8 src tests

# Type checking
mypy src

# Security checks
bandit -r src
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
