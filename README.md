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

Copyright 2025 MetaReason LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
