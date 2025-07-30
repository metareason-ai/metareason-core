# MetaReason Core

> **Quantify AI confidence through statistically rigorous evaluation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/metareason-ai/metareason-core.svg)](https://github.com/metareason-ai/metareason-core/issues)
[![GitHub Stars](https://img.shields.io/github/stars/metareason-ai/metareason-core.svg)](https://github.com/metareason-ai/metareason-core/stargazers)

MetaReason Core is the open-source evaluation engine that transforms LLM outputs from "it seems to work" to "it works with 94% confidence ±2%." Using Latin Hypercube Sampling and dual-oracle evaluation, it provides statistically rigorous confidence scores for AI systems in production.

## 🎯 What Problem Does This Solve?

- **Quantify AI Uncertainty**: Move beyond binary pass/fail to true confidence intervals
- **Statistical Rigor**: Bayesian analysis with proper uncertainty quantification
- **Real-World Testing**: Hundreds (or thousands!) of prompt variants reflect actual usage patterns
- **Regulatory Compliance**: ISO 42001 and EU AI Act ready
- **Open Methodology**: Transparent, auditable evaluation process

## 🚀 Quick Start

```bash
# Install MetaReason Core
pip install metareason-core

# Run your first evaluation
metareason evaluate examples/financial-qa.yaml

# View results
metareason report --latest
```

## 📋 Features

### 🔄 In Development (v1.1)

- YAML-based evaluation specifications
- Latin Hypercube Sampling for parameter space exploration
- Embedding similarity oracle (accuracy scoring)
- LLM-as-judge oracle (explainability scoring)
- Bayesian confidence intervals via PyMC
- CLI interface with rich output
- Detailed reporting and visualization
- Multi-model comparison

### 🎯 Planned Features (v2.0)

- Custom oracle development framework
- Performance benchmarking suite
- Docker containerization
- Real-time monitoring integration
- Drift detection algorithms
- Automated remediation workflows
- Enterprise dashboard API

### 📖 Documentation

- Getting Started Guide - Your first evaluation in 5 minutes
- YAML Schema Reference - Complete specification format
- Statistical Methodology - Mathematical foundations
- Oracle Development - Creating custom evaluation criteria
- API Reference - Python API documentation

## 🤝 Contributing

We welcome contributions! MetaReason Core is built by the community, for the community.

### Quick Contribution Steps

- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

## 🏢 Enterprise Version
Looking for enterprise features like:

- Dashboards
- Automated jobs
- VPC-native deployment
- Custom oracle development
- Compliance reporting

Check out MetaReason Enterprise or contact us at enterprise@metareason.ai

## 📜 License

Copyright 2025 MetaReason LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## 🌟 Community

GitHub Discussions: github.com/metareason/metareason-core/discussions

Discord Server: discord.gg/metareason

Twitter: @MetaReasonAI

Blog: blog.metareason.ai

Made with ❤️ by the MetaReason community
