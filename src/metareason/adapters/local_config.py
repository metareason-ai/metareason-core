"""Configuration utilities for local adapter setups."""

import asyncio
import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml

from .privacy import PrivacyAuditor

logger = logging.getLogger(__name__)


class LocalSetupValidator:
    """Validates local adapter setup and configuration."""

    @staticmethod
    async def validate_ollama_setup(
        base_url: str = "http://localhost:11434",
    ) -> Dict[str, Any]:
        """Validate Ollama server setup and connectivity.

        Args:
            base_url: Ollama server URL

        Returns:
            Validation results with status and recommendations
        """
        results = {
            "status": "unknown",
            "server_accessible": False,
            "models_available": [],
            "recommendations": [],
            "warnings": [],
            "system_info": {},
        }

        # System information
        results["system_info"] = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }

        try:
            # Check if server is accessible
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        results["server_accessible"] = True
                        data = await response.json()

                        # Extract available models
                        models = []
                        for model_data in data.get("models", []):
                            model_name = model_data.get("name", "")
                            if model_name:
                                models.append(
                                    {
                                        "name": model_name,
                                        "size": model_data.get("size", 0),
                                        "digest": model_data.get("digest", ""),
                                        "modified": model_data.get("modified_at", ""),
                                    }
                                )

                        results["models_available"] = models
                        results["status"] = "healthy" if models else "no_models"

                        if not models:
                            results["recommendations"].append(
                                "No models found. Run 'ollama pull llama3' to "
                                "get started."
                            )
                    else:
                        results["status"] = "server_error"
                        results["warnings"].append(
                            f"Ollama server returned status {response.status}"
                        )

        except aiohttp.ClientConnectorError:
            results["status"] = "not_accessible"
            results["recommendations"].extend(
                [
                    "Ollama server not accessible. Please ensure:",
                    "1. Ollama is installed (https://ollama.ai)",
                    "2. Ollama server is running ('ollama serve')",
                    "3. Server is accessible at the configured URL",
                ]
            )

        except asyncio.TimeoutError:
            results["status"] = "timeout"
            results["warnings"].append("Connection to Ollama server timed out")

        except Exception as e:
            results["status"] = "error"
            results["warnings"].append(f"Unexpected error: {str(e)}")

        # Network security check
        local_hosts = ["localhost", "127.0.0.1"]
        # Note: 0.0.0.0 bind address check for local development - safe in this context
        bind_all_addr = "0.0.0.0"  # nosec B104
        local_hosts.append(bind_all_addr)

        if base_url and not any(host in base_url for host in local_hosts):
            results["warnings"].append(
                "Ollama server appears to be remote - ensure network security is "
                "configured"
            )

        return results

    @staticmethod
    def check_local_resources() -> Dict[str, Any]:
        """Check local system resources for ML workloads.

        Returns:
            Resource availability assessment
        """
        import psutil

        resources = {
            "memory": {},
            "cpu": {},
            "disk": {},
            "recommendations": [],
        }

        # Memory check
        memory = psutil.virtual_memory()
        resources["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
        }

        if memory.total < 8 * (1024**3):  # Less than 8GB
            resources["recommendations"].append(
                "Consider adding more RAM for better model performance "
                "(8GB+ recommended)"
            )

        # CPU check
        resources["cpu"] = {
            "cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_usage": psutil.cpu_percent(interval=1),
        }

        # Disk space check for model storage
        try:
            disk_usage = psutil.disk_usage(Path.home())
            resources["disk"] = {
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "percent_used": round((disk_usage.used / disk_usage.total) * 100, 1),
            }

            if disk_usage.free < 20 * (1024**3):  # Less than 20GB free
                resources["recommendations"].append(
                    "Ensure adequate disk space for model storage (20GB+ recommended)"
                )
        except Exception:
            resources["recommendations"].append("Could not check disk space")

        return resources

    @staticmethod
    def generate_setup_guide(config: Dict[str, Any]) -> List[str]:
        """Generate step-by-step setup guide for local configuration.

        Args:
            config: Adapter configuration

        Returns:
            List of setup instructions
        """
        instructions = []

        adapter_type = config.get("type", "unknown")

        if adapter_type == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")
            default_model = config.get("default_model", "llama3")

            instructions.extend(
                [
                    "# Ollama Setup Guide",
                    "",
                    "## 1. Install Ollama",
                    (
                        "Visit https://ollama.ai and follow installation instructions "
                        "for your platform."
                    ),
                    "",
                    "## 2. Start Ollama Server",
                    "```bash",
                    "ollama serve",
                    "```",
                    "",
                    "## 3. Pull Required Models",
                    "```bash",
                    f"ollama pull {default_model}",
                    "```",
                    "",
                    "## 4. Verify Setup",
                    "```bash",
                    f"curl {base_url}/api/tags",
                    "```",
                    "",
                    "## 5. Test Configuration",
                    "Run your evaluation with the configuration file.",
                ]
            )

            # Add model-specific instructions
            if default_model in ["llama3", "llama3:70b"]:
                instructions.extend(
                    [
                        "",
                        "## Model Information",
                        f"- {default_model} is a high-quality open-source model",
                        "- Requires significant RAM for larger variants",
                        "- Excellent for general-purpose tasks",
                    ]
                )

            # Add privacy notes
            instructions.extend(
                [
                    "",
                    "## Privacy Benefits",
                    "- All processing happens locally",
                    "- No data sent to external servers",
                    "- Full control over data and models",
                    "- Compliant with data sovereignty requirements",
                ]
            )

        return instructions


class ConfigurationManager:
    """Manages local adapter configurations."""

    @staticmethod
    def load_local_config(config_path: Path) -> Dict[str, Any]:
        """Load and validate local configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Parsed and validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Validate configuration structure
        ConfigurationManager._validate_config_structure(config)

        return config

    @staticmethod
    def _validate_config_structure(config: Dict[str, Any]) -> None:
        """Validate basic configuration structure.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration structure is invalid
        """
        required_sections = ["adapters"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate adapters section
        adapters_config = config["adapters"]
        if "adapters" not in adapters_config:
            raise ValueError("No adapters defined in configuration")

        if "default_adapter" not in adapters_config:
            raise ValueError("No default_adapter specified")

        default_adapter = adapters_config["default_adapter"]
        if default_adapter not in adapters_config["adapters"]:
            raise ValueError(
                f"Default adapter '{default_adapter}' not found in adapters"
            )

    @staticmethod
    def create_minimal_ollama_config(
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Create a minimal Ollama configuration.

        Args:
            model: Default model to use
            base_url: Ollama server URL
            output_file: Optional path to save configuration

        Returns:
            Generated configuration
        """
        config = {
            "adapters": {
                "default_adapter": "ollama",
                "adapters": {
                    "ollama": {
                        "type": "ollama",
                        "base_url": base_url,
                        "default_model": model,
                        "pull_missing_models": True,
                        "timeout": 120.0,
                        "retry": {
                            "max_retries": 2,
                            "initial_delay": 2.0,
                            "max_delay": 10.0,
                        },
                        "rate_limit": {
                            "concurrent_requests": 3,
                            "burst_size": 5,
                        },
                    }
                },
            },
            "templates": {
                "system_prompt": "You are a helpful AI assistant.",
                "user_prompt": "{{input}}",
            },
            "schema": {"input": {"type": "categorical", "values": ["Hello, world!"]}},
            "sampling": {"method": "latin_hypercube", "sample_size": 5},
            "oracles": {
                "primary": {
                    "type": "llm_judge",
                    "adapter": "ollama",
                    "model": model,
                    "system_prompt": "Rate the response quality from 1-5.",
                    "prompt": "Input: {{input}}\nResponse: {{response}}\nRating:",
                }
            },
            "analysis": {"confidence_level": 0.95},
            "output": {"format": "json"},
            "privacy": {"level": "maximum", "data_retention": "local_only"},
        }

        if output_file:
            with open(output_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config

    @staticmethod
    async def validate_complete_setup(config_path: Path) -> Dict[str, Any]:
        """Perform complete validation of local setup.

        Args:
            config_path: Path to configuration file

        Returns:
            Comprehensive validation results
        """
        results = {
            "overall_status": "unknown",
            "config_validation": {},
            "adapter_validation": {},
            "privacy_assessment": {},
            "resource_check": {},
            "recommendations": [],
        }

        try:
            # Load and validate configuration
            config = ConfigurationManager.load_local_config(config_path)
            results["config_validation"] = {"status": "valid", "config": config}

            # Validate adapters
            adapters_config = config["adapters"]["adapters"]
            adapter_results = {}

            for adapter_name, adapter_config in adapters_config.items():
                if adapter_config.get("type") == "ollama":
                    validation = await LocalSetupValidator.validate_ollama_setup(
                        adapter_config.get("base_url", "http://localhost:11434")
                    )
                    adapter_results[adapter_name] = validation

            results["adapter_validation"] = adapter_results

            # Privacy assessment
            privacy_report = PrivacyAuditor.generate_privacy_report(adapters_config)
            results["privacy_assessment"] = privacy_report

            # Resource check
            resources = LocalSetupValidator.check_local_resources()
            results["resource_check"] = resources

            # Determine overall status
            all_adapters_healthy = all(
                result.get("status") == "healthy" for result in adapter_results.values()
            )

            if all_adapters_healthy:
                results["overall_status"] = "ready"
            elif any(
                result.get("server_accessible") for result in adapter_results.values()
            ):
                results["overall_status"] = "partial"
                results["recommendations"].append("Some adapters need attention")
            else:
                results["overall_status"] = "not_ready"
                results["recommendations"].append("Local servers need to be started")

        except Exception as e:
            results["config_validation"] = {"status": "error", "error": str(e)}
            results["overall_status"] = "error"

        return results
