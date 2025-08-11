"""Tests for local adapter configuration utilities."""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from aiohttp import ClientConnectorError

from metareason.adapters.local_config import (
    ConfigurationManager,
    LocalSetupValidator,
)


class TestLocalSetupValidator:
    """Test cases for LocalSetupValidator."""

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_success(self):
        """Test successful Ollama validation."""
        from aioresponses import aioresponses

        mock_response_data = {
            "models": [
                {
                    "name": "llama3:latest",
                    "size": 3825819519,
                    "digest": "sha256:abc123",
                    "modified_at": "2024-01-15T10:30:00Z",
                },
                {
                    "name": "mistral:latest",
                    "size": 2634567890,
                    "digest": "sha256:def456",
                    "modified_at": "2024-01-16T11:00:00Z",
                },
            ]
        }

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", payload=mock_response_data)

            result = await LocalSetupValidator.validate_ollama_setup()

            assert result["status"] == "healthy"
            assert result["server_accessible"] is True
            assert len(result["models_available"]) == 2
            assert result["models_available"][0]["name"] == "llama3:latest"
            assert result["models_available"][1]["name"] == "mistral:latest"
            assert len(result["recommendations"]) == 0

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_no_models(self):
        """Test Ollama validation with no models."""
        from aioresponses import aioresponses

        mock_response_data = {"models": []}

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", payload=mock_response_data)

            result = await LocalSetupValidator.validate_ollama_setup()

            assert result["status"] == "no_models"
            assert result["server_accessible"] is True
            assert len(result["models_available"]) == 0
            assert "ollama pull llama3" in result["recommendations"][0]

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_server_error(self):
        """Test Ollama validation with server error."""
        from aioresponses import aioresponses

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", status=500)

            result = await LocalSetupValidator.validate_ollama_setup()

            assert result["status"] == "server_error"
            assert result["server_accessible"] is False
            assert "status 500" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_not_accessible(self):
        """Test Ollama validation when server not accessible."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock ClientSession.get to raise ClientConnectorError
            mock_get.side_effect = ClientConnectorError(
                connection_key=None, os_error=OSError(111, "Connection refused")
            )

            result = await LocalSetupValidator.validate_ollama_setup()

            assert result["status"] == "not_accessible"
            assert result["server_accessible"] is False
            assert "Ollama is installed" in result["recommendations"][1]
            assert "ollama serve" in result["recommendations"][2]

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_timeout(self):
        """Test Ollama validation with timeout."""
        import asyncio

        from aioresponses import aioresponses

        with aioresponses() as m:
            m.get("http://localhost:11434/api/tags", exception=asyncio.TimeoutError())

            result = await LocalSetupValidator.validate_ollama_setup()

            assert result["status"] == "timeout"
            assert result["server_accessible"] is False
            assert "timed out" in result["warnings"][0]

    @pytest.mark.asyncio
    async def test_validate_ollama_setup_remote_warning(self):
        """Test Ollama validation with remote server warning."""
        from aioresponses import aioresponses

        mock_response_data = {"models": []}

        with aioresponses() as m:
            m.get("http://remote-server:11434/api/tags", payload=mock_response_data)

            result = await LocalSetupValidator.validate_ollama_setup(
                "http://remote-server:11434"
            )

            assert "remote" in result["warnings"][0]
            assert "network security" in result["warnings"][0]

    def test_check_local_resources(self):
        """Test local resource checking."""
        with patch("psutil.virtual_memory") as mock_memory:
            with patch("psutil.cpu_count") as mock_cpu_count:
                with patch("psutil.cpu_percent") as mock_cpu_percent:
                    with patch("psutil.disk_usage") as mock_disk_usage:
                        # Mock system information
                        mock_memory.return_value = MagicMock(
                            total=16 * 1024**3,  # 16GB
                            available=8 * 1024**3,  # 8GB
                            percent=50.0,
                        )
                        mock_cpu_count.side_effect = lambda logical=True: (
                            8 if logical else 4
                        )
                        mock_cpu_percent.return_value = 25.0
                        mock_disk_usage.return_value = MagicMock(
                            total=500 * 1024**3,  # 500GB
                            free=100 * 1024**3,  # 100GB
                            used=400 * 1024**3,  # 400GB
                        )

                        resources = LocalSetupValidator.check_local_resources()

                        assert resources["memory"]["total_gb"] == 16.0
                        assert resources["memory"]["available_gb"] == 8.0
                        assert resources["cpu"]["cores"] == 4
                        assert resources["cpu"]["logical_cores"] == 8
                        assert resources["disk"]["free_gb"] == 100.0

                        # Should have no recommendations for good resources
                        assert len(resources["recommendations"]) == 0

    def test_check_local_resources_low_memory(self):
        """Test resource checking with low memory."""
        with patch("psutil.virtual_memory") as mock_memory:
            with patch("psutil.cpu_count", return_value=4):
                with patch("psutil.cpu_percent", return_value=10.0):
                    with patch("psutil.disk_usage") as mock_disk_usage:
                        # Mock low memory
                        mock_memory.return_value = MagicMock(
                            total=4 * 1024**3,  # 4GB (low)
                            available=2 * 1024**3,  # 2GB
                            percent=50.0,
                        )
                        mock_disk_usage.return_value = MagicMock(
                            total=100 * 1024**3, free=50 * 1024**3, used=50 * 1024**3
                        )

                        resources = LocalSetupValidator.check_local_resources()

                        assert resources["memory"]["total_gb"] == 4.0
                        assert "adding more RAM" in resources["recommendations"][0]

    def test_check_local_resources_low_disk(self):
        """Test resource checking with low disk space."""
        with patch("psutil.virtual_memory") as mock_memory:
            with patch("psutil.cpu_count", return_value=4):
                with patch("psutil.cpu_percent", return_value=10.0):
                    with patch("psutil.disk_usage") as mock_disk_usage:
                        mock_memory.return_value = MagicMock(
                            total=16 * 1024**3, available=8 * 1024**3, percent=50.0
                        )
                        # Mock low disk space
                        mock_disk_usage.return_value = MagicMock(
                            total=100 * 1024**3,
                            free=10 * 1024**3,  # 10GB (low)
                            used=90 * 1024**3,
                        )

                        resources = LocalSetupValidator.check_local_resources()

                        assert resources["disk"]["free_gb"] == 10.0
                        assert "disk space" in resources["recommendations"][0]

    def test_generate_setup_guide_ollama(self):
        """Test setup guide generation for Ollama."""
        config = {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "default_model": "llama3",
        }

        guide = LocalSetupValidator.generate_setup_guide(config)

        assert "# Ollama Setup Guide" in guide
        assert "Visit https://ollama.ai" in " ".join(guide)
        assert "ollama serve" in " ".join(guide)
        assert "ollama pull llama3" in " ".join(guide)
        assert "Privacy Benefits" in " ".join(guide)
        assert "All processing happens locally" in " ".join(guide)

    def test_generate_setup_guide_custom_model(self):
        """Test setup guide with custom model."""
        config = {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "default_model": "mistral:7b",
        }

        guide = LocalSetupValidator.generate_setup_guide(config)

        assert "ollama pull mistral:7b" in " ".join(guide)

    def test_generate_setup_guide_llama3_info(self):
        """Test setup guide includes llama3 specific info."""
        config = {
            "type": "ollama",
            "default_model": "llama3:70b",
        }

        guide = LocalSetupValidator.generate_setup_guide(config)

        guide_text = " ".join(guide)
        assert "llama3:70b is a high-quality" in guide_text
        assert "Requires significant RAM" in guide_text


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config = {
            "adapters": {
                "default_adapter": "test_adapter",
                "adapters": {
                    "test_adapter": {
                        "type": "ollama",
                        "base_url": "http://localhost:11434",
                    }
                },
            }
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    @pytest.fixture
    def invalid_config_file(self, tmp_path):
        """Create a temporary invalid config file."""
        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content:\n  - this is\n    malformed")

        return config_file

    def test_load_local_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = ConfigurationManager.load_local_config(temp_config_file)

        assert "adapters" in config
        assert config["adapters"]["default_adapter"] == "test_adapter"
        assert "test_adapter" in config["adapters"]["adapters"]

    def test_load_local_config_file_not_found(self, tmp_path):
        """Test config loading with missing file."""
        missing_file = tmp_path / "missing.yaml"

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager.load_local_config(missing_file)

        assert "Configuration file not found" in str(exc_info.value)

    def test_load_local_config_invalid_yaml(self, invalid_config_file):
        """Test config loading with invalid YAML."""
        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager.load_local_config(invalid_config_file)

        assert "Invalid YAML configuration" in str(exc_info.value)

    def test_validate_config_structure_success(self):
        """Test successful config structure validation."""
        config = {
            "adapters": {
                "default_adapter": "test",
                "adapters": {"test": {"type": "ollama"}},
            }
        }

        # Should not raise exception
        ConfigurationManager._validate_config_structure(config)

    def test_validate_config_structure_missing_adapters(self):
        """Test config validation with missing adapters section."""
        config = {"other": "data"}

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager._validate_config_structure(config)

        assert "Missing required configuration section: adapters" in str(exc_info.value)

    def test_validate_config_structure_no_adapters_defined(self):
        """Test config validation with no adapters defined."""
        config = {"adapters": {}}

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager._validate_config_structure(config)

        assert "No adapters defined" in str(exc_info.value)

    def test_validate_config_structure_no_default_adapter(self):
        """Test config validation with no default adapter."""
        config = {"adapters": {"adapters": {"test": {"type": "ollama"}}}}

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager._validate_config_structure(config)

        assert "No default_adapter specified" in str(exc_info.value)

    def test_validate_config_structure_default_adapter_not_found(self):
        """Test config validation with invalid default adapter."""
        config = {
            "adapters": {
                "default_adapter": "missing",
                "adapters": {"test": {"type": "ollama"}},
            }
        }

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager._validate_config_structure(config)

        assert "Default adapter 'missing' not found" in str(exc_info.value)

    def test_create_minimal_ollama_config_defaults(self):
        """Test minimal config creation with defaults."""
        config = ConfigurationManager.create_minimal_ollama_config()

        assert config["adapters"]["default_adapter"] == "ollama"
        assert config["adapters"]["adapters"]["ollama"]["type"] == "ollama"
        assert config["adapters"]["adapters"]["ollama"]["default_model"] == "llama3"
        assert (
            config["adapters"]["adapters"]["ollama"]["base_url"]
            == "http://localhost:11434"
        )
        assert config["adapters"]["adapters"]["ollama"]["pull_missing_models"] is True

    def test_create_minimal_ollama_config_custom(self):
        """Test minimal config creation with custom parameters."""
        config = ConfigurationManager.create_minimal_ollama_config(
            model="mistral", base_url="http://custom:11434"
        )

        assert config["adapters"]["adapters"]["ollama"]["default_model"] == "mistral"
        assert (
            config["adapters"]["adapters"]["ollama"]["base_url"]
            == "http://custom:11434"
        )

    def test_create_minimal_ollama_config_save_file(self, tmp_path):
        """Test minimal config creation with file saving."""
        output_file = tmp_path / "generated_config.yaml"

        config = ConfigurationManager.create_minimal_ollama_config(
            output_file=output_file
        )

        assert output_file.exists()

        # Load and verify saved file
        with open(output_file, "r") as f:
            saved_config = yaml.safe_load(f)

        assert saved_config == config

    @pytest.mark.asyncio
    async def test_validate_complete_setup_success(self, temp_config_file):
        """Test complete setup validation success."""
        with patch.object(
            LocalSetupValidator, "validate_ollama_setup"
        ) as mock_validate:
            with patch.object(
                LocalSetupValidator, "check_local_resources"
            ) as mock_resources:
                # Mock successful validation
                mock_validate.return_value = {
                    "status": "healthy",
                    "server_accessible": True,
                }
                mock_resources.return_value = {
                    "memory": {"total_gb": 16.0},
                    "recommendations": [],
                }

                result = await ConfigurationManager.validate_complete_setup(
                    temp_config_file
                )

                assert result["overall_status"] == "ready"
                assert result["config_validation"]["status"] == "valid"
                assert "test_adapter" in result["adapter_validation"]
                assert (
                    result["adapter_validation"]["test_adapter"]["status"] == "healthy"
                )

    @pytest.mark.asyncio
    async def test_validate_complete_setup_partial(self, temp_config_file):
        """Test complete setup validation with partial success."""
        with patch.object(
            LocalSetupValidator, "validate_ollama_setup"
        ) as mock_validate:
            with patch.object(
                LocalSetupValidator, "check_local_resources"
            ) as mock_resources:
                # Mock server accessible but no models
                mock_validate.return_value = {
                    "status": "no_models",
                    "server_accessible": True,
                }
                mock_resources.return_value = {"recommendations": []}

                result = await ConfigurationManager.validate_complete_setup(
                    temp_config_file
                )

                assert result["overall_status"] == "partial"
                assert "Some adapters need attention" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_validate_complete_setup_not_ready(self, temp_config_file):
        """Test complete setup validation when not ready."""
        with patch.object(
            LocalSetupValidator, "validate_ollama_setup"
        ) as mock_validate:
            with patch.object(
                LocalSetupValidator, "check_local_resources"
            ) as mock_resources:
                # Mock server not accessible
                mock_validate.return_value = {
                    "status": "not_accessible",
                    "server_accessible": False,
                }
                mock_resources.return_value = {"recommendations": []}

                result = await ConfigurationManager.validate_complete_setup(
                    temp_config_file
                )

                assert result["overall_status"] == "not_ready"
                assert "Local servers need to be started" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_validate_complete_setup_config_error(self, tmp_path):
        """Test complete setup validation with config error."""
        missing_file = tmp_path / "missing.yaml"

        result = await ConfigurationManager.validate_complete_setup(missing_file)

        assert result["overall_status"] == "error"
        assert result["config_validation"]["status"] == "error"
        assert "Configuration file not found" in result["config_validation"]["error"]
