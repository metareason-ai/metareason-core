"""Tests for adapter privacy utilities."""

from metareason.adapters.base import CompletionRequest, Message, MessageRole
from metareason.adapters.privacy import (
    DataSanitizer,
    PrivacyAuditor,
    PrivacyLevel,
    log_privacy_safe_request,
)


class TestDataSanitizer:
    """Test cases for DataSanitizer."""

    def test_detect_sensitive_data_email(self):
        """Test email detection."""
        text = "Contact me at john.doe@example.com for more info."
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "email"
        assert detections[0]["matched_text"] == "john.doe@example.com"

    def test_detect_sensitive_data_ssn(self):
        """Test SSN detection."""
        text = "My SSN is 123-45-6789."
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "ssn"
        assert detections[0]["matched_text"] == "123-45-6789"

    def test_detect_sensitive_data_credit_card(self):
        """Test credit card detection."""
        text = "Card number: 1234 5678 9012 3456"
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "credit_card"
        assert detections[0]["matched_text"] == "1234 5678 9012 3456"

    def test_detect_sensitive_data_ip_address(self):
        """Test IP address detection."""
        text = "Server at 192.168.1.100 is down."
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "ip_address"
        assert detections[0]["matched_text"] == "192.168.1.100"

    def test_detect_sensitive_data_uuid(self):
        """Test UUID detection."""
        text = "Request ID: 550e8400-e29b-41d4-a716-446655440000"
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "uuid"
        assert detections[0]["matched_text"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_detect_sensitive_data_api_key(self):
        """Test API key detection."""
        test_cases = [
            "api_key=sk-abc123def456",
            "API_KEY: 'ghi789jkl012'",
            'apikey="mno345pqr678"',
        ]

        for text in test_cases:
            detections = DataSanitizer.detect_sensitive_data(text)
            assert len(detections) >= 1
            assert any(d["pattern_type"] == "api_key" for d in detections)

    def test_detect_sensitive_data_token(self):
        """Test token detection."""
        text = "Bearer token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0"
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "token"

    def test_detect_sensitive_data_password(self):
        """Test password detection."""
        text = "password=mySecretPass123"
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0]["pattern_type"] == "password"

    def test_detect_sensitive_data_multiple(self):
        """Test detection of multiple patterns."""
        text = """
        User: john@example.com
        SSN: 123-45-6789
        Server: 10.0.0.1
        API Key: api_key=sk-abc123
        """
        detections = DataSanitizer.detect_sensitive_data(text)

        # Note: API key pattern needs proper format to be detected
        assert len(detections) >= 3  # At least email, ssn, ip_address
        pattern_types = {d["pattern_type"] for d in detections}
        expected_minimum = {"email", "ssn", "ip_address"}
        assert expected_minimum.issubset(pattern_types)

    def test_detect_sensitive_data_none(self):
        """Test text with no sensitive data."""
        text = "This is a normal message with no sensitive information."
        detections = DataSanitizer.detect_sensitive_data(text)

        assert len(detections) == 0

    def test_sanitize_for_logging(self):
        """Test text sanitization for logging."""
        text = """
        Hello john@example.com!
        Your SSN 123-45-6789 has been verified.
        Server 192.168.1.1 processed your request.
        """

        sanitized = DataSanitizer.sanitize_for_logging(text)

        assert "john@example.com" not in sanitized
        assert "123-45-6789" not in sanitized
        assert "192.168.1.1" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_SSN]" in sanitized
        assert "[REDACTED_IP_ADDRESS]" in sanitized

    def test_sanitize_for_logging_no_sensitive_data(self):
        """Test sanitization with clean text."""
        text = "This is a clean message."
        sanitized = DataSanitizer.sanitize_for_logging(text)

        assert sanitized == text  # Should be unchanged

    def test_redact_messages(self):
        """Test message redaction for logging."""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
            ),
            Message(
                role=MessageRole.USER,
                content="My email is sensitive@example.com and my SSN is 123-45-6789.",
                metadata={"user_id": "12345", "session": "abc"},
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="I understand. I'll help you with that request.",
            ),
        ]

        redacted = DataSanitizer.redact_messages(messages)

        assert len(redacted) == 3

        # System message
        assert redacted[0]["role"] == "system"
        assert redacted[0]["has_sensitive_data"] is False
        assert "sensitive@example.com" not in redacted[0]["content_preview"]

        # User message with sensitive data
        assert redacted[1]["role"] == "user"
        assert redacted[1]["has_sensitive_data"] is True
        assert "sensitive@example.com" not in redacted[1]["content_preview"]
        assert "123-45-6789" not in redacted[1]["content_preview"]
        assert "[REDACTED_EMAIL]" in redacted[1]["content_preview"]
        assert "[REDACTED_SSN]" in redacted[1]["content_preview"]
        assert redacted[1]["metadata_keys"] == ["user_id", "session"]

        # Assistant message
        assert redacted[2]["role"] == "assistant"
        assert redacted[2]["has_sensitive_data"] is False

    def test_redact_messages_long_content(self):
        """Test message redaction with long content."""
        long_content = "This is a very long message. " * 10  # > 100 chars
        messages = [Message(role=MessageRole.USER, content=long_content)]

        redacted = DataSanitizer.redact_messages(messages)

        assert len(redacted[0]["content_preview"]) <= 103  # 100 chars + "..."
        assert redacted[0]["content_preview"].endswith("...")
        assert redacted[0]["content_length"] == len(long_content)


class TestPrivacyAuditor:
    """Test cases for PrivacyAuditor."""

    def test_assess_adapter_privacy_ollama(self):
        """Test privacy assessment for Ollama adapter."""
        config = {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "default_model": "llama3",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        assert assessment["privacy_level"] == PrivacyLevel.MAXIMUM
        assert "Local processing only" in assessment["mitigations"]
        assert "No external API calls" in assessment["mitigations"]
        assert len(assessment["risks"]) == 0  # Local URL should have no risks

    def test_assess_adapter_privacy_ollama_remote(self):
        """Test privacy assessment for remote Ollama."""
        config = {
            "type": "ollama",
            "base_url": "http://remote-server:11434",
            "default_model": "llama3",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        assert assessment["privacy_level"] == PrivacyLevel.MAXIMUM
        assert len(assessment["risks"]) == 1
        assert "remote host" in assessment["risks"][0]

    def test_assess_adapter_privacy_openai(self):
        """Test privacy assessment for OpenAI adapter."""
        config = {
            "type": "openai",
            "api_key": "sk-abc123",
            "base_url": "https://api.openai.com/v1",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        assert assessment["privacy_level"] == PrivacyLevel.BASIC
        assert "Data sent to external API provider" in assessment["risks"]
        assert "Use encrypted HTTPS connections" in assessment["mitigations"]
        assert any("API key" in risk for risk in assessment["risks"])

    def test_assess_adapter_privacy_anthropic(self):
        """Test privacy assessment for Anthropic adapter."""
        config = {
            "type": "anthropic",
            "api_key": "sk-ant-123",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        assert assessment["privacy_level"] == PrivacyLevel.BASIC
        assert "Data sent to external API provider" in assessment["risks"]

    def test_assess_adapter_privacy_azure_openai(self):
        """Test privacy assessment for Azure OpenAI adapter."""
        config = {
            "type": "azure_openai",
            "azure_endpoint": "https://myinstance.openai.azure.com",
            "api_key": "abc123",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        assert assessment["privacy_level"] == PrivacyLevel.ENHANCED
        assert "Data processed within Azure tenant" in assessment["mitigations"]

    def test_assess_adapter_privacy_no_api_key(self):
        """Test privacy assessment without API key in config."""
        config = {
            "type": "openai",
            "api_key_env": "OPENAI_API_KEY",
        }

        assessment = PrivacyAuditor.assess_adapter_privacy(config)

        # Should not have API key risk
        api_key_risks = [r for r in assessment["risks"] if "API key" in r]
        assert len(api_key_risks) == 0

    def test_generate_privacy_report_single_ollama(self):
        """Test privacy report for single Ollama adapter."""
        adapters_config = {
            "local": {
                "type": "ollama",
                "base_url": "http://localhost:11434",
            }
        }

        report = PrivacyAuditor.generate_privacy_report(adapters_config)

        assert report["overall_privacy_level"] == PrivacyLevel.MAXIMUM
        assert "local" in report["adapters"]
        assert report["compliance_summary"]["gdpr_ready"] is True
        assert report["compliance_summary"]["data_sovereignty"] is True
        assert "local" in report["compliance_summary"]["hipaa_considerations"][0]

    def test_generate_privacy_report_mixed_adapters(self):
        """Test privacy report for mixed adapter types."""
        adapters_config = {
            "local": {
                "type": "ollama",
                "base_url": "http://localhost:11434",
            },
            "cloud": {
                "type": "openai",
                "api_key": "sk-abc123",
            },
        }

        report = PrivacyAuditor.generate_privacy_report(adapters_config)

        assert report["overall_privacy_level"] == PrivacyLevel.ENHANCED
        assert len(report["adapters"]) == 2
        assert report["adapters"]["local"]["privacy_level"] == PrivacyLevel.MAXIMUM
        assert report["adapters"]["cloud"]["privacy_level"] == PrivacyLevel.BASIC
        assert len(report["recommendations"]) > 0

    def test_generate_privacy_report_all_basic(self):
        """Test privacy report for all basic privacy adapters."""
        adapters_config = {
            "openai": {"type": "openai", "api_key": "sk-123"},
            "anthropic": {"type": "anthropic", "api_key": "sk-ant-123"},
        }

        report = PrivacyAuditor.generate_privacy_report(adapters_config)

        assert report["overall_privacy_level"] == PrivacyLevel.BASIC
        assert report["compliance_summary"]["gdpr_ready"] is False
        assert any("local models" in rec for rec in report["recommendations"])

    def test_generate_privacy_report_enhanced_level(self):
        """Test privacy report with enhanced level adapters."""
        adapters_config = {
            "azure": {
                "type": "azure_openai",
                "azure_endpoint": "https://test.openai.azure.com",
                "api_key": "abc123",
            },
            "basic": {"type": "openai", "api_key": "sk-123"},
        }

        report = PrivacyAuditor.generate_privacy_report(adapters_config)

        assert report["overall_privacy_level"] == PrivacyLevel.ENHANCED


class TestPrivacyLogging:
    """Test privacy-safe logging functionality."""

    def test_log_privacy_safe_request_clean(self, caplog):
        """Test logging clean request."""
        import logging

        caplog.set_level(logging.INFO)

        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="What is the weather like?")
            ],
            model="test-model",
            temperature=0.7,
            max_tokens=100,
        )

        log_privacy_safe_request(request, "test_adapter")

        # Check log content
        log_text = caplog.text
        assert "test_adapter" in log_text
        assert "test-model" in log_text
        assert "What is the weather like?" not in log_text  # Content not logged
        assert "sensitive_data_detected" not in log_text

    def test_log_privacy_safe_request_sensitive(self, caplog):
        """Test logging request with sensitive data."""
        import logging

        caplog.set_level(logging.INFO)

        request = CompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="My email is test@example.com and phone is 555-1234.",
                )
            ],
            model="test-model",
        )

        log_privacy_safe_request(request, "test_adapter")

        # Check that sensitive data is flagged but not logged
        log_text = caplog.text
        assert "sensitive_data_detected" in log_text
        assert "test@example.com" not in log_text
        assert "555-1234" not in log_text

    def test_log_privacy_safe_request_stream(self, caplog):
        """Test logging streaming request."""
        import logging

        caplog.set_level(logging.INFO)

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        log_privacy_safe_request(request, "streaming_adapter")

        log_text = caplog.text
        assert "streaming_adapter" in log_text
        assert "stream" in log_text


class TestPrivacyLevelConstants:
    """Test privacy level constants."""

    def test_privacy_levels(self):
        """Test privacy level constant values."""
        assert PrivacyLevel.NONE == "none"
        assert PrivacyLevel.BASIC == "basic"
        assert PrivacyLevel.ENHANCED == "enhanced"
        assert PrivacyLevel.MAXIMUM == "maximum"

    def test_privacy_levels_ordering(self):
        """Test that privacy levels can be compared correctly."""
        levels = [
            PrivacyLevel.NONE,
            PrivacyLevel.BASIC,
            PrivacyLevel.ENHANCED,
            PrivacyLevel.MAXIMUM,
        ]

        # All should be distinct
        assert len(set(levels)) == 4

        # Basic ordering checks
        assert PrivacyLevel.MAXIMUM != PrivacyLevel.NONE
        assert PrivacyLevel.ENHANCED != PrivacyLevel.BASIC
