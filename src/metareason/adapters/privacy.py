"""Privacy utilities for LLM adapters."""

import logging
import re
from typing import Any, Dict, List, Pattern

from .base import CompletionRequest, Message

logger = logging.getLogger(__name__)


class PrivacyLevel:
    """Privacy level constants."""

    NONE = "none"  # No privacy protection
    BASIC = "basic"  # Basic data minimization
    ENHANCED = "enhanced"  # Enhanced privacy with local processing
    MAXIMUM = "maximum"  # Maximum privacy (local-only)


class DataSanitizer:
    """Utility class for sanitizing data for privacy."""

    # Common patterns that might contain sensitive information
    SENSITIVE_PATTERNS: List[Pattern[str]] = [
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),  # Credit card
        re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IP address
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),  # UUID
        re.compile(
            r'\bapi[_-]?key[_-]?[=:]\s*["\']?[a-zA-Z0-9_-]+["\']?', re.IGNORECASE
        ),  # API keys
        re.compile(
            r'\btoken[_-]?[=:]\s*["\']?[a-zA-Z0-9_.-]+["\']?', re.IGNORECASE
        ),  # Tokens
        re.compile(
            r'\bpassword[_-]?[=:]\s*["\']?[^\s"\']+["\']?', re.IGNORECASE
        ),  # Passwords
    ]

    @classmethod
    def detect_sensitive_data(cls, text: str) -> List[Dict[str, Any]]:
        """Detect potentially sensitive data in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected sensitive patterns
        """
        detections = []

        for i, pattern in enumerate(cls.SENSITIVE_PATTERNS):
            matches = pattern.finditer(text)
            for match in matches:
                detections.append(
                    {
                        "pattern_id": i,
                        "pattern_type": cls._get_pattern_type(i),
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group(),
                    }
                )

        return detections

    @classmethod
    def _get_pattern_type(cls, pattern_id: int) -> str:
        """Get human-readable pattern type."""
        pattern_types = [
            "email",
            "ssn",
            "credit_card",
            "ip_address",
            "uuid",
            "api_key",
            "token",
            "password",
        ]
        return (
            pattern_types[pattern_id] if pattern_id < len(pattern_types) else "unknown"
        )

    @classmethod
    def sanitize_for_logging(cls, text: str) -> str:
        """Sanitize text for safe logging by redacting sensitive patterns.

        Args:
            text: Original text

        Returns:
            Sanitized text with sensitive data redacted
        """
        sanitized = text

        for i, pattern in enumerate(cls.SENSITIVE_PATTERNS):
            pattern_type = cls._get_pattern_type(i)
            sanitized = pattern.sub(f"[REDACTED_{pattern_type.upper()}]", sanitized)

        return sanitized

    @classmethod
    def redact_messages(cls, messages: List[Message]) -> List[Dict[str, Any]]:
        """Redact sensitive content from messages for logging.

        Args:
            messages: Original messages

        Returns:
            Sanitized message representations
        """
        return [
            {
                "role": msg.role.value,
                "content_length": len(msg.content),
                "content_preview": (
                    cls.sanitize_for_logging(msg.content[:100]) + "..."
                    if len(msg.content) > 100
                    else cls.sanitize_for_logging(msg.content)
                ),
                "has_sensitive_data": bool(cls.detect_sensitive_data(msg.content)),
                "metadata_keys": list(msg.metadata.keys()) if msg.metadata else [],
            }
            for msg in messages
        ]


class PrivacyAuditor:
    """Audit privacy characteristics of adapter configurations."""

    @staticmethod
    def assess_adapter_privacy(adapter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy characteristics of an adapter configuration.

        Args:
            adapter_config: Adapter configuration

        Returns:
            Privacy assessment report
        """
        assessment = {
            "privacy_level": PrivacyLevel.NONE,
            "risks": [],
            "mitigations": [],
            "compliance_notes": [],
        }

        adapter_type = adapter_config.get("type", "unknown")
        base_url = adapter_config.get("base_url", "")

        # Assess based on adapter type
        if adapter_type == "ollama":
            assessment["privacy_level"] = PrivacyLevel.MAXIMUM
            assessment["mitigations"].extend(
                [
                    "Local processing only",
                    "No external API calls",
                    "No data leaves local environment",
                    "Full user control over data",
                ]
            )

            # Check for network exposure risks
            local_hosts = ["localhost", "127.0.0.1"]
            # Note: 0.0.0.0 bind address check for local development - safe in this context
            bind_all_addr = "0.0.0.0"  # nosec B104
            local_hosts.append(bind_all_addr)

            if base_url and not any(host in base_url for host in local_hosts):
                assessment["risks"].append(
                    "Ollama server appears to be on remote host - verify network security"
                )

        elif adapter_type in ["openai", "anthropic"]:
            assessment["privacy_level"] = PrivacyLevel.BASIC
            assessment["risks"].extend(
                [
                    "Data sent to external API provider",
                    "Subject to provider's privacy policy",
                    "Data retention by third party",
                    "Potential for data analysis by provider",
                ]
            )
            assessment["mitigations"].extend(
                [
                    "Use encrypted HTTPS connections",
                    "Review provider privacy policy",
                    "Consider data processing agreements",
                    "Implement request/response logging controls",
                ]
            )

        elif adapter_type == "azure_openai":
            assessment["privacy_level"] = PrivacyLevel.ENHANCED
            assessment["mitigations"].extend(
                [
                    "Data processed within Azure tenant",
                    "Enhanced enterprise privacy controls",
                    "Configurable data residency",
                ]
            )

        # API key assessment
        if adapter_config.get("api_key"):
            assessment["risks"].append(
                "API key in configuration - ensure secure storage"
            )
            assessment["compliance_notes"].append(
                "Secure credential management required"
            )

        return assessment

    @staticmethod
    def generate_privacy_report(
        adapters_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a comprehensive privacy report for all adapters.

        Args:
            adapters_config: Dictionary of adapter configurations

        Returns:
            Comprehensive privacy report
        """
        report = {
            "overall_privacy_level": PrivacyLevel.NONE,
            "adapters": {},
            "recommendations": [],
            "compliance_summary": {
                "gdpr_ready": False,
                "hipaa_considerations": [],
                "data_sovereignty": False,
            },
        }

        privacy_levels = []

        for adapter_name, config in adapters_config.items():
            assessment = PrivacyAuditor.assess_adapter_privacy(config)
            report["adapters"][adapter_name] = assessment
            privacy_levels.append(assessment["privacy_level"])

        # Determine overall privacy level (most restrictive)
        if PrivacyLevel.MAXIMUM in privacy_levels:
            if all(level == PrivacyLevel.MAXIMUM for level in privacy_levels):
                report["overall_privacy_level"] = PrivacyLevel.MAXIMUM
            else:
                report["overall_privacy_level"] = PrivacyLevel.ENHANCED
        elif PrivacyLevel.ENHANCED in privacy_levels:
            report["overall_privacy_level"] = PrivacyLevel.ENHANCED
        elif PrivacyLevel.BASIC in privacy_levels:
            report["overall_privacy_level"] = PrivacyLevel.BASIC

        # Generate recommendations
        if report["overall_privacy_level"] != PrivacyLevel.MAXIMUM:
            report["recommendations"].extend(
                [
                    "Consider using local models (Ollama) for maximum privacy",
                    "Implement data minimization practices",
                    "Review all external data processing agreements",
                    "Enable request/response logging controls",
                ]
            )

        # Compliance assessment
        max_privacy_adapters = [
            name
            for name, assessment in report["adapters"].items()
            if assessment["privacy_level"] == PrivacyLevel.MAXIMUM
        ]

        if max_privacy_adapters:
            report["compliance_summary"]["gdpr_ready"] = True
            report["compliance_summary"]["data_sovereignty"] = True
            report["compliance_summary"]["hipaa_considerations"].append(
                f"Local processing available via: {', '.join(max_privacy_adapters)}"
            )

        return report


def log_privacy_safe_request(request: CompletionRequest, adapter_name: str) -> None:
    """Log request information in a privacy-safe manner.

    Args:
        request: Completion request
        adapter_name: Name of the adapter being used
    """
    safe_info = {
        "adapter": adapter_name,
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "message_count": len(request.messages),
        "stream": request.stream,
    }

    # Check for sensitive data
    has_sensitive = any(
        DataSanitizer.detect_sensitive_data(msg.content) for msg in request.messages
    )

    if has_sensitive:
        safe_info["sensitive_data_detected"] = True
        logger.warning("Sensitive data detected in request - content not logged")

    logger.info(f"LLM request processed: {safe_info}")
