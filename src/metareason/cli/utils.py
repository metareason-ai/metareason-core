"""CLI utility functions for MetaReason."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.markup import escape
from rich.table import Table

from ..config.models import EvaluationConfig
from ..config.validator import ValidationReport


def find_config_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all YAML configuration files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of configuration file paths

    Raises:
        ValueError: If directory doesn't exist or contains no YAML files
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find YAML files
    yaml_files = []

    if recursive:
        yaml_files.extend(directory.rglob("*.yaml"))
        yaml_files.extend(directory.rglob("*.yml"))
    else:
        yaml_files.extend(directory.glob("*.yaml"))
        yaml_files.extend(directory.glob("*.yml"))

    # Filter out hidden files and temporary files
    yaml_files = [
        f for f in yaml_files if not f.name.startswith(".") and not f.name.endswith("~")
    ]

    if not yaml_files:
        raise ValueError(f"No YAML configuration files found in {directory}")

    return sorted(yaml_files)


def format_validation_report(
    report: ValidationReport, file_path: Optional[str] = None
) -> str:
    """Format a validation report for CLI output.

    Args:
        report: Validation report to format
        file_path: Optional file path for context

    Returns:
        Formatted report string
    """
    lines = []

    # Header
    if file_path:
        lines.append(f"📄 [bold]{escape(file_path)}[/bold]")

    # Status
    if report.is_valid:
        lines.append("✅ [green]Valid configuration[/green]")
    else:
        lines.append("❌ [red]Invalid configuration[/red]")

    # Errors
    if report.errors:
        lines.append("")
        lines.append("🚨 [bold red]Errors:[/bold red]")
        for i, error in enumerate(report.errors, 1):
            lines.append(
                f"  {i}. [red]{error['field']}[/red]: {escape(error['message'])}"
            )
            if error.get("suggestion"):
                lines.append(f"     💡 {escape(error['suggestion'])}")

    # Warnings
    if report.warnings:
        lines.append("")
        lines.append("⚠️  [bold yellow]Warnings:[/bold yellow]")
        for warning in report.warnings:
            lines.append(f"  • [yellow]{escape(warning)}[/yellow]")

    # Suggestions
    if report.suggestions:
        lines.append("")
        lines.append("💡 [bold blue]Suggestions:[/bold blue]")
        for suggestion in report.suggestions:
            lines.append(f"  • [blue]{escape(suggestion)}[/blue]")

    return "\n".join(lines)


def compare_configurations(
    config1: EvaluationConfig,
    config2: EvaluationConfig,
    ignore_fields: Optional[List[str]] = None,
    file1_name: str = "config1",
    file2_name: str = "config2",
) -> Dict[str, Any]:
    """Compare two configurations and return differences.

    Args:
        config1: First configuration
        config2: Second configuration
        ignore_fields: Fields to ignore in comparison
        file1_name: Name for first file in output
        file2_name: Name for second file in output

    Returns:
        Dictionary containing comparison results
    """
    ignore_set = set(ignore_fields or [])

    # Convert to dictionaries
    dict1 = config1.model_dump()
    dict2 = config2.model_dump()

    # Compare dictionaries
    changes = []
    _compare_dicts(dict1, dict2, "", changes, ignore_set)

    # Generate summary
    added_count = len([c for c in changes if c["type"] == "added"])
    removed_count = len([c for c in changes if c["type"] == "removed"])
    modified_count = len([c for c in changes if c["type"] == "modified"])

    return {
        "file1_name": file1_name,
        "file2_name": file2_name,
        "has_differences": len(changes) > 0,
        "changes": changes,
        "summary": {
            "total_changes": len(changes),
            "added": added_count,
            "removed": removed_count,
            "modified": modified_count,
        },
    }


def _compare_dicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    path: str,
    changes: List[Dict[str, Any]],
    ignore_fields: Set[str],
) -> None:
    """Recursively compare dictionaries and collect changes."""
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        # Skip ignored fields
        if current_path in ignore_fields:
            continue

        if key not in dict1:
            # Added in dict2
            changes.append(
                {"type": "added", "path": current_path, "new_value": dict2[key]}
            )
        elif key not in dict2:
            # Removed from dict1
            changes.append(
                {"type": "removed", "path": current_path, "old_value": dict1[key]}
            )
        else:
            # Key exists in both
            val1, val2 = dict1[key], dict2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                # Recursively compare nested dictionaries
                _compare_dicts(val1, val2, current_path, changes, ignore_fields)
            elif val1 != val2:
                # Values differ
                changes.append(
                    {
                        "type": "modified",
                        "path": current_path,
                        "old_value": val1,
                        "new_value": val2,
                    }
                )


def create_config_summary_table(config: EvaluationConfig) -> Table:
    """Create a rich table summarizing a configuration.

    Args:
        config: Configuration to summarize

    Returns:
        Rich table with configuration summary
    """
    table = Table(
        title="Configuration Summary", show_header=True, header_style="bold blue"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    # Basic information
    table.add_row("Prompt ID", config.prompt_id)
    table.add_row("Variants", str(config.n_variants))
    table.add_row("Sampling Method", config.sampling.method)

    # Axes
    axis_count = len(config.axes)
    categorical_count = sum(
        1 for axis in config.axes.values() if hasattr(axis, "values")
    )
    continuous_count = axis_count - categorical_count

    table.add_row("Total Axes", str(axis_count))
    table.add_row("Categorical Axes", str(categorical_count))
    table.add_row("Continuous Axes", str(continuous_count))

    # Oracles
    oracle_count = sum(
        [
            1 if config.oracles.accuracy else 0,
            1 if config.oracles.explainability else 0,
            1 if config.oracles.confidence_calibration else 0,
            len(config.oracles.custom_oracles) if config.oracles.custom_oracles else 0,
        ]
    )
    table.add_row("Oracles", str(oracle_count))

    # Optional sections
    table.add_row("Statistical Config", "✅" if config.statistical_config else "❌")
    table.add_row("Metadata", "✅" if config.metadata else "❌")
    table.add_row("Domain Context", "✅" if config.domain_context else "❌")

    return table


def discover_config_directories() -> List[Path]:
    """Discover common configuration directories.

    Returns:
        List of potential configuration directories
    """
    candidates = [
        Path.cwd() / "configs",
        Path.cwd() / "config",
        Path.cwd() / "examples",
        Path.home() / ".metareason",
        Path.home() / ".config" / "metareason",
    ]

    # Add system directories based on OS
    import os

    if os.name == "posix":
        candidates.extend(
            [
                Path("/etc/metareason"),
                Path("/usr/local/etc/metareason"),
            ]
        )
    elif os.name == "nt":
        candidates.extend(
            [
                Path("C:/ProgramData/metareason"),
                Path(os.environ.get("APPDATA", "")) / "metareason",
            ]
        )

    # Filter to existing directories
    return [d for d in candidates if d.exists() and d.is_dir()]


def suggest_config_location(config_name: str) -> Optional[Path]:
    """Suggest where to find a configuration file.

    Args:
        config_name: Name of configuration file

    Returns:
        Suggested path if found, None otherwise
    """
    directories = discover_config_directories()

    for directory in directories:
        # Try with .yaml extension
        candidate = directory / f"{config_name}.yaml"
        if candidate.exists():
            return candidate

        # Try with .yml extension
        candidate = directory / f"{config_name}.yml"
        if candidate.exists():
            return candidate

        # Try exact name
        candidate = directory / config_name
        if candidate.exists():
            return candidate

    return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to specified length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def create_progress_callback(console, total: int, description: str = "Processing"):
    """Create a progress callback for long-running operations.

    Args:
        console: Rich console instance
        total: Total number of items to process
        description: Description for progress bar

    Returns:
        Progress callback function
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )

    task = progress.add_task(description, total=total)

    def callback(completed: int = 1):
        """Update progress by specified amount."""
        progress.update(task, advance=completed)

    return progress, callback
