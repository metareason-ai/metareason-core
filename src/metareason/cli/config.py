"""Configuration-related CLI commands for MetaReason."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..config import load_yaml_config, validate_yaml_file, validate_yaml_directory
from .utils import find_config_files, format_validation_report, compare_configurations

console = Console()


@click.group(name="config")
def config_group() -> None:
    """Configuration management commands."""
    pass


@config_group.command()
@click.argument(
    "files", 
    nargs=-1, 
    type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--directory", "-d",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Validate all YAML files in directory"
)
@click.option(
    "--strict", 
    is_flag=True,
    help="Enable strict validation (warnings become errors)"
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json", "junit"]),
    default="text",
    help="Output format"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file for results"
)
@click.pass_context
def validate(
    ctx: click.Context, 
    files: tuple, 
    directory: Optional[Path], 
    strict: bool,
    output_format: str,
    output: Optional[Path]
) -> None:
    """Validate MetaReason configuration files.
    
    Validates YAML configuration files against the MetaReason schema,
    checking for syntax errors, missing required fields, and best practices.
    
    Examples:
      metareason config validate config.yaml
      metareason config validate -d ./configs --strict
      metareason config validate *.yaml --format json -o results.json
    """
    config_files = []
    
    # Collect files to validate
    if directory:
        try:
            config_files.extend(find_config_files(directory))
        except ValueError as e:
            console.print(f"❌ [red]Error:[/red] {e}")
            sys.exit(1)
    
    if files:
        config_files.extend(Path(f) for f in files)
    
    if not config_files:
        console.print("❌ [red]Error:[/red] No configuration files specified")
        console.print("Use --directory to validate a directory or specify individual files")
        sys.exit(1)
    
    # Validate files
    results = {}
    total_files = len(config_files)
    valid_files = 0
    
    with console.status(f"[bold blue]Validating {total_files} configuration files..."):
        for config_file in config_files:
            try:
                _, report = validate_yaml_file(config_file, strict=strict)
                results[str(config_file)] = report
                if report.is_valid:
                    valid_files += 1
            except Exception as e:
                console.print(f"❌ [red]Fatal error validating {config_file}:[/red] {e}")
                results[str(config_file)] = None
    
    # Output results
    if output_format == "json":
        json_results = {}
        for file_path, report in results.items():
            if report:
                json_results[file_path] = {
                    "valid": report.is_valid,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "suggestions": report.suggestions
                }
            else:
                json_results[file_path] = {"valid": False, "fatal_error": True}
        
        output_data = json.dumps(json_results, indent=2)
        
    elif output_format == "junit":
        # Generate JUnit XML format for CI/CD integration
        output_data = _generate_junit_xml(results)
        
    else:
        # Text format
        output_lines = []
        
        # Summary
        if valid_files == total_files:
            output_lines.append("✅ [bold green]All configuration files are valid![/bold green]")
        else:
            invalid_count = total_files - valid_files
            output_lines.append(f"❌ [bold red]{invalid_count} of {total_files} configuration files have issues[/bold red]")
        
        output_lines.append("")
        
        # Individual file results
        for file_path, report in results.items():
            if report:
                output_lines.append(format_validation_report(report, file_path))
            else:
                output_lines.append(f"💥 [red]{file_path}: Fatal error during validation[/red]")
            output_lines.append("")
        
        output_data = "\n".join(output_lines)
    
    # Write to file or console
    if output:
        output.write_text(output_data)
        console.print(f"💾 Results written to {output}")
    else:
        if output_format == "json":
            console.print_json(output_data)
        else:
            console.print(output_data)
    
    # Exit with error code if any files are invalid
    if valid_files < total_files:
        sys.exit(1)


@config_group.command()
@click.argument("file", type=click.Path(path_type=Path))
@click.option(
    "--format", "output_format",
    type=click.Choice(["yaml", "json", "toml"]),
    default="yaml",
    help="Output format"
)
@click.option(
    "--expand-includes", 
    is_flag=True,
    help="Expand !include statements inline"
)
@click.option(
    "--expand-env", 
    is_flag=True,
    help="Expand environment variables"
)
@click.option(
    "--show-defaults", 
    is_flag=True,
    help="Show default values for optional fields"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file"
)
def show(
    file: Path, 
    output_format: str, 
    expand_includes: bool,
    expand_env: bool,
    show_defaults: bool,
    output: Optional[Path]
) -> None:
    """Pretty-print a configuration file.
    
    Displays a MetaReason configuration file in a formatted, readable way.
    Can expand includes and environment variables for debugging.
    
    Examples:
      metareason config show config.yaml
      metareason config show config.yaml --format json
      metareason config show config.yaml --expand-includes --expand-env
    """
    try:
        # Load configuration with appropriate options
        config = load_yaml_config(
            file,
            enable_includes=expand_includes,
            enable_env_substitution=expand_env
        )
        
        # Convert to dictionary representation
        if show_defaults:
            config_dict = config.model_dump()
        else:
            config_dict = config.model_dump(exclude_unset=True)
        
        # Format output
        if output_format == "yaml":
            formatted_output = yaml.dump(
                config_dict, 
                default_flow_style=False, 
                sort_keys=False,
                indent=2
            )
            syntax = Syntax(formatted_output, "yaml", theme="monokai", line_numbers=True)
            
        elif output_format == "json":
            formatted_output = json.dumps(config_dict, indent=2, default=str)
            syntax = Syntax(formatted_output, "json", theme="monokai", line_numbers=True)
            
        elif output_format == "toml":
            try:
                import toml
                formatted_output = toml.dumps(config_dict)
                syntax = Syntax(formatted_output, "toml", theme="monokai", line_numbers=True)
            except ImportError:
                console.print("❌ [red]Error:[/red] TOML support requires 'toml' package")
                console.print("Install with: pip install toml")
                sys.exit(1)
        
        # Output
        if output:
            output.write_text(formatted_output)
            console.print(f"💾 Configuration written to {output}")
        else:
            panel = Panel(
                syntax,
                title=f"📄 {file.name}",
                title_align="left",
                border_style="blue"
            )
            console.print(panel)
    
    except Exception as e:
        console.print(f"❌ [red]Error loading configuration:[/red] {e}")
        sys.exit(1)


@config_group.command()
@click.argument("file1", type=click.Path(exists=True, path_type=Path))
@click.argument("file2", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json", "unified"]),
    default="text",
    help="Output format"
)
@click.option(
    "--ignore-fields",
    multiple=True,
    help="Fields to ignore in comparison (can be used multiple times)"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file for diff results"
)
def diff(
    file1: Path, 
    file2: Path, 
    output_format: str,
    ignore_fields: tuple,
    output: Optional[Path]
) -> None:
    """Compare two configuration files.
    
    Shows differences between two MetaReason configuration files,
    highlighting changes in structure, values, and metadata.
    
    Examples:
      metareason config diff old.yaml new.yaml
      metareason config diff config1.yaml config2.yaml --format json
      metareason config diff base.yaml modified.yaml --ignore-fields metadata.created_date
    """
    try:
        # Load both configurations
        config1 = load_yaml_config(file1)
        config2 = load_yaml_config(file2)
        
        # Compare configurations
        diff_result = compare_configurations(
            config1, config2, 
            ignore_fields=list(ignore_fields),
            file1_name=str(file1),
            file2_name=str(file2)
        )
        
        # Format output
        if output_format == "json":
            output_data = json.dumps(diff_result, indent=2, default=str)
        elif output_format == "unified":
            output_data = _format_unified_diff(diff_result)
        else:
            output_data = _format_text_diff(diff_result)
        
        # Output results
        if output:
            output.write_text(output_data)
            console.print(f"💾 Diff results written to {output}")
        else:
            if output_format == "json":
                console.print_json(output_data)
            else:
                console.print(output_data)
        
        # Exit with code 1 if differences found
        if diff_result.get("has_differences", False):
            sys.exit(1)
    
    except Exception as e:
        if output_format == "json":
            error_data = {
                "error": "Error comparing configurations",
                "message": str(e),
                "success": False
            }
            console.print_json(data=error_data)
        else:
            console.print(f"❌ [red]Error comparing configurations:[/red] {e}")
        sys.exit(1)


@config_group.command()
@click.option(
    "--clear", 
    is_flag=True,
    help="Clear the configuration cache"
)
@click.option(
    "--stats", 
    is_flag=True,
    help="Show cache statistics"
)
@click.option(
    "--disable", 
    is_flag=True,
    help="Disable configuration caching"
)
def cache(clear: bool, stats: bool, disable: bool) -> None:
    """Manage configuration cache.
    
    Controls the MetaReason configuration cache system for improved performance.
    """
    from ..config.cache import get_global_cache, clear_global_cache, disable_caching, is_caching_enabled
    
    if disable:
        disable_caching()
        console.print("🚫 Configuration caching disabled")
        return
    
    if not is_caching_enabled():
        console.print("ℹ️  Configuration caching is currently disabled")
        return
    
    cache_instance = get_global_cache()
    
    if clear:
        cache_instance.clear()
        console.print("🧹 Configuration cache cleared")
    
    if stats:
        stats_data = cache_instance.get_stats()
        
        table = Table(title="Configuration Cache Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Entries", str(stats_data["total_entries"]))
        table.add_row("Active Entries", str(stats_data["active_entries"]))
        table.add_row("Expired Entries", str(stats_data["expired_entries"]))
        table.add_row("TTL (seconds)", str(stats_data["ttl_seconds"]))
        table.add_row("Hot Reload", "✅" if stats_data["hot_reload_enabled"] else "❌")
        
        console.print(table)
    
    if not clear and not stats:
        # Default: show basic cache info
        stats_data = cache_instance.get_stats()
        console.print(f"💾 Cache: {stats_data['active_entries']} active entries")


def _generate_junit_xml(results: dict) -> str:
    """Generate JUnit XML format for test results."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    
    testsuites = Element("testsuites")
    testsuite = SubElement(testsuites, "testsuite", {
        "name": "metareason-config-validation",
        "tests": str(len(results)),
        "failures": str(sum(1 for r in results.values() if r and not r.is_valid)),
        "errors": str(sum(1 for r in results.values() if r is None))
    })
    
    for file_path, report in results.items():
        testcase = SubElement(testsuite, "testcase", {
            "name": Path(file_path).name,
            "classname": "config.validation"
        })
        
        if report is None:
            SubElement(testcase, "error", {"message": "Fatal validation error"})
        elif not report.is_valid:
            failure = SubElement(testcase, "failure", {"message": "Validation failed"})
            failure.text = "\n".join(f"{err['field']}: {err['message']}" for err in report.errors)
    
    return tostring(testsuites, encoding="unicode")


def _format_unified_diff(diff_result: dict) -> str:
    """Format diff result as unified diff."""
    lines = []
    lines.append(f"--- {diff_result.get('file1_name', 'file1')}")
    lines.append(f"+++ {diff_result.get('file2_name', 'file2')}")
    
    for change in diff_result.get("changes", []):
        if change["type"] == "added":
            lines.append(f"+ {change['path']}: {change['new_value']}")
        elif change["type"] == "removed":
            lines.append(f"- {change['path']}: {change['old_value']}")
        elif change["type"] == "modified":
            lines.append(f"- {change['path']}: {change['old_value']}")
            lines.append(f"+ {change['path']}: {change['new_value']}")
    
    return "\n".join(lines)


def _format_text_diff(diff_result: dict) -> str:
    """Format diff result as readable text."""
    lines = []
    
    if not diff_result.get("has_differences", False):
        lines.append("✅ [green]No differences found[/green]")
        return "\n".join(lines)
    
    lines.append(f"🔍 [bold]Comparing configurations:[/bold]")
    lines.append(f"   📄 {diff_result.get('file1_name', 'file1')}")
    lines.append(f"   📄 {diff_result.get('file2_name', 'file2')}")
    lines.append("")
    
    changes = diff_result.get("changes", [])
    
    # Group changes by type
    added = [c for c in changes if c["type"] == "added"]
    removed = [c for c in changes if c["type"] == "removed"]
    modified = [c for c in changes if c["type"] == "modified"]
    
    if added:
        lines.append("➕ [green]Added:[/green]")
        for change in added:
            lines.append(f"   {change['path']}: {change['new_value']}")
        lines.append("")
    
    if removed:
        lines.append("➖ [red]Removed:[/red]")
        for change in removed:
            lines.append(f"   {change['path']}: {change['old_value']}")
        lines.append("")
    
    if modified:
        lines.append("🔄 [yellow]Modified:[/yellow]")
        for change in modified:
            lines.append(f"   {change['path']}:")
            lines.append(f"     - {change['old_value']}")
            lines.append(f"     + {change['new_value']}")
        lines.append("")
    
    summary = diff_result.get("summary", {})
    lines.append(f"📊 [bold]Summary:[/bold] {summary.get('total_changes', 0)} changes")
    
    return "\n".join(lines)