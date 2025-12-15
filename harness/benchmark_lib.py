#!/usr/bin/env python3
"""
Library functions for running CUDA benchmarks with configurations.

This module provides utilities for parsing benchmark output, running
benchmarks with JSON configurations, loading results, and formatting
config summaries.
"""

import json
import subprocess
import tempfile
import os
import re
from pathlib import Path


def parse_output_line(line):
    """
    Parse output lines matching the pattern "'<key>':<json>"

    Returns (key, value) tuple or None if line doesn't match pattern.
    """
    # Match pattern: 'key' (single-quoted), colon, then JSON
    match = re.match(r"^'(\w+)':(.+)$", line)
    if not match:
        return None

    key = match.group(1)
    value_str = match.group(2).strip()

    # Try to parse as JSON
    try:
        value = json.loads(value_str)
        return (key, value)
    except json.JSONDecodeError:
        # If it's not valid JSON, return as string
        return (key, value_str)


def run_benchmark(config, build_dir=None):
    """
    Run benchmark with the given configuration.

    Args:
        config: Dictionary with hierarchical structure:
                - executable: dict with "make_target" key
                - runtime: dict with runtime configuration parameters
        build_dir: Directory containing the executable (default: parent of harness)

    Returns:
        Dictionary containing config and parsed output values
    """
    # Determine working directory
    if build_dir is None:
        # Default to parent directory of harness
        build_dir = Path(__file__).parent.parent

    # Extract executable and runtime config
    executable_info = config.get("executable", {})
    make_target = executable_info.get("make_target")
    runtime_config = config.get("runtime", {})

    if not make_target:
        return {
            "config": config,
            "error": "missing_make_target",
            "message": "Config must include executable.make_target"
        }

    # Create temporary config file with runtime configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(runtime_config, f)
        config_file = f.name

    try:
        # Build the project (suppress output unless there's an error)
        make_result = subprocess.run(
            ['make', make_target],
            cwd=build_dir,
            capture_output=True,
            text=True
        )

        if make_result.returncode != 0:
            return {
                "config": config,
                "error": "build_failed",
                "stderr": make_result.stderr
            }

        # Run the executable (same name as make target)
        executable_path = f"./{make_target}"
        result = subprocess.run(
            [executable_path, config_file],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {
                "config": config,
                "error": "execution_failed",
                "returncode": result.returncode,
                "stderr": result.stderr
            }

        # Parse output
        output_data = {"config": config}

        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue

            parsed = parse_output_line(line)
            if parsed:
                key, value = parsed
                # Don't overwrite our hierarchical config with the binary's runtime config
                if key != 'config':
                    output_data[key] = value

        return output_data

    except subprocess.TimeoutExpired:
        return {
            "config": config,
            "error": "timeout"
        }
    except Exception as e:
        return {
            "config": config,
            "error": "exception",
            "message": str(e)
        }
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_file)
        except:
            pass


def load_existing_results(output_file):
    """Load existing results from output file into a set of config hashes."""
    if not output_file.exists():
        return {}

    existing = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                if 'config' in result:
                    # Use JSON string of config as key for comparison
                    config_key = json.dumps(result['config'], sort_keys=True)
                    existing[config_key] = result
            except json.JSONDecodeError:
                continue

    return existing


def format_config_summary(config):
    """Format a config into a human-readable summary."""
    runtime = config.get('runtime', {})
    executable = config.get('executable', {})

    parts = []
    parts.append(f"[cyan]{executable.get('make_target', 'unknown')}[/cyan]")

    if 'vector_size' in runtime:
        size = runtime['vector_size']
        if size >= 1024*1024:
            parts.append(f"size={size/(1024*1024):.1f}M")
        elif size >= 1024:
            parts.append(f"size={size/1024:.1f}K")
        else:
            parts.append(f"size={size}")

    if 'threads_per_block' in runtime:
        parts.append(f"threads={runtime['threads_per_block']}")

    if 'experiment' in runtime:
        exp_name = runtime['experiment'].get('name', '')
        if exp_name:
            parts.append(f"[yellow]{exp_name}[/yellow]")

    return " ".join(parts)
