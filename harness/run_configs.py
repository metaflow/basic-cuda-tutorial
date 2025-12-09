#!/usr/bin/env python3
"""
Run CUDA benchmarks with multiple configurations.

This script reads configuration JSON lines from an input file, runs the
benchmark for each config, and outputs combined results as JSON lines.
"""

import json
import subprocess
import sys
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


def run_benchmark(config, executable="./01-vector-addition", build_dir=None):
    """
    Run benchmark with the given configuration.

    Args:
        config: Dictionary with configuration parameters
        executable: Path to the executable to run
        build_dir: Directory containing the executable (default: parent of harness)

    Returns:
        Dictionary containing config and parsed output values
    """
    # Determine working directory
    if build_dir is None:
        # Default to parent directory of harness
        build_dir = Path(__file__).parent.parent

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Build the project (suppress output unless there's an error)
        make_result = subprocess.run(
            ['make', '01-vector-addition'],
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

        # Run the executable
        result = subprocess.run(
            [executable, config_file],
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


def main():
    """
    Read configs from stdin (or file), run benchmarks, output results to stdout.

    Usage:
        python3 run_configs.py < configs.jsonl > results.jsonl
        python3 generate_configs.py | python3 run_configs.py > results.jsonl
    """

    # Determine build directory (parent of harness directory)
    build_dir = Path(__file__).parent.parent

    # Read configs from stdin
    for line_num, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue

        try:
            config = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({
                "error": "invalid_config",
                "line_num": line_num,
                "message": str(e)
            }), file=sys.stderr)
            continue

        # Run benchmark
        result = run_benchmark(config, build_dir=build_dir)

        # Output result as single JSON line
        print(json.dumps(result))
        sys.stdout.flush()  # Ensure output is written immediately


if __name__ == "__main__":
    main()
