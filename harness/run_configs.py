#!/usr/bin/env python3
"""
Run CUDA benchmarks with multiple configurations.

This script reads configuration JSON lines from an input file, runs the
benchmark for each config, and outputs combined results as JSON lines.
"""

import json
import sys
import argparse
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from benchmark_lib import (
    run_benchmark,
    load_existing_results,
    format_config_summary
)


def main():
    """
    Read configs from stdin, run benchmarks, output results to file.

    Usage:
        python3 run_configs.py -o results.jsonl < configs.jsonl
        python3 generate_configs.py | python3 run_configs.py -o results.jsonl
    """
    parser = argparse.ArgumentParser(description='Run CUDA benchmarks with configurations')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output file for results (JSONL format)')
    args = parser.parse_args()

    # Determine build directory (parent of harness directory)
    build_dir = Path(__file__).parent.parent

    # Load existing results to avoid re-running
    existing_results = load_existing_results(args.output)

    console = Console()

    # First pass: count total configs
    configs = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            config = json.loads(line)
            configs.append(config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing config:[/red] {e}", file=sys.stderr)
            continue

    total_configs = len(configs)
    skipped = 0
    run = 0
    errors = 0

    # Open output file in append mode
    with open(args.output, 'a') as outfile:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Running benchmarks...", total=total_configs)

            for config in configs:
                # Check if this config already exists
                config_key = json.dumps(config, sort_keys=True)

                if config_key in existing_results:
                    skipped += 1
                    progress.update(task, advance=1,
                                    description=f"[yellow]Skipped:[/yellow] {format_config_summary(config)}")
                    continue

                # Update progress with current config
                run += 1
                progress.update(task, description=f"[green]Running:[/green] {format_config_summary(config)}")

                # Run benchmark
                result = run_benchmark(config, build_dir=build_dir)

                # Check for errors
                if 'error' in result:
                    errors += 1
                    progress.update(task, description=f"[red]Error:[/red] {format_config_summary(config)}")

                # Write result to file
                outfile.write(json.dumps(result) + '\n')
                outfile.flush()

                progress.update(task, advance=1)

    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total configs: {total_configs}")
    console.print(f"  [green]Executed: {run}[/green]")
    console.print(f"  [yellow]Skipped (already done): {skipped}[/yellow]")
    console.print(f"  [red]Errors: {errors}[/red]")
    console.print(f"  Results written to: [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    main()
