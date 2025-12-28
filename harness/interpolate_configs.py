#!/usr/bin/env python3
"""
Run CUDA benchmarks with multiple configurations and plot results in real-time.

This script reads configuration JSON lines from an input file, runs the
benchmark for each config, and outputs combined results as JSON lines.
It displays a live plot of results using plotext.
"""

import json
import sys
import argparse
import re
import shutil
from pathlib import Path
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import plotext as plt

from benchmark_lib import (
    run_benchmark,
    load_existing_results,
    format_config_summary
)


def get_value_from_path(data, path):
    """
    Extract a value from nested dictionary using dot notation.

    Examples:
        get_value_from_path(data, "vector_size") -> data["config"]["runtime"]["vector_size"]
        get_value_from_path(data, "throughput_gib_s") -> data["gpu_perf"]["throughput_gib_s"]
    """
    # Try direct lookup first
    if path in data:
        return data[path]

    # Try in nested structures (e.g., "gpu_perf.throughput_gib_s")
    parts = path.split('.')
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def create_plot(results, x_axis, y_axis, width=80, height=20, x_log=False, y_log=False):
    """
    Create a plotext plot from results.

    Args:
        results: List of result dictionaries
        x_axis: Key to use for x-axis (e.g., "vector_size")
        y_axis: Key to use for y-axis (e.g., "throughput_gib_s")
        width: Plot width in characters
        height: Plot height in characters
        x_log: Use logarithmic scale for x-axis
        y_log: Use logarithmic scale for y-axis

    Returns:
        String representation of the plot
    """
    if not results:
        return "No data to plot yet"

    # Extract x and y values
    points = []
    for result in results:
        # Skip error results
        if 'error' in result:
            continue

        x_val = get_value_from_path(result, x_axis)
        y_val = get_value_from_path(result, y_axis)

        if x_val is not None and y_val is not None:
            points.append((x_val, y_val))

    if not points:
        return f"No valid data points for x={x_axis}, y={y_axis}"

    # Sort by x value
    points.sort(key=lambda p: p[0])

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    # Create plot
    plt.clf()
    plt.plotsize(width, height)
    plt.theme('clear')  # Use plain text theme without colors

    # Set logarithmic scales if requested
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    plt.plot(x_vals, y_vals, marker="braille")

    # Add axis labels with log notation if applicable
    x_label = f"{x_axis} (log)" if x_log else x_axis
    y_label = f"{y_axis} (log)" if y_log else y_axis

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")

    # Build plot and strip ANSI escape codes
    plot_str = plt.build()
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', plot_str)


def main():
    """
    Read configs from stdin, run benchmarks, output results to file with live plotting.

    Usage:
        python3 interpolate_configs.py -o results.jsonl -x vector_size -y throughput_gib_s < configs.jsonl
        python3 generate_configs.py | python3 interpolate_configs.py -o results.jsonl -x vector_size -y throughput_gib_s
    """
    parser = argparse.ArgumentParser(description='Run CUDA benchmarks with real-time plotting')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output file for results (JSONL format)')
    parser.add_argument('-x', '--x-axis', required=True,
                        help='Config/result field for x-axis (e.g., "vector_size")')
    parser.add_argument('-y', '--y-axis', required=True,
                        help='Result field for y-axis (e.g., "throughput_gib_s" or "gpu_perf.throughput_gib_s")')
    parser.add_argument('--x-log', action='store_true',
                        help='Use logarithmic scale for x-axis')
    parser.add_argument('--y-log', action='store_true',
                        help='Use logarithmic scale for y-axis')
    parser.add_argument('--plot-width', type=int, default=None,
                        help='Plot width in characters (default: auto-detect terminal width)')
    parser.add_argument('--plot-height', type=int, default=None,
                        help='Plot height in characters (default: auto-detect terminal height)')
    args = parser.parse_args()

    # Auto-detect terminal size if not specified
    terminal_size = shutil.get_terminal_size(fallback=(80, 24))
    if args.plot_width is None:
        # Use full terminal width minus some padding for borders
        args.plot_width = max(40, terminal_size.columns - 4)
    if args.plot_height is None:
        args.plot_height = max(15, int(terminal_size.lines) - 4)

    # Determine build directory (parent of harness directory)
    build_dir = Path(__file__).parent.parent

    # Load existing results to avoid re-running
    existing_results = load_existing_results(args.output)

    # Convert to list for plotting
    all_results = list(existing_results.values())

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

    # Create layout for plot and progress
    layout = Layout()
    layout.split_column(
        Layout(name="plot", size=args.plot_height + 4),
        Layout(name="progress", size=10)
    )

    # Initial plot with existing results
    plot_text = create_plot(all_results, args.x_axis, args.y_axis, args.plot_width, args.plot_height,
                           args.x_log, args.y_log)

    # Open output file in append mode
    with open(args.output, 'a') as outfile:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            refresh_per_second=1,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmarks...", total=total_configs)
            # Update layout
            layout["plot"].update(Panel(plot_text, title=f"Results: {args.y_axis} vs {args.x_axis}"))
            layout["progress"].update(progress)

            with Live(layout, console=console, refresh_per_second=1, screen=False):
                for config in configs:
                    # Check if this config already exists
                    config_key = json.dumps(config, sort_keys=True)

                    if config_key in existing_results:
                        skipped += 1
                        progress.update(task, advance=1)
                        continue

                    # Update progress with current config
                    run += 1
                    current_status = f"[{run}/{total_configs-skipped}] {format_config_summary(config)}"
                    progress.update(task, description=current_status)

                    # Run benchmark
                    result = run_benchmark(config, build_dir=build_dir)

                    # Check for errors
                    if 'error' in result:
                        errors += 1

                    # Add to results list
                    all_results.append(result)

                    # Write result to file
                    outfile.write(json.dumps(result) + '\n')
                    outfile.flush()

                    # Update plot
                    plot_text = create_plot(all_results, args.x_axis, args.y_axis, args.plot_width, args.plot_height,
                                          args.x_log, args.y_log)
                    layout["plot"].update(Panel(plot_text, title=f"Results: {args.y_axis} vs {args.x_axis}"))

                    progress.update(task, advance=1)

    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total configs: {total_configs}")
    console.print(f"  [green]Executed: {run}[/green]")
    console.print(f"  [yellow]Skipped (already done): {skipped}[/yellow]")
    console.print(f"  [red]Errors: {errors}[/red]")
    console.print(f"  Results written to: [cyan]{args.output}[/cyan]")

    # Show final plot
    console.print("\n[bold]Final Plot:[/bold]")
    plot_text = create_plot(all_results, args.x_axis, args.y_axis, args.plot_width, args.plot_height,
                           args.x_log, args.y_log)
    console.print(plot_text)


if __name__ == "__main__":
    main()
