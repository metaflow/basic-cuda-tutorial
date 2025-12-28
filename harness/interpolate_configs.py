#!/usr/bin/env python3
"""
Run CUDA benchmarks with multiple configurations and plot results in real-time.

This script reads configuration JSON lines from an input file, runs the
benchmark for each config, and outputs combined results as JSON lines.
It displays a live plot of results using plotext.

After running all initial configs, it automatically generates new configs
at "interesting points" (slope changes and biggest steps) and continues
running until max-results is reached or user cancels.
"""

import json
import sys
import argparse
import re
import shutil
import copy
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
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


def set_value_in_path(data, path, value):
    """
    Set a value in nested dictionary using dot notation.

    Examples:
        set_value_in_path(config, "config.runtime.vector_size", 1024)
    """
    parts = path.split('.')
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def generate_midpoint_config(result1, result2, x_path, log_x=True):
    """
    Generate a single midpoint config between two result dicts.

    Args:
        result1: First result dict (must have 'config' field and x_path value)
        result2: Second result dict (must have 'config' field and x_path value)
        x_path: Path to the parameter to interpolate (e.g., "config.runtime.vector_size")
        log_x: Whether to use log spacing (True) or linear spacing (False)

    Returns:
        A new config dict with the interpolated x value
    """
    x1 = get_value_from_path(result1, x_path)
    x2 = get_value_from_path(result2, x_path)

    if x1 is None or x2 is None:
        return None

    # Calculate midpoint based on spacing mode
    if log_x:
        midpoint_x = int(np.exp((np.log(x1) + np.log(x2)) / 2))
    else:
        midpoint_x = int((x1 + x2) / 2)

    # Use first result's config as template
    new_config = copy.deepcopy(result1['config'])

    # Set the new value in the config
    # x_path is typically "config.runtime.vector_size", but we only want "runtime.vector_size" in config
    if x_path.startswith('config.'):
        config_path = x_path[7:]  # Remove "config." prefix
    else:
        config_path = x_path

    set_value_in_path(new_config, config_path, midpoint_x)

    return new_config


def find_slope_changes(results, x_path, y_path, log_x=True):
    """
    Find points with biggest changes in slope (second derivative).
    Weights slope changes by the interval length to prioritize changes over larger ranges.

    Args:
        results: List of result dicts (must be sorted by x_path)
        x_path: Path for x-axis (e.g., "config.runtime.vector_size")
        y_path: Path for y-axis (e.g., "gpu_perf.throughput_gib_s")
        log_x: Whether to use log scale for x-axis

    Returns:
        List of dicts with slope change info and generated interpolation configs
    """
    if len(results) < 3:
        return []

    # Extract x and y values
    x_orig = np.array([get_value_from_path(r, x_path) for r in results])
    y = np.array([get_value_from_path(r, y_path) for r in results])

    # Apply log to x for slope calculation if requested
    x = np.log(x_orig) if log_x else x_orig

    # Calculate first derivative (slope between consecutive points)
    slopes = np.diff(y) / np.diff(x)

    # Calculate second derivative (change in slope)
    slope_changes = np.abs(np.diff(slopes))

    # Calculate interval lengths (in the appropriate scale)
    if log_x:
        interval_lengths = np.log(x_orig[2:] / x_orig[:-2])
    else:
        interval_lengths = x_orig[2:] - x_orig[:-2]

    # Weight slope changes by interval length
    weighted_slope_changes = slope_changes * interval_lengths

    # Find indices sorted by weighted slope change magnitude
    sorted_indices = np.argsort(weighted_slope_changes)[::-1]

    output = []
    for idx in sorted_indices:
        middle_idx = idx + 1

        result_before = results[middle_idx - 1]
        result_middle = results[middle_idx]
        result_after = results[middle_idx + 1]

        # Generate interpolation configs (two midpoints)
        interp_configs = [
            generate_midpoint_config(result_before, result_middle, x_path, log_x),
            generate_midpoint_config(result_middle, result_after, x_path, log_x)
        ]

        # Filter out None configs
        interp_configs = [c for c in interp_configs if c is not None]

        if interp_configs:
            output.append({
                'middle_index': middle_idx,
                'weighted_slope_change': weighted_slope_changes[idx],
                'interpolation_configs': interp_configs
            })

    return output


def find_biggest_step(results, x_path, y_path, log_x=True):
    """
    Find the pair of consecutive points with the biggest absolute step in y value.

    Args:
        results: List of result dicts (must be sorted by x_path)
        x_path: Path for x-axis (e.g., "config.runtime.vector_size")
        y_path: Path for y-axis (e.g., "gpu_perf.throughput_gib_s")
        log_x: Whether to use log spacing for interpolation

    Returns:
        List of dicts with step info and generated interpolation config
    """
    if len(results) < 2:
        return []

    y = np.array([get_value_from_path(r, y_path) for r in results])

    # Calculate absolute step between consecutive points
    steps = np.abs(np.diff(y))

    # Find indices sorted by step magnitude
    sorted_indices = np.argsort(steps)[::-1]

    output = []
    for idx in sorted_indices:
        result1 = results[idx]
        result2 = results[idx + 1]

        # Generate single interpolation config
        interp_config = generate_midpoint_config(result1, result2, x_path, log_x)

        if interp_config is not None:
            output.append({
                'pair_index': idx,
                'step_magnitude': steps[idx],
                'interpolation_configs': [interp_config]
            })

    return output


def find_biggest_x_gap(results, x_path, log_x=True):
    """
    Find the pair of consecutive points with the biggest gap in x-axis values.
    This helps fill in sparse regions of the parameter space.

    Args:
        results: List of result dicts (must be sorted by x_path)
        x_path: Path for x-axis (e.g., "config.runtime.vector_size")
        log_x: Whether to use log spacing for gap calculation

    Returns:
        List of dicts with gap info and generated interpolation config
    """
    if len(results) < 2:
        return []

    x_orig = np.array([get_value_from_path(r, x_path) for r in results])

    # Calculate gap between consecutive points
    if log_x:
        # In log space, gap is the ratio between consecutive points
        gaps = np.log(x_orig[1:] / x_orig[:-1])
    else:
        # In linear space, gap is the difference
        gaps = x_orig[1:] - x_orig[:-1]

    # Find indices sorted by gap size
    sorted_indices = np.argsort(gaps)[::-1]

    output = []
    for idx in sorted_indices:
        result1 = results[idx]
        result2 = results[idx + 1]

        # Generate single interpolation config
        interp_config = generate_midpoint_config(result1, result2, x_path, log_x)

        if interp_config is not None:
            output.append({
                'pair_index': idx,
                'gap_size': gaps[idx],
                'x_values': (x_orig[idx], x_orig[idx + 1]),
                'interpolation_configs': [interp_config]
            })

    return output


def generate_interesting_configs(results, x_axis, y_axis, x_log=False):
    """
    Generate new configs at interesting points based on:
    1. Slope changes (2nd derivative) - 2 configs
    2. Biggest Y-step (1st derivative) - 1 config
    3. Biggest X-gap (sparse regions) - 1 config

    Args:
        results: List of result dicts
        x_axis: X-axis field (e.g., "config.runtime.vector_size")
        y_axis: Y-axis field (e.g., "gpu_perf.throughput_gib_s")
        x_log: Whether x-axis uses log scale

    Returns:
        List of up to 4 new configs
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATE_INTERESTING_CONFIGS: Starting analysis")
    logger.info(f"  Total results: {len(results)}")

    # Filter out error results and results without required fields
    valid_results = []
    for r in results:
        if 'error' in r:
            continue
        x_val = get_value_from_path(r, x_axis)
        y_val = get_value_from_path(r, y_axis)
        if x_val is not None and y_val is not None:
            valid_results.append(r)

    logger.info(f"  Valid results (no errors): {len(valid_results)}")

    if len(valid_results) < 2:
        logger.warning(f"  Not enough valid results ({len(valid_results)} < 2) - cannot generate configs")
        return []

    # Sort by x-axis
    valid_results.sort(key=lambda r: get_value_from_path(r, x_axis))

    # Log current x-values
    x_values = [get_value_from_path(r, x_axis) for r in valid_results]
    logger.info(f"  Current x-values: {x_values}")

    new_configs = []

    # Strategy 1: Get 2 configs from biggest slope changes (second derivative)
    if len(valid_results) >= 3:
        slope_changes = find_slope_changes(valid_results, x_axis, y_axis, log_x=x_log)
        logger.info(f"  Found {len(slope_changes)} slope change points")

        if slope_changes:
            # Take configs from the top slope change
            top_slope = slope_changes[0]
            logger.info(f"    Top slope change:")
            logger.info(f"      Middle index: {top_slope['middle_index']}")
            logger.info(f"      Weighted score: {top_slope['weighted_slope_change']:.4f}")
            logger.info(f"      Generated {len(top_slope['interpolation_configs'])} configs from slope change")

            for i, cfg in enumerate(top_slope['interpolation_configs']):
                x_val = get_value_from_path(cfg, x_axis.replace('config.', '') if x_axis.startswith('config.') else x_axis)
                logger.info(f"        Config {i+1}: {x_axis}={x_val}")

            new_configs.extend(top_slope['interpolation_configs'])
    else:
        logger.info(f"  Not enough valid results for slope changes ({len(valid_results)} < 3)")

    # Strategy 2: Get 1 config from biggest Y-step (first derivative)
    if len(valid_results) >= 2:
        biggest_steps = find_biggest_step(valid_results, x_axis, y_axis, log_x=x_log)
        logger.info(f"  Found {len(biggest_steps)} Y-step pairs")

        if biggest_steps:
            top_step = biggest_steps[0]
            logger.info(f"    Top Y-step:")
            logger.info(f"      Pair index: {top_step['pair_index']}")
            logger.info(f"      Step magnitude: {top_step['step_magnitude']:.4f}")
            logger.info(f"      Generated {len(top_step['interpolation_configs'])} configs from biggest Y-step")

            for i, cfg in enumerate(top_step['interpolation_configs']):
                x_val = get_value_from_path(cfg, x_axis.replace('config.', '') if x_axis.startswith('config.') else x_axis)
                logger.info(f"        Config {i+1}: {x_axis}={x_val}")

            new_configs.extend(top_step['interpolation_configs'])

    # Strategy 3: Get 1 config from biggest X-gap (fill sparse regions)
    if len(valid_results) >= 2:
        biggest_gaps = find_biggest_x_gap(valid_results, x_axis, log_x=x_log)
        logger.info(f"  Found {len(biggest_gaps)} X-gap pairs")

        if biggest_gaps:
            top_gap = biggest_gaps[0]
            logger.info(f"    Top X-gap:")
            logger.info(f"      Pair index: {top_gap['pair_index']}")
            logger.info(f"      Gap size: {top_gap['gap_size']:.4f} ({'log' if x_log else 'linear'} scale)")
            logger.info(f"      X range: {top_gap['x_values'][0]} to {top_gap['x_values'][1]}")
            logger.info(f"      Generated {len(top_gap['interpolation_configs'])} configs from biggest X-gap")

            for i, cfg in enumerate(top_gap['interpolation_configs']):
                x_val = get_value_from_path(cfg, x_axis.replace('config.', '') if x_axis.startswith('config.') else x_axis)
                logger.info(f"        Config {i+1}: {x_axis}={x_val}")

            new_configs.extend(top_gap['interpolation_configs'])

    logger.info(f"  TOTAL generated configs: {len(new_configs)}")
    logger.info(f"{'='*80}\n")

    return new_configs


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


def create_markdown_table(results, x_axis, y_axis):
    """
    Create a markdown table from results, sorted by x-axis.

    Args:
        results: List of result dictionaries
        x_axis: Key to use for x-axis (e.g., "config.runtime.vector_size")
        y_axis: Key to use for y-axis (e.g., "gpu_perf.throughput_gib_s")

    Returns:
        String containing markdown table
    """
    # Extract valid points
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
        return f"No valid data points for x={x_axis}, y={y_axis}\n"

    # Sort by x value
    points.sort(key=lambda p: p[0])

    # Build markdown table
    lines = []
    lines.append(f"# Results: {y_axis} vs {x_axis}\n")
    lines.append("")
    lines.append(f"| {x_axis} | {y_axis} |")
    lines.append(f"|{'-' * (len(x_axis) + 2)}|{'-' * (len(y_axis) + 2)}|")

    for x_val, y_val in points:
        # Format numbers appropriately
        if isinstance(x_val, float):
            x_str = f"{x_val:.6g}"
        else:
            x_str = str(x_val)

        if isinstance(y_val, float):
            y_str = f"{y_val:.6g}"
        else:
            y_str = str(y_val)

        lines.append(f"| {x_str} | {y_str} |")

    return "\n".join(lines) + "\n"


def main():
    """
    Read configs from stdin, run benchmarks, output results to file with live plotting.

    After running all initial configs, automatically generates new configs at interesting
    points (slope changes and biggest steps) and continues until max-results is reached.

    Usage:
        python3 interpolate_configs.py -o results.jsonl -x config.runtime.vector_size -y gpu_perf.throughput_gib_s < configs.jsonl
        python3 generate_configs.py | python3 interpolate_configs.py -o results.jsonl -x config.runtime.vector_size -y gpu_perf.throughput_gib_s
    """
    parser = argparse.ArgumentParser(description='Run CUDA benchmarks with real-time plotting')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output file for results (JSONL format)')
    parser.add_argument('-omd', '--output-markdown', type=Path,
                        help='Optional output file for markdown table of results')
    parser.add_argument('-x', '--x-axis', required=True,
                        help='Config/result field for x-axis (e.g., "config.runtime.vector_size")')
    parser.add_argument('-y', '--y-axis', required=True,
                        help='Result field for y-axis (e.g., "gpu_perf.throughput_gib_s")')
    parser.add_argument('--x-log', action='store_true',
                        help='Use logarithmic scale for x-axis')
    parser.add_argument('--y-log', action='store_true',
                        help='Use logarithmic scale for y-axis')
    parser.add_argument('--max-results', type=int, default=100,
                        help='Maximum total results before stopping auto-generation (default: 100)')
    parser.add_argument('--plot-width', type=int, default=None,
                        help='Plot width in characters (default: auto-detect terminal width)')
    parser.add_argument('--plot-height', type=int, default=None,
                        help='Plot height in characters (default: auto-detect terminal height)')
    args = parser.parse_args()

    # Setup logging
    log_filename = f"interpolate_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("INTERPOLATE_CONFIGS: Starting new session")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  X-axis: {args.x_axis} (log={args.x_log})")
    logger.info(f"  Y-axis: {args.y_axis} (log={args.y_log})")
    logger.info(f"  Max results: {args.max_results}")
    logger.info(f"  Log file: {log_filename}")
    logger.info("="*80)

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
    logger.info(f"\nLoaded {len(existing_results)} existing results from {args.output}")

    # Convert to list for plotting
    all_results = list(existing_results.values())

    console = Console()

    # Read initial configs from stdin
    initial_configs = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            config = json.loads(line)
            initial_configs.append(config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing config:[/red] {e}", file=sys.stderr)
            logger.error(f"Error parsing config: {e}")
            continue

    logger.info(f"\nRead {len(initial_configs)} initial configs from stdin")

    # Queue of configs to run (start with initial configs)
    config_queue = initial_configs.copy()
    initial_count = len(initial_configs)

    skipped = 0
    run = 0
    errors = 0
    auto_generated = 0

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
            task = progress.add_task("[cyan]Running benchmarks...", total=None)
            # Update layout
            layout["plot"].update(Panel(plot_text, title=f"Results: {args.y_axis} vs {args.x_axis}"))
            layout["progress"].update(progress)

            with Live(layout, console=console, refresh_per_second=1, screen=False):
                iteration = 0

                while len(all_results) < args.max_results:
                    # Check if queue is empty and generate new configs if needed
                    if len(config_queue) == 0:
                        logger.info(f"\n[After {iteration} iterations] CONFIG QUEUE EMPTY - generating new configs")
                        logger.info(f"  Current results: {len(all_results)}/{args.max_results}")

                        # Generate interesting configs
                        new_configs = generate_interesting_configs(all_results, args.x_axis, args.y_axis, args.x_log)

                        # Filter out configs that already exist
                        unique_new_configs = []
                        duplicate_count = 0
                        for new_config in new_configs:
                            new_key = json.dumps(new_config, sort_keys=True)
                            if new_key not in existing_results:
                                unique_new_configs.append(new_config)
                            else:
                                duplicate_count += 1

                        logger.info(f"  Filtered duplicates: {duplicate_count}")
                        logger.info(f"  Unique new configs: {len(unique_new_configs)}")

                        if unique_new_configs:
                            auto_generated += len(unique_new_configs)
                            config_queue.extend(unique_new_configs)
                            logger.info(f"  Added {len(unique_new_configs)} configs to queue")
                        else:
                            logger.warning(f"  NO UNIQUE CONFIGS GENERATED - stopping")
                            break

                    iteration += 1
                    config = config_queue.pop(0)
                    config_key = json.dumps(config, sort_keys=True)
                    logger.info(f"processing config {config_key}")

                    if config_key in existing_results:
                        skipped += 1
                        logger.info(f"[Iteration {iteration}] SKIPPED - config already exists")
                        continue

                    run += 1
                    is_auto = run > (initial_count - skipped)
                    status_prefix = "[yellow]AUTO[/yellow]" if is_auto else "[cyan]INIT[/cyan]"
                    config_type = "AUTO" if is_auto else "INIT"

                    # Extract x value for display
                    x_val = get_value_from_path(config, args.x_axis.replace('config.', '') if args.x_axis.startswith('config.') else args.x_axis)

                    # Format x value for display
                    if isinstance(x_val, float):
                        x_display = f"{x_val:.6g}"
                    else:
                        x_display = str(x_val)

                    # Get short x-axis name (last component after dots)
                    x_name = args.x_axis.split('.')[-1]

                    current_status = f"{status_prefix} [{len(all_results)+1}/{args.max_results}] {x_name}={x_display}"
                    progress.update(task, description=current_status)

                    logger.info(f"[Iteration {iteration}] RUNNING {config_type} config")
                    logger.info(f"  {args.x_axis} = {x_val}")
                    logger.info(f"  Results so far: {len(all_results)}/{args.max_results}")
                    logger.info(f"  Queue size: {len(config_queue)}")

                    result = run_benchmark(config, build_dir=build_dir)

                    if 'error' in result:
                        errors += 1
                        logger.error(f"  BENCHMARK ERROR: {result.get('error')}")
                    else:
                        y_val = get_value_from_path(result, args.y_axis)
                        logger.info(f"  Result: {args.y_axis} = {y_val}")

                    all_results.append(result)
                    existing_results[config_key] = result

                    # Write result to file
                    outfile.write(json.dumps(result) + '\n')
                    outfile.flush()

                    # Update plot
                    plot_text = create_plot(all_results, args.x_axis, args.y_axis, args.plot_width, args.plot_height,
                                          args.x_log, args.y_log)
                    layout["plot"].update(Panel(plot_text, title=f"Results: {args.y_axis} vs {args.x_axis}"))

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info(f"  Initial configs: {initial_count}")
    logger.info(f"  Auto-generated configs: {auto_generated}")
    logger.info(f"  Executed: {run}")
    logger.info(f"  Skipped (already done): {skipped}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Total results: {len(all_results)}")
    logger.info(f"  Results written to: {args.output}")
    logger.info("="*80)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Initial configs: {initial_count}")
    console.print(f"  [yellow]Auto-generated configs: {auto_generated}[/yellow]")
    console.print(f"  [green]Executed: {run}[/green]")
    console.print(f"  [yellow]Skipped (already done): {skipped}[/yellow]")
    console.print(f"  [red]Errors: {errors}[/red]")
    console.print(f"  Total results: {len(all_results)}")
    console.print(f"  Results written to: [cyan]{args.output}[/cyan]")
    console.print(f"  [dim]Log written to: {log_filename}[/dim]")

    # Show final plot
    console.print("\n[bold]Final Plot:[/bold]")
    plot_text = create_plot(all_results, args.x_axis, args.y_axis, args.plot_width, args.plot_height,
                           args.x_log, args.y_log)
    console.print(plot_text)

    # Write markdown table if requested
    if args.output_markdown:
        markdown_table = create_markdown_table(all_results, args.x_axis, args.y_axis)
        with open(args.output_markdown, 'w') as md_file:
            md_file.write(markdown_table)
        console.print(f"\n  Markdown table written to: [cyan]{args.output_markdown}[/cyan]")
        logger.info(f"Markdown table written to: {args.output_markdown}")


if __name__ == "__main__":
    main()
