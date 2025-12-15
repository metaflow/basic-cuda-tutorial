# CUDA Benchmark Harness

Python scripts for automated benchmarking of CUDA programs with multiple configurations.

## Setup

Activate the Python virtual environment before running scripts:
```bash
source uvenv-cuda/bin/activate
```

## Scripts

### `generate_configs_01.py`

Generates configuration JSON lines for benchmarking the 01-vector-addition program.

**Usage:**
```bash
python3 harness/generate_configs_01.py | tee configs.jsonl
```

**Output Format:**
Each line is a hierarchical JSON object with:
- `executable`: Object containing `make_target` (the binary to build and run)
- `runtime`: Object with runtime configuration parameters

**Example Output:**
```json
{"executable": {"make_target": "01-vector-addition"}, "runtime": {"vector_size": 1024, "validate": true, "profile_gpu": true, "profile_cpu": true, "threads_per_block": 256, "timeout_s": 10, "experiment": {"name": "vector_size_sweep", "threads": 256}}}
{"executable": {"make_target": "01-vector-addition"}, "runtime": {"vector_size": 2048, "validate": true, "profile_gpu": true, "profile_cpu": true, "threads_per_block": 256, "timeout_s": 10, "experiment": {"name": "vector_size_sweep", "threads": 256}}}
```

### `run_configs.py`

Runs benchmarks for each configuration with progress display and duplicate detection.

**Features:**
- Rich progress bar with real-time status updates
- Automatic skipping of already-completed configs
- Colored output (green=running, yellow=skipped, red=error)
- Human-readable config summaries
- Detailed summary statistics

**Usage:**
```bash
# Run from config file
python3 harness/run_configs.py -o results.jsonl < configs.jsonl

# Pipeline from generation
python3 harness/generate_configs_01.py | python3 harness/run_configs.py -o results.jsonl

# Run subset of configs
python3 harness/generate_configs_01.py | head -5 | python3 harness/run_configs.py -o results.jsonl

# Re-running will skip already completed configs
python3 harness/generate_configs_01.py | python3 harness/run_configs.py -o results.jsonl
```

**Input Format:**
JSON lines with hierarchical configuration objects:
- `executable`: Object with `make_target` field
- `runtime`: Object with runtime parameters

**Output Format:**
Each line in the output file is a JSON object containing:
- `config`: The full hierarchical input configuration (with `executable` and `runtime`)
- Parsed output fields matching pattern `'<key>':<json>` from program output:
  - `valid`: Validation result (bool)
  - `gpu_perf`: GPU performance metrics (object with `time_us`, `iterations`, `throughput_gib_s`)
  - `cpu_perf`: CPU performance metrics (object with `time_us`, `iterations`, `throughput_gib_s`)
- `error`: Error message if execution failed

**Example Output:**
```json
{"config": {"executable": {"make_target": "01-vector-addition"}, "runtime": {"vector_size": 1024, "validate": true, "profile_gpu": true, "profile_cpu": true, "threads_per_block": 256, "timeout_s": 10, "experiment": {"name": "vector_size_sweep", "threads": 256}}}, "valid": true, "gpu_perf": {"time_us": 2, "iterations": 13, "throughput_gib_s": 5.722}, "cpu_perf": {"time_us": 0, "iterations": 12, "throughput_gib_s": null}}
```

## Examples

### Quick Test
```bash
# Test with 3 configurations
python3 harness/generate_configs_01.py | head -3 | python3 harness/run_configs.py -o test.jsonl
```

### Full Benchmark Suite
```bash
# Run all configurations and save results
python3 harness/generate_configs_01.py | python3 harness/run_configs.py -o 01-vector-addition-perf.jsonl

# Re-running the same command will skip already completed configs
python3 harness/generate_configs_01.py | python3 harness/run_configs.py -o 01-vector-addition-perf.jsonl

# Count successful runs
grep -c '"valid":true' 01-vector-addition-perf.jsonl

# Extract GPU throughput values
grep 'gpu_perf' 01-vector-addition-perf.jsonl | python3 -c '
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    if "gpu_perf" in data:
        print(f"Size: {data[\"config\"][\"runtime\"][\"vector_size\"]:>10}, GPU: {data[\"gpu_perf\"][\"throughput_gib_s\"]:>8.2f} GiB/s")
'
```

### Analysis
```bash
# Extract specific metrics with jq
cat 01-vector-addition-perf.jsonl | jq -c '{size: .config.runtime.vector_size, threads: .config.runtime.threads_per_block, gpu_throughput: .gpu_perf.throughput_gib_s, cpu_throughput: .cpu_perf.throughput_gib_s}'

# Filter by experiment type
cat 01-vector-addition-perf.jsonl | jq 'select(.config.runtime.experiment.name == "vector_size_sweep")'
```

## Notes

- The harness automatically builds the project using the `make_target` before running benchmarks
- Each configuration runs in isolation with a temporary config file containing only the runtime parameters
- Already-completed configs are automatically skipped (based on exact config match)
- Results are written immediately (line-buffered) and appended to the output file
- Timeouts are set to 120 seconds per configuration
- Build errors and execution failures are captured in the output JSON with an `error` field
- Progress display requires the `rich` Python library (installed in the uv environment)
