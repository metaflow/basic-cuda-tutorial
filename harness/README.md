# CUDA Benchmark Harness

Python scripts for automated benchmarking of CUDA programs with multiple configurations.

## Scripts

### `generate_configs.py`

Generates configuration JSON lines for benchmarking.

**Usage:**
```bash
./generate_configs_01.py | tee configs.jsonl
```

**Output Format:**
Each line is a JSON object with fields: `vector_size`, `validate`, `profile`, `threads_per_block`

**Example Output:**
```json
{"vector_size": 1000, "validate": true, "profile": true, "threads_per_block": 256}
{"vector_size": 10000, "validate": true, "profile": true, "threads_per_block": 256}
```

### `run_configs.py`

Runs benchmarks for each configuration and outputs results.

**Usage:**
```bash
# Run from config file
python3 run_configs.py < configs.jsonl > results.jsonl

# Pipeline from generation
python3 generate_configs.py | python3 run_configs.py > results.jsonl

# Run subset of configs
python3 generate_configs.py | head -5 | python3 run_configs.py
```

**Input Format:**
JSON lines with configuration objects (same format as `generate_configs.py` output)

**Output Format:**
Each line is a JSON object containing:
- `config`: The input configuration
- Parsed output fields matching pattern `<key>:<json>` from program output:
  - `valid`: Validation result (bool)
  - `gpu_perf`: GPU performance metrics (object with `time_us`, `iterations`, `throughput_gib_s`)
  - `cpu_perf`: CPU performance metrics (object with `time_us`, `iterations`, `throughput_gib_s`)
  - `GPU`: Human-readable GPU stats (string)
  - `CPU`: Human-readable CPU stats (string)
- `error`: Error message if execution failed

**Example Output:**
```json
{"config": {"vector_size": 100000, "validate": true, "profile": true, "threads_per_block": 256}, "valid": true, "gpu_perf": {"time_us": 2, "iterations": 14, "throughput_gib_s": 558.794}, "cpu_perf": {"time_us": 10, "iterations": 12, "throughput_gib_s": 111.759}}
```

## Examples

### Quick Test
```bash
# Test with 3 configurations
python3 generate_configs.py | head -3 | python3 run_configs.py
```

### Full Benchmark Suite
```bash
# Run all configurations and save results
python3 generate_configs.py | python3 run_configs.py > results.jsonl

# Count successful runs
grep -c '"valid":true' results.jsonl

# Extract GPU throughput values
grep 'gpu_perf' results.jsonl | python3 -c '
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    if "gpu_perf" in data:
        print(f"Size: {data[\"config\"][\"vector_size\"]:>10}, GPU: {data[\"gpu_perf\"][\"throughput_gib_s\"]:>8.2f} GiB/s")
'
```

### Analysis
```bash
# Extract specific metrics with jq
cat results.jsonl | jq -c '{size: .config.vector_size, gpu_throughput: .gpu_perf.throughput_gib_s, cpu_throughput: .cpu_perf.throughput_gib_s}'
```

## Notes

- The harness automatically builds the project before running benchmarks
- Each configuration runs in isolation with a temporary config file
- Results are output immediately (line-buffered) for streaming
- Timeouts are set to 60 seconds per configuration
- Build errors and execution failures are captured in the output JSON
