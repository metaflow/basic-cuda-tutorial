# Claude Development Notes

## General notes

Do not do "additonal" work that user have not asked, such as adding extra features.

## Python Environment

This project uses a uv virtual environment. Before running Python scripts:
```bash
source uvenv-cuda/bin/activate
```

## Repository Overview

This is a CUDA tutorial collection with examples covering GPU programming fundamentals, PTX assembly, memory optimization, neural networks, profiling, and low-latency techniques for ML/AI workloads.

## JSON Output Format

Programs use a structured JSON line output format: `prefix:<jsonline>`

Design Rationale: this format enables

- Easy parsing by test harnesses and automation tools
- Grep-friendly filtering (`grep "^gpu_perf:" output.log`)
- Human-readable debugging output
- Structured data extraction without complex parsing

### Configuration System

- Runtime config via JSON file (single command-line argument)
- All config fields are required
- Uses nlohmann/json library (header-only, in `external/nlohmann/`)
- Config struct in `lib/config.h`
