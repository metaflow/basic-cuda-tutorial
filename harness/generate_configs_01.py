#!/usr/bin/env python3
"""
Generate configuration files for CUDA benchmark harness.

This script generates JSON configuration lines for testing different
parameter combinations (vector sizes, thread counts, etc.).
"""

import json
import sys


def generate_configs():
    """Generate a variety of configurations for benchmarking."""

    # Vector sizes to test (powers of 2 and some intermediate values)
    vector_sizes = [
        2**i for i in range(10, 31) # 2**30 is the limit on my machine.
    ]

    # Generate configs with different vector sizes (default threads)
    for size in vector_sizes:
        config = {
            "executable": {
                "make_target": "01-vector-addition"
            },
            "runtime": {
                "vector_size": size,
                "validate": True,
                "profile_gpu": True,
                "profile_cpu": True,
                "threads_per_block": 256,
                "timeout_s": 10,
                "experiment": {"name": "vector_size_sweep", "threads": 256}
            }
        }
        print(json.dumps(config))



    # Thread block sizes to test
    threads_per_block_options = [64, 128, 256, 512, 1024]

    # Generate configs testing different thread counts (medium vector size)
    for threads in threads_per_block_options:
        config = {
            "executable": {
                "make_target": "01-vector-addition"
            },
            "runtime": {
                "vector_size": 1000000,
                "validate": True,
                "profile_gpu": True,
                "profile_cpu": True,
                "threads_per_block": threads,
                "timeout_s": 10,
                "experiment": {"name": "threads_sweep", "size": 1000000}
            }
        }
        print(json.dumps(config))


def main():
    generate_configs()


if __name__ == "__main__":
    main()
