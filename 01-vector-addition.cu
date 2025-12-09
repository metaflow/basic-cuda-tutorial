/**
 * CUDA Basic Example - Vector Addition
 *
 * This program demonstrates the fundamental concepts of CUDA programming:
 * 1. Allocating memory on the GPU
 * 2. Copying data between CPU and GPU
 * 3. Executing a kernel function on the GPU
 * 4. Synchronization between CPU and GPU
 * 5. Cleaning up resources
 */

#include <stdio.h>
#include <chrono>
#include <vector>
#include <stdlib.h>
#include <utils.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <config.h>

// CUDA kernel function to add two vectors
__global__ void addGPU(const float *A, const float *B, float *C, int numElements) {
    // Get the unique thread ID, which is the index in the vector
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void addCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    // Load configuration from JSON file
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <config.json>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Config config = Config::loadFromFile(argv[1]);
    if (!config.isValid()) {
        return EXIT_FAILURE;
    }

    // Print configuration as single-line JSON
    printf("'config':%s\n", config.toJson().dump().c_str());

    // Vector size and memory size
    int numElements = config.vector_size;
    size_t size = numElements * sizeof(float);

    printf("Vector addition of %d elements\n", numElements);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_V = (float *)malloc(size);

    // Initialize host arrays with random data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch the CUDA kernel
    int threadsPerBlock = config.threads_per_block;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch kernel and check correctness.
    if (config.validate) {
        addGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

        addCPU(h_A, h_B, h_V, numElements);

        // Verify the result
        for (int i = 0; i < numElements; ++i) {
            if (fabs(h_V[i] - h_C[i]) > 1e-5) {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }
        printf("'valid':true\n");
    }

    // GPU perf.
    if (config.profile_gpu) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double bytes_transferred = 3.0 * size;  // Read A, B and write C

        {
        long long gpu_start = time_ns();
        long long cuda_best = LONG_MAX;
        int interation_counter = 0;
        int keep_counter = 10;
        long long timeout_ns = config.timeout_s * 1000000000LL;
        while (true) {
            interation_counter++;
            CUDA_CHECK(cudaEventRecord(start));
            addGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            long long us = round(ms * 1000);
            if (us >= cuda_best) {
                keep_counter--;
                if (keep_counter < 0) break;
            } else {
                cuda_best = us;
            }
            if (time_ns() - gpu_start > timeout_ns) break;
        }
            double gpu_throughput = bytes_transferred / (cuda_best / 1e6) / (1024.0 * 1024.0 * 1024.0);  // GiB/s
            printf("GPU: %lld us from %d iterations (%.2f GiB/s)\n", cuda_best, interation_counter, gpu_throughput);

            json gpu_perf = {
                {"time_us", cuda_best},
                {"iterations", interation_counter},
                {"throughput_gib_s", std::round(gpu_throughput * 1000.0) / 1000.0}
            };
            printf("'gpu_perf':%s\n", gpu_perf.dump().c_str());
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // CPU perf.
    if (config.profile_cpu) {
        double bytes_transferred = 3.0 * size;  // Read A, B and write C

        long long cpu_start = time_ns();
        long long cpu_best_us = LONG_MAX;
        int iterations_counter = 0;
        int keep_counter = 10;
        long long timeout_ns = config.timeout_s * 1000000000LL;
        while (true) {
            iterations_counter++;
            auto s = time_ns();
            addCPU(h_A, h_B, h_V, numElements);
            auto e = time_ns();
            long long us = (e - s) / 1000LL;
            if (us >= cpu_best_us) {
                keep_counter--;
                if (keep_counter < 0) break;
            } else {
                cpu_best_us = us;
            }
            if (e - cpu_start > timeout_ns) break;
        }
        double cpu_throughput = bytes_transferred / (cpu_best_us / 1e6) / (1024.0 * 1024.0 * 1024.0);  // GiB/s
        printf("CPU: %lld us from %d iterations (%.2f GiB/s)\n",
               cpu_best_us, iterations_counter, cpu_throughput);

        json cpu_perf = {
            {"time_us", cpu_best_us},
            {"iterations", iterations_counter},
            {"throughput_gib_s", std::round(cpu_throughput * 1000.0) / 1000.0}
        };
        printf("'cpu_perf':%s\n", cpu_perf.dump().c_str());
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_V);

    printf("Done\n");
    return 0;
}