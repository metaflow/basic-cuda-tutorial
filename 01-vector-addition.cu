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
#include <stdlib.h>
#include <utils.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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

int main() {
    // Vector size and memory size
    int numElements = 50000;
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    addGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    CUDA_CHECK(cudaEventRecord(stop));

    // Check for errors in kernel launch
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventSynchronize(stop));

    {
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("kernel execution time: %.3f ms\n", ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        addCPU(h_A, h_B, h_V, numElements);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        printf("CPU execution time: %.3f ms\n",
               std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());
    }
    // Verify the result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_V[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

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