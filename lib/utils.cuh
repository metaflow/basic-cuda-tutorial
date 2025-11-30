#ifndef LIB_UTILS_H
#define LIB_UTILS_H

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

long long time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long long t = ts.tv_sec;
  t = t * 1000000000LL + ts.tv_nsec;
  return t;
}

#endif  // LIB_UTILS_H