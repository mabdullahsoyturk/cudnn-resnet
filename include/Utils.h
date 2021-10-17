#pragma once
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#define CUDNN_CALL(func) {                                                                         \
  cudnnStatus_t status = (func);                                                                   \
  if (status != CUDNN_STATUS_SUCCESS) {                                                            \
    std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                                                                       \
  }                                                                                                \
}

#define CUDA_CALL(func) {                                                                           \
  cudaError_t status = (func);                                                                      \
  if (status != cudaSuccess) {                                                                         \
    std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl;  \
    std::exit(1);                                                                                   \
  }                                                                                                 \
}

__global__ void fill_constant(float *px, float k);
__global__ void add_identity(float* orig, float* identity, int size);
__global__ void copy(float* in, float* out, int size);

void print(const float *data, int n, int c, int h, int w);
