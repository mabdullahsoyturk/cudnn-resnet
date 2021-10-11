#include "Utils.h"

__global__ void fill_constant(float *px, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k; 
}

void print(const float *data, int n, int c, int h, int w) {
    std::vector<float> buffer(1 << 20);
    CUDA_CALL(cudaMemcpy(buffer.data(), data, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));

    int a = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    std::cout << std::setw(10) << std::right << buffer[a];
                    ++a;
                }
                std::cout << std::endl;
            }
        }
    }

    std::cout << std::endl;
}
