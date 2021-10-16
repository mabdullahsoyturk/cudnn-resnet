#include "Block.h"

Block::Block(cudnnHandle_t handle, float* data, int N, int C, int H, int W): handle(handle), input_data(data) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;
}

float* Block::GetOutputData() {
    return output_data;
}