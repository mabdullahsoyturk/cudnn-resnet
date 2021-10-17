#pragma once
#include "ConvolutionLayer.h"
#include "BatchNorm.h"
#include "RELU.h"

class Block {
    cudnnHandle_t handle;
    float* input_data;
    float* output_data;
    float* identity_data;
    
    int input_n, input_c, input_h, input_w;
    int stride;
    int intermediate_c;

    public:
        Block(cudnnHandle_t handle, float* data, int N, int C, int H, int W, int intermediate_channels, int stride);
    
    float* GetOutputData();
    void Forward();
};