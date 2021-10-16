#pragma once
#include "ConvolutionLayer.h"
#include "BatchNorm.h"
#include "RELU.h"

class Block {
    /*ConvolutionLayer conv1;
    BatchNorm batch_norm1;
    RELU relu1;
    
    ConvolutionLayer conv2;
    BatchNorm batch_norm2;
    RELU relu2;

    RELU relu3;*/

    cudnnHandle_t handle;
    float* input_data;
    float* output_data;
    
    int input_n, input_c, input_h, input_w;

    public:
        Block(cudnnHandle_t handle, float* data, int N, int C, int H, int W);
    
    float* GetOutputData();
    void Forward();
};