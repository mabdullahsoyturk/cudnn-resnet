#pragma once
#include "Utils.h"
#include <cudnn.h>

class PoolingLayer {
    public:
        cudnnHandle_t handle;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnPoolingDescriptor_t pooling_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        int window_height, window_width;
        int stride_vertical, stride_horizontal;

        int input_n, input_c, input_h, input_w;
        int output_n, output_c, output_h, output_w;

        float* input_data;
        float* output_data;

        const float alpha = 1.f;
        const float beta = 0.f;

        PoolingLayer();
        PoolingLayer(cudnnHandle_t handle);

        void SetInputDescriptor(int N, int C, int H, int W);
        void SetPoolingDescriptor(int window_H, int window_W, int stride_V, int stride_H);
        void SetOutputDescriptor(int N, int C, int H, int W);
        float* GetOutputData();
        void SetInputData(float* data);
        void AllocateMemory();
        void Forward();
        void Free();
};
