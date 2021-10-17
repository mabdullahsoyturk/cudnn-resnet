#pragma once
#include "Utils.h"
#include <cudnn.h>

class RELU {
    public:
        cudnnHandle_t handle;

        cudnnActivationDescriptor_t activation_descriptor;
        cudnnTensorDescriptor_t input_descriptor;

        int input_n, input_c, input_h, input_w;
        int output_n, output_c, output_h, output_w;

        float* input_data;

        RELU();
        RELU(cudnnHandle_t handle, float* data);

        void SetInputDescriptor(int N, int C, int H, int W);
        void SetOutputDescriptor();
        void Forward();
        void Free();
};
