#pragma once
#include "Utils.h"
#include <cudnn.h>

class BatchNorm {
    private:
        void EstimateMeanAndVariance();
    
    public:
        cudnnHandle_t handle;
        float* input_data;
        
        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t batch_norm_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        int input_n, input_c, input_h, input_w;

        const float alpha = 1.f;
        const float beta = 0.f;

        const float estimated_mean = 1.f;
        const float estimated_variance = 0.f;
        const double epsilon = 0.0001;

        float* bn_scale;
        float* bn_bias;

        BatchNorm(cudnnHandle_t handle, float* data);

        void SetInputDescriptor(int N, int C, int H, int W);
        void SetBatchNormDescriptor(int N, int C, int H, int W);
        void SetScaleAndBias();
        void SetOutputDescriptor();
        float* GetOutputData();
        void Forward();
};