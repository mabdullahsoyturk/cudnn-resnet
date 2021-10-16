#pragma once
#include "Utils.h"
#include <cudnn.h>

class BatchNorm {
    private:
        void EstimateMeanAndVariance();
    
    public:
        cudnnHandle_t handle;
        float* input_data;
        float* output_data;
        
        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t batch_norm_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        int input_n, input_c, input_h, input_w;

        float* estimated_mean;
        float* estimated_variance;
        const double epsilon = 0.0001;

        float* bn_scale;
        float* bn_bias;
        float* d_bn_scale;
        float* d_bn_bias;

        BatchNorm(cudnnHandle_t handle, float* data);

        void SetInputDescriptor(int N, int C, int H, int W);
        void SetBatchNormDescriptor();
        void SetOutputDescriptor();
        void SetScaleAndBias();
        float* GetOutputData();
        void Forward();
        void Free();
};