#pragma once
#include "Utils.h"
#include <cudnn.h>

class ConvolutionLayer {
    public:
        cudnnHandle_t handle;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnFilterDescriptor_t filter_descriptor;
        cudnnConvolutionDescriptor_t convolution_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        cudnnConvolutionFwdAlgo_t algorithm;

        size_t workspace_size;
        float *workspace_data;
        
        int input_n, input_c, input_h, input_w;
        int filter_n, filter_c, filter_h, filter_w;
        int output_n, output_c, output_h, output_w;

        int padding_h, padding_w;
        int stride_h, stride_w;
        int dilation_h, dilation_w;

        const float alpha = 1.f;
        const float beta = 0.f;

        float *input_data;
        float *filter_data;
        float *output_data;

        ConvolutionLayer();
        ConvolutionLayer(cudnnHandle_t handle, float* data);
        void SetInputDescriptor(int N, int C, int H, int W);
        void SetFilterDescriptor(int N, int C, int H, int W);
        void SetConvolutionDescriptor(int H_padding, int W_padding, int H_stride, int W_stride, int H_dilation, int W_dilation);
        void SetOutputDescriptor();
        float* GetOutputData();
        void SetAlgorithm();
        void AllocateMemory();
        void AllocateWorkspace();
        void Forward();
        void Free();
};
