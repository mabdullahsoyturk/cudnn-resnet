#include "BatchNorm.h"

BatchNorm::BatchNorm(cudnnHandle_t handle, float* data): handle(handle), input_data(data) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&batch_norm_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
}

void BatchNorm::SetScaleAndBias() {
    bn_scale = (float*) malloc(input_c * sizeof(float));
    bn_bias = (float*) malloc(input_c * sizeof(float));
    CUDA_CALL(cudaMalloc((void**)&d_bn_scale, input_c * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_bn_bias, input_c * sizeof(float)));

    for(int i = 0; i < input_c; i++) {
        bn_scale[i] = 1;
        bn_bias[i] = 0;
    }

    CUDA_CALL(cudaMemcpy(d_bn_scale, bn_scale, input_c * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bn_bias, bn_bias, input_c * sizeof(float), cudaMemcpyHostToDevice));
}

void BatchNorm::SetInputDescriptor(int N, int C, int H, int W) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          input_n, input_c, input_h, input_w));
    
    #if DEBUG
    printf("Batch Norm Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
    #endif
}

void BatchNorm::SetBatchNormDescriptor() {
    CUDNN_CALL(cudnnSetTensor4dDescriptor(batch_norm_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          1, input_c, 1, 1));
}

void BatchNorm::SetOutputDescriptor() {
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          input_n, input_c, input_h, input_w));
    
    #if DEBUG
    printf("Batch Norm Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
    #endif

    CUDA_CALL(cudaMalloc(&estimated_mean, input_n * input_c * input_h * input_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&estimated_variance, input_n * input_c * input_h * input_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&output_data, input_n * input_c * input_h * input_w * sizeof(float)));
}

float* BatchNorm::GetOutputData() {
    return input_data;
}

void BatchNorm::Forward() {
    float one = 1;
    float zero = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL, /*cudnnBatchNormMode_t mode*/
        &one,
        &zero,
        input_descriptor,/*const cudnnTensorDescriptor_t xDesc*/
        input_data,/*const void *x*/
        output_descriptor,/*const cudnnTensorDescriptor_t yDesc*/
        output_data, /*void *y*/
        batch_norm_descriptor,/*const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc*/
        d_bn_scale,
        d_bn_bias,
        estimated_mean,
        estimated_variance,
        epsilon
    ));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f,", milliseconds);
}

void BatchNorm::Free() {
    CUDA_CALL(cudaFree(d_bn_scale));
    CUDA_CALL(cudaFree(d_bn_bias));
    CUDA_CALL(cudaFree(estimated_mean));
    CUDA_CALL(cudaFree(estimated_variance));
    CUDA_CALL(cudaFree(input_data));
    free(bn_scale);
    free(bn_bias);

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(batch_norm_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
}