#include "BatchNorm.h"

BatchNorm::BatchNorm(cudnnHandle_t handle, float* data): handle(handle), input_data(data) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
}

void BatchNorm::SetScaleAndBias() {
    bn_scale = (float*) malloc(input_c * sizeof(float));
    for(int i = 0; i < input_c; i++) {
        bn_scale[i] = 1;
    }
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

void BatchNorm::SetBatchNormDescriptor(int N, int C, int H, int W) {
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
}

float* BatchNorm::GetOutputData() {
    return input_data;
}

void BatchNorm::Forward() {
    //cudnnStatus_t cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL, /*cudnnBatchNormMode_t mode*/
        &alpha,
        &beta,
        input_descriptor,/*const cudnnTensorDescriptor_t xDesc*/
        input_data,/*const void *x*/
        output_descriptor,/*const cudnnTensorDescriptor_t yDesc*/
        input_data, /*void *y*/
        batch_norm_descriptor,/*const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc*/
        /*const void *bnScale*/
        /*const void *bnBias*/
        &estimatedMean,
        &estimatedVariance,
        double epsilon
    //);
}