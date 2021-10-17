#include "PoolingLayer.h"

PoolingLayer::PoolingLayer() {}

PoolingLayer::PoolingLayer(cudnnHandle_t handle): handle(handle) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor))
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
}

void PoolingLayer::SetInputDescriptor(int N, int C, int H, int W) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          input_n, input_c, input_h, input_w));
    #if DEBUG
    printf("Pooling Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
    #endif
}

void PoolingLayer::SetInputData(float* data) {
    input_data = data;
}

void PoolingLayer::SetPoolingDescriptor(int window_H, int window_W, int stride_V, int stride_H) {
    window_height = window_H;
    window_width = window_W;
    stride_vertical = stride_V;
    stride_horizontal = stride_H;

    CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           window_height,
                                           window_width,
                                           /*Pad H*/1,
                                           /*Pad W*/1,
                                           stride_vertical,
                                           stride_horizontal));
}

void PoolingLayer::SetOutputDescriptor(int N, int C, int H, int W) {
    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(pooling_descriptor, 
                                                 input_descriptor,
                                                 &output_n, &output_c, &output_h, &output_w));

    #if DEBUG
    printf("Pooling Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);
    #endif

    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          output_n, output_c, output_h, output_w));
}

float* PoolingLayer::GetOutputData() {
    return output_data;
}
 
void PoolingLayer::AllocateMemory() {
    CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));
}

void PoolingLayer::Forward() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CUDNN_CALL(cudnnPoolingForward(handle,
                                   pooling_descriptor,
                                   &alpha,
                                   input_descriptor, input_data,
                                   &beta,
                                   output_descriptor, output_data));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    #if !DEBUG
    printf("%f,", milliseconds);
    #endif
}

void PoolingLayer::Free() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDA_CALL(cudaFree(input_data));
}
