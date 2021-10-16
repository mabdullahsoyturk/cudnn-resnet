#include "RELU.h"

RELU::RELU() {}

RELU::RELU(cudnnHandle_t handle, float *data) : 
        handle(handle), input_data(data) {
    
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_descriptor));
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN,
                                            /*RELU_coef=*/0));
}

void RELU::SetInputDescriptor(int N, int C, int H, int W) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                        CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT,
                                        input_n, input_c, input_h, input_w));
    
    #if DEBUG
    printf("RELU Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
    #endif

    RELU::SetOutputDescriptor();

    #if DEBUG
    printf("RELU Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);
    #endif
}

void RELU::SetOutputDescriptor() {
    output_n = input_n;
    output_c = input_c;
    output_w = input_w;
    output_h = input_h;
}

void RELU::Forward() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CUDNN_CALL(cudnnActivationForward(handle,
                                      activation_descriptor,
                                      &alpha,
                                      input_descriptor,
                                      input_data,
                                      &beta,
                                      input_descriptor,
                                      input_data));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f,", milliseconds);
    cudaDeviceSynchronize();
}

void RELU::Free() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_descriptor));
}
