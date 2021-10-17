#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer() {}

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t handle, float* data): handle(handle), input_data(data) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
}

void ConvolutionLayer::SetInputDescriptor(int N, int C, int H, int W) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          input_n, input_c, input_h, input_w));
    
    #if DEBUG
    printf("Convolution Input Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", input_n, input_c, input_h, input_w);
    #endif
}

void ConvolutionLayer::SetFilterDescriptor(int N, int C, int H, int W) {
    filter_n = N;
    filter_c = C;
    filter_h = H;
    filter_w = W;

    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                    CUDNN_DATA_FLOAT, 
                                    CUDNN_TENSOR_NCHW,
                                    filter_n, filter_c, filter_h, filter_w));

    #if DEBUG
    printf("Convolution Filter Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", filter_n, filter_c, filter_h, filter_w);
    #endif
}

void ConvolutionLayer::SetOutputDescriptor() {
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, 
                                                     input_descriptor, filter_descriptor,
                                                     &output_n, &output_c, &output_h, &output_w));

    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          CUDNN_TENSOR_NCHW, 
                                          CUDNN_DATA_FLOAT,
                                          output_n, output_c, output_h, output_w));
    
    #if DEBUG
    printf("Convolution Output Shape (NCHW) => N: %d, C: %d, H: %d, W: %d\n", output_n, output_c, output_h, output_w);
    #endif
}

float* ConvolutionLayer::GetOutputData() {
    return output_data;
}

void ConvolutionLayer::SetConvolutionDescriptor(int H_padding, int W_padding, int H_stride, int W_stride, int H_dilation, int W_dilation) {
    padding_h = H_padding;
    padding_w = W_padding;
    stride_h = H_stride;
    stride_w = W_stride;
    dilation_h = H_dilation;
    dilation_w = W_dilation;

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                             CUDNN_CONVOLUTION, 
                                             CUDNN_DATA_FLOAT));
    
    #if DEBUG
    printf("Convolution parameters => Padding h: %d, Padding w: %d, Stride h: %d, Stride w: %d, Dilation h: %d, Dilation w: %d\n",
                                    padding_h,     padding_w,     stride_h,     stride_w,     dilation_h,     dilation_w);
    #endif
}

void ConvolutionLayer::SetAlgorithm() {
    cudnnConvolutionFwdAlgoPerf_t convolution_algo_perf;
    int algo_count;

    cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                           input_descriptor,
                                           filter_descriptor,
                                           convolution_descriptor,
                                           output_descriptor,
                                           /*requested algo count*/1,
                                           /*returned algo count*/&algo_count,
                                           &convolution_algo_perf);
    
    algorithm = convolution_algo_perf.algo;
    //algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

void ConvolutionLayer::AllocateMemory() {
    CUDA_CALL(cudaMalloc(&filter_data, filter_n * filter_c * filter_h * filter_w * sizeof(float)));
    CUDA_CALL(cudaMalloc(&output_data, output_n * output_c * output_h * output_w * sizeof(float)));

    fill_constant<<<filter_w * filter_h, filter_n * filter_c>>>(filter_data, 3.f);
    cudaDeviceSynchronize();
}

void ConvolutionLayer::AllocateWorkspace() {
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, 
                                                       input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 
                                                       algorithm, 
                                                       &workspace_size));

    CUDA_CALL(cudaMalloc(&workspace_data, workspace_size));
    #if DEBUG
    printf("Workspace allocated: %ld bytes\n", workspace_size);
    #endif
}

void ConvolutionLayer::Forward() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CUDNN_CALL(cudnnConvolutionForward(handle,
                                       &alpha, 
                                       input_descriptor, input_data, 
                                       filter_descriptor, filter_data,
                                       convolution_descriptor, algorithm, workspace_data, workspace_size,
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

void ConvolutionLayer::Free() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
    CUDA_CALL(cudaFree(input_data));
    CUDA_CALL(cudaFree(filter_data));
    CUDA_CALL(cudaFree(workspace_data));
}
