#include "Block.h"

Block::Block(cudnnHandle_t handle, float* data, int N, int C, int H, int W, int intermediate_channels, int stride) : 
    handle(handle), input_data(data), intermediate_c(intermediate_channels), stride(stride) {
    input_n = N;
    input_c = C;
    input_h = H;
    input_w = W;

    cudaMalloc((void **)&identity_data, N * C * H * W * sizeof(float));

    int size = N * C * H * W;
    int THREADS = 512;
    int BLOCKS = (size + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    
    copy<<<BLOCKS, THREADS>>>(input_data, identity_data, size);
    cudaDeviceSynchronize();
}

void Block::Forward() {
    // First
    ConvolutionLayer conv1(handle, input_data);
    conv1.SetInputDescriptor(input_n, input_c, input_h, input_w);
    conv1.SetFilterDescriptor(intermediate_c, input_c, 3, 3);
    conv1.SetConvolutionDescriptor(1, 1, stride, stride, 1, 1);
    conv1.SetOutputDescriptor();
    conv1.SetAlgorithm();
    conv1.AllocateWorkspace();
    conv1.AllocateMemory();
    conv1.Forward();
    conv1.Free();

    int new_w = ((input_w - 3 + 2) / stride) + 1;
    int new_h = new_w;

    BatchNorm batch_norm1(handle, conv1.GetOutputData());
    batch_norm1.SetInputDescriptor(1, intermediate_c, new_h, new_w);
    batch_norm1.SetBatchNormDescriptor();
    batch_norm1.SetOutputDescriptor();
    batch_norm1.SetScaleAndBias();
    batch_norm1.Forward();
    batch_norm1.Free();

    RELU relu1(handle, batch_norm1.GetOutputData());
    relu1.SetInputDescriptor(1, intermediate_c, new_h, new_w);
    relu1.Forward();
    relu1.Free();
    
    // Second
    ConvolutionLayer conv2(handle, batch_norm1.GetOutputData());
    conv2.SetInputDescriptor(input_n, intermediate_c, new_h, new_w);
    conv2.SetFilterDescriptor(intermediate_c, intermediate_c, 3, 3);
    conv2.SetConvolutionDescriptor(1, 1, 1, 1, 1, 1);
    conv2.SetOutputDescriptor();
    conv2.SetAlgorithm();
    conv2.AllocateWorkspace();
    conv2.AllocateMemory();
    conv2.Forward();
    conv2.Free();

    BatchNorm batch_norm2(handle, conv2.GetOutputData());
    batch_norm2.SetInputDescriptor(1, intermediate_c, new_h, new_w);
    batch_norm2.SetBatchNormDescriptor();
    batch_norm2.SetOutputDescriptor();
    batch_norm2.SetScaleAndBias();
    batch_norm2.Forward();
    batch_norm2.Free();

    RELU relu2(handle, batch_norm2.GetOutputData());
    relu2.SetInputDescriptor(1, intermediate_c, new_h, new_w);
    relu2.Forward();
    relu2.Free();

    output_data = batch_norm2.GetOutputData();

    int N = 1 * intermediate_c * new_h * new_w;
    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    add_identity<<<BLOCKS, THREADS>>>(output_data, identity_data, N);
    cudaDeviceSynchronize();
    
    RELU relu3(handle, output_data);
}

float* Block::GetOutputData() {
    return output_data;
}