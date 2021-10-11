#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "BatchNorm.h"
#include "RELU.h"

#define IMAGE_N 1
#define IMAGE_C 3
#define IMAGE_H 224
#define IMAGE_W 224

#define ITERATIONS 10000

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    for(int i = 0; i < ITERATIONS; i++) {
        float *input_data;
        CUDA_CALL(cudaMalloc(&input_data, IMAGE_N * IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float)));
        fill_constant<<<IMAGE_W * IMAGE_H, IMAGE_N * IMAGE_C>>>(input_data, 1.f);
        cudaDeviceSynchronize();

        // Convolution Layer 1
        ConvolutionLayer convolution1(cudnn, input_data);
        convolution1.SetInputDescriptor(IMAGE_N, IMAGE_C, IMAGE_H, IMAGE_W);
        convolution1.SetFilterDescriptor(64, 3, 7, 7);
        convolution1.SetConvolutionDescriptor(3, 3, 2, 2, 1, 1);
        convolution1.SetOutputDescriptor();
        convolution1.SetAlgorithm();
        convolution1.AllocateWorkspace();
        convolution1.AllocateMemory();
        convolution1.Forward();
        convolution1.Free();
        
        
        // Batch Norm Layer 1
        BatchNorm batch_norm1(cudnn, convolution1.GetOutputData());
        batch_norm1.SetInputDescriptor(1, 64, 112, 112);
        batch_norm1.SetBatchNormDescriptor();
        batch_norm1.SetOutputDescriptor();
        batch_norm1.SetScaleAndBias();
        batch_norm1.Forward();
        batch_norm1.Free();
        /*

         // ReLU 1
        RELU relu1(cudnn, convolution1.GetOutputData());
        relu1.SetInputDescriptor(1, 64, 112, 112);
        relu1.Forward();
        relu1.Free();

        // Pooling Layer 1
        PoolingLayer pooling1(cudnn);
        pooling1.SetInputDescriptor(1, 64, 112, 112);
        pooling1.SetInputData(convolution1.GetOutputData());
        pooling1.SetPoolingDescriptor(3, 3, 2, 2);
        pooling1.SetOutputDescriptor(1, 64, 56, 56);
        pooling1.AllocateMemory();
        pooling1.Forward();
        pooling1.Free();
        */
        
        /*
        // Convolution Layer 3
        ConvolutionLayer convolution2(cudnn, pooling1.GetOutputData());
        convolution2.SetInputDescriptor(1, 96, 27, 27);
        convolution2.SetFilterDescriptor(256, 96, 5, 5);
        convolution2.SetConvolutionDescriptor(2, 2, 1, 1, 1, 1);
        convolution2.SetOutputDescriptor();
        convolution2.SetAlgorithm();
        convolution2.AllocateWorkspace();
        convolution2.AllocateMemory();
        convolution2.Forward();
        convolution2.Free();

        // ReLU 3
        RELU relu2(cudnn, convolution2.GetOutputData());
        relu2.SetInputDescriptor(1, 256, 27, 27);
        relu2.Forward();
        relu2.Free();

        // Pooling Layer 4
        PoolingLayer pooling2(cudnn);
        pooling2.SetInputDescriptor(1, 256, 27, 27);
        pooling2.SetInputData(convolution2.GetOutputData());
        pooling2.SetPoolingDescriptor(3, 3, 2, 2);
        pooling2.SetOutputDescriptor(1, 256, 13, 13);
        pooling2.AllocateMemory();
        pooling2.Forward();
        pooling2.Free();

        // Convolution Layer 5
        ConvolutionLayer convolution3(cudnn, pooling2.GetOutputData());
        convolution3.SetInputDescriptor(1, 256, 13, 13);
        convolution3.SetFilterDescriptor(384, 256, 3, 3);
        convolution3.SetConvolutionDescriptor(1, 1, 1, 1, 1, 1);
        convolution3.SetOutputDescriptor();
        convolution3.SetAlgorithm();
        convolution3.AllocateWorkspace();
        convolution3.AllocateMemory();
        convolution3.Forward();
        convolution3.Free();

        // ReLU 5
        RELU relu3(cudnn, convolution3.GetOutputData());
        relu3.SetInputDescriptor(1, 384, 13, 13);
        relu3.Forward();
        relu3.Free();

        // Convolution Layer 6
        ConvolutionLayer convolution4(cudnn, convolution3.GetOutputData());
        convolution4.SetInputDescriptor(1, 384, 13, 13);
        convolution4.SetFilterDescriptor(384, 384, 3, 3);
        convolution4.SetConvolutionDescriptor(1, 1, 1, 1, 1, 1);
        convolution4.SetOutputDescriptor();
        convolution4.SetAlgorithm();
        convolution4.AllocateWorkspace();
        convolution4.AllocateMemory();
        convolution4.Forward();
        convolution4.Free();

        // ReLU 6
        RELU relu4(cudnn, convolution4.GetOutputData());
        relu4.SetInputDescriptor(1, 384, 13, 13);
        relu4.Forward();
        relu4.Free();

        // Convolution Layer 7
        ConvolutionLayer convolution5(cudnn, convolution4.GetOutputData());
        convolution5.SetInputDescriptor(1, 384, 13, 13);
        convolution5.SetFilterDescriptor(256, 384, 3, 3);
        convolution5.SetConvolutionDescriptor(1, 1, 1, 1, 1, 1);
        convolution5.SetOutputDescriptor();
        convolution5.SetAlgorithm();
        convolution5.AllocateWorkspace();
        convolution5.AllocateMemory();
        convolution5.Forward();
        convolution5.Free();

        // ReLU 7
        RELU relu5(cudnn, convolution5.GetOutputData());
        relu5.SetInputDescriptor(1, 256, 13, 13);
        relu5.Forward();
        relu5.Free();

        // Pooling Layer 8
        PoolingLayer pooling3(cudnn);
        pooling3.SetInputDescriptor(1, 256, 13, 13);
        pooling3.SetInputData(convolution5.GetOutputData());
        pooling3.SetPoolingDescriptor(3, 3, 2, 2);
        pooling3.SetOutputDescriptor(1, 256, 6, 6);
        pooling3.AllocateMemory();
        pooling3.Forward();
        pooling3.Free();
        printf("\n");
        */
    }

     // Cleanup
     CUDNN_CALL(cudnnDestroy(cudnn));
}
