#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "BatchNorm.h"
#include "RELU.h"
#include "Block.h"

#define IMAGE_N 1
#define IMAGE_C 3
#define IMAGE_H 224
#define IMAGE_W 224

#define ITERATIONS 1000

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    for(int i = 0; i < ITERATIONS; i++) {
        float *input_data;
        CUDA_CALL(cudaMalloc(&input_data, IMAGE_N * IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float)));
        fill_constant<<<IMAGE_W * IMAGE_H, IMAGE_N * IMAGE_C>>>(input_data, 1.f);
        cudaDeviceSynchronize();


        ///////////////////////////////////////////// FIRST LAYER OF RESNET18 /////////////////////////////////////////////
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

         // ReLU 1
        RELU relu1(cudnn, batch_norm1.GetOutputData());
        relu1.SetInputDescriptor(1, 64, 112, 112);
        relu1.Forward();
        relu1.Free();

        // Pooling Layer 1
        PoolingLayer pooling1(cudnn);
        pooling1.SetInputDescriptor(1, 64, 112, 112);
        pooling1.SetInputData(batch_norm1.GetOutputData());
        pooling1.SetPoolingDescriptor(3, 3, 2, 2);
        pooling1.SetOutputDescriptor(1, 64, 56, 56);
        pooling1.AllocateMemory();
        pooling1.Forward();
        pooling1.Free();

        ///////////////////////////////////////////// START OF RESNET BLOCKS /////////////////////////////////////////////

        ///////////////////////////////////////////// FIRST BLOCK /////////////////////////////////////////////
        Block block1(cudnn, pooling1.GetOutputData(), 1, 64, 56, 56, 64, 1);
        block1.Forward();

        ///////////////////////////////////////////// SECOND BLOCK /////////////////////////////////////////////
        Block block2(cudnn, block1.GetOutputData(), 1, 64, 56, 56, 128, 2);
        block2.Forward();

        ///////////////////////////////////////////// THIRD BLOCK /////////////////////////////////////////////
        Block block3(cudnn, block2.GetOutputData(), 1, 128, 28, 28, 256, 2);
        block3.Forward();

        ///////////////////////////////////////////// FOURTH BLOCK /////////////////////////////////////////////
        Block block4(cudnn, block3.GetOutputData(), 1, 256, 14, 14, 512, 2);
        block4.Forward();
        
        printf("\n");
    }

     // Cleanup
     CUDNN_CALL(cudnnDestroy(cudnn));
}
