CXX := nvcc
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include -I ./include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_75 -std=c++11 -DDEBUG=1
SOURCES := src/Resnet.cu src/ConvolutionLayer.cu src/PoolingLayer.cu src/BatchNorm.cu src/RELU.cu src/Utils.cu src/Block.cu

all: resnet

resnet: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(SOURCES) -o resnet -lcudnn

.phony: clean

clean:
	rm resnet || echo -n ""
