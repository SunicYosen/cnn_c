#ifndef CNN_H_
#define CNN_H_

#include <stdbool.h>
#include "mat.h"
#include "mnist.h"

#define AVG_POOLING 0  //Pooling with average
#define MAX_POOLING 1  //Pooling with Maximum
#define MIN_POOLING 2  //Pooling with Minimum

//Define Structure of Convolutional Layer
typedef struct ConvolutionalLayer{
	int16_t input_width;  //
	int16_t input_height; //
	int16_t map_size;     //kernel feature map size {map_size}

	int16_t input_channels;  //
	int16_t output_channels; //

	int16_t**** map_data;     //Kernel Data
	int16_t**** dmap_data;    //存放特征模块的数据的局部梯度

	int16_t* bias_data;     //偏置
	bool is_full_connect;   //
	bool* connect_model;    //

  // 下面三者的大小同输出的维度相同
	int16_t*** v;            //进入激活函数的输入值
	int16_t*** y;            //激活函数后神经元的输出
	int16_t*** d;            //网络的局部梯度,δ值      // 输出像素的局部梯度

}ConvolutionLayer;

//Define pooling layer
typedef struct PoolingLayer{
	int16_t input_width;
	int16_t input_height;
	int16_t map_size;

	int16_t input_channels;
	int16_t output_channels;

	int16_t pooling_type;
	int16_t* bias_data;

	int16_t*** y;  // 采样函数后神经元的输出,无激活函数
	int16_t*** d;  // 网络的局部梯度,δ值
}PoolingLayer;

//Define Output layer
typedef struct OutputLayer{
	int16_t input_num;
	int16_t output_num;

	int16_t** weight_data;
	int16_t* bias_data;  

	int16_t* v;
	int16_t* y; 
	int16_t* d;

	bool is_full_connect;
} OutputLayer;

//Define CNN Architectrue
typedef struct ConvolutionalNeuralNetwork{
	int16_t layer_num;    //Layer num
	ConvolutionLayer* C1;        //Convolution Layer1
	PoolingLayer* S2;    //Pooling Layer2
	ConvolutionLayer* C3;        //Convolution Layer3
	PoolingLayer* S4;    //Pooling Layer4
	OutputLayer* O5;     //Output Layer5

	int16_t* e;            //// 训练误差
	int16_t* L;            // // 瞬时误差能量

}Cnn;

//Train Options
typedef struct TrainOptions{
	int16_t numepochs; 
	int16_t alpha; 
}TrainOptions;

void CnnSetup(Cnn* cnn, MatSize inputSize, int16_t outputSize);

void CnnTrain(Cnn* cnn,	ImageArray inputData, LabelArray outputData, \
              TrainOptions opts, int32_t num_trains);

int32_t CnnTest(Cnn* cnn, ImageArray inputData, \
                LabelArray outputData, int32_t testNum);

void SaveCnn(Cnn* cnn, const char* filename);

void ImportCnn(Cnn* cnn, const char* filename);

ConvolutionLayer* InitialConvolutionLayer(int16_t input_width, \
                                          int16_t input_height, \
																					int16_t map_size, \
																					int16_t input_channels, \
																					int16_t output_channels);

void ConvolutionLayerConnect(ConvolutionLayer* cov_layer,bool* connect_model);

PoolingLayer* InitialPoolingLayer(int16_t input_width, \
                                  int16_t input_height, \
                                  int16_t map_size, \
																	int16_t input_channels, \
																	int16_t output_channels, \
																	int16_t pooling_type);

void PoolingLayerConnect(PoolingLayer* poolL, bool* connect_model);

OutputLayer* InitOutputLayer(int16_t input_num, int16_t output_num);

int16_t ActivationSigma(int16_t input, int16_t bias); 

int16_t ActivationReLU(int16_t input, int16_t bias);

void CnnFF(Cnn* cnn, int16_t** inputData); 

void CnnBP(Cnn* cnn, int16_t* outputData);

void CnnApplyGradients(Cnn* cnn, TrainOptions opts, int16_t** inputData);

void CnnClear(Cnn* cnn); 

void AvgPooling(int16_t** output,MatSize outputSize, int16_t** input,
                MatSize inputSize, int16_t map_size);

void MaxPooling(int16_t** output,MatSize outputSize, int16_t** input,
                MatSize inputSize, int16_t map_size);

void nnff(int16_t* output, int16_t* input, int16_t** wdata, \
          int16_t* bias, MatSize nnSize);

void SaveCnnData(Cnn* cnn, const char* filename, int16_t** inputdata);

#endif
