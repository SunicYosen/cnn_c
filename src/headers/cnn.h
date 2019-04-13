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
	int8_t input_width;  //
	int8_t input_height; //
	int8_t map_size;     //kernel feature map size {map_size}

	int8_t input_channels;  //
	int8_t output_channels; //

	float**** map_data;     //Kernel Data
	float**** dmap_data;    //

	float* basic_data;
	bool is_full_connect;
	bool* connect_model;

	float*** v;
	float*** y;

	float*** d;
}ConvolutionLayer;

//Define pooling layer
typedef struct PoolingLayer{
	int input_width;
	int input_height;
	int map_size;

	int input_channels;
	int output_channels;

	int pooling_type;
	float* basic_data;

	float*** y;
	float*** d; 
}PoolingLayer;

//Define Output layer
typedef struct OutputLayer{
	int8_t input_num;
	int8_t output_num;

	float** weight_data;
	float* basic_data;  

	float* v;
	float* y; 
	float* d;

	bool is_full_connect;
} OutputLayer;

//Define CNN Architectrue
typedef struct ConvolutionalNeuralNetwork{
	int8_t layer_num;    //Layer num
	ConvolutionLayer* C1;        //Convolution Layer1
	PoolingLayer* S2;    //Pooling Layer2
	ConvolutionLayer* C3;        //Convolution Layer3
	PoolingLayer* S4;    //Pooling Layer4
	OutputLayer* O5;     //Output Layer5

	float* e;            //
	float* L;            //
}Cnn;

//Train Options
typedef struct TrainOptions{
	int8_t numepochs; 
	float alpha; 
}TrainOptions;

void CnnSetup(Cnn* cnn, MatSize inputSize, int8_t outputSize);

void CnnTrain(Cnn* cnn,	ImageArray inputData,LabelArray outputData, \
              TrainOptions opts, int32_t num_trains);

float CnnTest(Cnn* cnn, ImageArray inputData,LabelArray outputData,int testNum);

void SaveCnn(Cnn* cnn, const char* filename);

void ImportCnn(Cnn* cnn, const char* filename);

ConvolutionLayer* InitialConvolutionLayer(int8_t input_width, int8_t input_height, \
          int8_t map_size, int8_t input_channels, int8_t output_channels);

void ConvolutionLayerConnect(ConvolutionLayer* cov_layer,bool* connect_model);

PoolingLayer* InitialPoolingLayer(int8_t input_width, int8_t input_height, \
                                  int8_t map_size, int8_t input_channels, \
																	int8_t output_channels, int8_t pooling_type);

void PoolingLayerConnect(PoolingLayer* poolL, bool* connect_model);

OutputLayer* InitOutputLayer(int8_t input_num, int8_t output_num);

float ActivationSigma(float input,float bas); 

void CnnFF(Cnn* cnn,float** inputData); 
void CnnBP(Cnn* cnn,float* outputData); 
void CnnApplyGrads(Cnn* cnn, TrainOptions opts,float** inputData);
void CnnClear(Cnn* cnn); 

void AvgPooling(float** output,MatSize outputSize, float** input, 
                MatSize inputSize, int map_size);

void nnff(float* output, float* input, float** wdata, float* bas, MatSize nnSize);

void SaveCnnData(Cnn* cnn,const char* filename,float** inputdata);

#endif
