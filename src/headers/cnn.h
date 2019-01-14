#ifndef CNN_H_
#define CNN_H_

#include <stdbool.h>
#include "mat.h"
#include "mnist.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

//Define Structure of Convolutional Layer
typedef struct ConvolutionalLayer{
	int input_width; 
	int input_height;
	int map_size;

	int input_channels;
	int output_channels;
	float**** map_data;
	float**** dmap_data;

	float* basic_data;
	bool is_full_connect;
	bool* connect_model;

	float*** v;
	float*** y;

	float*** d;
}CovLayer;

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
	int inputNum;
	int outputNum;

	float** wData; 
	float* basic_data;  

	float* v;
	float* y; 
	float* d; 

	bool is_full_connect;
} OutputLayer;

//Define CNN Architectrue
typedef struct ConvolutionalNeuralNetwork{
	int layer_num;
	CovLayer* C1;
	PoolingLayer* S2;
	CovLayer* C3;
	PoolingLayer* S4;
	OutputLayer* O5;
	float* e;
	float* L; 
}Cnn;

//Train Options
typedef struct TrainOptions{
	int numepochs; 
	float alpha; 
}TrainOptions;

void CnnSetup(Cnn* cnn,MatSize inputSize,int outputSize);

void CnnTrain(Cnn* cnn,	ImageArray inputData,LabelArray outputData,TrainOptions opts,int trainNum);

float CnnTest(Cnn* cnn, ImageArray inputData,LabelArray outputData,int testNum);

void SaveCnn(Cnn* cnn, const char* filename);

void ImportCnn(Cnn* cnn, const char* filename);

CovLayer* InitialCovLayer(int input_width,int input_height,int map_size,int input_channels,int output_channels);
void CovLayerConnect(CovLayer* covL,bool* connect_model);

PoolingLayer* InitialPoolingLayer(int input_width,int inputHeigh,int map_size,int input_channels,int output_channels,int pooling_type);
void PoolingLayerConnect(PoolingLayer* poolL,bool* connect_model);

OutputLayer* InitOutputLayer(int inputNum,int outputNum);

float ActivationSigma(float input,float bas); 

void CnnFF(Cnn* cnn,float** inputData); 
void CnnBP(Cnn* cnn,float* outputData); 
void CnnApplyGrads(Cnn* cnn,TrainOptions opts,float** inputData);
void CnnClear(Cnn* cnn); 

void AvgPooling(float** output,MatSize outputSize,float** input,MatSize inputSize,int map_size);

void nnff(float* output,float* input,float** wdata,float* bas,MatSize nnSize);

void SaveCnnData(Cnn* cnn,const char* filename,float** inputdata);

#endif
