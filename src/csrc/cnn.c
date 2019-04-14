//
//CNN Functions and the CNN Architectrue
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>  //
#include <time.h>  //Random seed
#include <stdint.h>

#include "cnn.h" 

void CnnSetup(Cnn* cnn, MatSize input_size, int16_t output_size)
{
	int16_t map_size = 5;
	cnn->layer_num = 5;      //layers = 5
  
	MatSize temp_input_size;

  //Layer1 Convolution input size: {28,28}
	temp_input_size.columns = input_size.columns;
	temp_input_size.rows = input_size.rows;
	cnn->C1 = InitialConvolutionLayer(temp_input_size.columns, \
	                          temp_input_size.rows, 5, 1, 6);

	//Layer2 Pooling with average. Input size: {24,24}
	temp_input_size.columns = temp_input_size.columns - map_size + 1;
	temp_input_size.rows = temp_input_size.rows - map_size + 1;
	cnn->S2 = InitialPoolingLayer(temp_input_size.columns, \
	                              temp_input_size.rows, 2, 6, 6, AVG_POOLING);

	//Layer3 Convolution input size: {12,12}
	temp_input_size.columns = temp_input_size.columns / 2;
	temp_input_size.rows = temp_input_size.rows / 2;
	cnn->C3 = InitialConvolutionLayer(temp_input_size.columns,  \
	                          temp_input_size.rows, 5, 6, 12);

	//Layer4 Pooling with average. Input size: {8,8}
	temp_input_size.columns = temp_input_size.columns - map_size + 1;
	temp_input_size.rows = temp_input_size.rows - map_size + 1;
	cnn->S4 = InitialPoolingLayer(temp_input_size.columns, 
	                              temp_input_size.rows, 2, 12, 12, AVG_POOLING);
	
	//Layer5 Output layer. Input size: {4,4}
	temp_input_size.columns = temp_input_size.columns / 2;
	temp_input_size.rows = temp_input_size.rows / 2;
	cnn->O5 = InitOutputLayer(temp_input_size.columns * temp_input_size.rows*12,\
	                          output_size);

	cnn->e = (int16_t*)calloc(cnn->O5->output_num, sizeof(int16_t));
}

ConvolutionLayer* InitialConvolutionLayer(int16_t input_width, int16_t input_height, \
          int16_t map_size, int16_t input_channels, int16_t output_channels)
{
	ConvolutionLayer* cov_layer = (ConvolutionLayer*)malloc(sizeof(ConvolutionLayer));

	cov_layer->input_height = input_height;
	cov_layer->input_width  = input_width;
	cov_layer->map_size     = map_size;   //Kernel Featrue map size

	cov_layer->input_channels  = input_channels;
	cov_layer->output_channels = output_channels;

	cov_layer->is_full_connect = true; //
	
	srand((unsigned)time(NULL));
  
	//Define & initial weight
	cov_layer->map_data = (int16_t****)malloc(input_channels * sizeof(int16_t***));

	for(int32_t i=0; i<input_channels; i++)
	{
		cov_layer->map_data[i] = (int16_t***)malloc(output_channels * sizeof(int16_t**));
		for(int32_t j=0; j<output_channels; j++)
		{
			cov_layer->map_data[i][j] = (int16_t**)malloc(map_size * sizeof(int16_t*));
			for(int32_t r=0; r<map_size; r++)
			{
				cov_layer->map_data[i][j][r] = (int16_t*)malloc(map_size * sizeof(int16_t));
				for(int32_t c=0;c<map_size;c++)
				{
					//[-1,1] -> [-128,127]
					//float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2;
					int16_t randnum = rand()%(uint16_t)256 - 128;

					cov_layer->map_data[i][j][r][c] = randnum * 3 / \
					      (map_size * map_size * (input_channels + output_channels));
				}
			}
		}
	}

  //Define  dmap_data for Backward Propagation
	cov_layer->dmap_data = (int16_t****)malloc(input_channels * sizeof(int16_t***));

	for(int32_t i=0; i<input_channels; i++)
	{
		cov_layer->dmap_data[i] = (int16_t***)malloc(output_channels * sizeof(int16_t**));
		
		for(int32_t j=0; j<output_channels; j++)
		{
			cov_layer->dmap_data[i][j] = (int16_t**)malloc(map_size *  sizeof(int16_t*));
			
			for(int32_t r=0;r<map_size;r++)
			{
				cov_layer->dmap_data[i][j][r] = (int16_t*)calloc(map_size,sizeof(int16_t));
			}
		}
	}

	cov_layer -> bias_data=(int16_t*)calloc(output_channels, sizeof(int16_t));

  //Define output data arrary
	int16_t output_width = input_width   - map_size + 1;
	int16_t output_height = input_height - map_size + 1;

	cov_layer->d = (int16_t***)malloc(output_channels*sizeof(int16_t**));
	cov_layer->v = (int16_t***)malloc(output_channels*sizeof(int16_t**));
	cov_layer->y = (int16_t***)malloc(output_channels*sizeof(int16_t**));

	for(int j=0; j<output_channels; j++)
	{
		cov_layer->d[j] = (int16_t**)malloc(output_height * sizeof(int16_t*));
		cov_layer->v[j] = (int16_t**)malloc(output_height * sizeof(int16_t*));
		cov_layer->y[j] = (int16_t**)malloc(output_height * sizeof(int16_t*));
		
		for(int r=0; r<output_height; r++)
		{
			cov_layer->d[j][r] = (int16_t*)calloc(output_width, sizeof(int16_t));
			cov_layer->v[j][r] = (int16_t*)calloc(output_width, sizeof(int16_t));
			cov_layer->y[j][r] = (int16_t*)calloc(output_width, sizeof(int16_t));
		}
	}

	return cov_layer;
}

PoolingLayer* InitialPoolingLayer(int16_t input_width, int16_t input_height, \
                                  int16_t map_size, int16_t input_channels, \
																	int16_t output_channels, int16_t pooling_type)
{
	PoolingLayer* poolL = (PoolingLayer*)malloc(sizeof(PoolingLayer));

	poolL->input_height    = input_height;
	poolL->input_width     = input_width;
	poolL->map_size        = map_size;
	poolL->input_channels  = input_channels;
	poolL->output_channels = output_channels;
	poolL->pooling_type    = pooling_type; 

	poolL->bias_data      = (int16_t*)calloc(output_channels,sizeof(int16_t));

	int16_t output_width    = input_width/map_size;
	int16_t output_height   = input_height/map_size;

	poolL->d               = (int16_t***)malloc(output_channels*sizeof(int16_t**));
	poolL->y               = (int16_t***)malloc(output_channels*sizeof(int16_t**));
 
	for(int16_t j=0; j<output_channels; j++)
	{
		poolL->d[j]           = (int16_t**)malloc(output_height * sizeof(int16_t*));
		poolL->y[j]           = (int16_t**)malloc(output_height * sizeof(int16_t*));
		for(int32_t r=0; r<output_height; r++)
		{
			poolL->d[j][r]      = (int16_t*)calloc(output_width,sizeof(int16_t));
			poolL->y[j][r]      =(int16_t*)calloc(output_width,sizeof(int16_t));
		}
	}

	return poolL;
}

OutputLayer* InitOutputLayer(int16_t input_num,int16_t output_num)
{
	OutputLayer* outLayer = (OutputLayer*)malloc(sizeof(OutputLayer));

	outLayer->input_num   = input_num;
	outLayer->output_num  = output_num;

	outLayer->bias_data  = (int16_t*)calloc(output_num, sizeof(int16_t));

	outLayer->d           = (int16_t*)calloc(output_num, sizeof(int16_t));
	outLayer->v           = (int16_t*)calloc(output_num, sizeof(int16_t));
	outLayer->y           = (int16_t*)calloc(output_num, sizeof(int16_t));

	//
	outLayer->weight_data = (int16_t**)malloc(output_num * sizeof(int16_t*)); //

	srand((unsigned)time(NULL));

	for(int32_t i=0; i<output_num; i++)
	{
		outLayer->weight_data[i] = (int16_t*)malloc(input_num*sizeof(int16_t));
		
		for(int32_t j=0; j<input_num; j++)
		{
			//[-1,1] -> [-128,127]
			//float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2;
			int16_t randnum = rand()%(uint16_t)256 - 128;
			outLayer->weight_data[i][j] = randnum * 2 / (input_num + output_num);
		}
	}

	outLayer->is_full_connect = true;

	return outLayer;
}


//Vector Max
int32_t VectorMaxIndex(int8_t* vector, int32_t vector_length)
{
	int8_t max_num   = 0;
	int32_t max_index = 0;

	for(int32_t i=0; i<vector_length; i++)
	{
		if(max_num < vector[i])
		{
			max_num   = vector[i];
			max_index = i;
		}
	}
	return max_index;
}

int32_t CnnTest(Cnn* cnn, ImageArray input_data, \
              LabelArray output_data, int32_t test_num)
{
	int32_t incorrect_num=0; 

	for(int32_t n=0; n<test_num; n++)
	{
		printf("=== Testing ... : %d/%d%c", n, test_num, (char)13);

		CnnFF(cnn, input_data->image_point[n].image_data);

		if(VectorMaxIndex(cnn->O5->y, cnn->O5->output_num) != \
		   VectorMaxIndex(output_data->label_point[n].label_data, cnn->O5->output_num))
		{
			incorrect_num ++;
		}

		CnnClear(cnn);
	}

	return incorrect_num;
}

//Save CNN
void SaveCnn(Cnn* cnn, const char* filename)
{
	FILE  *file_point=NULL;
	file_point=fopen(filename,"wb");

	if(file_point==NULL)
		printf("[-] <SaveCnn> Open write file failed <%s>\n",filename);

	for(int32_t i=0; i<cnn->C1->input_channels; i++)
		for(int32_t j=0; j<cnn->C1->output_channels; j++)
			for(int32_t m=0; m<cnn->C1->map_size; m++)
				fwrite(cnn->C1->map_data[i][j][m], sizeof(int16_t), \
				       cnn->C1->map_size, file_point);

	fwrite(cnn->C1->bias_data, sizeof(int16_t), 
	       cnn->C1->output_channels, file_point);

	for(int i=0; i<cnn->C3->input_channels; i++)
		for(int j=0; j<cnn->C3->output_channels; j++)
			for(int m=0; m<cnn->C3->map_size; m++)
				fwrite(cnn->C3->map_data[i][j][m], sizeof(int16_t), \
				       cnn->C3->map_size, file_point);

	fwrite(cnn->C3->bias_data, sizeof(int16_t), \
	       cnn->C3->output_channels, file_point);

	for(int i=0; i<cnn->O5->output_num; i++)
		fwrite(cnn->O5->weight_data[i], sizeof(int16_t), \
		       cnn->O5->input_num, file_point);

	fwrite(cnn->O5->bias_data, sizeof(int16_t), \
	       cnn->O5->output_num, file_point);

	fclose(file_point);
}

//import cnn from file
void ImportCnn(Cnn* cnn, const char* filename)
{
	FILE  *file_point=NULL;
	file_point = fopen(filename,"rb");

	if(file_point==NULL)
		printf("[-] <ImportCnn> Open file failed! <%s>\n",filename);

	for(int32_t i=0; i<cnn->C1->input_channels; i++)
		for(int32_t j=0; j<cnn->C1->output_channels; j++)
			for(int32_t r=0; r<cnn->C1->map_size; r++)
				for(int32_t c=0; c<cnn->C1->map_size; c++)
				{
					int16_t* in = (int16_t*)malloc(sizeof(int16_t));
					fread(in,sizeof(int16_t),1,file_point);
					cnn->C1->map_data[i][j][r][c] = *in;
				}

	for(int32_t i=0; i<cnn->C1->output_channels; i++)
		fread(&cnn->C1->bias_data[i], sizeof(int16_t), 1, file_point);

	for(int32_t i=0;i<cnn->C3->input_channels;i++)
		for(int32_t j=0;j<cnn->C3->output_channels;j++)
			for(int32_t r=0;r<cnn->C3->map_size;r++)
				for(int32_t c=0;c<cnn->C3->map_size;c++)
					fread(&cnn->C3->map_data[i][j][r][c], sizeof(int16_t), 1, file_point);

	for(int32_t i=0; i<cnn->C3->output_channels; i++)
		fread(&cnn->C3->bias_data[i], sizeof(int16_t), 1, file_point);

	for(int32_t i=0; i<cnn->O5->output_num; i++)
		for(int32_t j=0; j<cnn->O5->input_num; j++)
			fread(&cnn->O5->weight_data[i][j], sizeof(int16_t), 1, file_point);

	for(int32_t i=0; i<cnn->O5->output_num; i++)
		fread(&cnn->O5->bias_data[i], sizeof(int16_t), 1, file_point);

	fclose(file_point);
}

void CnnTrain(Cnn* cnn,	ImageArray input_data, LabelArray output_data, \
              TrainOptions opts, int32_t num_trains)
{
	cnn->L = (int16_t*)malloc(num_trains * sizeof(int16_t));

	for(int32_t epoch = 0; epoch<opts.numepochs; epoch++)
	{
		printf("[+] --Training ... %d/%d\n", epoch, opts.numepochs);
		
		for(int32_t train=0; train<num_trains; train++)
		{
			printf("[+] Training Process: %d / %d%c", train, num_trains, (char)13);

			CnnFF(cnn, input_data->image_point[train].image_data); //one image
			CnnBP(cnn, output_data->label_point[train].label_data); //one label

      #if SAVECNNDATA
				char* filedir="output/cnn_data/";
				const char* filename=CombineStrings(filedir, \
				                     CombineStrings(IntToChar(n), ".cnn"));

				SaveCnnData(cnn,filename,input_data->image_point[n].ImgData);
			#endif

			CnnApplyGradients(cnn,opts,input_data->image_point[train].image_data);

			CnnClear(cnn);

			int16_t l=0;
			for(int32_t i=0; i<cnn->O5->output_num; i++)
				l = l + cnn->e[i] * cnn->e[i];  //variance l + error^2

			if(train==0)
				cnn->L[train] = l/2;

			else
				cnn->L[train] = cnn->L[train-1] * (int16_t)99 / (int16_t)100 + l/(int16_t)2/(int16_t)100;
		}
	}
}

void CnnFF(Cnn* cnn, uint8_t** input_data)
{
	//First C1 Convolution
	MatSize map_size    = {cnn->C1->map_size,    cnn->C1->map_size};
	MatSize input_size  = {cnn->C1->input_width, cnn->C1->input_height};
	MatSize output_size = {cnn->S2->input_width, cnn->S2->input_height};

	for(int32_t i=0; i<(cnn->C1->output_channels); i++)
	{
		for(int32_t j=0; j<(cnn->C1->input_channels); j++)
		{
			int16_t** mapout = MatConvolution(cnn->C1->map_data[j][i], map_size, \
			                      input_data, input_size, VALID);

			MatAdd(cnn->C1->v[i], cnn->C1->v[i], output_size,mapout, output_size);
			
			for(int32_t row=0; row<output_size.rows; row++)
				free(mapout[row]);
			
			free(mapout);
		}

		for(int row=0; row<output_size.rows; row++)
			for(int col=0; col<output_size.columns; col++)
				cnn->C1->y[i][row][col] = ActivationSigma(cnn->C1->v[i][row][col], \
				                                      cnn->C1->bias_data[i]);
	}


	//S2 Pooling 
	output_size.columns = cnn->C3->input_width;
	output_size.rows = cnn->C3->input_height;

	input_size.columns = cnn->S2->input_width;
	input_size.rows = cnn->S2->input_height;

	for(int i=0; i<(cnn->S2->output_channels); i++)
	{
		if(cnn->S2->pooling_type == AVG_POOLING)
			AvgPooling(cnn->S2->y[i],output_size,cnn->C1->y[i],
			                         input_size,cnn->S2->map_size);
	}

	output_size.columns = cnn->S4->input_width;
	output_size.rows = cnn->S4->input_height;
	input_size.columns = cnn->C3->input_width;
	input_size.rows = cnn->C3->input_height;
	map_size.columns = cnn->C3->map_size;
	map_size.rows = cnn->C3->map_size;

	for(int i=0; i<(cnn->C3->output_channels); i++)
	{
		for(int j=0; j<(cnn->C3->input_channels); j++)
		{
			int16_t** mapout = MatConvolution(cnn->C3->map_data[j][i], map_size, 
			                        cnn->S2->y[j], input_size, VALID);
			MatAdd(cnn->C3->v[i], cnn->C3->v[i], output_size, mapout, output_size);
			
			for(int r=0; r<output_size.rows; r++)
			  free(mapout[r]);
			
			free(mapout);
		}
		for(int r=0; r<output_size.rows; r++)
			for(int c=0; c<output_size.columns; c++)
				cnn->C3->y[i][r][c] = ActivationSigma(cnn->C3->v[i][r][c], \
				                                    cnn->C3->bias_data[i]);
	}

	input_size.columns = cnn->S4->input_width;
	input_size.rows = cnn->S4->input_height;
	output_size.columns = input_size.columns / cnn->S4->map_size;
	output_size.rows = input_size.rows / cnn->S4->map_size;
	for(int i=0;i<(cnn->S4->output_channels);i++)
	{
		if(cnn->S4->pooling_type == AVG_POOLING)
			AvgPooling(cnn->S4->y[i], output_size, cnn->C3->y[i], \
			                          input_size, cnn->S4->map_size);
	}

	int16_t* O5inData=(int16_t*)malloc((cnn->O5->input_num)*sizeof(int16_t)); 

	for(int32_t i=0; i<(cnn->S4->output_channels); i++) 
		for(int32_t row=0; row<output_size.rows; row++)
			for(int32_t col=0; col<output_size.columns; col ++)
				O5inData[i * output_size.rows * output_size.columns + \
				         row * output_size.columns + col] = cnn->S4->y[i][row][col];

	MatSize nnSize = {cnn->O5->input_num, cnn->O5->output_num};
	nnff(cnn->O5->v, O5inData, cnn->O5->weight_data, cnn->O5->bias_data, nnSize);
	
	for(int32_t i=0; i<cnn->O5->output_num; i++)
		cnn->O5->y[i] = ActivationSigma(cnn->O5->v[i], cnn->O5->bias_data[i]);

	free(O5inData);
}

//
int16_t ActivationSigma(int16_t input,int16_t bias) //sigma activatiion function
{
	int16_t temp = input + bias;
	//return (temp>0)?temp:0;
	return (temp>0)?1:0;
	//return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

int16_t ActivationReLU(int16_t input, int16_t bias)
{
	int16_t temp = input + bias;
	return (temp>0)?temp:0;
}

void AvgPooling(int16_t** output, MatSize output_size, int16_t** input, \
                               MatSize input_size, int16_t map_size)
{
	int output_width  = input_size.columns / map_size;
	int output_height = input_size.rows / map_size;

	if(output_size.columns != output_width || output_size.rows != output_height)
	  printf("[-] ERROR: Output size is wrong! <AvgPooling> \n");

	for(int32_t i=0; i<output_height; i++)
		for(int32_t j=0; j<output_width; j++)
		{
			int16_t sum=0;
			for(int32_t m=i*map_size; m<i*map_size+map_size; m++)
				for(int32_t n=j*map_size; n<j*map_size+map_size; n++)
					sum = sum+input[m][n];

			output[i][j] = sum/(map_size * map_size);
		}
}

// 
int16_t VectorMultiply(int16_t* vector1, int16_t* vector2, int32_t vector_length)//
{
	int16_t result=0;

	for(int32_t i=0; i<vector_length; i++)
		result = result + vector1[i]*vector2[i];

	return result;
}

//Vector * Mat + bias
void nnff(int16_t* output, int16_t* input, int16_t** wdata, int16_t* bias, MatSize nnSize)
{
	int width  = nnSize.columns;
	int height = nnSize.rows;

	for(int32_t i=0; i<height; i++)
		output[i] = VectorMultiply(input, wdata[i], width) + bias[i];
}

float sigma_derivation(float y)
{ //
	return y*(1-y); 
}

//
void CnnBP(Cnn* cnn, int8_t* output_data)
{
	for(int32_t i=0; i<cnn->O5->output_num; i++)
		cnn->e[i] = cnn->O5->y[i] - output_data[i];

	for(int32_t i=0; i<cnn->O5->output_num; i++)
		cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);

	MatSize output_size = {cnn->S4->input_width / cnn->S4->map_size, \
	                       cnn->S4->input_height / cnn->S4->map_size};

	for(int32_t i=0; i<cnn->S4->output_channels; i++)
		for(int32_t row=0; row<output_size.rows; row++)
			for(int32_t col=0; col<output_size.columns; col++)
				for(int32_t j=0; j<cnn->O5->output_num; j++)
				{
					int32_t wInt = i * output_size.columns * output_size.rows + \
					           row * output_size.columns + col;

					cnn->S4->d[i][row][col] = cnn->S4->d[i][row][col] + cnn->O5->d[j] * \
					                          cnn->O5->weight_data[j][wInt];
				}

	int mapdata = cnn->S4->map_size;
	MatSize S4dSize;
	S4dSize.columns = cnn->S4->input_width  / mapdata;
	S4dSize.rows    = cnn->S4->input_height /  mapdata;

	for(int i=0; i<cnn->C3->output_channels; i++)
	{
		int16_t** C3e = MatUpSample(cnn->S4->d[i], S4dSize, \
		                        cnn->S4->map_size, cnn->S4->map_size);

		for(int row=0; row<cnn->S4->input_height; row++)
			for(int col=0; col<cnn->S4->input_width; col++)
			{
				cnn->C3->d[i][row][col] = C3e[row][col] * \
				  sigma_derivation(cnn->C3->y[i][row][col])/ \
					(float)(cnn->S4->map_size * cnn->S4->map_size);
			}

		for(int row=0; row<cnn->S4->input_height; row++)
			free(C3e[row]);

		free(C3e);
	}

	output_size.columns = cnn->C3->input_width;
	output_size.rows    = cnn->C3->input_height;
	MatSize input_size  = {cnn->S4->input_width, cnn->S4->input_height};
	MatSize map_size    = {cnn->C3->map_size, cnn->C3->map_size};
	
	for(int i=0; i<cnn->S2->output_channels; i++)
	{
		for(int j=0; j<cnn->C3->output_channels; j++)
		{
			int16_t** corr = MatCorrelation(cnn->C3->map_data[i][j], map_size, 
			                              cnn->C3->d[j], input_size, FULL);
			MatAdd(cnn->S2->d[i], cnn->S2->d[i], output_size, corr, output_size);
			for(int row=0; row<output_size.rows; row++)
				free(corr[row]);
			free(corr);
		}
		/*
		for(r=0;r<cnn->C3->input_height;r++)
			for(c=0;c<cnn->C3->input_width;c++)
		*/
	}


	mapdata = cnn->S2->map_size;

	MatSize S2dSize;
	S2dSize.columns = cnn->S2->input_width / mapdata;
  S2dSize.rows = cnn->S2->input_height / mapdata;

	for(int i=0; i<cnn->C1->output_channels; i++)
	{
		int16_t** C1e = MatUpSample(cnn->S2->d[i], S2dSize, \
		                          cnn->S2->map_size, cnn->S2->map_size);
		for(int row=0; row<cnn->S2->input_height; row++)
			for(int col=0; col<cnn->S2->input_width; col++)
			{
				cnn->C1->d[i][row][col] = C1e[row][col] * \
				     sigma_derivation(cnn->C1->y[i][row][col]) / \
						 (float)(cnn->S2->map_size * cnn->S2->map_size);
			}
		for(int row=0; row<cnn->S2->input_height; row++)
			free(C1e[row]);
			
		free(C1e);
	}
}

//Apply Gradient
void CnnApplyGradients(Cnn* cnn, TrainOptions opts, int16_t** input_data)
{
	MatSize dSize = {cnn->S2->input_height, cnn->S2->input_width};
	MatSize ySize = {cnn->C1->input_height, cnn->C1->input_width};
	MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};

	for(int i=0; i<cnn->C1->output_channels; i++)
	{
		for(int j=0; j<cnn->C1->input_channels; j++)
		{
			int16_t** flipinput_data = MatRotate180(input_data, ySize);
			int16_t** C1dk = MatConvolution(cnn->C1->d[i], dSize, flipinput_data, ySize,VALID);
			MatMultiplyImmediate(C1dk, C1dk, map_size, -1*opts.alpha);
			MatAdd(cnn->C1->map_data[j][i], cnn->C1->map_data[j][i], \
			                                map_size, C1dk, map_size);
			for(int row=0; row<(dSize.rows-(ySize.rows-1)); row++)
				free(C1dk[row]);

			free(C1dk);

			for(int row=0; row<ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C1->bias_data[i] = cnn->C1->bias_data[i] - \
		                         opts.alpha * MatSum(cnn->C1->d[i], dSize);
	}

	dSize.columns = cnn->S4->input_width;
	dSize.rows = cnn->S4->input_height;
	ySize.columns = cnn->C3->input_width;
	ySize.rows = cnn->C3->input_height;

	map_size.columns = cnn->C3->map_size;
	map_size.rows = cnn->C3->map_size;
	
	for(int i=0; i<cnn->C3->output_channels; i++)
	{
		for(int j=0; j<cnn->C3->input_channels; j++)
		{
			int16_t** flipinput_data = MatRotate180(cnn->S2->y[j], ySize);
			int16_t** C3dk = MatConvolution(cnn->C3->d[i], dSize, \
			                               flipinput_data, ySize, VALID);
			
			MatMultiplyImmediate(C3dk, C3dk, map_size, -1*opts.alpha);
			MatAdd(cnn->C3->map_data[j][i], cnn->C3->map_data[j][i], \
			                                map_size, C3dk, map_size);
			for(int row=0; row<(dSize.rows-(ySize.rows-1)); row++)
				free(C3dk[row]);

			free(C3dk);

			for(int row=0; row<ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C3->bias_data[i] = cnn->C3->bias_data[i] - \
		                         opts.alpha * MatSum(cnn->C3->d[i], dSize);
	}

	int16_t* O5inData = (int16_t*)malloc((cnn->O5->input_num) * sizeof(int16_t));

	MatSize output_size;
	output_size.columns = cnn->S4->input_width/cnn->S4->map_size;
	output_size.rows = cnn->S4->input_height/cnn->S4->map_size;
	
	for(int32_t i=0; i<(cnn->S4->output_channels); i++)
		for(int32_t row=0; row<output_size.rows; row++)
			for(int32_t col=0; col<output_size.columns; col++)
				O5inData[i*output_size.rows * output_size.columns + \
				         row*output_size.columns + col] = cnn->S4->y[i][row][col];

	for(int32_t j=0; j<cnn->O5->output_num; j++)
	{
		for(int32_t i=0; i<cnn->O5->input_num; i++)
			cnn->O5->weight_data[j][i] = cnn->O5->weight_data[j][i] - \
			                       opts.alpha * cnn->O5->d[j] * O5inData[i];
		
		cnn->O5->bias_data[j] = cnn->O5->bias_data[j] - \
		                         opts.alpha * cnn->O5->d[j];
	}
	free(O5inData);
}

void CnnClear(Cnn* cnn)
{
	for(int32_t j=0; j<cnn->C1->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->S2->input_height; row++)
		{
			for(int32_t col=0; col<cnn->S2->input_width; col++)
			{
				cnn->C1->d[j][row][col] =  0;
				cnn->C1->v[j][row][col] =  0;
				cnn->C1->y[j][row][col] =  0;
			}
		}
	}

	for(int j=0; j<cnn->S2->output_channels; j++)
	{
		for(int row=0;row<cnn->C3->input_height; row++)
		{
			for(int col=0;col<cnn->C3->input_width; col++)
			{
				cnn->S2->d[j][row][col] = 0;
				cnn->S2->y[j][row][col] = 0;
			}
		}
	}

	for(int32_t j=0; j<cnn->C3->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->S4->input_height; row++)
		{
			for(int32_t col=0; col<cnn->S4->input_width; col++)
			{
				cnn->C3->d[j][row][col] = 0;
				cnn->C3->v[j][row][col] = 0;
				cnn->C3->y[j][row][col] = 0;
			}
		}
	}

	for(int j=0; j<cnn->S4->output_channels; j++)
	{
		for(int row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			for(int col=0; col<cnn->S4->input_width/cnn->S4->map_size; col++)
			{
				cnn->S4->d[j][row][col] = 0;
				cnn->S4->y[j][row][col] = 0;
			}
		}
	}

	for(int n=0; n<cnn->O5->output_num; n++){
		cnn->O5->d[n] = 0;
		cnn->O5->v[n] = 0;
		cnn->O5->y[n] = 0;
	}
}

void SaveCnnData(Cnn* cnn, const char* filename, int16_t** inputdata)
{
	FILE  *file_point = NULL;
	file_point = fopen(filename,"wb");

	if(file_point==NULL)
		printf("[-] <SvaeCNNData> Open Write file failed! <%s>\n", filename);

	for(int32_t i=0; i<cnn->C1->input_height; i++)
		fwrite(inputdata[i], sizeof(int16_t), cnn->C1->input_width, file_point);

	for(int32_t i=0; i<cnn->C1->input_channels; i++)
		for(int32_t j=0; j<cnn->C1->output_channels; j++)
			for(int32_t s=0; s<cnn->C1->map_size; s++)
				fwrite(cnn->C1->map_data[i][j][s], sizeof(int16_t), \
				       cnn->C1->map_size, file_point);

	fwrite(cnn->C1->bias_data, sizeof(int16_t), \
	       cnn->C1->output_channels, file_point);

	for(int32_t j=0; j<cnn->C1->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->v[j][row], sizeof(int16_t), \
			       cnn->S2->input_width, file_point);
		}

		for(int32_t row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->d[j][row], sizeof(int16_t), \
			       cnn->S2->input_width, file_point);
		}

		for(int32_t row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->y[j][row], sizeof(int16_t), 
			       cnn->S2->input_width, file_point);
		}
	}

	for(int32_t j=0; j<cnn->S2->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->d[j][row], sizeof(int16_t), \
			       cnn->C3->input_width, file_point);
		}
		for(int32_t row=0; row<cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->y[j][row], sizeof(int16_t), \
			       cnn->C3->input_width, file_point);
		}
	}

	for(int32_t i=0; i<cnn->C3->input_channels; i++)
		for(int32_t j=0; j<cnn->C3->output_channels; j++)
			for(int32_t row=0; row<cnn->C3->map_size; row++)
				fwrite(cnn->C3->map_data[i][j][row], sizeof(int16_t), \
				       cnn->C3->map_size, file_point);

	fwrite(cnn->C3->bias_data, sizeof(int16_t), \
	       cnn->C3->output_channels, file_point);

	for(int32_t j=0; j<cnn->C3->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->v[j][row], sizeof(int16_t), 
			       cnn->S4->input_width, file_point);
		}

		for(int32_t row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->d[j][row], sizeof(int16_t), 
			       cnn->S4->input_width, file_point);
		}

		for(int32_t row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->y[j][row], sizeof(int16_t), \
			       cnn->S4->input_width, file_point);
		}
	}

	for(int32_t j=0; j<cnn->S4->output_channels; j++)
	{
		for(int32_t row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->d[j][row], sizeof(int16_t), \
			       cnn->S4->input_width/cnn->S4->map_size, file_point);
		}

		for(int32_t row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->y[j][row], sizeof(int16_t), \
			       cnn->S4->input_width/cnn->S4->map_size, file_point);
		}
	}

	for(int32_t i=0; i<cnn->O5->output_num; i++)
		fwrite(cnn->O5->weight_data[i], sizeof(int16_t), cnn->O5->input_num, file_point);

	fwrite(cnn->O5->bias_data, sizeof(int16_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->v, sizeof(int16_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->d, sizeof(int16_t), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->y, sizeof(int16_t), cnn->O5->output_num, file_point);

	fclose(file_point);
}
