//
//CNN Functions and the CNN Architectrue
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>  //
#include <time.h>  //Random seed
#include "cnn.h" 

void CnnSetup(Cnn* cnn,MatSize input_size,int output_size)
{
	int map_size = 5;
	cnn->layer_num = 5;      //layers = 5
  
	MatSize temp_input_size;

  //Layer1 Cov input size: {28,28}
	temp_input_size.columns = input_size.columns;
	temp_input_size.rows = input_size.rows;
	cnn->C1 = InitialCovLayer(temp_input_size.columns, \
	                          temp_input_size.rows, 5, 1, 6);

	//Layer2 Pooling input size: {24,24}
	temp_input_size.columns = temp_input_size.columns - map_size + 1;
	temp_input_size.rows = temp_input_size.rows - map_size + 1;
	cnn->S2 = InitialPoolingLayer(temp_input_size.columns, \
	                              temp_input_size.rows, 2, 6, 6, AVG_POOLING);

	//Layer3 Cov input size: {12,12}
	temp_input_size.columns = temp_input_size.columns / 2;
	temp_input_size.rows = temp_input_size.rows / 2;
	cnn->C3 = InitialCovLayer(temp_input_size.columns,  \
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

	cnn->e = (float*)calloc(cnn->O5->output_num, sizeof(float));
}

CovLayer* InitialCovLayer(int input_width, int input_height, int map_size, \
                          int input_channels, int output_channels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->input_height=input_height;
	covL->input_width=input_width;
	covL->map_size=map_size;

	covL->input_channels=input_channels;
	covL->output_channels=output_channels;

	covL->is_full_connect=true; //
	int i,j,c,r;
	srand((unsigned)time(NULL));
	covL->map_data=(float****)malloc(input_channels*sizeof(float***));
	for(i=0;i<input_channels;i++)
	{
		covL->map_data[i]=(float***)malloc(output_channels*sizeof(float**));
		for(j=0;j<output_channels;j++)
		{
			covL->map_data[i][j]=(float**)malloc(map_size*sizeof(float*));
			for(r=0;r<map_size;r++)
			{
				covL->map_data[i][j][r]=(float*)malloc(map_size*sizeof(float));
				for(c=0;c<map_size;c++)
				{
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
					covL->map_data[i][j][r][c] = randnum * (sqrt((float)6.0 / (float) \
					      (map_size * map_size * (input_channels + output_channels))));
				}
			}
		}
	}
	covL->dmap_data=(float****)malloc(input_channels*sizeof(float***));
	for(i=0;i<input_channels;i++)
	{
		covL->dmap_data[i]=(float***)malloc(output_channels*sizeof(float**));
		for(j=0;j<output_channels;j++)
		{
			covL->dmap_data[i][j]=(float**)malloc(map_size*sizeof(float*));
			for(r=0;r<map_size;r++)
			{
				covL->dmap_data[i][j][r]=(float*)calloc(map_size,sizeof(float));
			}
		}
	}

	covL->basic_data=(float*)calloc(output_channels,sizeof(float));

	int outW=input_width-map_size+1;
	int outH=input_height-map_size+1;


	covL->d=(float***)malloc(output_channels*sizeof(float**));
	covL->v=(float***)malloc(output_channels*sizeof(float**));
	covL->y=(float***)malloc(output_channels*sizeof(float**));
	for(j=0;j<output_channels;j++)
	{
		covL->d[j]=(float**)malloc(outH*sizeof(float*));
		covL->v[j]=(float**)malloc(outH*sizeof(float*));
		covL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++)
		{
			covL->d[j][r]=(float*)calloc(outW,sizeof(float));
			covL->v[j][r]=(float*)calloc(outW,sizeof(float));
			covL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return covL;
}

PoolingLayer* InitialPoolingLayer(int input_width, int input_height, \
      int map_size, int input_channels, int output_channels, int pooling_type)
{
	PoolingLayer* poolL = (PoolingLayer*)malloc(sizeof(PoolingLayer));

	poolL->input_height=input_height;
	poolL->input_width=input_width;
	poolL->map_size=map_size;
	poolL->input_channels=input_channels;
	poolL->output_channels=output_channels;
	poolL->pooling_type=pooling_type; 

	poolL->basic_data=(float*)calloc(output_channels,sizeof(float));

	int outW=input_width/map_size;
	int outH=input_height/map_size;

	int j,r;
	poolL->d=(float***)malloc(output_channels*sizeof(float**));
	poolL->y=(float***)malloc(output_channels*sizeof(float**));
	for(j=0;j<output_channels;j++)
	{
		poolL->d[j]=(float**)malloc(outH*sizeof(float*));
		poolL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++)
		{
			poolL->d[j][r]=(float*)calloc(outW,sizeof(float));
			poolL->y[j][r]=(float*)calloc(outW,sizeof(float));
		}
	}

	return poolL;
}

OutputLayer* InitOutputLayer(int input_num,int output_num)
{
	OutputLayer* outL=(OutputLayer*)malloc(sizeof(OutputLayer));

	outL->input_num=input_num;
	outL->output_num=output_num;


	outL->basic_data=(float*)calloc(output_num,sizeof(float));

	outL->d=(float*)calloc(output_num,sizeof(float));
	outL->v=(float*)calloc(output_num,sizeof(float));
	outL->y=(float*)calloc(output_num,sizeof(float));

	//
	outL->wData=(float**)malloc(output_num*sizeof(float*)); //

	srand((unsigned)time(NULL));
	for(int i=0;i<output_num;i++)
	{
		outL->wData[i]=(float*)malloc(input_num*sizeof(float));
		for(int j=0;j<input_num;j++)
		{
			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; //
			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(input_num+output_num));
		}
	}

	outL->is_full_connect=true;

	return outL;
}

int vecmaxIndex(float* vec, int veclength)
{
	int i;
	float maxnum=-1.0;
	int maxIndex=0;
	for(i=0;i<veclength;i++)
	{
		if(maxnum<vec[i])
		{
			maxnum=vec[i];
			maxIndex=i;
		}
	}
	return maxIndex;
}

float CnnTest(Cnn* cnn, ImageArray input_data, \
              LabelArray output_data,int test_num)
{
	int n=0;
	int incorrect_num=0; 

	for(n=0; n<test_num; n++)
	{
		printf("=== Testing ... : %d/%d%c",n,test_num,(char)13);

		CnnFF(cnn, input_data->image_point[n].image_data);

		if(vecmaxIndex(cnn->O5->y,cnn->O5->output_num) != \
		   vecmaxIndex(output_data->label_point[n].LabelData, cnn->O5->output_num))
		{
			incorrect_num++;
		}
		CnnClear(cnn);
	}
	return (float)incorrect_num/(float)test_num;
}

//Save CNN
void SaveCnn(Cnn* cnn, const char* filename)
{
	FILE  *file_point=NULL;
	file_point=fopen(filename,"wb");
	if(file_point==NULL)
		printf("write file failed\n");

	for(int i=0; i<cnn->C1->input_channels; i++)
		for(int j=0; j<cnn->C1->output_channels; j++)
			for(int m=0; m<cnn->C1->map_size; m++)
				fwrite(cnn->C1->map_data[i][j][m], sizeof(float), \
				       cnn->C1->map_size, file_point);

	fwrite(cnn->C1->basic_data, sizeof(float), 
	       cnn->C1->output_channels, file_point);

	for(int i=0; i<cnn->C3->input_channels; i++)
		for(int j=0; j<cnn->C3->output_channels; j++)
			for(int m=0; m<cnn->C3->map_size; m++)
				fwrite(cnn->C3->map_data[i][j][m], sizeof(float), \
				       cnn->C3->map_size, file_point);

	fwrite(cnn->C3->basic_data, sizeof(float), \
	       cnn->C3->output_channels, file_point);

	for(int i=0; i<cnn->O5->output_num; i++)
		fwrite(cnn->O5->wData[i], sizeof(float), \
		       cnn->O5->input_num, file_point);

	fwrite(cnn->O5->basic_data, sizeof(float), \
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

	int i,j,c,r;

	for(i=0;i<cnn->C1->input_channels;i++)
		for(j=0;j<cnn->C1->output_channels;j++)
			for(r=0;r<cnn->C1->map_size;r++)
				for(c=0;c<cnn->C1->map_size;c++)
				{
					float* in=(float*)malloc(sizeof(float));
					fread(in,sizeof(float),1,file_point);
					cnn->C1->map_data[i][j][r][c]=*in;
				}

	for(i=0;i<cnn->C1->output_channels;i++)
		fread(&cnn->C1->basic_data[i],sizeof(float),1,file_point);

	for(i=0;i<cnn->C3->input_channels;i++)
		for(j=0;j<cnn->C3->output_channels;j++)
			for(r=0;r<cnn->C3->map_size;r++)
				for(c=0;c<cnn->C3->map_size;c++)
				fread(&cnn->C3->map_data[i][j][r][c],sizeof(float),1,file_point);

	for(i=0;i<cnn->C3->output_channels;i++)
		fread(&cnn->C3->basic_data[i],sizeof(float),1,file_point);

	for(i=0;i<cnn->O5->output_num;i++)
		for(j=0;j<cnn->O5->input_num;j++)
			fread(&cnn->O5->wData[i][j],sizeof(float),1,file_point);

	for(i=0;i<cnn->O5->output_num;i++)
		fread(&cnn->O5->basic_data[i],sizeof(float),1,file_point);

	fclose(file_point);
}

void CnnTrain(Cnn* cnn,	ImageArray input_data, LabelArray output_data, \
              TrainOptions opts, int trainNum)
{
	cnn->L=(float*)malloc(trainNum*sizeof(float));
	int e;
	for(e=0;e<opts.numepochs;e++)
	{
		printf("[+] --Training ... %d/%d\n",e,opts.numepochs);
		int n=0;
		for(n=0;n<trainNum;n++)
		{
			printf("[+] Training Process: %d / %d%c",n,trainNum,(char)13);

			CnnFF(cnn,input_data->image_point[n].image_data);
			CnnBP(cnn,output_data->label_point[n].LabelData);

      #if SAVECNNDATA
				char* filedir="output/cnn_data/";
				const char* filename=CombineStrings(filedir, \
				                     CombineStrings(IntToChar(n), ".cnn"));
				SaveCnnData(cnn,filename,input_data->image_point[n].ImgData);
			#endif

			CnnApplyGrads(cnn,opts,input_data->image_point[n].image_data);

			CnnClear(cnn);
			float l=0.0;
			int i;
			for(i=0;i<cnn->O5->output_num;i++)
				l=l+cnn->e[i]*cnn->e[i];
			if(n==0)
				cnn->L[n]=l/(float)2.0;
			else
				cnn->L[n]=cnn->L[n-1]*0.99+0.01*l/(float)2.0;
		}
	}
}

void CnnFF(Cnn* cnn,float** input_data)
{
	int output_sizeW=cnn->S2->input_width;
	int output_sizeH=cnn->S2->input_height;

	MatSize map_size={cnn->C1->map_size,cnn->C1->map_size};
	MatSize input_size={cnn->C1->input_width,cnn->C1->input_height};
	MatSize output_size={cnn->S2->input_width,cnn->S2->input_height};
	for(int i=0;i<(cnn->C1->output_channels);i++)
	{
		for(int j=0;j<(cnn->C1->input_channels);j++)
		{
			float** mapout=MatCov(cnn->C1->map_data[j][i], map_size, \
			                      input_data, input_size, VALID);

			MatAdd(cnn->C1->v[i],cnn->C1->v[i],output_size,mapout,output_size);
			
			for(int row=0; row<output_size.rows; row++)
				free(mapout[row]);
			
			free(mapout);
		}

		for(int row=0; row<output_size.rows; row++)
			for(int col=0; col<output_size.columns; col++)
				cnn->C1->y[i][row][col] = ActivationSigma(cnn->C1->v[i][row][col], \
				                                      cnn->C1->basic_data[i]);
	}

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
			float** mapout = MatCov(cnn->C3->map_data[j][i], map_size, 
			                        cnn->S2->y[j], input_size, VALID);
			MatAdd(cnn->C3->v[i], cnn->C3->v[i], output_size, mapout, output_size);
			
			for(int r=0; r<output_size.rows; r++)
			  free(mapout[r]);
			
			free(mapout);
		}
		for(int r=0; r<output_size.rows; r++)
			for(int c=0; c<output_size.columns; c++)
				cnn->C3->y[i][r][c] = ActivationSigma(cnn->C3->v[i][r][c], \
				                                    cnn->C3->basic_data[i]);
	}

	input_size.columns = cnn->S4->input_width;
	input_size.rows = cnn->S4->input_height;
	output_size.columns = input_size.columns / cnn->S4->map_size;
	output_size.rows = input_size.rows / cnn->S4->map_size;
	for(int i=0;i<(cnn->S4->output_channels);i++)
	{
		if(cnn->S4->pooling_type==AVG_POOLING)
			AvgPooling(cnn->S4->y[i], output_size, cnn->C3->y[i], \
			                          input_size, cnn->S4->map_size);
	}

	float* O5inData=(float*)malloc((cnn->O5->input_num)*sizeof(float)); 

	for(int i=0; i<(cnn->S4->output_channels); i++) 
		for(int row=0; row<output_size.rows; row++)
			for(int col=0; col<output_size.columns; col ++)
				O5inData[i * output_size.rows * output_size.columns + \
				         row * output_size.columns + col] = cnn->S4->y[i][row][col];

	MatSize nnSize = {cnn->O5->input_num, cnn->O5->output_num};
	nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->basic_data, nnSize);
	
	for(int i=0; i<cnn->O5->output_num; i++)
		cnn->O5->y[i] = ActivationSigma(cnn->O5->v[i], cnn->O5->basic_data[i]);

	free(O5inData);
}

//
float ActivationSigma(float input,float bas) //sigma activatiion function
{
	float temp = input + bas;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

void AvgPooling(float** output, MatSize output_size, float** input, \
                               MatSize input_size, int map_size)
{
	int outputW = input_size.columns / map_size;
	int outputH=input_size.rows / map_size;

	if(output_size.columns != outputW || output_size.rows != outputH)
	  printf("[-] ERROR: Output size is wrong! <AvgPooling> \n");

	int i,j,m,n;
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float sum=0.0;
			for(m=i*map_size;m<i*map_size+map_size;m++)
				for(n=j*map_size;n<j*map_size+map_size;n++)
					sum=sum+input[m][n];

			output[i][j]=sum/(float)(map_size*map_size);
		}
}

// 
float vecMulti(float* vec1,float* vec2,int vecL)//
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}

void nnff(float* output,float* input,float** wdata,float* bas,MatSize nnSize)
{
	int w=nnSize.columns;
	int h=nnSize.rows;
	
	int i;
	for(i=0;i<h;i++)
		output[i]=vecMulti(input,wdata[i],w)+bas[i];
}

float sigma_derivation(float y){ //
	return y*(1-y); 
}

void CnnBP(Cnn* cnn,float* output_data)
{
	for(int i=0; i<cnn->O5->output_num; i++)
		cnn->e[i] = cnn->O5->y[i]-output_data[i];

	for(int i=0; i<cnn->O5->output_num; i++)
		cnn->O5->d[i] = cnn->e[i]*sigma_derivation(cnn->O5->y[i]);

	MatSize output_size = {cnn->S4->input_width / cnn->S4->map_size, \
	                       cnn->S4->input_height / cnn->S4->map_size};

	for(int i=0; i<cnn->S4->output_channels; i++)
		for(int row=0; row<output_size.rows; row++)
			for(int col=0; col<output_size.columns; col++)
				for(int j=0; j<cnn->O5->output_num; j++)
				{
					int wInt = i*output_size.columns * output_size.rows + \
					           row*output_size.columns + col;
					cnn->S4->d[i][row][col] = cnn->S4->d[i][row][col] +  \
					                          cnn->O5->d[j] * cnn->O5->wData[j][wInt];
				}

	int mapdata = cnn->S4->map_size;
	MatSize S4dSize;
	S4dSize.columns = cnn->S4->input_width / mapdata;
	S4dSize.rows = cnn->S4->input_height  /  mapdata;

	for(int i=0; i<cnn->C3->output_channels; i++)
	{
		float** C3e=MatUpSample(cnn->S4->d[i], S4dSize, \
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
	output_size.rows = cnn->C3->input_height;
	MatSize input_size = {cnn->S4->input_width, cnn->S4->input_height};
	MatSize map_size = {cnn->C3->map_size, cnn->C3->map_size};
	
	for(int i=0; i<cnn->S2->output_channels; i++)
	{
		for(int j=0; j<cnn->C3->output_channels; j++)
		{
			float** corr = MatCorrelation(cnn->C3->map_data[i][j], map_size, 
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
		float** C1e = MatUpSample(cnn->S2->d[i], S2dSize, \
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

//Apply Grad
void CnnApplyGrads(Cnn* cnn,TrainOptions opts,float** input_data)
{
	MatSize dSize = {cnn->S2->input_height, cnn->S2->input_width};
	MatSize ySize = {cnn->C1->input_height, cnn->C1->input_width};
	MatSize map_size = {cnn->C1->map_size, cnn->C1->map_size};

	for(int i=0; i<cnn->C1->output_channels; i++)
	{
		for(int j=0; j<cnn->C1->input_channels; j++)
		{
			float** flipinput_data = MatRotate180(input_data, ySize);
			float** C1dk = MatCov(cnn->C1->d[i], dSize, flipinput_data, ySize,VALID);
			MatMultifactor(C1dk, C1dk, map_size, -1*opts.alpha);
			MatAdd(cnn->C1->map_data[j][i], cnn->C1->map_data[j][i], \
			                                map_size, C1dk, map_size);
			for(int row=0; row<(dSize.rows-(ySize.rows-1)); row++)
				free(C1dk[row]);

			free(C1dk);

			for(int row=0; row<ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C1->basic_data[i] = cnn->C1->basic_data[i] - \
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
			float** flipinput_data = MatRotate180(cnn->S2->y[j], ySize);
			float** C3dk = MatCov(cnn->C3->d[i], dSize, flipinput_data, ySize, VALID);
			
			MatMultifactor(C3dk, C3dk, map_size, -1.0*opts.alpha);
			MatAdd(cnn->C3->map_data[j][i], cnn->C3->map_data[j][i], \
			                                map_size, C3dk, map_size);
			for(int row=0; row<(dSize.rows-(ySize.rows-1)); row++)
				free(C3dk[row]);

			free(C3dk);

			for(int row=0; row<ySize.rows; row++)
				free(flipinput_data[row]);

			free(flipinput_data);
		}
		cnn->C3->basic_data[i] = cnn->C3->basic_data[i] - \
		                         opts.alpha * MatSum(cnn->C3->d[i], dSize);
	}

	float* O5inData=(float*)malloc((cnn->O5->input_num)*sizeof(float));

	MatSize output_size;
	output_size.columns = cnn->S4->input_width/cnn->S4->map_size;
	output_size.rows = cnn->S4->input_height/cnn->S4->map_size;
	
	for(int i=0; i<(cnn->S4->output_channels); i++)
		for(int row=0; row<output_size.rows; row++)
			for(int col=0; col<output_size.columns; col++)
				O5inData[i*output_size.rows * output_size.columns + \
				         row*output_size.columns + col] = cnn->S4->y[i][row][col];

	for(int j=0; j<cnn->O5->output_num; j++)
	{
		for(int i=0; i<cnn->O5->input_num; i++)
			cnn->O5->wData[j][i] = cnn->O5->wData[j][i] - \
			                       opts.alpha * cnn->O5->d[j] * O5inData[i];
		
		cnn->O5->basic_data[j] = cnn->O5->basic_data[j] - \
		                         opts.alpha * cnn->O5->d[j];
	}
	free(O5inData);
}

void CnnClear(Cnn* cnn)
{
	for(int j=0; j<cnn->C1->output_channels; j++)
	{
		for(int row=0; row<cnn->S2->input_height; row++)
		{
			for(int col=0; col<cnn->S2->input_width; col++)
			{
				cnn->C1->d[j][row][col] = (float)0.0;
				cnn->C1->v[j][row][col] = (float)0.0;
				cnn->C1->y[j][row][col] = (float)0.0;
			}
		}
	}

	for(int j=0; j<cnn->S2->output_channels; j++)
	{
		for(int row=0;row<cnn->C3->input_height; row++)
		{
			for(int col=0;col<cnn->C3->input_width; col++)
			{
				cnn->S2->d[j][row][col] = (float)0.0;
				cnn->S2->y[j][row][col] = (float)0.0;
			}
		}
	}

	for(int j=0; j<cnn->C3->output_channels; j++)
	{
		for(int row=0; row<cnn->S4->input_height; row++)
		{
			for(int col=0; col<cnn->S4->input_width; col++)
			{
				cnn->C3->d[j][row][col] = (float)0.0;
				cnn->C3->v[j][row][col] = (float)0.0;
				cnn->C3->y[j][row][col] = (float)0.0;
			}
		}
	}

	for(int j=0; j<cnn->S4->output_channels; j++)
	{
		for(int row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			for(int col=0; col<cnn->S4->input_width/cnn->S4->map_size; col++)
			{
				cnn->S4->d[j][row][col] = (float)0.0;
				cnn->S4->y[j][row][col] = (float)0.0;
			}
		}
	}

	for(int n=0; n<cnn->O5->output_num; n++){
		cnn->O5->d[n] = (float)0.0;
		cnn->O5->v[n] = (float)0.0;
		cnn->O5->y[n] = (float)0.0;
	}
}

void SaveCnnData(Cnn* cnn, const char* filename, float** inputdata)
{
	FILE  *file_point=NULL;
	file_point=fopen(filename,"wb");
	if(file_point==NULL)
		printf("[-] <SvaeCNNData> Open Write file failed! <%s>\n", filename);

	for(int i=0; i<cnn->C1->input_height; i++)
		fwrite(inputdata[i], sizeof(float), cnn->C1->input_width, file_point);

	for(int i=0; i<cnn->C1->input_channels; i++)
		for(int j=0; j<cnn->C1->output_channels; j++)
			for(int s=0; s<cnn->C1->map_size; s++)
				fwrite(cnn->C1->map_data[i][j][s], sizeof(float), \
				       cnn->C1->map_size, file_point);

	fwrite(cnn->C1->basic_data, sizeof(float), \
	       cnn->C1->output_channels, file_point);

	for(int j=0; j<cnn->C1->output_channels; j++)
	{
		for(int row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->v[j][row], sizeof(float), \
			       cnn->S2->input_width, file_point);
		}

		for(int row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->d[j][row], sizeof(float), \
			       cnn->S2->input_width, file_point);
		}

		for(int row=0; row<cnn->S2->input_height; row++)
		{
			fwrite(cnn->C1->y[j][row], sizeof(float), 
			       cnn->S2->input_width, file_point);
		}
	}

	for(int j=0; j<cnn->S2->output_channels; j++)
	{
		for(int row=0; row<cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->d[j][row], sizeof(float), \
			       cnn->C3->input_width, file_point);
		}
		for(int row=0; row<cnn->C3->input_height; row++)
		{
			fwrite(cnn->S2->y[j][row], sizeof(float), \
			       cnn->C3->input_width, file_point);
		}
	}

	for(int i=0; i<cnn->C3->input_channels; i++)
		for(int j=0; j<cnn->C3->output_channels; j++)
			for(int row=0; row<cnn->C3->map_size; row++)
				fwrite(cnn->C3->map_data[i][j][row], sizeof(float), \
				       cnn->C3->map_size, file_point);

	fwrite(cnn->C3->basic_data, sizeof(float), \
	       cnn->C3->output_channels, file_point);

	for(int j=0; j<cnn->C3->output_channels; j++)
	{
		for(int row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->v[j][row], sizeof(float), 
			       cnn->S4->input_width, file_point);
		}

		for(int row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->d[j][row], sizeof(float), 
			       cnn->S4->input_width, file_point);
		}

		for(int row=0; row<cnn->S4->input_height; row++)
		{
			fwrite(cnn->C3->y[j][row], sizeof(float), \
			       cnn->S4->input_width, file_point);
		}
	}

	for(int j=0; j<cnn->S4->output_channels; j++)
	{
		for(int row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->d[j][row], sizeof(float), \
			       cnn->S4->input_width/cnn->S4->map_size, file_point);
		}

		for(int row=0; row<cnn->S4->input_height/cnn->S4->map_size; row++)
		{
			fwrite(cnn->S4->y[j][row], sizeof(float), \
			       cnn->S4->input_width/cnn->S4->map_size, file_point);
		}
	}

	for(int i=0; i<cnn->O5->output_num; i++)
		fwrite(cnn->O5->wData[i], sizeof(float), cnn->O5->input_num, file_point);

	fwrite(cnn->O5->basic_data, sizeof(float), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->v, sizeof(float), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->d, sizeof(float), cnn->O5->output_num, file_point);
	fwrite(cnn->O5->y, sizeof(float), cnn->O5->output_num, file_point);

	fclose(file_point);
}
