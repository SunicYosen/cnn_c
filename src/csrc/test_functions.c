#include <stdlib.h>  //random
#include <stdio.h>
#include <time.h>    //random seed

#include "cnn.h" 

// Test Mnist data. Save the images to local.
void TestMnist()
{
  LabelArray test_label = ReadLabels("mnist/t10k-labels-idx1-ubyte");
  ImageArray test_image = ReadImages("mnist/t10k-images-idx3-ubyte");
	SaveImage(test_image, "mnist/test_images"); //Save images
}

//Test Mat functions.
void TestMat()
{
	MatSize srcSize = {6,6};
	MatSize map_size = {4,4};

  //time to random seed.
	srand((unsigned)time(NULL)); 
  
	//define and random src mat
	float** src = (float**)malloc(srcSize.rows * sizeof(float*));
	for(int i=0; i<srcSize.rows; i++)
	{
		src[i] = (float*)malloc(srcSize.columns * sizeof(float));

		for(int j=0; j<srcSize.columns; j++)
		{
			//Generate float from [-1:1]
			src[i][j] = (((float)rand() / (float)RAND_MAX)-0.5) * 2; 
		}
	}

  //define and random map mat
	float** map = (float**)malloc(map_size.rows * sizeof(float*));
	for(int i=0; i<map_size.rows; i++)
	{
		map[i] = (float*)malloc(map_size.columns * sizeof(float));

		for(int j=0; j<map_size.columns; j++)
		{
			//Generate float from [-1:1]
			map[i][j] = (((float)rand() / (float)RAND_MAX)-0.5) * 2; 
		}
	}

	MatSize cov1size={srcSize.columns+map_size.columns-1,srcSize.rows+map_size.rows-1};
	float** cov1=MatConvolution(map,map_size,src,srcSize,FULL);
	//MatSize cov2size={srcSize.columns,srcSize.rows};
	//float** cov2=MatConvolution(map,map_size,src,srcSize,SAME);
	MatSize cov3size={srcSize.columns-(map_size.columns-1),srcSize.rows-(map_size.rows-1)};
	float** cov3=MatConvolution(map,map_size,src,srcSize,VALID);

	MatSaving(src,srcSize,"output/src.ma");
	MatSaving(map,map_size,"output/map.ma");
	MatSaving(cov1,cov1size,"output/cov1.ma");
	//MatSaving(cov2,cov2size,"output/cov2.ma");
	MatSaving(cov3,cov3size,"output/cov3.ma");

	float** sample=MatUpSample(src,srcSize,2,2);
	MatSize samSize={srcSize.columns*2,srcSize.rows*2};
	MatSaving(sample,samSize,"output/sam.ma");
}

void TestMat1()
{
	int i,j;
	MatSize srcSize={12,12};
	MatSize map_size={5,5};
	float** src=(float**)malloc(srcSize.rows*sizeof(float*));
	for(i=0;i<srcSize.rows;i++){
		src[i]=(float*)malloc(srcSize.columns*sizeof(float));
		for(j=0;j<srcSize.columns;j++){
			src[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map1=(float**)malloc(map_size.rows*sizeof(float*));
	for(i=0;i<map_size.rows;i++){
		map1[i]=(float*)malloc(map_size.columns*sizeof(float));
		for(j=0;j<map_size.columns;j++){
			map1[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map2=(float**)malloc(map_size.rows*sizeof(float*));
	for(i=0;i<map_size.rows;i++){
		map2[i]=(float*)malloc(map_size.columns*sizeof(float));
		for(j=0;j<map_size.columns;j++){
			map2[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map3=(float**)malloc(map_size.rows*sizeof(float*));
	for(i=0;i<map_size.rows;i++){
		map3[i]=(float*)malloc(map_size.columns*sizeof(float));
		for(j=0;j<map_size.columns;j++){
			map3[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}

	float** cov1=MatConvolution(map1,map_size,src,srcSize,VALID);
	float** cov2=MatConvolution(map2,map_size,src,srcSize,VALID);
	MatSize covsize={srcSize.columns-(map_size.columns-1),srcSize.rows-(map_size.rows-1)};
	float** cov3=MatConvolution(map3,map_size,src,srcSize,VALID);
	MatAdd(cov1,cov1,covsize,cov2,covsize);
	MatAdd(cov1,cov1,covsize,cov3,covsize);


	MatSaving(src,srcSize,"output/src.ma");
	MatSaving(map1,map_size,"output/map1.ma");
	MatSaving(map2,map_size,"output/map2.ma");
	MatSaving(map3,map_size,"output/map3.ma");
	MatSaving(cov1,covsize,"output/cov1.ma");
	MatSaving(cov2,covsize,"output/cov2.ma");
	MatSaving(cov3,covsize,"output/cov3.ma");

}

void test_cnn()
{

	LabelArray test_label = ReadLabels("mnist/train-labels-idx1-ubyte");
	ImageArray test_image = ReadImages("mnist/train-images-idx3-ubyte");

	MatSize input_size = {test_image->image_point[0].number_of_columns, \
	                  test_image->image_point[0].number_of_rows};
  
	//Output size
	int output_size = test_label->label_point[0].label_length; 

	Cnn* cnn=(Cnn*)malloc(sizeof(Cnn));

  //Setup the CNN
	CnnSetup(cnn,input_size,output_size);

  //Train the CNN
	TrainOptions opts;
	opts.numepochs=1;  //train epochs
	opts.alpha=1;
	int num_train=5000; //Train number
	CnnTrain(cnn,test_image,test_label,opts,num_train);
  
	//output
	FILE *file_point = NULL;
	file_point = fopen("output/cnn_layer.ma","wb");
	if(file_point == NULL)
		printf("[-] Write file failed! <output/cnn_layer.ma>\n");

	fwrite(cnn->L, sizeof(float), num_train, file_point);
	fclose(file_point);
}