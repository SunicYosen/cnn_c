//Main function of CNN Train and Test.
//

#include <stdlib.h>  //random
#include <stdio.h>
#include <time.h>    //random seed
#include "cnn.h" 

int main()
{
	//Read train and test data.
	LabelArray trainLabel = ReadLabels("mnist/train-labels-idx1-ubyte");
	ImageArray trainImg = ReadImages("mnist/train-images-idx3-ubyte");
	LabelArray test_label = ReadLabels("mnist/t10k-labels-idx1-ubyte");
	ImageArray test_image = ReadImages("mnist/t10k-images-idx3-ubyte");
	printf("[+] Read data finished!\n");

  //Input image mat size {columns,rows}{28,28}
	MatSize input_size;
	input_size.columns = test_image->image_point[0].number_of_columns;
	input_size.rows = test_image->image_point[0].number_of_rows;
  printf("[+] Input size: {%d,%d}\n",input_size.columns,input_size.rows);

  //Output Label array size {label_length} {10}
	int output_size = test_label->label_point[0].label_length;
	printf("[+] Output size: %d\n",output_size);
  
	//Setup CNN
	Cnn* cnn=(Cnn*)malloc(sizeof(Cnn));
	CnnSetup(cnn,input_size,output_size);
	printf("[+] CNN setup finished!\n");
    
  #if TRAIN
		TrainOptions opts;
		opts.numepochs=1;
		opts.alpha=1.0;
		int trainNum=55000;
		CnnTrain(cnn,trainImg,trainLabel,opts,trainNum);
		printf("[+] Train finished!!\n");
		
		SaveCnn(cnn,"output/mnist.cnn");
		FILE  *fp=NULL;
		fp=fopen("output/cnnL.ma","wb");
		if(fp==NULL)
			printf("write file failed\n");
		fwrite(cnn->L,sizeof(float),trainNum,fp);
		fclose(fp);
  #endif

	#if TEST
		ImportCnn(cnn,"output/mnist.cnn");
		printf("[+] Import CNN finished!\n");
		int testNum=10000;
		float incorrectRatio=1.0;
		incorrectRatio=CnnTest(cnn,test_image,test_label,testNum);
		printf("[+] Correct Ratio = %f\n",(1-incorrectRatio));
		printf("[+] CNN Test finished!\n");
	#endif

	return 0;
}