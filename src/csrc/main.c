//Main function of CNN Train and Test.
//

#include <stdlib.h>  //random
#include <stdio.h>
#include <time.h>    //random seed
#include <stdint.h>
#include "cnn.h" 

int main()
{
	//Read train and test data.
	LabelArray train_labels = ReadLabels("mnist/train-labels-idx1-ubyte");
	ImageArray train_images = ReadImages("mnist/train-images-idx3-ubyte");
	LabelArray test_labels = ReadLabels("mnist/t10k-labels-idx1-ubyte");
	ImageArray test_images = ReadImages("mnist/t10k-images-idx3-ubyte");
	printf("[+] Read data finished!\n");
  
	//Save one image to file.
	//SaveImage(train_images,"output/train_images/");
	//SaveImage(test_images,"output/test_images/");

	const char* cnn_arch_filename = "output/mnist.cnn";
	const char* cnn_layer_filename = "output/cnn_layer.ma";


  //Input image mat size {columns,rows}{28,28}
	MatSize input_size;
	input_size.columns = test_images->image_point[0].number_of_columns;
	input_size.rows = test_images->image_point[0].number_of_rows;

  printf("[+] Input size: {%d,%d}\n",input_size.columns,input_size.rows);

  //Output Label array size {label_length} {10}
	int16_t output_size = test_labels->label_point[0].label_length;
	printf("[+] Output size: %d\n",output_size);
  
	//Setup CNN
	Cnn* cnn=(Cnn*)malloc(sizeof(Cnn));

	CnnSetup(cnn, input_size, output_size);
	printf("[+] CNN setup finished!\n");
    
  #if TRAIN
		TrainOptions opts;
		opts.numepochs = 1;
		opts.alpha     = 1;
		int32_t train_images_num = 55000;
		// Train
		CnnTrain(cnn, train_images, train_labels, opts, train_images_num);
		printf("[+] Train finished!\n");
		
		//Save cnn arch to file
		SaveCnn(cnn,cnn_arch_filename);
    
		//save cnn layer info to file
		FILE  *file_point = NULL;
		file_point = fopen(cnn_layer_filename, "wb");
		if(file_point == NULL)
			printf("[-] <main.TRAIN> Open Write file failed! <%s>\n",cnn_layer_filename);

		fwrite(cnn->L, sizeof(int16_t), train_images_num, file_point);

		fclose(file_point);
  #endif

	#if TEST
		ImportCnn(cnn, cnn_arch_filename);
		printf("[+] Import CNN finished!\n");

		int32_t test_images_num = 10000;
		int32_t incorrect_num = test_images_num;
		incorrect_num = CnnTest(cnn, test_images, test_labels, test_images_num);

		printf("[+] Correct Ratio = %d/%d\n", test_images_num - incorrect_num, test_images_num);
		printf("[+] CNN Test finished!\n");
	#endif

	return 0;
}