//
// Header for mnist. Define the data architecture and functions.
//

#ifndef MNIST_H_
#define MNIST_H_

typedef struct MnistImage{  //Structrue for Mnist Image.
	int number_of_columns;             //Define the width of image. --- columns
	int number_of_rows;                //Define the highth ot image. --- rows
	float** image_data;      //Dynamic Define 2D Image Data Arrary. 
}MnistImage;

typedef struct MnistImageArray{ //Structrue for Mnist Images Array
	int number_of_images;             //Define Number of Images.    
	MnistImage* image_point;       //Point for Images Array.
}*ImageArray;                   //Images Array.

typedef struct MnistLabel{  //Strutrue for Mnist Label
	int label_length;                    //The length of Label 
	float* LabelData;         //The data of label
}MnistLabel;

typedef struct MnistLabelArray{ //Structrue for Mnist Label Arrary
	int number_of_labels;         //Numers of Labels
	MnistLabel* label_point;      //Point for Mnist labels array
} *LabelArray;                  //Mnist labels array

LabelArray ReadLabels(const char* filename);  //Read labels function
ImageArray ReadImages(const char* filename);   //Read Images function
void SaveImage(ImageArray imgarr,char* filedir); //Save Images function.

#endif