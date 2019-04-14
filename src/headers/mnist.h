//
// Header for mnist. Define the data architecture and functions.
//

#ifndef MNIST_H_
#define MNIST_H_

#include <stdint.h>

typedef struct MnistImage{  //Structrue for Mnist Image.
	int16_t number_of_columns;             //Define the width of image. --- columns
	int16_t number_of_rows;                //Define the highth ot image. --- rows
	int16_t** image_data;      //Dynamic Define 2D Image Data Arrary. 
}MnistImage;

typedef struct MnistImageArray{ //Structrue for Mnist Images Array
	int32_t number_of_images;     //Define Number of Images.    
	MnistImage* image_point;      //Point for Images Array.
}*ImageArray;                   //Images Array.

typedef struct MnistLabel{  //Strutrue for Mnist Label
	int16_t label_length;        //The length of Label 
	int16_t* label_data;         //The data of label
}MnistLabel;

typedef struct MnistLabelArray{ //Structrue for Mnist Label Arrary
	int32_t number_of_labels;         //Numers of Labels
	MnistLabel* label_point;      //Point for Mnist labels array
} *LabelArray;                  //Mnist labels array

LabelArray ReadLabels(const char* filename);  //Read labels function
ImageArray ReadImages(const char* filename);   //Read Images function
void SaveImage(ImageArray imgarr,char* filedir); //Save Images function.

#endif