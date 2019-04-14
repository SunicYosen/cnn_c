//
// Mnist Data Process Functions.
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "mnist.h"

#define COLORTH 127

//Reverse an int32 For Big-End
//eg: 0x11223344 --> 0x44332211
int32_t ReverseInt32(int32_t integer)
{  
	unsigned char bits_0_7;
	unsigned char bits_8_15;
	unsigned char bits_16_23;
	unsigned char bits_24_31;

	bits_0_7   = integer & 255;         //Get the low 8-bit of int32 integer
	bits_8_15  = (integer >> 8) & 255;  //
	bits_16_23 = (integer >> 16) & 255; //
	bits_24_31 = (integer >> 24) & 255; //
	return ((int32_t)bits_0_7 << 24) + ((int32_t)bits_8_15 << 16) \
	     + ((int16_t)bits_16_23 << 8) + bits_24_31;  
}

//Read Images from data <filename>
ImageArray ReadImages(const char* filename)
{
	//Open images file 
	FILE  *file_point = NULL;
	file_point = fopen(filename,"rb"); 
	if(file_point == NULL)  //Failed
	{
		printf("[-] ReadImages() Open file [%s] failed!\n",filename);
		assert(file_point);
	}
  
	//Read images from file with file_point
	int32_t magic_number = 0;     //magic number
	int32_t number_of_images = 0; //Images' number
	int32_t n_rows = 0;  //number of rows of an image<image hight>
	int32_t n_columns = 0;  //number of cols of an image<image width>
	
	// >Big-End Style, So Reverse the Integer. Read magic number
	fread((char*)&magic_number, sizeof(magic_number), 1, file_point);
	magic_number = ReverseInt32(magic_number);
	
	// >Big-End. Read the number of images.
	fread((char*)&number_of_images, sizeof(number_of_images), 1, file_point);  
	number_of_images = ReverseInt32(number_of_images);
	
	//Read the rows and cols of an image
	fread((char*)&n_rows, sizeof(n_rows), 1, file_point);
	fread((char*)&n_columns, sizeof(n_columns), 1, file_point); 

	n_rows = ReverseInt32(n_rows);                  
	n_columns = ReverseInt32(n_columns);  

  //Define strutrue of image array
	ImageArray image_array = (ImageArray)malloc(sizeof(ImageArray));
	image_array->number_of_images = number_of_images;  //number of images

	//array of all images.
	image_array->image_point = (MnistImage*)malloc(number_of_images * sizeof(MnistImage));
  
	//int row,column;  // Temp for row and column 
	for(uint32_t i=0; i<number_of_images; ++i) //Images from 0 -> number_of_images-1
	{  
		image_array->image_point[i].number_of_rows = n_rows;     //
		image_array->image_point[i].number_of_columns = n_columns;  //set 
		image_array->image_point[i].image_data = (int16_t** ) malloc(n_rows * sizeof(int16_t *));

		for(uint32_t row = 0; row < n_rows; ++row) //from 0 -> n_rows-1
		{
			image_array->image_point[i].image_data[row] = (int16_t* )malloc(n_columns * sizeof(int16_t));

			for(uint32_t column = 0; column < n_columns; ++column)  //from 0 -> n_columns-1
			{ 
				unsigned char temp_pixel = 0;   
				//read a pixel 0-255 with 8-bit
				fread((char*) &temp_pixel, sizeof(temp_pixel), 1, file_point); 
				
				//Change color to 1/0 by color Threshold.
				image_array->image_point[i].image_data[row][column] = \
						(int16_t)((temp_pixel>(int16_t)COLORTH) ? (int16_t)1 : (int16_t)0);
				
				/*
					//Change 8-bit pixel to float. 
					image_array->image_point[i].image_data[row][column]= (float)temp_pixel/255;
				*/
			  
			}
		}
	}

	fclose(file_point);
	return image_array;
}

LabelArray ReadLabels(const char* filename)
{
	FILE *file_point=NULL;
	file_point = fopen(filename, "rb");
	if(file_point==NULL)
	{
		printf("[-] <ReadLabels> Open file failed! <%s>\n",filename);
	  assert(file_point);
	}

	int32_t magic_number = 0;      //A 32-bit interger
	int32_t number_of_labels = 0;  //60000 for training. 10000 for testing
	int16_t label_long = 10;

	fread((char*)&magic_number, sizeof(magic_number), 1, file_point); 
	magic_number = ReverseInt32(magic_number);  

	fread((char*)&number_of_labels,sizeof(number_of_labels),1,file_point);  
	number_of_labels = ReverseInt32(number_of_labels);

	//LabelArray labels_array=(LabelArray)malloc(sizeof(MnistLabelArray));
	LabelArray labels_array=(LabelArray)malloc(sizeof(LabelArray));
	labels_array->number_of_labels = number_of_labels;
	labels_array->label_point = (MnistLabel*)malloc(number_of_labels*sizeof(MnistLabel));

	for(int32_t i = 0; i < number_of_labels; ++i)  
	{  
		labels_array->label_point[i].label_length = 10;
		labels_array->label_point[i].label_data = (int16_t *)calloc(label_long, sizeof(int16_t));
		
		unsigned char temp = 0;  
		fread((char*) &temp, sizeof(temp), 1, file_point); 
		
		//printf("%d\n",labels_array->label_point[i].label_data[(int16_t)temp]);
		labels_array->label_point[i].label_data[(int16_t)temp] = 1;
	
	}

	fclose(file_point);
	return labels_array;	
}

char* IntToChar(int32_t i)
{
	int32_t i_temp = i;
	int32_t w_max  = 0;

	while(i_temp>=10)
	{
		i_temp      = i_temp/10;
		w_max++;
	}

	char* ptr=(char*)malloc((w_max+2) * sizeof(char));

	ptr[w_max+1] = '\0';
	int value;

	while(i>=10)
	{
		value = i%10;
		i     = i/10;		
		ptr[w_max] = (char)(value+48);
		w_max--;
	}

	ptr[w_max] = (char)(i+48);

	return ptr;
}

char * CombineStrings(char *a, char *b) 
{
	char *ptr;
	int32_t lena=strlen(a);
	int32_t lenb=strlen(b);
	
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));
  
	int32_t l=0;
	for(int32_t i=0; i<lena; i++)
		ptr[l++] = a[i];

	for(int32_t i=0; i<lenb; i++)
		ptr[l++]=b[i];

	ptr[l]='\0';
	return(ptr);
}

void SaveImage(ImageArray image_array, char* filedir)
{
	int img_number = image_array->number_of_images;

	for(int32_t i=0; i<img_number; i++)
	{
		const char* filename = CombineStrings(filedir, CombineStrings(IntToChar(i), ".gray"));
		
		FILE  *file_point    = NULL;
		file_point           = fopen(filename,"wb");

		if(file_point == NULL)
		{
			printf("write file failed\n");
		  assert(file_point);
		}

		for(int32_t r=0; r<image_array->image_point[i].number_of_rows; r++)
		{
			
			for(int32_t m=0; m<image_array->image_point[i].number_of_columns; m++)
			{
				int16_t * temp;
				* temp = image_array->image_point[i].image_data[r][m];
				fwrite(&temp, sizeof(int16_t), 1, file_point);

				printf("%0x",*temp);
			}			
			printf("\n");
			//fwrite((int16_t)'\n',sizeof(int16_t),1,file_point);
		}
		
		fclose(file_point);
	}	
}