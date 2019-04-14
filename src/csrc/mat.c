// 
// 

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "mat.h" 

//Rotate
int16_t** MatRotate180(int16_t** mat, MatSize mat_size)
{
	int16_t output_size_width  = mat_size.columns;
	int16_t output_size_height = mat_size.rows;
	int16_t** output_data = (int16_t**)malloc(output_size_height * sizeof(int16_t*));

	for(int32_t i=0; i<output_size_height; i++)
		output_data[i] = (int16_t*)malloc(output_size_width * sizeof(int16_t));

	for(int32_t row=0; row<output_size_height; row++)
		for(int32_t column=0; column<output_size_width; column++)
			output_data[row][column] = \
			        mat[output_size_height-row-1][output_size_width-column-1];

	return output_data;
}

//Correlatoin
int16_t** MatCorrelation(int16_t** map, MatSize map_size, \
                       int16_t** input_data, MatSize input_size, int16_t type)
{

	//Define MAX output size with all types
	int16_t output_size_width  = input_size.columns + (map_size.columns-1); //
	int16_t output_size_height = input_size.rows + (map_size.rows-1);

	int16_t** output_data = (int16_t**)malloc(output_size_height * sizeof(int16_t*)); //

	for(int i=0; i<output_size_height; i++)
		output_data[i] = (int16_t*)calloc(output_size_width, sizeof(int16_t));

	//Expand mat with 0
	int16_t** expand_input_data = MatEdgeExpand(input_data, input_size, \
	                                    map_size.columns-1, map_size.rows-1);
  
	//Calculate the mat with MAX output size
	for(int32_t j=0; j<output_size_height; j++)
		for(int32_t i=0; i<output_size_width; i++)
			for(int32_t r=0; r<map_size.rows; r++)
				for(int32_t c=0; c<map_size.columns; c++)
				{
					output_data[j][i] = output_data[j][i] + map[r][c] * \
					                    expand_input_data[j+r][i+c];
				}

	for(int32_t i=0; i<input_size.rows + 2*(map_size.rows-1); i++)
		free(expand_input_data[i]);

	free(expand_input_data);

  //For different type
  int32_t half_map_size_width;
	int32_t half_map_size_height;

	if(map_size.rows%2==0 && map_size.columns%2==0)
	{
		half_map_size_width  = (map_size.columns)/2;
		half_map_size_height = (map_size.rows)/2;
	}
	else
	{
		half_map_size_width  = (map_size.columns-1)/2; 
		half_map_size_height = (map_size.rows-1)/2;
	}

	MatSize outSize = {output_size_width, output_size_height};

	switch(type)
	{

		case FULL:
			return output_data;

		case SAME:
		{
			int16_t** same_size_result = MatEdgeShrink(output_data, outSize, \
			                           half_map_size_width, half_map_size_height);

			for(int32_t i=0; i<outSize.rows; i++)
				free(output_data[i]);
				
			free(output_data);

			return same_size_result;
			}

		case VALID:
		{
			int16_t** valid_result;
			if((map_size.rows%2==0) && (map_size.columns%2==0))
				valid_result = MatEdgeShrink(output_data, outSize, \
				               half_map_size_width*2-1, half_map_size_height*2-1);
			
			else
				valid_result = MatEdgeShrink(output_data, outSize, \
				               half_map_size_width*2, half_map_size_height*2);
			
			for(int32_t i=0; i<outSize.rows; i++)
				free(output_data[i]);
			free(output_data);

			return valid_result;
		}

		default:
			return output_data;
	}
}

int16_t** MatConvolution(int16_t** map, MatSize map_size, \
                       int16_t** input_data, MatSize input_size, int16_t type)
{
	int16_t** flipmap = MatRotate180(map, map_size);
	int16_t** result = MatCorrelation(flipmap, map_size, input_data, \
	                                input_size, type);

	for(int32_t i=0; i<map_size.rows; i++)
		free(flipmap[i]);

	free(flipmap);

	return result;
}


//Mat up sample 用于反向传播, pooling的反过程
int16_t** MatUpSample(int16_t** mat, MatSize mat_size, int16_t up_cols, int16_t up_rows)
{ 
	int16_t cols = mat_size.columns;
	int16_t rows = mat_size.rows;
	int16_t** result_mat = (int16_t**)malloc((rows*up_rows) * sizeof(int16_t*));

	for(int row=0; row<(rows*up_rows); row++)
		result_mat[row] = (int16_t*)malloc((cols * up_cols) * sizeof(int16_t));

	for(int row=0; row<rows*up_rows; row=row+up_rows)
	{
		for(int col=0; col<cols*up_cols; col=col+up_cols)
			for(int m=0; m<up_cols; m++)
				result_mat[row][col+m] = mat[row/up_rows][col/up_cols];

		for(int n=1; n<up_rows; n++)
			for(int col=0; col<cols*up_cols; col++)
				result_mat[row+n][col] = result_mat[row][col];
	}

	return result_mat;
}

//Expand mat edge by add 0 rows/cols
int16_t** MatEdgeExpand(int16_t** mat, MatSize mat_size, \
                      int16_t expand_col, int16_t expand_row)
{
	int columns = mat_size.columns;
	int rows    = mat_size.rows;
	int16_t** result = (int16_t**)malloc((rows + 2*expand_row) * sizeof(int16_t*));

	for(int i=0; i<(rows + 2*expand_row); i++)
		result[i] = (int16_t*)malloc((columns + 2*expand_col) * sizeof(int16_t));

	for(int j=0; j<rows+2*expand_row; j++)
	{
		for(int i=0; i<columns+2*expand_col; i++)
		{
			if(j<expand_row || i<expand_col || \
			   j>=(rows+expand_row) || i>=(columns+expand_col))
				result[j][i] = 0;

			else
				result[j][i] = mat[j-expand_row][i-expand_col];
		}
	}
	return result;
}

//Mat edge shrink
int16_t** MatEdgeShrink(int16_t** mat, MatSize mat_size, \
                      int32_t shrink_cols, int32_t shrink_rows)
{
	int16_t columns = mat_size.columns;
	int16_t rows    = mat_size.rows;
	int16_t** result_mat = (int16_t**)malloc((rows - 2*shrink_rows) * sizeof(int16_t*));
	
	for(int32_t i=0; i<(rows-2*shrink_rows); i++)
		result_mat[i] = (int16_t*)malloc((columns - 2*shrink_cols) * sizeof(int16_t));

	
	for(int32_t j=0; j<rows; j++)
	{
		for(int32_t i=0; i<columns; i++)
		{
			if(j>=shrink_rows && i>=shrink_cols && \
			   j<(rows-shrink_rows) && i<(columns-shrink_cols))

				result_mat[j-shrink_rows][i-shrink_cols] = mat[j][i];
		}
	}
	return result_mat;
}

//Save mat to file by binary
void MatSaving(int16_t** mat, MatSize mat_size, const char* filename)
{
	FILE  *file_point = NULL;
	file_point = fopen(filename,"wb");

	if(file_point == NULL)
		printf("write file failed\n");

	for(int i=0; i<mat_size.rows; i++)
		fwrite(mat[i],sizeof(int16_t), mat_size.columns, file_point);

	fclose(file_point);
}

//Add Mat
void MatAdd(int16_t** result_mat, int16_t** mat1, MatSize mat_size1, \
            int16_t** mat2, MatSize mat_size2)
{
	//Check 
	if((mat_size1.columns != mat_size2.columns) || (mat_size1.rows != mat_size2.rows))
		printf("ERROR: Size is not same!");

	for(int32_t row=0; row<mat_size1.rows; row++)
		for(int32_t col=0; col<mat_size1.columns; col++)
			result_mat[row][col]=mat1[row][col] + mat2[row][col];
}

//Multiply immediate
void MatMultiplyImmediate(int16_t** res, int16_t** mat, MatSize mat_size, int16_t immediate)
{
	for(int row=0; row<mat_size.rows; row++)
		for(int col=0; col<mat_size.columns; col++)
			res[row][col] = mat[row][col] * immediate;
}

int16_t MatSum(int16_t** mat, MatSize mat_size) 
{
	int16_t sum=0;

	for(int32_t row=0; row<mat_size.rows; row++)
		for(int32_t col=0; col<mat_size.columns; col++)
			sum = sum + mat[row][col];

	return sum;
}