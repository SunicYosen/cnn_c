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
float** MatRotate180(float** mat, MatSize mat_size)
{
	int column,row;
	int output_size_width  = mat_size.columns;
	int output_size_height = mat_size.rows;
	float** output_data = (float**)malloc(output_size_height * sizeof(float*));

	for(int i=0; i<output_size_height; i++)
		output_data[i] = (float*)malloc(output_size_width * sizeof(float));

	for(row=0; row<output_size_height; row++)
		for(column=0; column<output_size_width; column++)
			output_data[row][column] = \
			        mat[output_size_height-row-1][output_size_width-column-1];

	return output_data;
}

//Correlatoin
float** MatCorrelation(float** map, MatSize map_size, \
                       float** input_data, MatSize input_size, int type)
{

	//Define MAX output size with all types
	int output_size_width  = input_size.columns + (map_size.columns-1); //
	int output_size_height = input_size.rows + (map_size.rows-1);

	float** output_data = (float**)malloc(output_size_height * sizeof(float*)); //

	for(int i=0; i<output_size_height; i++)
		output_data[i] = (float*)calloc(output_size_width, sizeof(float));

	//Expand mat with 0
	float** expand_input_data = MatEdgeExpand(input_data, input_size, \
	                                    map_size.columns-1, map_size.rows-1);
  
	//Calculate the mat with MAX output size
	for(int j=0; j<output_size_height; j++)
		for(int i=0; i<output_size_width; i++)
			for(int r=0; r<map_size.rows; r++)
				for(int c=0; c<map_size.columns; c++)
				{
					output_data[j][i] = output_data[j][i] + map[r][c] * \
					                    expand_input_data[j+r][i+c];
				}

	for(int i=0; i<input_size.rows + 2*(map_size.rows-1); i++)
		free(expand_input_data[i]);

	free(expand_input_data);

  //For different type
  int half_map_size_width;
	int half_map_size_height;

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
			float** same_size_result = MatEdgeShrink(output_data, outSize, \
			                           half_map_size_width, half_map_size_height);

			for(int i=0; i<outSize.rows; i++)
				free(output_data[i]);
				
			free(output_data);

			return same_size_result;
			}

		case VALID:
		{
			float** valid_result;
			if((map_size.rows%2==0) && (map_size.columns%2==0))
				valid_result = MatEdgeShrink(output_data, outSize, \
				               half_map_size_width*2-1, half_map_size_height*2-1);
			
			else
				valid_result = MatEdgeShrink(output_data, outSize, \
				               half_map_size_width*2, half_map_size_height*2);
			
			for(int i=0; i<outSize.rows; i++)
				free(output_data[i]);
			free(output_data);

			return valid_result;
		}

		default:
			return output_data;
	}
}

float** MatConvolution(float** map, MatSize map_size, \
                       float** input_data, MatSize input_size, int type)
{
	float** flipmap = MatRotate180(map, map_size);
	float** result = MatCorrelation(flipmap, map_size, input_data, \
	                                input_size, type);

	for(int i=0; i<map_size.rows; i++)
		free(flipmap[i]);
	free(flipmap);

	return result;
}


//Mat up sample
float** MatUpSample(float** mat, MatSize mat_size, int up_cols, int up_rows)
{ 
	int cols = mat_size.columns;
	int rows = mat_size.rows;
	float** result_mat = (float**)malloc((rows*up_rows) * sizeof(float*));

	for(int row=0; row<(rows*up_rows); row++)
		result_mat[row] = (float*)malloc((cols * up_cols) * sizeof(float));

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
float** MatEdgeExpand(float** mat, MatSize mat_size, \
                      int expand_col, int expand_row)
{
	int columns = mat_size.columns;
	int rows    = mat_size.rows;
	float** result = (float**)malloc((rows + 2*expand_row) * sizeof(float*));

	for(int i=0; i<(rows + 2*expand_row); i++)
		result[i] = (float*)malloc((columns + 2*expand_col) * sizeof(float));

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
float** MatEdgeShrink(float** mat, MatSize mat_size, \
                      int shrink_cols, int shrink_rows)
{
	int columns = mat_size.columns;
	int rows    = mat_size.rows;
	float** result_mat = (float**)malloc((rows - 2*shrink_rows) * sizeof(float*));
	
	for(int i=0; i<(rows-2*shrink_rows); i++)
		result_mat[i] = (float*)malloc((columns-2*shrink_cols) * sizeof(float));

	
	for(int j=0; j<rows; j++)
	{
		for(int i=0; i<columns; i++)
		{
			if(j>=shrink_rows && i>=shrink_cols && \
			   j<(rows-shrink_rows) && i<(columns-shrink_cols))

				result_mat[j-shrink_rows][i-shrink_cols] = mat[j][i];
		}
	}
	return result_mat;
}

//Save mat to file by binary
void MatSaving(float** mat, MatSize mat_size, const char* filename)
{
	FILE  *file_point = NULL;
	file_point = fopen(filename,"wb");

	if(file_point == NULL)
		printf("write file failed\n");

	for(int i=0; i<mat_size.rows; i++)
		fwrite(mat[i],sizeof(float),mat_size.columns,file_point);

	fclose(file_point);
}

//Add Mat
void MatAdd(float** result_mat, float** mat1, MatSize mat_size1, \
            float** mat2, MatSize mat_size2)
{
	//Check 
	if((mat_size1.columns != mat_size2.columns) || (mat_size1.rows != mat_size2.rows))
		printf("ERROR: Size is not same!");

	for(int row=0; row<mat_size1.rows; row++)
		for(int col=0; col<mat_size1.columns; col++)

			result_mat[row][col]=mat1[row][col] + mat2[row][col];
}

//Multiply immediate
void MatMultiplyImmediate(float** res, float** mat, MatSize mat_size, float immediate)
{
	for(int row=0; row<mat_size.rows; row++)
		for(int col=0; col<mat_size.columns; col++)
			res[row][col]=mat[row][col] * immediate;
}

float MatSum(float** mat, MatSize mat_size) 
{
	float sum=0;

	for(int row=0; row<mat_size.rows; row++)
		for(int col=0; col<mat_size.columns; col++)
			sum = sum + mat[row][col];

	return sum;
}