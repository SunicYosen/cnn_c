
#ifndef MAT_H_
#define MAT_H_

#define FULL 0
#define SAME 1
#define VALID 2

typedef struct Mat2DSize{
	int8_t columns;
	int8_t rows;
}MatSize;

float** MatRotate180(float** mat, MatSize mat_size);
void MatAdd(float** res, float** mat1, MatSize mat_size1, float** mat2, MatSize mat_size2);

float** MatCorrelation(float** map, MatSize map_size, float** inputData, MatSize inSize, int type);

float** MatConvolution(float** map,MatSize map_size,float** inputData,MatSize inSize,int type);

float** MatUpSample(float** mat,MatSize mat_size,int upc,int upr);

float** MatEdgeExpand(float** mat,MatSize mat_size,int addc,int addr);

float** MatEdgeShrink(float** mat,MatSize mat_size,int shrinkc,int shrinkr);

void MatSaving(float** mat,MatSize mat_size,const char* filename);

void MatMultiplyImmediate(float** res, float** mat, MatSize mat_size, float factor);

float MatSum(float** mat,MatSize mat_size);

char * CombineStrings(char *a, char *b);

char* IntToChar(int i);

#endif