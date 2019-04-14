
#ifndef MAT_H_
#define MAT_H_

#define FULL 0
#define SAME 1
#define VALID 2

typedef struct Mat2DSize{
	int16_t columns;
	int16_t rows;
}MatSize;

// Rotate Mat 180
int16_t** MatRotate180(int16_t** mat, MatSize mat_size);

// Add two mat one by one.
void MatAdd(int16_t** res, int16_t** mat1, MatSize mat_size1, \
                          int16_t** mat2, MatSize mat_size2);

//
int16_t** MatCorrelation(int16_t** map, MatSize map_size, int16_t** inputData, \
                                      MatSize inSize,   int16_t type);

//Convolution two mats
int16_t** MatConvolution(int16_t** map, MatSize map_size, int16_t** inputData, \
                                      MatSize inSize,   int16_t type);

//Expand Mat for up sample
int16_t** MatUpSample(int16_t** mat, MatSize mat_size, int16_t upc, int16_t upr);

//Expand the edge of mat by 0;
int16_t** MatEdgeExpand(int16_t** mat, MatSize mat_size, int16_t addc, int16_t addr);

//Shrink mat edge
int16_t** MatEdgeShrink(int16_t** mat,MatSize mat_size,int32_t shrink_cols,int32_t shrink_rows);

// Save mat data to file
void MatSaving(int16_t** mat, MatSize mat_size, const char* filename);

//Multiply mat by Immediate
void MatMultiplyImmediate(int16_t** res,  int16_t** mat, MatSize mat_size, int16_t immediate);

//Calmulate the sum of one mat
int16_t MatSum(int16_t** mat, MatSize mat_size);

//Combine strings
char * CombineStrings(char *a, char *b);

//Transfer int to char
char* IntToChar(int32_t i);

#endif