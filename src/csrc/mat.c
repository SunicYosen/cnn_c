// 
// 

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mat.h" 

float** MatRotate180(float** mat, MatSize mat_size)
{
	int column,row;
	int outSizeW=mat_size.columns;
	int outSizeH=mat_size.rows;
	float** outputData=(float**)malloc(outSizeH*sizeof(float*));

	for(int i=0; i<outSizeH; i++)
		outputData[i]=(float*)malloc(outSizeW*sizeof(float));

	for(row=0; row<outSizeH; row++)
		for(column=0; column<outSizeW; column++)
			outputData[row][column] = mat[outSizeH-row-1][outSizeW-column-1];

	return outputData;
}

float** MatCorrelation(float** map,MatSize map_size,float** inputData,MatSize inSize,int type)
{
	int halfmapsizew;
	int halfmapsizeh;
	if(map_size.rows%2==0&&map_size.columns%2==0){
		halfmapsizew=(map_size.columns)/2;
		halfmapsizeh=(map_size.rows)/2;
	}else{
		halfmapsizew=(map_size.columns-1)/2; 
		halfmapsizeh=(map_size.rows-1)/2;
	}

	//
	int outSizeW=inSize.columns+(map_size.columns-1); //
	int outSizeH=inSize.rows+(map_size.rows-1);
	float** outputData=(float**)malloc(outSizeH*sizeof(float*)); //
	for(int i=0;i<outSizeH;i++)
		outputData[i]=(float*)calloc(outSizeW,sizeof(float));

	//
	float** exInputData=MatEdgeExpand(inputData,inSize,map_size.columns-1,map_size.rows-1);

	for(int j=0;j<outSizeH;j++)
		for(int i=0;i<outSizeW;i++)
			for(int r=0;r<map_size.rows;r++)
				for(int c=0;c<map_size.columns;c++){
					outputData[j][i]=outputData[j][i]+map[r][c]*exInputData[j+r][i+c];
				}

	for(int i=0;i<inSize.rows+2*(map_size.rows-1);i++)
		free(exInputData[i]);
	free(exInputData);

	MatSize outSize={outSizeW,outSizeH};
	switch(type){ 
	case FULL:
		return outputData;
	case SAME:{
		float** sameres=MatEdgeShrink(outputData,outSize,halfmapsizew,halfmapsizeh);
		for(int i=0;i<outSize.rows;i++)
			free(outputData[i]);
		free(outputData);
		return sameres;
		}
	case VALID:{
		float** validres;
		if(map_size.rows%2==0&&map_size.columns%2==0)
			validres=MatEdgeShrink(outputData,outSize,halfmapsizew*2-1,halfmapsizeh*2-1);
		else
			validres=MatEdgeShrink(outputData,outSize,halfmapsizew*2,halfmapsizeh*2);
		for(int i=0;i<outSize.rows;i++)
			free(outputData[i]);
		free(outputData);
		return validres;
		}
	default:
		return outputData;
	}
}

float** MatCov(float** map,MatSize map_size,float** inputData,MatSize inSize,int type)
{
	float** flipmap=MatRotate180(map,map_size);
	float** res=MatCorrelation(flipmap,map_size,inputData,inSize,type);
	int i;
	for(i=0;i<map_size.rows;i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

float** MatUpSample(float** mat,MatSize mat_size,int upc,int upr)
{ 
	int i,j,m,n;
	int c=mat_size.columns;
	int r=mat_size.rows;
	float** res=(float**)malloc((r*upr)*sizeof(float*));
	for(i=0;i<(r*upr);i++)
		res[i]=(float*)malloc((c*upc)*sizeof(float));

	for(j=0;j<r*upr;j=j+upr){
		for(i=0;i<c*upc;i=i+upc)
			for(m=0;m<upc;m++)
				res[j][i+m]=mat[j/upr][i/upc];

		for(n=1;n<upr;n++)
			for(i=0;i<c*upc;i++)
				res[j+n][i]=res[j][i];
	}
	return res;
}

float** MatEdgeExpand(float** mat,MatSize mat_size,int addc,int addr)
{
	int i,j;
	int c=mat_size.columns;
	int r=mat_size.rows;
	float** res=(float**)malloc((r+2*addr)*sizeof(float*));
	for(i=0;i<(r+2*addr);i++)
		res[i]=(float*)malloc((c+2*addc)*sizeof(float));

	for(j=0;j<r+2*addr;j++){
		for(i=0;i<c+2*addc;i++){
			if(j<addr||i<addc||j>=(r+addr)||i>=(c+addc))
				res[j][i]=(float)0.0;
			else
				res[j][i]=mat[j-addr][i-addc];
		}
	}
	return res;
}

float** MatEdgeShrink(float** mat,MatSize mat_size,int shrinkc,int shrinkr)
{
	int i,j;
	int c=mat_size.columns;
	int r=mat_size.rows;
	float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*));
	for(i=0;i<(r-2*shrinkr);i++)
		res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));

	
	for(j=0;j<r;j++){
		for(i=0;i<c;i++){
			if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
				res[j-shrinkr][i-shrinkc]=mat[j][i];
		}
	}
	return res;
}

void MatSaving(float** mat,MatSize mat_size,const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	int i;
	for(i=0;i<mat_size.rows;i++)
		fwrite(mat[i],sizeof(float),mat_size.columns,fp);
	fclose(fp);
}

void MatAdd(float** res, float** mat1, MatSize mat_size1, float** mat2, MatSize mat_size2)
{
	int i,j;
	if(mat_size1.columns!=mat_size2.columns||mat_size1.rows!=mat_size2.rows)
		printf("ERROR: Size is not same!");

	for(i=0;i<mat_size1.rows;i++)
		for(j=0;j<mat_size1.columns;j++)
			res[i][j]=mat1[i][j]+mat2[i][j];
}

void MatMultifactor(float** res, float** mat, MatSize mat_size, float factor)
{
	int i,j;
	for(i=0;i<mat_size.rows;i++)
		for(j=0;j<mat_size.columns;j++)
			res[i][j]=mat[i][j]*factor;
}

float MatSum(float** mat,MatSize mat_size) 
{
	float sum=0.0;
	int i,j;
	for(i=0;i<mat_size.rows;i++)
		for(j=0;j<mat_size.columns;j++)
			sum=sum+mat[i][j];
	return sum;
}