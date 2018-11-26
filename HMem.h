#ifndef HMEM_H_
#define HMEM_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

typedef enum {FALSE = 0, TRUE = 1} Boolean;

#ifdef DOUBLEANN
typedef double NFloat;
#else
typedef float NFloat;
#endif

//typedef half NFloatHalf;

typedef struct _NVector {
    size_t vecLen;
    NFloat *vecElems;   /* index starts from 0 */
    NFloat *devElems;   /* the elements on the GPU */
    //NFloatHalf *vecElemsHalf;
    //NFloatHalf *devElemsHalf;
} NVector;

typedef struct _NMatrix {
    size_t rowNum;
    size_t colNum;
    NFloat *matElems;   /* row is leading; index starts from 0 */
    NFloat *devElems;   /* the elements on the GPU */
    //NFloatHalf *matElemsHalf;
    //NFloatHalf *devElemsHalf;
} NMatrix;


NVector *CreateNVector(int nlen);
NMatrix *CreateNMatrix(int nrows, int ncols);
void SyncNVectorDev2Host(NVector *v);
void SyncNVectorHost2Dev(NVector *v);
void SyncNMatrixDev2Host(NMatrix *m);
void SyncNMatrixHost2Dev(NMatrix *m);
void ShowNVector(NVector *inVec);
void ShowNMatrix(NMatrix *inMat, int nrows);
#endif

