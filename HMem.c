#include "HCUDA.h"

NVector *CreateNVector(int nlen) {
    NVector *v;

    v = (NVector *) malloc(sizeof(NVector));
    /*memset(v, 0, sizeof(NVector));*/
    v->vecLen = nlen;
    v->vecElems = (NFloat *) malloc(nlen * sizeof(NFloat));
    DevNew((void **)&v->devElems, nlen * sizeof(NFloat));

    //v->vecElemsHalf = (NFloatHalf *) malloc(nlen * sizeof(NFloatHalf));
    //DevNew((void **)&v->devElemsHalf, nlen * sizeof(NFloatHalf));

    return v;
}

NMatrix *CreateNMatrix(int nrows, int ncols) {
    NMatrix *m;

    m = (NMatrix *) malloc(sizeof(NMatrix));
    /*memset(m, 0, sizeof(NMatrix));*/
    m->rowNum = nrows;
    m->colNum = ncols;
    m->matElems = (NFloat *) malloc(nrows * ncols * sizeof(NFloat));
    DevNew((void **) &m->devElems, nrows * ncols * sizeof(NFloat));
    
    //m->matElemsHalf = (NFloatHalf *) malloc(nrows * ncols * sizeof(NFloatHalf));
    //DevNew((void **) &m->devElemsHalf, nrows * ncols * sizeof(NFloatHalf));

    return m;
}



void SyncNVectorDev2Host(NVector *v) {
    SyncDev2Host(v->devElems, v->vecElems, v->vecLen * sizeof(NFloat));
    //SyncDev2Host(v->devElemsHalf, v->vecElemsHalf, v->vecLen * sizeof(NFloatHalf));
}

void SyncNVectorHost2Dev(NVector *v) {
    SyncHost2Dev(v->vecElems, v->devElems, v->vecLen * sizeof(NFloat));
    //SyncHost2Dev(v->vecElemsHalf, v->devElemsHalf, v->vecLen * sizeof(NFloatHalf));
}

void SyncNMatrixDev2Host(NMatrix *m) {
    SyncDev2Host(m->devElems, m->matElems, m->rowNum * m->colNum * sizeof(NFloat));
    //SyncDev2Host(m->devElemsHalf, m->matElemsHalf, m->rowNum * m->colNum * sizeof(NFloatHalf));
}

void SyncNMatrixHost2Dev(NMatrix *m) {
    SyncHost2Dev(m->matElems, m->devElems, m->rowNum * m->colNum * sizeof(NFloat));
    //SyncHost2Dev(m->matElemsHalf, m->devElemsHalf, m->rowNum * m->colNum * sizeof(NFloatHalf));
}



void ShowNVector(NVector *inVec) {
    int i;

    SyncNVectorDev2Host(inVec);
    for (i = 0; i < inVec->vecLen; ++i)
        printf(" %e",inVec->vecElems[i]);
    printf("\n");
}

void ShowNMatrix(NMatrix *inMat, int nrows) {
    int i, j;

    SyncNMatrixDev2Host(inMat);
    if (nrows <= 0)
        nrows = inMat->rowNum;

    for (i = 0; i < nrows; ++i) {
        printf("Line %d\n", i);
        for (j = 0; j < inMat->colNum; ++j)
            printf(" %e", inMat->matElems[i * inMat->colNum + j]);
        printf("\n");
    }
}

