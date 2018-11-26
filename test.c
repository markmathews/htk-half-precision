#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HCUDA.h"
#include "HMem.h"

#define MAXSTRLEN 1024

NVector *ReadVector(char *path) {
    FILE *fp;
    char buf[MAXSTRLEN];
    int nlen, i;
    NVector *vec;

    /* read a vector given in path */
    if ((fp = fopen(path, "r")) == NULL) {
        printf("ERROR: Cannot open input file %s!\n", path);
        exit(1);
    }
    fscanf(fp, "%s", buf);
    if (strcmp(buf, "<VECTOR>") != 0) {
        printf("ERROR: keyword <VECTOR> rather than %s expected!\n", buf);
        exit(1);
    }
    fscanf(fp, "%d", &nlen);
    vec = CreateNVector(nlen);
    for (i = 0; i < nlen; ++i)
        fscanf(fp, "%e", &vec->vecElems[i]);

    //for (i = 0; i < nlen; ++i)
    //    &vec->vecElemsHalf[i] = approx_float_to_half((float) &vec->vecElems[i]);

    fclose(fp);
    /* synchronise PC memory to GPU memory */
    SyncNVectorHost2Dev(vec);

    return vec;
}

NMatrix *ReadMatrix(char *path) {
    FILE *fp;
    char buf[MAXSTRLEN];
    int nrows, ncols, i;
    NMatrix *mat;

    /* read a vector given in path */
    if ((fp = fopen(path, "r")) == NULL) {
        printf("ERROR: Cannot open input file %s!\n", path);
        exit(1);
    }

    fscanf(fp, "%s", buf);
    if (strcmp(buf, "<MATRIX>") != 0) {
        printf("ERROR: keyword <MATRIX> rather than %s expected!\n", buf);
        exit(1);
    }
    fscanf(fp, "%d %d", &nrows, &ncols);
    mat = CreateNMatrix(nrows, ncols);
    for (i = 0; i < nrows * ncols; ++i)
        fscanf(fp, "%e", &mat->matElems[i]);
    fclose(fp);

    //for (i = 0; i < nrows * ncols; ++i)
    //    &mat->matElemsHalf[i] = approx_float_to_half((float) &mat->matElems[i]);

    /* synchronise PC memory to GPU memory */
    SyncNMatrixHost2Dev(mat);

    return mat;
}


void main(int argc, char **argv) {
    NMatrix *mat1, *mat2, *mat3;
    clock_t start_t, end_t;
    double total_t;

    if (argc < 3) {
        printf("ERROR: Two operands are expected as two input files and one for result display\n");
        exit(1);
    }
    /* start GPU device */
    StartCUDA();

    /* load input files */
    mat1 = ReadMatrix(argv[1]);
    mat2 = ReadMatrix(argv[2]);
    if (mat1->rowNum != mat2->rowNum || mat1->colNum != mat2->colNum) {
        printf("ERROR: Input matrix dimensions not equal!\n");
        exit(1);
    }

    ShowNMatrix(mat1, mat1->rowNum);
    printf("\n");
    ShowNMatrix(mat2, mat2->rowNum);
    printf("\n");

    /* matrix to store result of multiplication */
    mat3 = CreateNMatrix(mat1->rowNum, mat2->colNum);


    /* do the operation */
    int i;
    
    //for(i = 0; i < 5; i++) {
    //SelfAddNSegmentCUDA(mat1->devElems, (int) mat1->rowNum * mat1->colNum, mat2->devElems);
    //SelfAddNSegmentCUDAHalf(mat1->devElems, (int) mat1->rowNum * mat1->colNum, mat2->devElems);
    //SelfAddNSegmentCUDAHalf2(mat1->devElems, (int) mat1->rowNum * mat1->colNum, mat2->devElems);
    //}


    for(i = 0; i < 5; i++) {
        //MulMatricesCUDA(mat1->devElems, mat2->devElems, mat3->devElems, (int) mat1->colNum, (int) mat1->rowNum * mat2->colNum); 
        //MulMatricesCUDAHalf(mat1->devElems, mat2->devElems, mat3->devElems, (int) mat1->colNum, (int) mat1->rowNum * mat2->colNum);
        MulMatricesCUDAHalf2(mat1->devElems, mat2->devElems, mat3->devElems, (int) mat1->colNum, (int) mat1->rowNum * mat2->colNum);
    }


    /* synchronise to PC memory */
    //SyncNMatrixDev2Host(mat1);
    //SyncNMatrixDev2Host(mat2);
    SyncNMatrixDev2Host(mat3);


	ShowNMatrix(mat3, mat3->rowNum);


    /* stop GPU device */
    StopCUDA();
}
