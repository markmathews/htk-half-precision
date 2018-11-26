#ifndef HCUDA_H_
#define HCUDA_H_

#include "HMem.h"

void StartCUDA(void);
void StopCUDA(void);
void SyncDev2Host(void *devPtr, void *hostPtr, size_t size);
void SyncHost2Dev(void *hostPtr, void *devPtr, size_t size);
void DevDispose(void *devPtr, size_t size);
Boolean DevNew(void **devAddr, size_t size);


void SelfAddNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat *lhPtr);
void SelfAddNSegmentCUDAHalf(NFloat *rhPtr, int segLen, NFloat *lhPtr);
void SelfAddNSegmentCUDAHalf2(NFloat *rhPtr, int segLen, NFloat *lhPtr);


void MulMatricesCUDA(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen);
void MulMatricesCUDAHalf(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen);
void MulMatricesCUDAHalf2(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen);

//void SelfMulNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat *lhPtr);
//void MulNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr);
#endif
